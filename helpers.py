import pickle
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

from data import STS


class AllAP(nn.Module):
    def forward(self, x):  # shape (batch_size, 1, max_length + width - 1, height)
        pool_width = x.shape[2]
        out = F.avg_pool2d(x, (pool_width, 1))  # shape (batch_size, 1, 1, height)
        out = torch.squeeze(out, dim=2)  # shape (batch_size, 1, height)
        # out = torch.squeeze(out, dim=1)  # shape (batch_size, height)
        return out  # shape (batch_size, height)


class WidthAP(nn.Module):
    def __init__(self, filter_width):
        super().__init__()
        self.wp = nn.AvgPool2d((filter_width, 1), stride=1)

    def forward(self, x):  # shape (batch_size, 1, max_length + width - 1, height)
        return self.wp(x)  # shape (batch_size, 1, max_length, height)


class ConvBlock(nn.Module):
    def __init__(self, sentence_length, conv_kern_count, filter_width, d0, activation_fn, model_type):
        super(ConvBlock, self).__init__()
        self.model_type = model_type
        if model_type == "ABCNN1" or model_type == "ABCNN3":
            self.conv = nn.Conv2d(2, conv_kern_count, kernel_size=(filter_width, d0), stride=1,
                                  padding=(filter_width - 1, 0))
        else:
            self.conv = nn.Conv2d(1, conv_kern_count, kernel_size=(filter_width, d0), stride=1,
                                  padding=(filter_width - 1, 0))
        self.activation_fn = activation_fn
        self.wp = WidthAP(filter_width)
        self.ap = AllAP()
        self.attn1 = AttentionConvInput(d0, sentence_length)
        self.attn2 = AttentionWPooling(sentence_length, filter_width)

    def forward(self, x0, x1):
        if self.model_type == "ABCNN1" or self.model_type == "ABCNN3":
            x0, x1 = self.attn1(x0, x1)
        x0, x1 = self.conv(x0), self.conv(x1)
        x0, x1 = self.activation_fn(x0), self.activation_fn(x1)
        x0, x1 = x0.permute(0, 3, 2, 1), x1.permute(0, 3, 2, 1)
        if self.model_type == "ABCNN2" or self.model_type == "ABCNN3":
            wp0, wp1 = self.attn2(x0, x1)
        else:
            wp0, wp1 = self.wp(x0), self.wp(x1)
        ap0, ap1 = self.ap(x0), self.ap(x1)
        return wp0, wp1, ap0, ap1


def euclidean_similarity_score_for_attention(x0, x1):
    return 1.0 / (1.0 + torch.norm(x0 - x1, p=2, dim=2))  # shape (batch_size, 1)


def euclidean_similarity_score(x0, x1):
    return 1.0 / (1.0 + torch.norm(x0 - x1, p=2, dim=1))


def create_attention_matrix(x0, x1):
    batch_size = x0.shape[0]
    sentence_length = x0.shape[2]
    A = torch.empty((batch_size, 1, sentence_length, sentence_length), dtype=torch.float)
    for i in range(sentence_length):
        for j in range(sentence_length):
            a0 = x0[:, :, i, :]
            a1 = x1[:, :, j, :]
            A[:, :, i, j] = euclidean_similarity_score_for_attention(a0, a1)
    return A


class AttentionConvInput(nn.Module):
    def __init__(self, input_size, sentence_length):
        super(AttentionConvInput, self).__init__()
        W0 = nn.Parameter(torch.Tensor(sentence_length, input_size))
        W1 = nn.Parameter(torch.Tensor(sentence_length, input_size))
        self.W0 = W0
        self.W1 = W1

    def forward(self, x0, x1):  # shape (batch_size, 1, max_length, input_size)
        A = create_attention_matrix(x0, x1)
        A = A.cuda() if self.W1.is_cuda else A
        A_t = A.permute(0, 1, 3, 2)

        a0 = torch.matmul(A, self.W0)
        a1 = torch.matmul(A_t, self.W1)

        f0 = torch.cat([x0, a0], dim=1)
        f1 = torch.cat([x1, a1], dim=1)
        return f0, f1  # shape (batch_size, 2, max_length, input_size)


class AttentionWPooling(nn.Module):
    def __init__(self, sentence_length, filter_width):
        super(AttentionWPooling, self).__init__()
        self.sentence_length = sentence_length
        self.filter_width = filter_width

    def forward(self, x0, x1):  # shape (batch_size, 1, max_length + width - 1, output_size)
        A = create_attention_matrix(x0, x1)

        batch_size = x0.shape[0]
        height = x0.shape[3]
        wp0 = torch.zeros((batch_size, 1, self.sentence_length, height))
        wp1 = torch.zeros((batch_size, 1, self.sentence_length, height))
        wp0 = wp0.cuda() if x0.is_cuda else wp0
        wp1 = wp1.cuda() if x1.is_cuda else wp1

        for j in range(self.sentence_length):
            for k in range(j, j + self.filter_width):
                row_sum = torch.sum(A[:, :, :, k], dim=2, keepdim=True)
                col_sum = torch.sum(A[:, :, k, :], dim=2, keepdim=True)
                row_sum = row_sum.cuda() if x0.is_cuda else row_sum
                col_sum = col_sum.cuda() if x1.is_cuda else col_sum
                wp0[:, :, j, :] += row_sum * x0[:, :, k, :]
                wp1[:, :, j, :] += col_sum * x1[:, :, k, :]
        return wp0, wp1  # shape (batch_size, 1, max_length, output_size)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif classname.find("AttentionConvInput") != -1:
        nn.init.xavier_normal_(m.W0)
        nn.init.xavier_normal_(m.W1)


def remove_stopwords_and_special_characters(x):
    stop_words = set(stopwords.words('english'))
    x_without_special_chars = re.sub('[^A-Za-z0-9 ]+', '', x)
    tokens = word_tokenize(x_without_special_chars)
    x_filtered = [w for w in tokens if not w.lower() in stop_words]
    return x_filtered


class Word2Vec():
    def __init__(self):
        # Load Google's pre-trained Word2Vec model.
        self.model = KeyedVectors.load_word2vec_format(
            'C:/Users/leva/Thesis/Embeddings/GoogleNews-vectors-negative300.bin',
            binary=True)
        self.unknowns = np.random.uniform(-0.01, 0.01, 300).astype("float32")

    def get(self, word):
        if word not in self.model.key_to_index:
            word_embedded = self.unknowns
        else:
            word_embedded = self.model.get_vector(word)
        return word_embedded


class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def metrics_sklearn(prediction, truth):
    accuracy = accuracy_score(truth, prediction)
    f1 = f1_score(truth, prediction)
    return accuracy, f1


def save_ckp(state, path):
    f_path = path + 'checkpoint.pt'
    torch.save(state, f_path)


def from_pickle_to_dataloader(path, batch_size):
    data = STS()

    with open(path, 'rb') as file:
        data.embedded = pickle.load(file)

    # transpose list into correct format, from [[s0,s1,score],...,[s0,s1,score]] to [[s0s],[s1s],[scores]]
    ready_data = list(map(list, zip(*data.embedded)))
    x = [np.array(ready_data[0]), np.array(ready_data[1])]
    y = np.array(ready_data[2])
    tensor_x = torch.Tensor(x).permute(1, 0, 3, 2)
    tensor_y = torch.Tensor(y).unsqueeze(1)
    tensor_dataset = TensorDataset(tensor_x, tensor_y)
    data_loader = DataLoader(dataset=tensor_dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def pearson_correlation_coefficient(prediction, target):
    difPred = prediction - torch.mean(prediction)
    difTarg = target - torch.mean(target)
    p = torch.sum(difPred * difTarg) / (torch.sqrt(torch.sum(difPred ** 2)) * torch.sqrt(torch.sum(difTarg ** 2)))
    return p


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']