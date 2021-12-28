import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader

from data import Data


def metrics_sklearn(prediction, truth):
    report = classification_report(truth, prediction, labels=[0, 1], output_dict=True)
    accuracy = report.get('accuracy')
    f1_0 = report.get('0').get('f1-score')
    f1_1 = report.get('1').get('f1-score')
    precision_0 = report.get('0').get('precision')
    precision_1 = report.get('1').get('precision')
    recall_0 = report.get('0').get('recall')
    recall_1 = report.get('1').get('recall')
    f1_weighted = report.get('weighted avg').get('f1-score')
    return accuracy, f1_0, f1_1, f1_weighted, precision_0, precision_1, recall_0, recall_1


def save_checkpoint(state, path):
    torch.save(state, path)


def from_pickle_to_dataloader(path, batch_size, get_sentence_dimensions=False):
    data = Data()

    with open(path, 'rb') as file:
        data.embedded = pickle.load(file)

    # transpose list into correct format, from [[s0,s1,mt_features,score],...,[s0,s1,mt_features, score]] to [[s0s],[s1s],[mt_features],[scores]]
    ready_data = list(map(list, zip(*data.embedded)))
    x = [np.array(ready_data[0]), np.array(ready_data[1])]
    y = np.array(ready_data[2])
    tensor_x = torch.Tensor(x).permute(1, 0, 3, 2)
    tensor_y = torch.Tensor(y)
    tensor_dataset = TensorDataset(tensor_x, tensor_y)
    data_loader = DataLoader(dataset=tensor_dataset, batch_size=batch_size, shuffle=True)
    if get_sentence_dimensions:
        sentence_length, d0 = len(x[0][0][0]), len(x[0][0])
        return data_loader, sentence_length, d0
    else:
        return data_loader


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def load_checkpoint_for_test(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def update_epoch_metrics(epoch_sum, total_batch, logs, epoch_metrics, prefix=''):
    epoch_loss = float(epoch_sum.loss / total_batch)
    epoch_accuracy = float(epoch_sum.accuracy / total_batch)
    epoch_f1_0 = float(epoch_sum.f1_0 / total_batch)
    epoch_f1_1 = float(epoch_sum.f1_1 / total_batch)
    epoch_precision_0 = float(epoch_sum.precision_0 / total_batch)
    epoch_precision_1 = float(epoch_sum.precision_1 / total_batch)
    epoch_recall_0 = float(epoch_sum.recall_0 / total_batch)
    epoch_recall_1 = float(epoch_sum.recall_1 / total_batch)
    epoch_f1_weighted = float(epoch_sum.f1_weighted / total_batch)

    metrics = {prefix + 'loss': epoch_loss,
               prefix + 'accuracy': epoch_accuracy,
               prefix + 'f1_weighted': epoch_f1_weighted,
               prefix + 'f1_0': epoch_f1_0,
               prefix + 'f1_1': epoch_f1_1,
               prefix + 'precision_0': epoch_precision_0,
               prefix + 'precision_1': epoch_precision_1,
               prefix + 'recall_0': epoch_recall_0,
               prefix + 'recall_1': epoch_recall_1}

    logs.update(metrics)
    epoch_metrics.update(metrics)


def predict(X, Y, device, include_additional_features, model, loss_fn, threshold, epoch_sum, mode, optimizer):
    x0 = X[:, 0, :, :].unsqueeze(1)
    x1 = X[:, 1, :, :].unsqueeze(1)

    x0 = x0.to(device)
    x1 = x1.to(device)
    labels = Y[:, -1].unsqueeze(1)
    if include_additional_features:
        additional_features = Y[:, 0:-1]
        x2 = additional_features.to(device)
    else:
        x2 = 0
    labels = labels.to(device)

    predictions, features = model(x0, x1, x2)
    loss = loss_fn(predictions.float(), labels)
    loss = loss.to(device)

    if device != "cpu":
        predictions = predictions.cpu()
        labels = labels.cpu()

    logits = (predictions > threshold).int()
    epoch_sum.update(loss.item(), logits.numpy().flatten(), labels.numpy().flatten())

    if mode == 'train':
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class EpochSum():
    def __init__(self):
        self.loss = 0.0
        self.accuracy = 0.0
        self.f1_0 = 0.0
        self.f1_1 = 0.0
        self.precision_0 = 0.0
        self.precision_1 = 0.0
        self.recall_0 = 0.0
        self.recall_1 = 0.0
        self.f1_weighted = 0.0

    def update(self, new_loss, prediction, truth):
        accuracy, f1_0, f1_1, f1_weighted, precision_0, precision_1, recall_0, recall_1 = metrics_sklearn(
            prediction, truth)
        self.loss += new_loss
        self.accuracy += accuracy
        self.f1_0 += f1_0
        self.f1_1 += f1_1
        self.precision_0 += precision_0
        self.precision_1 += precision_1
        self.recall_0 += recall_0
        self.recall_1 += recall_1
        self.f1_weighted += f1_weighted


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif classname.find("AttentionConvInput") != -1:
        nn.init.xavier_normal_(m.W)
