import pickle

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader

from data import Data


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
    report = classification_report(truth, prediction, labels=[0,1], output_dict=True)
    accuracy = report.get('accuracy')
    f1_0 = report.get('0').get('f1-score')
    f1_1 = report.get('1').get('f1-score')
    f1_weighted = report.get('weighted avg').get('f1-score')
    return accuracy, f1_0, f1_1, f1_weighted


def save_checkpoint(state, path):
    f_path = path + 'checkpoint.pt'
    torch.save(state, f_path)


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


def pearson_correlation_coefficient(prediction, target):
    difPred = prediction - torch.mean(prediction)
    difTarg = target - torch.mean(target)
    p = torch.sum(difPred * difTarg) / (torch.sqrt(torch.sum(difPred ** 2)) * torch.sqrt(torch.sum(difTarg ** 2)))
    return p


def load_checkpoint(path, model, optimizer):
    f_path = path + 'checkpoint.pt'
    checkpoint = torch.load(f_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']
