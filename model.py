import torch
import torch.nn as nn
from helpers import AllAP, ConvBlock, euclidean_similarity_score


class ABCNN(nn.Module):
    def __init__(self, model_type, d0, sentence_length, filter_width, conv_layers_count, conv_kernels_count, activation_fn,
                 include_input_ap):
        super(ABCNN, self).__init__()
        self.model_type = model_type
        self.d0 = d0
        self.include_input_ap = include_input_ap
        self.convs = []
        for i in range(conv_layers_count):
            if i == 0:
                self.convs.append(
                    ConvBlock(sentence_length, conv_kernels_count, filter_width, d0, activation_fn, model_type))
            else:
                self.convs.append(
                    ConvBlock(sentence_length, conv_kernels_count, filter_width, conv_kernels_count, activation_fn, model_type))
        self.convs_list = nn.ModuleList(self.convs)
        self.ap = AllAP()
        self.fc_input_shape = conv_kernels_count * conv_layers_count
        if include_input_ap:
            self.fc_input_shape += d0
        self.fc = nn.Linear(self.fc_input_shape, 1)

    def forward(self, x0, x1):
        scores = []
        if (self.include_input_ap):
            input_a0, input_a1 = self.ap(x0), self.ap(x1)
            scores.append(euclidean_similarity_score(input_a0, input_a1))
        for f in self.convs_list:
            x0, x1, a0, a1 = f(x0, x1)
            score = euclidean_similarity_score(a0, a1)
            scores.append(euclidean_similarity_score(a0, a1))
        features = torch.cat(scores, dim=1)
        output = torch.sigmoid(self.fc(features))
        return output, features
