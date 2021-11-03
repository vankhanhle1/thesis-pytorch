import torch
import torch.nn as nn
import torch.nn.functional as F


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
