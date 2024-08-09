import torch
from torch import nn
from dataset import MathClassificationDataset


class AttentionBlock(nn.Module):  # TODO multihead
    def __init__(self, embedding_dim):
        super().__init__()  # TODO good initialization + Read
        self.embedding_dim = embedding_dim
        self.Wq = torch.randn(embedding_dim, embedding_dim, requires_grad=True)
        self.Wk = torch.randn(embedding_dim, embedding_dim, requires_grad=True)
        self.Wv = torch.randn(embedding_dim, embedding_dim, requires_grad=True)

    def forward(self, x):  # TODO how to work with Len x Dim
        # Batch x Len x Dim | Len x Dim
        Q = torch.matmul(x, self.Wq)  # Batch x Len x Dim
        K = torch.matmul(x, self.Wk)  # Batch x Len x Dim
        V = torch.matmul(x, self.Wv)  # Batch x Len x Dim

        softmax = nn.Softmax(dim=2)
        attention_scores = softmax(
            torch.matmul(Q, torch.transpose(K, 1, 2)) / torch.sqrt(self.embedding_dim)  # Batch x Len x Len
        )  # Batch x Len x Len
        attention_vectors = torch.matmul(attention_scores, V)  # Batch x Len x Dim
        return torch.concat([x, attention_vectors], dim=2)  # Batch x Len x 2Dim


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()


class TransformerClassificationModel(nn.Module):
    def __init__(self, dataset: MathClassificationDataset, embedding_dim):
        super().__init__()
        self.dataset = dataset
        self.embedding_dim = embedding_dim
        
        # TODO слой, который конкатенирует позиционный эмбеддинг (как-то обучать его)

    def forward(self):
        pass


