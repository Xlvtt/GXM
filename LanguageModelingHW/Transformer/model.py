import math

import torch
from torch import nn
from dataset import MathClassificationDataset


class AttentionBlock(nn.Module):  # маска для каждого объекта (например, паддинговая)
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.Wq = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.Wk = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.Wv = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:  # x of Batch x Len x Dim
        if mask is not None:
            assert mask.shape == x.shape
        Q = self.Wq(x)  # Batch x Len x Dim
        K = self.Wk(x)  # Batch x Len x Dim
        V = self.Wv(x)  # Batch x Len x Dim

        attention_scores = nn.functional.softmax(
            torch.matmul(Q, torch.transpose(K, 1, 2)) / math.sqrt(self.embedding_dim),  # Batch x Len x Len
            dim=-1
        )  # Batch x Len x Len
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, -math.inf)
        attention_vectors = torch.matmul(attention_scores, V)  # Batch x Len x Dim
        return attention_vectors


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.heads = []
        for i in range(num_heads):
            self.heads.append(AttentionBlock(int(self.embedding_dim / self.num_heads)))
        self.linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        results = [head(x, mask) for head in self.heads]  # heads x Batch x Len x Dim
        attention_vectors = torch.concatenate(results, dim=-1)  # Batch x Len x Dim
        return self.linear(attention_vectors)


class PositionalEncoding(nn.Module):  # concat or append positional vector
    def __init__(self, embedding_dim: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        # Хотим получить вектор размерности
        pos = torch.arange(start=0, end=max_len, step=1).unsqueeze(1).repeat(1, embedding_dim)  # max_len x embedding_dim
        index = torch.arange(start=0, end=embedding_dim, step=1).unsqueeze(0).repeat(max_len, 1)  # max_len x embedding_dim
        weight = 1 / torch.exp(2 * index * (-math.log(1000) / embedding_dim))

        encoded_seq = torch.zeros(size=(max_len, embedding_dim), dtype=torch.float64)
        encoded_seq[:, 0::2] = torch.sin(pos * weight)[:, 0::2]
        encoded_seq[:, 1::2] = torch.cos(pos * weight)[:, 1::2]

        self.encoded_seq = nn.Parameter(encoded_seq, requires_grad=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = x + self.encoded_seq.unsqueeze(0).repeat(x.shape[0], 1, 1)
        return self.dropout(outputs)


def get_padding_mask(batch: torch.Tensor, pad_id: int) -> torch.Tensor:  # get mask with 0 and 1
    # batch of Batch x Len
    length = batch.shape[1]
    mask = (batch != pad_id)
    mask = mask.unsqueeze(dim=1)  # Batch x 1 x Len
    mask = mask.repeat(1, length, 1)  # Размножили каждую строку Len раз Batch x Len x Len
    return mask * mask.transpose(1, 2)  # Умножили на себя транспонированные, чтобы все аттеншены с паддингами были 0


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, pad_id: int, dropout: float = 0.1):
        super().__init__()
        self.pad_id = pad_id
        self.dropout = nn.Dropout(p=dropout)

        self.encoder_encoder_attention = MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads)
        self.first_norm = nn.LayerNorm(normalized_shape=embedding_dim)  # TODO for what
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        )
        self.second_norm = nn.LayerNorm(normalized_shape=embedding_dim)  # TODO for what

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # TODO concat with attention mode
        attention = self.encoder_encoder_attention(x, get_padding_mask(x, self.pad_id))
        attention = self.dropout(attention)

        x = self.first_norm(attention + x) + x

        feed_forward = torch.feed_forward(x)
        feed_forward = self.dropout(feed_forward)

        x = self.second_norm(feed_forward) + x
        return x


class TransformerClassificationModel(nn.Module):
    def __init__(self, dataset: MathClassificationDataset, embedding_dim: int, num_layers: int, num_heads, dropout: float = 0.1):
        super().__init__()
        self.dataset = dataset
        self.max_len = self.dataset.max_length
        self.embedding_dim = embedding_dim

        self.positional_encoder = PositionalEncoding(embedding_dim=self.embedding_dim, max_len=self.max_len)
        self.encoder_layers = []
        for i in range(num_layers):
            self.encoder_layers.append(TransformerEncoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                pad_id=self.dataset.pad_id
            ))
        self.linear = nn.Linear(in_features=embedding_dim, out_features=self.dataset.vocab_size)

    def forward(self):  # TODO
        pass

# TODO what is BAtchnorm and LayerNorm used to


class LogisticRegression(nn.Module):
    def __init__(self, dataset: MathClassificationDataset, text_embedding_dim: int):
        super().__init__()
        self.dataset = dataset
        self.text_embedding_dim = text_embedding_dim
        self.logreg = nn.Linear(in_features=text_embedding_dim, out_features=self.dataset.num_classes)

    def forward(self, x: torch.Tensor):
        return self.logreg(x)

