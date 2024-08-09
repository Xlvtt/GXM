import torch
from typing import Type, Optional
from torch import nn
from dataset import TextDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.categorical import Categorical


# TODO Коэффициент для случайности teacher forcing в декодере
class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1, dropout: int = 0):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        self.embedding = nn.Embedding(embedding_dim=embed_size, num_embeddings=self.vocab_size)  # Обучаем Эмбеддинги вместе с моделью
        self.rnn = rnn_type(
            batch_first=True,
            input_size=embed_size, hidden_size=hidden_size,
            num_layers=rnn_layers, dropout=dropout
        )  # Всегда ожидает на вход хотя бы двумерный тензор (последовательность)
        # TODO надобавлять дропауты, слои внимания и тд
        self.linear = nn.Linear(in_features=hidden_size, out_features=self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        Уже используем teacher forcing, передавая истинные слова в качестве следующих
        """

        embeddings = self.embedding(indices)  # (Batch x Len x EmbedDim)

        embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)

        # rnn requires h0 of size = (Batch x NumLayers x EmbedDim), zeros by default
        outputs, hn = self.rnn(embeddings)
        outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
        # outputs --> (Batch x MaxLen x HiddenDim), где MaxLen - максимальная длина в батче
        # last_output = outputs[:, -1, :]

        logits = self.linear(outputs)  # logits --> (Batch x Len x VocabSize)
        print()

        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1., top_k : Optional[int] = None) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        :top_k: num of best tokens for sampling
        Мы считаем префикс началом последовательности, так что помещаем токен <bos> перед ним
        """
        self.eval()
        # encode text
        indices = [self.dataset.bos_id] + self.dataset.text2ids(prefix)  # Len
        generated_list = indices[1:]
        indices = torch.tensor(indices, dtype=torch.int64)  # Len + 1

        # generate first token
        context_embeddings = self.embedding(indices)  # Len X EmbedDim, посл контекста
        output, h = self.rnn(context_embeddings)  # output - seq of outputs, h -> NuwLayers * HiddenDim (одним вектором)
        logits = self.linear(output[-1, :])
        token = self.__sample_token(logits, top_k)

        while len(generated_list) < self.max_length:
            embedding = self.embedding(token).unsqueeze(0)  # rnn есть двумернй тензор
            output, h = self.rnn(embedding, h)
            logits = self.linear(output[-1, :]) / temp

            token = self.__sample_token(logits, top_k)
            while token == self.dataset.bos_id or token == self.dataset.unk_id:
                token = self.__sample_token(logits, top_k)  # не хотим генерировать unk и второй bos

            if token.item() == self.dataset.eos_id:
                break
            generated_list.append(token.item())

        return ''.join(self.dataset.ids2text(generated_list))

    def __sample_token(self, logits: torch.Tensor, top_k):
        if top_k is None:
            return Categorical(logits=logits).sample()
        else:
            sampling_logits = torch.topk(logits, dim=0, k=top_k)
            index = Categorical(logits=sampling_logits.values).sample()
            return sampling_logits.indices[index]