import os

import pandas as pd
import torch
from typing import Union, List, Tuple
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor  # TODO поменяем при файнтюнинге
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
from abc import abstractmethod
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class BaseTokenizer:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def text2ids(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        pass

    @abstractmethod
    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def vocab_size(self):
        pass

    @abstractmethod
    def pad_id(self):
        pass

    @abstractmethod
    def unk_id(self):
        pass

    @abstractmethod
    def eos_id(self):
        pass

    @abstractmethod
    def bos_id(self):
        pass


class NltkTokenizer(BaseTokenizer):  # Словарь по датасету строится вот тут
    def __init__(self, dataset_path: str, min_word_freq: int = 2):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        texts = pd.read_csv(dataset_path)["problem_text"]

        self.vocab = build_vocab_from_iterator(
            iterator=self.__tokenize_texts(texts),
            min_freq=min_word_freq,
            specials=["<pad>", "<unk>", "<bos>", "<eos>"],
        )
        self.vocab.set_default_index(self.unk_id())
        self.vocab_decoder = self.vocab.get_itos()

    def __tokenize_texts(self, texts: Union[List[str], pd.Series]):
        for text in texts:
            yield self.__tokenize_text(text)

    def __tokenize_text(self, text: str) -> List[str]:
        tokenized_text = []
        for word in word_tokenize(text.lower()):
            word = self.lemmatizer.lemmatize(word)
            if str.isalpha(word) and word not in self.stop_words:
                tokenized_text.append(word)
        return tokenized_text

    def __encode_text(self, text: str) -> List[int]:
        return [self.vocab[word] for word in self.__tokenize_text(text)]  # каждому слову свой токен по словарю

    def text2ids(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        if isinstance(texts, list):
            encoded = []
            for text in texts:
                encoded.append(self.__encode_text(text))
            return encoded
        else:
            return self.__encode_text(texts)

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        if ids and isinstance(ids[0], list):
            texts = []
            for text in ids:
                texts.append(self.__decode_text(text))
            return texts
        else:
            return self.__decode_text(ids)

    def __decode_text(self, ids: List[int]) -> str:
        return ' '.join([self.vocab_decoder[ind] for ind in ids])

    def vocab_size(self):
        return len(self.vocab)

    def pad_id(self):
        return self.vocab["<pad>"]

    def unk_id(self):
        return self.vocab["<unk>"]

    def eos_id(self):
        return self.vocab["<eos>"]

    def bos_id(self):
        return self.vocab["<bos>"]


class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, vocab_size, tokenizer_model_name: str, dataset_path: str, model_type: str = "bpe",
                 normalization_rule_name: str = "nmt_nfkc_cf"):
        super().__init__()
        if not os.path.isfile(tokenizer_model_name + ".model"):
            SentencePieceTrainer.train(
                input=dataset_path, vocab_size=vocab_size,
                model_type=model_type, model_prefix=tokenizer_model_name,
                normalization_rule_name=normalization_rule_name,
                pad_id=3
            )
        self.tokenizer = SentencePieceProcessor(model_file=tokenizer_model_name + ".model")

    def text2ids(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        return self.tokenizer.encode(texts)

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()
        return self.tokenizer.decode(ids)

    def vocab_size(self):
        return self.tokenizer.vocab_size()

    def pad_id(self):
        return self.tokenizer.pad_id()

    def unk_id(self):
        return self.tokenizer.unk_id()

    def eos_id(self):
        return self.tokenizer.eos_id()

    def bos_id(self):
        return self.tokenizer.bos_id()


class MathClassificationDataset(Dataset):
    train_size = 0.85
    split_seed = 666

    def __init__(self, dataset_path: str, train: bool, max_length: int, tokenizer: BaseTokenizer):
        super().__init__()
        train_set, val_set = train_test_split(
            pd.read_csv(dataset_path), train_size=self.train_size, random_state=self.split_seed
        )
        self.texts = train_set if train else val_set
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.ind_to_class = list(self.texts["topic"].unique())
        self.num_classes = len(self.ind_to_class)
        self.class_to_ind = {classname: index for index, classname in enumerate(self.ind_to_class)}
        self.indices = self.texts.copy()
        self.indices["problem_text"] = self.indices["problem_text"].map(self.tokenizer.text2ids)
        self.indices["topic"] = self.indices["topic"].map(lambda classname: self.class_to_ind[classname])
        self.indices.reset_index(inplace=True, drop=True)

        self.pad_id, self.unk_id, self.bos_id, self.eos_id = \
            self.tokenizer.pad_id(), self.tokenizer.unk_id(), \
            self.tokenizer.bos_id(), self.tokenizer.eos_id()
        self.vocab_size = self.tokenizer.vocab_size()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        indices = list(self.indices["problem_text"][item])
        target = self.indices["topic"][item]

        if len(indices) + 2 < self.max_length:
            indices = [self.bos_id] + indices + [self.eos_id] + [self.pad_id] * (self.max_length - len(indices) - 2)
        else:
            indices = [self.bos_id] + indices[:(self.max_length - 2)] + [self.eos_id]
        return torch.tensor(indices, dtype=torch.int64), target
