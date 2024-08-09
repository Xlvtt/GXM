import os

import pandas as pd
import torch
from typing import Union, List, Tuple
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor  # TODO поменяем при файнтюнинге
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class MathClassificationDataset(Dataset):
    train_size = 0.85
    split_seed = 666

    def __init__(self, dataset_path, vocab_size, tokenizer_model_name: str, model_type: str = "bpe",
                 normalization_rule_name: str = "nmt_nfkc_cf", max_length: int = 128, train: bool = True):
        super().__init__()
        if not os.path.isfile(tokenizer_model_name + ".model"):
            SentencePieceTrainer.train(
                input=dataset_path, vocab_size=vocab_size,
                model_type=model_type, model_prefix=tokenizer_model_name,
                normalization_rule_name=normalization_rule_name,
                pad_id=3
            )

        self.tokenizer = SentencePieceProcessor(model_file=tokenizer_model_name + ".model")
        train_set, val_set = train_test_split(pd.read_csv(dataset_path), train_size=self.train_size, random_state=self.split_seed)
        self.texts = train_set if train else val_set

        self.ind_to_class = list(self.texts["topic"].unique())
        self.class_to_ind = {classname: index for index, classname in enumerate(self.ind_to_class)}

        self.indices = self.texts.copy()
        self.indices["problem_text"] = self.indices["problem_text"].map(self.text2ids)
        self.indices["topic"] = self.indices["topic"].map(lambda classname: self.class_to_ind[classname])

        self.pad_id, self.unk_id, self.bos_id, self.eos_id = \
            self.tokenizer.pad_id(), self.tokenizer.unk_id(), \
            self.tokenizer.bos_id(), self.tokenizer.eos_id()
        self.max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size()

    def __getitem__(self, item):
        indices = list(self.indices["problem_text"][item])
        target = self.indices["topic"][item]

        if len(indices) + 2 < self.max_length:
            length = len(indices) + 2
            indices = [self.bos_id] + indices + [self.eos_id] + [self.pad_id] * (self.max_length - len(indices) - 2)
        else:
            length = self.max_length
            indices = [self.bos_id] + indices[:(self.max_length - 2)] + [self.eos_id]
        return torch.tensor(indices, dtype=torch.int64), length, target

    def __len__(self):
        return len(self.indices)

    def text2ids(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        return self.tokenizer.encode(texts)

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()
        return self.tokenizer.decode(ids)

