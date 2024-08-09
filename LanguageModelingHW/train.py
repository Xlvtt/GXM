import torch
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm
from model import LanguageModel


sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})


# TODO USE TENSORBOARD
# TODO USE DEVICE
def plot_losses(train_losses: List[float], val_losses: List[float]):
    """
    Plot loss and perplexity of train and validation samples
    :param train_losses: list of train losses at each epoch
    :param val_losses: list of validation losses at each epoch
    """
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')

    train_perplexities = torch.exp(torch.tensor(train_losses, dtype=torch.float64))
    val_perplexities = torch.exp(torch.tensor(val_losses, dtype=torch.float64))

    axs[1].plot(range(1, len(train_perplexities) + 1), train_perplexities, label='train')
    axs[1].plot(range(1, len(val_perplexities) + 1), val_perplexities, label='val')
    axs[1].set_ylabel('perplexity')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()

def training_epoch(model: LanguageModel, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str):
    """
    Process one training epoch
    :param model: language model to train
    :param optimizer: optimizer instance
    :param criterion: loss function class
    :param loader: training dataloader
    :param tqdm_desc: progress bar description
    :return: running train loss
    """
    device = next(model.parameters()).device
    train_loss = 0.0

    model.train()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        optimizer.zero_grad()
        logits = model(indices, lengths)[:, :-1, :]  # Batch x (MaxLen-1) x VocabSize, не берем bos
        logits = torch.transpose(logits, 1, 2)  # Batch x VocabSize x (MaxLen-1) - такова воля CrossEntropyLoss

        y_true = indices[:, 1:logits.shape[2]+1]  # Batch x Len-1, тут отрезаем eos
        loss = criterion(logits, y_true)  # Получаем желаемое соответствие
        loss.backward()
        optimizer.step()

        train_loss += loss.item()  # лосс на батче

    train_loss /= len(loader)  # Усредняем лосс по числу батчей
    return train_loss


@torch.no_grad()
def validation_epoch(model: LanguageModel, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):
    """
    Process one validation epoch
    :param model: language model to validate
    :param criterion: loss function class
    :param loader: validation dataloader
    :param tqdm_desc: progress bar description
    :return: validation loss
    """
    device = next(model.parameters()).device
    val_loss = 0.0

    model.eval()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        logits = model(indices, lengths)[:, :-1, :]  # Batch x Len x VocabSize
        logits = torch.transpose(logits, 1, 2)  # Batch x VocabSize x (MaxLen-1) - такова воля CrossEntropyLoss

        y_true = indices[:, 1:logits.shape[2]+1]
        loss = criterion(logits, y_true)

        val_loss += loss.item()

    val_loss /= len(loader)
    return val_loss


def train(model: LanguageModel, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int,
          saving_path: Optional[str] = None, num_examples: int = 5):
    """
    Train language model for several epochs
    :param model: language model to train
    :param optimizer: optimizer instance
    :param scheduler: optional scheduler
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param num_epochs: number of training epochs
    :param num_examples: number of generation examples to print after each epoch
    :param saving_path
    """
    train_losses, val_losses = [], []
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_id)

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        val_losses += [val_loss]
        plot_losses(train_losses, val_losses)

        if saving_path is not None:
            state_dict = {
                "epoсh": epoch,  # Последняя завершенная эпоха
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': torch.tensor(train_losses),
                'val_losses': torch.tensor(val_losses),
                'embed_size': torch.tensor(model.embedding.embedding_dim, dtype=torch.int64),
                'hidden_size': torch.tensor(model.rnn.hidden_size, dtype=torch.int64),
                'rnn_layers': torch.tensor(model.rnn.num_layers, dtype=torch.int64),
                'dropout': torch.tensor(model.rnn.dropout)
            }
            if scheduler is not None:
                state_dict['scheduler'] = scheduler.state_dict()
            torch.save(state_dict, saving_path)

        generate_examples(model, num_examples)


def generate_examples(model: torch.nn.Module, num_examples: int = 5):
    print('Generation examples:')
    for i in range(num_examples):
        print(f"{i + 1}.", model.inference())