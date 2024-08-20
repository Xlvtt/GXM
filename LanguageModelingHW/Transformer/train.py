import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from IPython.display import clear_output


def plot_losses(train_losses, train_accuracies, val_losses, val_accuracies):
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')

    axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='train')
    axs[1].plot(range(1, len(val_accuracies) + 1), val_accuracies, label='val')
    axs[1].set_ylabel('accuracy')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()


def training_epoch(model, loader, optimizer, criterion, tqdm_desc):
    pass  # TODO


@torch.no_grad()
def validating_epoch(model, loader, optimizer, criterion, tqdm_desc):
    pass  # TODO


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = training_epoch(
            model, train_loader, optimizer, criterion,
            tqdm_desc=f"training on epoch {epoch} / {num_epochs}"
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = validating_epoch(
            model, val_loader, optimizer, criterion,
            tqdm_desc=f"validating on epoch {epoch} / {num_epochs}"
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        plot_losses(train_losses, train_accuracies, val_losses, val_accuracies)