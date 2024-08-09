# TODO FastApi

import gradio as gr
from model import LanguageModel
from dataset import TextDataset
import torch


def generate_joke(prefix: str) -> str:
    tensors = torch.load("lstm_checkpoint.pt")
    vocab_size = 6000

    dataset = TextDataset(data_file='mega_jokes_dataset.txt', train=True, sp_model_prefix='bpe', vocab_size=vocab_size)
    model = LanguageModel(
        dataset,
        embed_size=tensors["embed_size"].item(),
        hidden_size=tensors["hidden_size"].item(),
        rnn_layers=tensors["rnn_layers"].item(),
        dropout=tensors["dropout"].item(),
        rnn_type=torch.nn.LSTM
    )
    model.load_state_dict(tensors["model_state_dict"])

    return model.inference(prefix)


prefix_input = gr.Textbox(label="Введи начала анекдота")
output = gr.Textbox(label="Получи фашист гранату")

app = gr.Interface(fn=generate_joke, inputs=prefix_input, outputs=output)
app.launch()
