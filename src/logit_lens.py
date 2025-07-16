import numpy as np
import plotly.graph_objects as go
import torch
from torch import nn
from torch.return_types import topk

from _transformers.model_with_alias import AutoModelForCausalLMWithAliases
from utils.typing import Tensor


class LogitLens:
    def __init__(self, model: AutoModelForCausalLMWithAliases):
        self.norm = model.model.norm
        self.lm_head = model.lm_head

    @torch.no_grad()
    def lens(self, x: Tensor, prob=True, topk=None) -> Tensor | topk:
        lens_result = self.lm_head(self.norm(x))

        if prob:
            lens_result = nn.functional.softmax(lens_result, dim=-1)

        if topk is not None:
            lens_result = torch.topk(
                lens_result, k=topk, dim=-1, largest=True, sorted=True
            )

        return lens_result


def visualize_logit_lens(
    probs: np.ndarray,
    inputs: list[str],
    decoded: list[list[str]],
    y_labels: list[str] = None,
):
    # Assume probs is of shape (LAYER, SEQUENCE, N)
    layers, sequences, top_k = probs.shape

    # We will create a heatmap for each layer showing top-k logits
    fig = go.Figure()

    # top1 data
    z_data = probs[:, :, 0]
    x_labels = inputs
    if y_labels is None:
        y_labels = np.arange(layers)
    text = [[position[0] for position in layer] for layer in decoded]

    # topk text data
    hover_text = [
        [
            "<br>".join(
                [
                    f"{probs[layer_idx][position][k]:.2f}:{decoded[layer_idx][position][k]}"
                    for k in range(top_k)
                ]
            )
            for position in range(sequences)
        ]
        for layer_idx in range(layers)
    ]

    fig.add_trace(
        go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            hovertext=hover_text,
            hovertemplate="<b>Input:</b> %{x}<br><b>Layer:</b> %{y}<br><br>%{hovertext}",
            name="",
        )
    )

    # Update layout to make the plot more readable
    fig.update_layout(
        title="Logit Lens Visualization",
        xaxis_title="Sequence",
        yaxis_title="Layer",
        height=30 * layers,  # Set a fixed height for the plot
        showlegend=False,
    )

    # Show the figure
    return fig


def prepare_data_for_logit_lens_with_mid_layer(
    logit_lens_prob_topk, logit_lens_prob_topk_mid_layer
):
    layers, sequences, topk = logit_lens_prob_topk_mid_layer.values.shape
    probs = [logit_lens_prob_topk.values[0]]
    indices = [logit_lens_prob_topk.indices[0]]
    y_labels = ["emb"]

    for layer in range(layers):
        probs.append(logit_lens_prob_topk_mid_layer.values[layer])
        probs.append(logit_lens_prob_topk.values[layer + 1])
        indices.append(logit_lens_prob_topk_mid_layer.indices[layer])
        indices.append(logit_lens_prob_topk.indices[layer + 1])
        y_labels.append(f"{layer}.mid")
        y_labels.append(f"{layer}.out")

    return torch.stack(probs), torch.stack(indices), y_labels
