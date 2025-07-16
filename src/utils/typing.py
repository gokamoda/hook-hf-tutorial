from typing import Annotated

import torch
from typing_extensions import Generic, TypeVarTuple

T = TypeVarTuple("T")


class Tensor(Generic[*T], torch.Tensor):  # type: ignore
    pass


N = Annotated[int, "n"]
BATCH = Annotated[int, "batch_size"]
LAYER = Annotated[int, "layer"]
SEQUENCE = Annotated[int, "length"]
HEAD = Annotated[int, "head"]
HIDDEN_DIM = Annotated[int, "hidden_dim"]
HEAD_DIM = Annotated[int, "head_dim"]
LAYER_PLUS_1 = Annotated[int, "layer_plus_1"]
