from dataclasses import dataclass

import torch
from transformers import (
    AutoTokenizer,
)

from _transformers.model_with_alias import AutoModelForCausalLMWithAliases
from hook import AbstractResult, Hook
from utils.logger import init_logging
from utils.typing import BATCH, HIDDEN_DIM, LAYER, LAYER_PLUS_1, SEQUENCE, Tensor

logger = init_logging(__name__)


@dataclass(repr=False, init=False)
class BatchHiddenStateObservationResult(AbstractResult):
    hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


def main(
    model: AutoModelForCausalLMWithAliases,
    tokenizer: AutoTokenizer,
    prompt: str,
    positional_args_keys: list[str] = None,
    output_keys: list[str] = None,
    mid_layers=True,
) -> BatchHiddenStateObservationResult:
    """Get the inputs and outputs of a specific layer.

    Parameters
    ----------
    model : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    prompt : str
    layer_index : int, optional

    Returns
    -------
    dict
        A dictionary containing the inputs and outputs of the specified layer.
        The keys are 'args', 'kwargs', and 'output'.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    num_layers = len(model.model.layers)

    # Register the hook for last hidden layer
    last_hidden_state_hook = Hook(
        model.model.layers[-1],
        result_class=BatchHiddenStateObservationResult,
        to_cpu=True,
        output_keys=["hidden_states"],
        with_kwargs=False,
    )

    if mid_layers:
        logger.warn_once(
            "Assuming mid_layer is the sum of Attention output and input to the layer."
        )
        mid_layer_hidden_state_hooks = [
            Hook(
                model.model.layers[layer_index].self_attn,
                result_class=BatchHiddenStateObservationResult,
                to_cpu=True,
                output_keys=["hidden_states", "self_attn_weights"],
                with_kwargs=False,
            )
            for layer_index in range(num_layers)
        ]
    else:
        mid_layer_hidden_state_hooks = []

    with Hook.context([last_hidden_state_hook] + mid_layer_hidden_state_hooks):
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=1,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    hidden_states: Tensor[LAYER_PLUS_1, SEQUENCE, HIDDEN_DIM] = (
        torch.stack(outputs["hidden_states"][0]).squeeze(1).cpu()
    )
    hidden_states[-1] = last_hidden_state_hook.result.hidden_states

    if mid_layers:
        mid_layer_hidden_states: Tensor[LAYER, SEQUENCE, HIDDEN_DIM] = torch.stack(
            [
                hook.result.hidden_states.squeeze(0) + layer_input
                for hook, layer_input in zip(
                    mid_layer_hidden_state_hooks, hidden_states
                )
            ]
        )
    else:
        mid_layer_hidden_states = None

    # Get the result of the hook
    return {
        "hidden_states": hidden_states,
        "mid_layer_hidden_states": mid_layer_hidden_states,
    }


if __name__ == "__main__":
    model_name_or_path = "gpt2"
    prompt = "Tokyo is the capital of Japan."

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Load model
    model = AutoModelForCausalLMWithAliases.from_pretrained(model_name_or_path)
    model.eval()

    main(model=model, tokenizer=tokenizer, prompt=prompt, output_keys=["hidden_states"])
