from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from hook import AbstractResult, ObservationHook
from utils.typing import BATCH, HIDDEN_DIM, SEQUENCE, Tensor


@dataclass(repr=False, init=False)
class BatchHiddenStateObservationResult(AbstractResult):
    hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


def main(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    positional_args_keys: list[str] = None,
    output_keys: list[str] = None,
    layer_index=0,
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

    # Register the hook on the specified layer
    hook = ObservationHook(
        model.transformer.h[layer_index],
        result_class=BatchHiddenStateObservationResult,
        positional_args_keys=positional_args_keys,
        output_keys=output_keys,
    )

    model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=1,
        do_sample=False,
    )

    # Remove the hook to prevent memory leaks
    hook.remove()

    # Get the result of the hook
    return hook.result


if __name__ == "__main__":
    model_name_or_path = "gpt2"
    prompt = "Tokyo is the capital of Japan."

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.eval()

    main(model=model, tokenizer=tokenizer, prompt=prompt, output_keys=["hidden_states"])
