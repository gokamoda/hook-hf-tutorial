from transformers import AutoModelForCausalLM, AutoTokenizer

from hook import ObservationHook


def main(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer_index=0,
) -> dict:
    """layer_index 層目の入出力を取得する

    Parameters
    ----------
    model : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    prompt : str
    layer_index : int, optional

    Returns
    -------
    dict
        入力のargs, およびkwargsと、出力のoutputを含む辞書
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    # フックを登録
    hook = ObservationHook(model.transformer.h[layer_index])

    model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=1,
        do_sample=False,
    )

    # フックを解除
    hook.remove()

    # フックの結果を取得
    return hook.result


if __name__ == "__main__":
    model_name_or_path = "gpt2"
    prompt = "Tokyo is the capital of Japan."

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.eval()

    main(model=model, tokenizer=tokenizer, prompt=prompt)
