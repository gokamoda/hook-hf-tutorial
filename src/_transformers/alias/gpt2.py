GPT2LMHEADMODEL_ALIAS = [
    ("model", "transformer"),
    ("model.embed_tokens", "transformer.wte"),
    ("model.lm_head", "transformer.lm_head"),
    ("model.layers", "transformer.h"),
    ("model.layers[*].self_attn", "transformer.h[*].attn"),
    ("model.norm", "transformer.ln_f"),
]
