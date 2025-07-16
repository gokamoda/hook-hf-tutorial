# Hook を使った Transformers モデルの内部分析


## 環境構築
uv, zsh であれば以下のコマンドで環境を構築できます。
```zsh
uv sync --no-dev --extra default
```


## `simple_get_hidden_state`
- とりあえずhookは使えるようになった
- 課題: Hookで取ってきた位置引数、出力に名前がついていないので後から使いづらい