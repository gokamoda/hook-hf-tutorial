# Hook を使った Transformers モデルの内部分析


## 環境構築
uv, zsh であれば以下のコマンドで環境を構築できます。
```zsh
uv sync --no-dev --extra default
```

## ブランチ
リポジトリが育っていく過程を追うために、以下のブランチを用意しています。

### `simple_get_hidden_state`
- とりあえずhookは使えるようになった
- 課題: Hookで取ってきた位置引数、出力に名前がついていないので後から使いづらい

### `hook_with_kwargs`
- Hookで取ってきた位置引数、出力に名前をつけられるようになった
- 課題: Hookを使うたびにすべての情報を取得す必要はないので非効率的

### `hook_with_result_class`
- どの情報をHook取得するかをコントロールできるようになった
- Tensorの場合はcpuメモリ上で保持するようにできるようになった
- with 構文で、`remove` を書かなくても良くなった

### `logit_lens` (応用例)
- `hook_with_result_class` を使ってLogit Lens の可視化を行う
- モデルアーキテクチャによって層の名前が異なるので、Aliasを付加して同じ名前で呼び出せるようにした