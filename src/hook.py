from torch import nn
from torch.utils.hooks import RemovableHandle


class ObservationHook:
    hook: RemovableHandle
    result = None

    def __init__(self, module: nn.Module):
        # フック関数を登録する
        self.hook = module.register_forward_hook(self.hook_fn, with_kwargs=True)

    def hook_fn(self, module, args, kwargs, output) -> None:
        """module の forward 関数の入出力を取得する

        Parameters
        ----------
        module : nn.Module
            フックを登録したモジュール
        args : tuple
            モジュールに渡された位置引数
        kwargs : dict
            モジュールに渡されたキーワード引数
        output : Tensor
            モジュールの出力
        """
        self.result = {"args": args, "kwargs": kwargs, "output": output}

    def remove(self):
        """
        フックを削除する
        これを呼び出すと、フックが削除され、以降の forward 呼び出しではフックが適用されなくなる。
        """
        self.hook.remove()
