from torch import nn
from torch.utils.hooks import RemovableHandle


class ObservationHook:
    hook: RemovableHandle
    result = None

    def __init__(self, module: nn.Module):
        # Register a forward hook on the module
        self.hook = module.register_forward_hook(self.hook_fn, with_kwargs=True)

    def hook_fn(self, module, args, kwargs, output) -> None:
        """Forward hook function to capture inputs and outputs.

        Parameters
        ----------
        module : nn.Module
            Module with hook
        args : tuple
            Position arguments passed to the module
        kwargs : dict
            Keyword arguments passed to the module
        output : Tensor
            Output of the module
        """
        self.result = {"args": args, "kwargs": kwargs, "output": output}

    def remove(self):
        """
        Remove the hook.
        Calling this will remove the hook, and it will no longer be applied in subsequent forward calls.
        """
        self.hook.remove()
