from torch import nn
from torch.utils.hooks import RemovableHandle


class ObservationHook:
    hook: RemovableHandle
    result = None

    def __init__(
        self,
        module: nn.Module,
        positional_args_keys: list[str] = None,
        output_keys: list[str] = None,
    ):
        # Register a forward hook on the module
        self.hook = module.register_forward_hook(self.hook_fn, with_kwargs=True)

        # Register keys for positional arguments and output
        self.positional_args_keys = positional_args_keys
        self.output_keys = output_keys

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

        hook_result = {}

        # Add positional arguments to the hook result
        if self.positional_args_keys:
            assert len(self.positional_args_keys) == len(args), (
                f"Positional args length {len(args)} does not match expected "
                f"length {len(self.positional_args_keys)}."
            )
            for k, v in zip(self.positional_args_keys, args):
                assert k not in kwargs, f"Key {k} already exists in kwargs."
                hook_result[k] = v

        # Add keyword arguments to the hook result
        hook_result.update(kwargs)

        # Add output to the hook result
        if isinstance(output, tuple):
            assert len(output) == len(self.output_keys), (
                f"Output tuple length {len(output)} does not match expected "
                f"length {len(self.output_keys)}"
            )
            for k, v in zip(self.output_keys, output):
                assert k not in kwargs, f"Key {k} already exists in kwargs."
                hook_result[self.output_keys[0]] = v
        else:
            assert len(self.output_keys) == 1, (
                f"Output keys length {len(self.output_keys)} does not match expected "
                f"length 1 for single output."
            )
            hook_result[self.output_keys[0]] = output

        self.result = hook_result

    def remove(self):
        """
        Remove the hook.
        Calling this will remove the hook, and it will no longer be applied in subsequent forward calls.
        """
        self.hook.remove()
