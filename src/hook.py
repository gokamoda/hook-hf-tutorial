import contextlib
from dataclasses import dataclass, fields

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


@dataclass
class AbstractResult:
    def __init__(self, **kwargs):
        # Get the field names from the dataclass
        field_names = {f.name for f in fields(self.__class__)}
        for key, value in kwargs.items():
            if key in field_names:
                setattr(self, key, value)

    def __repr__(self):
        """
        String to show at print.
        """
        msg = self.__class__.__name__ + ":\n"
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                msg += f"\t{k}: {v.shape}\n"
            elif isinstance(v, AbstractResult):
                msg += f"\t{k}: {v.__class__.__name__}\n"
            else:
                msg += f"\t{k}: {v}\n"
        return msg


class Hook:
    hook: RemovableHandle
    result = None

    def __init__(
        self,
        module: nn.Module,
        result_class: AbstractResult,
        positional_args_keys: list[str] = None,
        output_keys: list[str] = None,
        to_cpu: bool = True,
    ):
        # Register a forward hook on the module
        self.hook = module.register_forward_hook(self.hook_fn, with_kwargs=True)

        # Register keys for positional arguments and output
        self.positional_args_keys = positional_args_keys
        self.output_keys = output_keys

        self.result_class = result_class
        self.to_cpu = to_cpu

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
                hook_result[k] = (
                    v.cpu().clone()
                    if self.to_cpu and isinstance(v, torch.Tensor)
                    else v
                )

        # Add keyword arguments to the hook result
        hook_result.update(
            {
                k: v.cpu().clone() if self.to_cpu and isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
        )

        # Add output to the hook result
        if self.output_keys is not None:
            if isinstance(output, tuple):
                assert len(output) == len(self.output_keys), (
                    f"Output tuple length {len(output)} does not match expected "
                    f"length {len(self.output_keys)}"
                )
                for k, v in zip(self.output_keys, output):
                    assert k not in kwargs, f"Key {k} already exists in kwargs."
                    hook_result[self.output_keys[0]] = (
                        v.cpu().clone()
                        if self.to_cpu and isinstance(v, torch.Tensor)
                        else v
                    )
            else:
                assert len(self.output_keys) == 1, (
                    f"Output keys length {len(self.output_keys)} does not match expected "
                    f"length 1 for single output."
                )
                hook_result[self.output_keys[0]] = (
                    output.cpu().clone()
                    if self.to_cpu and isinstance(output, torch.Tensor)
                    else output
                )

        self.result = self.result_class(**hook_result)

    def remove(self):
        """
        Remove the hook.
        Calling this will remove the hook, and it will no longer be applied in subsequent forward calls.
        """
        self.hook.remove()

    @classmethod
    @contextlib.contextmanager
    def context(cls, hooks: "list[Hook]"):
        """Context manager to use the hook."""
        try:
            yield
        finally:
            for hook in hooks:
                hook.remove()
