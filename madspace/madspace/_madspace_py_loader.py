import ctypes
import logging
import os
import platform
from collections import namedtuple

# pre-load libmadspace
ctypes.CDLL(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "lib",
        "libmadspace.dylib" if platform.system() == "Darwin" else "libmadspace.so",
    ),
    mode=ctypes.RTLD_GLOBAL,
)

from ._madspace_py import *


def _init():
    """
    Monkey-patch classes for a more pythonic experience.
    """
    set_lib_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

    def call_and_convert(runtime, args):
        if len(args) == 0:
            tensorlib = "numpy"
        else:
            tensorlib = type(args[0]).__module__
        outputs = runtime.call(args)
        # Convert outputs, lazy-loading torch or numpy
        if tensorlib == "torch":
            import torch

            return tuple(torch.from_dlpack(out) for out in outputs)
        else:
            import numpy

            return tuple(numpy.from_dlpack(out) for out in outputs)

    def runtime_call(self, *args):
        outputs = call_and_convert(self, args)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def function_call(self, *args, **kwargs):
        if not hasattr(self, "runtime"):
            self.runtime = FunctionRuntime(self)
            self.ret_tuple = namedtuple("Result", self.outputs.keys())
            self.arg_keys = self.inputs.keys()
        if len(args) == 0:
            args = [kwargs[key] for key in self.arg_keys]
        outputs = call_and_convert(self.runtime, args)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return self.ret_tuple(*outputs)

    def function_generator_call(self, *args, **kwargs):
        if not hasattr(self, "runtime"):
            func = self.function()
            self.runtime = FunctionRuntime(func)
            self.ret_tuple = namedtuple("Result", func.outputs.keys())
            self.arg_keys = func.inputs.keys()
        if len(args) == 0:
            args = [kwargs[key] for key in self.arg_keys]
        outputs = call_and_convert(self.runtime, args)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return self.ret_tuple(*outputs)

    def map_forward(self, inputs=[], conditions=[], **kwargs):
        if not hasattr(self, "forward_runtime"):
            func = self.forward_function()
            self.forward_runtime = FunctionRuntime(func)
            self.forward_tuple = namedtuple("Result", func.outputs.keys())
            self.forward_keys = func.inputs.keys()
        if len(inputs) + len(conditions) == 0:
            args = [kwargs[key] for key in self.forward_keys]
        else:
            args = [*inputs, *conditions]
        outputs = call_and_convert(self.forward_runtime, args)
        return self.forward_tuple(*outputs)

    def map_inverse(self, inputs=[], conditions=[], **kwargs):
        if not hasattr(self, "inverse_runtime"):
            func = self.inverse_function()
            self.inverse_runtime = FunctionRuntime(func)
            self.inverse_tuple = namedtuple("Result", func.outputs.keys())
            self.inverse_keys = func.inputs.keys()
        if len(inputs) + len(conditions) == 0:
            args = [kwargs[key] for key in self.inverse_keys]
        else:
            args = [*inputs, *conditions]
        outputs = call_and_convert(self.inverse_runtime, args)
        return self.inverse_tuple(*outputs)

    def tensor_numpy(tensor):
        import numpy  # Lazy-load numpy, to make it optional dependency

        return numpy.from_dlpack(tensor)

    def tensor_torch(tensor):
        import torch  # Lazy-load torch, to make it optional dependency

        return torch.from_dlpack(tensor)

    py_logger = logging.getLogger("madspace")

    def log_handler(level, message):
        match level:
            case Logger.level_debug:
                py_logger.debug(message)
            case Logger.level_info:
                py_logger.info(message)
            case Logger.level_warning:
                py_logger.warning(message)
            case Logger.level_error:
                py_logger.error(message)

    FunctionRuntime.__call__ = runtime_call
    Function.__call__ = function_call
    FunctionGenerator.__call__ = function_generator_call
    Mapping.map_forward = map_forward
    Mapping.map_inverse = map_inverse
    Tensor.numpy = tensor_numpy
    Tensor.torch = tensor_torch
    # Logger.set_log_handler(log_handler)


_init()
