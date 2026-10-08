"""A stand-in for torchaudio, which has no build for the NGC image's pre-release torch. ComfyUI imports it at module
level; text to video never calls it. Any attribute is a placeholder that raises when used, and submodules import."""

import importlib.abc
import importlib.machinery
import sys
import types


class _Missing:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        return _Missing(f"{self._name}.{attr}")

    def __call__(self, *a, **k):
        raise RuntimeError(f"torchaudio is a stub in the video twin's environment: {self._name} was called")

    def __mro_entries__(self, bases):
        return (object,)


class _Module(types.ModuleType):
    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Missing(f"{self.__name__}.{attr}")


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.startswith("torchaudio."):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _Module(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Finder())
__path__ = []
__version__ = "0.0.0+stub"


def __getattr__(attr):
    if attr.startswith("__"):
        raise AttributeError(attr)
    return _Missing(f"torchaudio.{attr}")
