"""Compatibility launcher for running InternVL SFT with ms-swift.

The wrapper patches a narrow import/version issue before delegating to the normal Swift training entry point."""

import runpy

from transformers.modeling_utils import PreTrainedModel


def normalize_tied_keys(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple, set)):
        return {str(key): None for key in value}
    if hasattr(value, "keys"):
        return value
    return {}


def get_all_tied_weights_keys(self):
    value = self.__dict__.get("_all_tied_weights_keys_compat", None)
    if value is None:
        value = getattr(self, "_tied_weights_keys", None)
    return normalize_tied_keys(value)


def set_all_tied_weights_keys(self, value):
    self.__dict__["_all_tied_weights_keys_compat"] = normalize_tied_keys(value)


PreTrainedModel.all_tied_weights_keys = property(
    get_all_tied_weights_keys,
    set_all_tied_weights_keys,
)


def main():
    runpy.run_module("swift.cli.sft", run_name="__main__")


if __name__ == "__main__":
    main()
