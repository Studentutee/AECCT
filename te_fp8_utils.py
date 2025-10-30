
import torch
import torch.nn as nn

try:
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch import recipe as te_recipe
    _TE_AVAILABLE = True
except Exception as e:
    te = None
    te_recipe = None
    _TE_AVAILABLE = False
    _TE_ERR = e

__all__ = [
    "is_te_available",
    "build_fp8_recipe",
    "convert_linear_to_te",
    "fp8_context",
]

def is_te_available():
    return _TE_AVAILABLE, _TE_ERR if not _TE_AVAILABLE else None

def build_fp8_recipe(fmt: str = "e5m2"):
    if not _TE_AVAILABLE:
        raise RuntimeError(f"TransformerEngine not available: {_TE_ERR}")
    fmt = fmt.lower()
    if fmt not in ("e5m2", "e4m3"):
        raise ValueError("fmt must be 'e5m2' or 'e4m3'")
    fp8_format = te_recipe.Format.E5M2 if fmt == "e5m2" else te_recipe.Format.E4M3
    rec = te_recipe.DelayedScaling(
        fp8_format=fp8_format,
        amax_history_len=16,
        amax_compute_algo="max",
        margin=0
    )
    return rec

def _convert_module(m: nn.Module):
    if isinstance(m, nn.Linear):
        out_features = m.out_features
        in_features = m.in_features
        bias = m.bias is not None
        te_linear = te.Linear(in_features, out_features, bias=bias, params_dtype=m.weight.dtype)
        with torch.no_grad():
            te_linear.weight.copy_(m.weight)
            if bias:
                te_linear.bias.copy_(m.bias)
        te_linear.to(next(m.parameters()).device)
        te_linear.eval()
        return te_linear
    return None

def convert_linear_to_te(model: nn.Module):
    if not _TE_AVAILABLE:
        raise RuntimeError(f"TransformerEngine not available: {_TE_ERR}")
    for name, child in list(model.named_children()):
        new_child = _convert_module(child)
        if new_child is not None:
            setattr(model, name, new_child)
        else:
            convert_linear_to_te(child)
    return model

class fp8_context:
    def __init__(self, recipe):
        if not _TE_AVAILABLE:
            raise RuntimeError(f"TransformerEngine not available: {_TE_ERR}")
        self.recipe = recipe
        self._ctx = None

    def __enter__(self):
        self._ctx = te.fp8_autocast(enabled=True, fp8_recipe=self.recipe)
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)
