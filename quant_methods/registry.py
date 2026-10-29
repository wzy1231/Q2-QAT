# -*- coding: utf-8 -*-
from .cnn import LSQFakeQuantizer, PACTActivation, N2UQActivation
from .transformer import (
    GPLQActivationQuantizer, VVTQFakeQuantizer, QMHSA,
    QDETRQuantizer, AQDETR,
)

_METHODS = {
    "lsq": LSQFakeQuantizer,
    "pact": PACTActivation,
    "n2uq": N2UQActivation,
    "gplq": GPLQActivationQuantizer,
    "vvtq": VVTQFakeQuantizer,
    "q_mhsa": QMHSA,
    "q_detr": QDETRQuantizer,
    "aq_detr": AQDETR,
}


def available_methods():
    return tuple(sorted(_METHODS))


def build_quantizer(name, **kwargs):
    key = name.lower().replace("-", "_")
    if key not in _METHODS:
        raise KeyError(f"Unknown method {name!r}; available={available_methods()}")
    return _METHODS[key](**kwargs)
