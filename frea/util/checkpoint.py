#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch


def load_trusted_checkpoint(path, map_location=None):
    """Load project checkpoints that are expected to come from trusted local files."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
