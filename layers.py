"""
Created on Sun Mar 26 07:57:43 PM CEST 2023

@file: layers.py

@author: Yoquec

@desc: File that contains layers for different kinds of Networks
"""
from typing import Callable


# 📦 Data structures -----------------------------------------------------------------
class PreceptronLayer(object):
    """
    Object that represents a Neuron layer.
    """

    def __init__(
        self,
        size: int,
        activation: Callable,
        dropout: bool = False,
        dropout_rate: float = 0.0,
    ):
        # Initial safety checks
        if (size is None) or not isinstance(size, int):
            raise ValueError(f"Value {size} is not a proper value for the size")

        self._size = size
        self._activation = activation
        self._dropout = dropout
        self._dropout_rate = 0.0 if not self.dropout else dropout_rate

    @property
    def size(self):
        """The size property."""
        return self._size

    @property
    def activation(self):
        """The activation property."""
        return self._activation

    @property
    def dropout(self):
        """The dropout property."""
        return self._dropout

    @property
    def dropout_rate(self):
        """The dropout_rate property."""
        return self._dropout_rate
