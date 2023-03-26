"""
Created on Sun Mar 26 01:44:54 PM CEST 2023

@file: networks.py

@author: Yoquec

"""
from torch import nn, optim, Tensor
import functools
from typing import List, Callable, Union
from .layers import PreceptronLayer
from .errors import (
    NoCriterionAssigned,
    NoOptimizerAssigned,
    CriterionAlreadyAssignedError,
    OptimizerAlreadyAssignedError
)




# 🧠 Neural Networks -----------------------------------------------------------------
class MultiLayerPreceptron(nn.Module):
    def __init__(self, input_size: int) -> None:
        if not isinstance(input_size, int):
            raise ValueError(f"Invalid value {input_size} for argument input_size")
        super().__init__()
        self.dim_in = input_size
        self.module_layers = nn.ModuleList()
        self.layers: List[PreceptronLayer] = []
        self.forward_pipe: List[Callable] = []
        self.compiled_pipe: Union[Callable, None] = None
        self.loss_during_training: Union[List[float], None] = None
        self.validation_loss_during_training: Union[List[float], None] = None
        self._optim = None
        self._criterion = None
        self.__optimizers = ["adam"]
        return

    def _compose(self, pipe: List[Callable]) -> Callable:
        """
        Small function composition to build composite functions
        """
        return functools.reduce(lambda f, g: lambda x: g(f(x)), pipe)

    def addLayer(self, layer: PreceptronLayer):
        # ⛑  safety checks
        if self._optim is not None:
            raise ValueError(
                "⚠️ No layers can be added after having" "set up the optimizer"
            )

        # add the new layer to the list of layers
        if len(self.layers) == 0:
            new_layer = nn.Linear(self.dim_in, layer.size)
        else:
            new_layer = nn.Linear(self.layers[-1].size, layer.size)

        # NOTE: By adding new_layer to two lists, we are NOT creating new objects,
        # but adding references to the same object, so tensors and data will be the unique

        # append the layer dataclass to have info about the layers.
        self.layers.append(layer)
        # register the linear layer as a module
        self.module_layers.append(new_layer)
        # add it to the forward_pipe
        self.forward_pipe.append(new_layer)

        # add the proper activation function
        if layer.activation == "relu":
            new_act_func = nn.ReLU()
        elif layer.activation == "logsoftmax":
            new_act_func = nn.LogSoftmax(dim=1)
        else:  # back up with a Relu
            new_act_func = nn.ReLU()

        # add the activation function to the pipe to the pipe
        self.forward_pipe.append(new_act_func)

        # TODO: Think about adding activation functions as modules (they get repeated)
        # same thing for the dropout
        # self.module_act_funcs.append(new_act_func)

        # add dropout regularization
        if layer.dropout:
            self.forward_pipe.append(nn.Dropout(p=layer.dropout_rate))

        return self

    def addCriterion(self, criterion: Callable):
        if self._criterion is not None:
            raise CriterionAlreadyAssignedError(
                f"🛑 There is already an assigned criterion ({type(self._criterion)})"
            )
        else:
            self._criterion = criterion

        return self

    def addOptim(self, optimizer: "str", learning_rate: float):
        if self._optim is not None:
            raise OptimizerAlreadyAssignedError(
                f"🛑 There is already an optimizer set ({type(self._optim)})"
            )

        # set the optimizer
        if optimizer.lower() not in self.__optimizers:
            raise ValueError(
                "Optimizer {optimizer} is not available."
                f"Options are: {self.__optimizers}"
            )
        elif optimizer.lower() == "adam":
            self._optim = optim.Adam(self.parameters(), learning_rate)
        else:
            raise ValueError(
                "Optimizer {optimizer} is not available."
                f"Options are: {self.__optimizers}"
            )
        return self

    def _compile_pipe(self, x: Tensor):
        """
        Method that forwards an input tensor through
        a pipe built using functool's reduce.

        If the pipe hasn't been "compiled" yet, it will compile
        it and use it.
        """
        if self.compiled_pipe is None:
            self.compiled_pipe = self._compose(self.forward_pipe)

        return self.compiled_pipe(x)

    def forward(self, x: Tensor) -> Tensor:
        """
        Method that forwards an input vector through the network
        """
        if self._optim is None:
            raise NoOptimizerAssigned("🛑 You haven't assigned an optimizer yet!")
        elif self._criterion is None:
            raise NoCriterionAssigned("🛑 You haven't assigned a criterion yet!")
        else:
            return self._compile_pipe(x)
