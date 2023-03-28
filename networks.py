"""
Created on Sun Mar 26 01:44:54 PM CEST 2023

@file: networks.py

@author: Yoquec

"""
from torch import nn, optim, Tensor
from torch import no_grad
from tqdm import tqdm
import numpy as np
from numpy._typing import NDArray
from torch.utils.data.dataloader import DataLoader
import functools
from typing import List, Callable, Union
from .layers import PerceptronLayer
from .errors import (
    CodedLayerAlreadyAssignedError,
    NoCodedLayerAssignedError,
    NoCriterionAssignedError,
    NoOptimizerAssignedError,
    CriterionAlreadyAssignedError,
    OptimizerAlreadyAssignedError,
)


# 🧠 Neural Networks -----------------------------------------------------------------
class MultiLayerPerceptron(nn.Module):
    """
    A class that represents Multi layer Perceptron (MLP)
    architectures.
    """

    def __init__(self, input_size: int) -> None:
        if not isinstance(input_size, int):
            raise ValueError(f"Invalid value {input_size} for argument input_size")
        super().__init__()
        self.dim_in = input_size
        self.module_layers = nn.ModuleList()
        self.layers: List[PerceptronLayer] = []
        self.forward_pipe: List[Callable] = []
        self.compiled_pipe: Union[Callable, None] = None
        self.loss_during_training: Union[List[float], NDArray, None] = None
        self.validation_loss_during_training: Union[List[float], NDArray, None] = None
        self._optim = None
        self._criterion = None
        self.__optimizers = ["adam"]
        return

    def _compose(self, pipe: List[Callable]) -> Callable:
        """
        Small method for building composite functions.
        Used for processing forward pipes.
        """
        return functools.reduce(lambda f, g: lambda x: g(f(x)), pipe)

    def addLayer(self, layer: PerceptronLayer):
        # ⛑  safety checks
        if self._optim is not None:
            raise ValueError(
                "⚠️ No layers can be added after having set up the optimizer"
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
        if layer.activation is not None:
            new_act_func = layer.activation
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
            raise NoOptimizerAssignedError("🛑 You haven't assigned an optimizer yet!")
        elif self._criterion is None:
            raise NoCriterionAssignedError("🛑 You haven't assigned a criterion yet!")
        else:
            return self._compile_pipe(x)

    def train(
        self,
        trainloader: DataLoader,
        epochs: int,
        validloader: Union[DataLoader, None] = None,
        loss_modifier: Callable = lambda x: x,
    ):
        """
        NOTE: loss_modifier is a function that will be used as an injected dependency to
        modify the output value of the loss function if needed before adding it to the running loss
        """
        self.loss_during_training = np.empty(epochs, dtype=float)
        self.validation_loss_during_training = (
            np.empty(epochs, dtype=float) if validloader else None
        )

        for e in tqdm(range(epochs)):
            # WARNING: How should we handle computing running_loss for non-scalar
            # loss functions?
            # TODO: For now we will introduce a loss_modifier function as parameter to let the user handle it

            # backpropagate and
            running_loss = self.propagateLoss(trainloader, loss_modifier)
            epoch_loss = running_loss / len(trainloader)

            # store the loss
            self.loss_during_training[e] = epoch_loss

            # compute validation scores
            if validloader is not None:
                validation_loss = self.computeValidationLoss(validloader, loss_modifier)
                self.validation_loss_during_training[e] = validation_loss  # type:ignore

        print(f"Training Loss: {self.loss_during_training[-1]}")
        if validloader:
            print(
                f"Validation Loss: {self.validation_loss_during_training[-1]}"  # type:ignore
            )

    def propagateLoss(self, trainloader: DataLoader, loss_modifier: Callable) -> float:
        """
        # IMPORTANT
        This is a method that should be modularized and modified by the user depending on the
        network arquitecture

        This method must load the training data of the epoch (whether in batches or not), propagate
        the loss backwards, take the needed step in the optimizer, and return a loss value.
        """
        running_loss = 0.0

        for data, labels in trainloader:
            # reset gradients
            self._optim.zero_grad()  # type:ignore

            # compute the loss
            out = self.forward(data.view(data.shape[0], -1))
            loss = self._criterion(out, labels)  # type:ignore

            # propagate the loss and take a step in the optimizer
            loss.backward()
            self._optim.step()  # type:ignore

            # Store the current batch loss
            running_loss += loss_modifier(loss).item()

        return running_loss

    def computeValidationLoss(self, validloader: DataLoader, loss_modifier) -> float:
        """
        Method that computes validation loss in the event that a validation loader
        was provided to the training method.

        #NOTE: If modified by the user, it must return a float.
        """
        running_loss = 0.0
        with no_grad():
            for data_val, labels_val in validloader:
                out_val = self.forward(data_val.view(data_val.shape[0], -1))
                loss_val = self._criterion(out_val, labels_val)  # type:ignore

                running_loss += loss_modifier(loss_val).item()
        return running_loss


class DeterministicAutoEncoder(MultiLayerPerceptron):
    """
    Class that represents Deterministic Autoencoders.
    """

    def __init__(self, input_sequence_size: int) -> None:
        # Initialize MLP methods
        super().__init__(input_sequence_size)
        self.coded_layer_index: int = -1
        self.coded_layer_index_pipe: int = -1
        self.compiled_pipe_embedding: Union[Callable, None] = None
        return

    def propagateLoss(self, trainloader: DataLoader, loss_modifier: Callable) -> float:
        """
        # IMPORTANT
        This is a method that should be modularized and modified by the user depending on the
        network arquitecture

        This method must load the training data of the epoch (whether in batches or not), propagate
        the loss backwards, take the needed step in the optimizer, and return a loss value.
        """
        running_loss = 0.0
        for sequence, _ in trainloader:
            # reset gradients
            self._optim.zero_grad()  # type: ignore

            # compute the loss (difference between an sequence and itself)
            out = self.forward(sequence.view(sequence.shape[0], -1))
            loss = self._criterion(
                out, sequence.view(sequence.shape[0], -1)
            )  # type:ignore

            # propagate the loss
            loss.backward()
            self._optim.step()  # type: ignore

            # Store the current batch loss
            running_loss += loss_modifier(loss).item()

        return running_loss

    def computeValidationLoss(
        self, validloader: DataLoader, loss_modifier: Callable
    ) -> float:
        """
        Method that computes validation difference in the case that a validation loader
        was provided to the training method.

        #NOTE: If modified by the user, it must return a float accounting for the difference
        between the original sequence and the reconstructed one.
        """
        running_loss = 0.0
        with no_grad():
            for sequence, _ in validloader:
                out = self.forward(sequence.view(sequence.shape[0], -1))
                loss = self._criterion(out, sequence.view(sequence.shape[0], -1))  # type: ignore

                running_loss += loss_modifier(loss).item()
            return running_loss

    def disimilarity(self, dataloader: DataLoader, loss_modifier: Callable) -> float:
        #TODO: RENAME THIS FUNCTION TO dissimilarity
        """
        Score function for AutoEncoders that quantifies the differences (the dissimilarity) between the original
        sequence and the reconstructed one.
        """
        disimilarity = 0.0
        with no_grad():
            for sequence, _ in dataloader:
                # get the reconstructed sequence and compute the difference
                reconstructed_seq = self.forward(sequence.view(sequence.shape[0], -1))
                loss = self._criterion(reconstructed_seq, sequence)  # type:ignore

                disimilarity += loss_modifier(loss).item()
            return disimilarity / len(dataloader)

    def addLayer(self, layer: PerceptronLayer, coded_layer: bool = False):
        # ⛑  safety checks
        if self._optim is not None:
            raise ValueError(
                "⚠️ No layers can be added after having set up the optimizer"
            )

        # add the new layer to the list of layers
        if len(self.layers) == 0:
            new_layer = nn.Linear(self.dim_in, layer.size)
        else:
            new_layer = nn.Linear(self.layers[-1].size, layer.size)

        # Check for the feature_layer flag
        if coded_layer:
            # coded layer is still unset
            if self.coded_layer_index == -1 and self.coded_layer_index_pipe == -1:
                self.coded_layer_index = len(self.layers)
                self.coded_layer_index_pipe = len(self.forward_pipe)

            # coded layer has already been set
            else:
                raise CodedLayerAlreadyAssignedError(
                    f"🛑 The auto encoder layer has already been assigned to layer {self.coded_layer_index}"
                )

        # NOTE: By adding new_layer to two lists, we are NOT creating new objects,
        # but adding references to the same object, so tensors and data will be the unique

        # append the layer dataclass to have info about the layers.
        self.layers.append(layer)
        # register the linear layer as a module
        self.module_layers.append(new_layer)
        # add it to the forward_pipe
        self.forward_pipe.append(new_layer)

        # add the proper activation function
        if layer.activation is not None:
            new_act_func = layer.activation
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

    def _pre_forward_checks(self) -> None:
        if self._optim is None:
            raise NoOptimizerAssignedError("🛑 You haven't assigned an optimizer yet!")
        elif self._criterion is None:
            raise NoCriterionAssignedError("🛑 You haven't assigned a criterion yet!")
        elif self.coded_layer_index == -1:
            raise NoCodedLayerAssignedError(
                "🛑 You still haven't assigned a coded layer.\n"
                "\t👉 Assign one by adding `coded_layer=True` in the `addLayer()` method"
            )

    def _compile_pipe_embedding(self, x: Tensor):
        # if the pipe is not yet built, build it
        if self.compiled_pipe_embedding is None:
            offset = 0
            # add offsets if needed for the activation and dropout layers
            if self.layers[self.coded_layer_index].activation:
                offset += 1
            if self.layers[self.coded_layer_index].dropout:
                offset += 1

            # build the pipe
            self.compiled_pipe_embedding = self._compose(
                # build the pipe
                self.forward_pipe[: self.coded_layer_index_pipe + 1 + offset]
            )

        return self.compiled_pipe_embedding(x)

    def forward(self, x: Tensor) -> Tensor:
        self._pre_forward_checks()
        return self._compile_pipe(x)

    def forwardEmbedding(self, x: Tensor) -> Tensor:
        self._pre_forward_checks()
        return self._compile_pipe_embedding(x)
