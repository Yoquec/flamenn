# Flamenn 🔥👷
Flamenn is a high-level [pytorch](pytorch.org) wrapper for rapid prototyping of models.
It uses a builder pattern for creating network arquitectures using pytorch's components.

## Example usage
```python
import torch
from flamenn.networks import MultiLayerPreceptron
from flamenn.layers import PreceptronLayer

testNN = (
    MultiLayerPreceptron(input_size=5)
    .addLayer(PreceptronLayer(5, activation="relu", dropout=False))
    .addLayer(PreceptronLayer(3, "relu", False))
    .addCriterion(torch.nn.NLLLoss())
    .addOptim("adam", learning_rate=10e-3)
)

testNN.forward(torch.rand((1,5)))

# Result:
# >>> tensor([[0.4108, 0.2620, 0.0000]], grad_fn=<ReluBackward0>)
```
