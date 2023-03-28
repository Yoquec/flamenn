# Flamenn 🔥👷
Flamenn is a high-level [PyTorch](pytorch.org) wrapper for rapid model protyping.
It uses a builder pattern for creating network arquitectures using PyTorch's components.

## Example usage
```python
import torch
from flamenn.networks import MultiLayerPerceptron
from flamenn.layers import PerceptronLayer

testNN = (
    MultiLayerPerceptron(input_size=5)
    .addLayer(PerceptronLayer(5, activation=torch.nn.ReLU(), dropout=False))
    .addLayer(PerceptronLayer(3, torch.nn.ReLU(), False))
    .addCriterion(torch.nn.NLLLoss())
    .addOptim("adam", learning_rate=10e-3)
)

testNN.forward(torch.rand((1,5)))

# Result:
# >>> tensor([[0.4108, 0.2620, 0.0000]], grad_fn=<ReluBackward0>)
```
