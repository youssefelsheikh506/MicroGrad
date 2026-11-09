# Micrograd (from scratch)

A minimal autograd engine and neural network library implemented from scratch in Python, inspired by Andrej Karpathy's micrograd. It builds a computation graph of scalar `Value` objects, supports backpropagation via the chain rule, and trains small MLPs.

## Features

- Scalar autograd with a dynamically built DAG
- Reverse-mode differentiation via topological sort
- Ops: `+`, `-`, `*`, `/`, `**`, `tanh`, `exp`, `relu`
- Tiny NN library: `Neuron`, `Layer`, `MLP`
- No dependencies beyond the Python standard library

## Files

- **`engine.py`** — the autograd core. Defines `Value`, a scalar that tracks its data, gradient, and the operation that produced it. Each op stores a local backward closure. Calling `backward()` topologically sorts the graph and accumulates gradients into every node.
- **`nn.py`** — the neural network layer: `Module`, `Neuron`, `Layer`, `MLP`. Each `Neuron` computes `tanh(w·x + b)` and exposes its parameters for optimization.

## Quick start

```python
from engine import Value

x = Value(2.0)
y = Value(3.0)
z = x * y + x          # z.data == 8.0
z.backward()
print(x.grad, y.grad)  # dz/dx, dz/dy
```

## Train a small MLP

```python
import random
from engine import Value
from nn import MLP

random.seed(42)

xs = [
    [2.0, 3.0, -1.0],
    [3.0, 5.0, 1.0],
    [-2.0, 1.0, 1.5],
    [0.5, -0.5, 0.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

n = MLP(3, [4, 4, 1])   # 3 inputs -> two hidden layers of 4 -> 1 output

for step in range(100):
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    n.zero_grad()
    loss.backward()

    for p in n.parameters():
        p.data -= 0.05 * p.grad

    if step % 10 == 0:
        print(f"step {step:3d}  loss {loss.data:.4f}")
```

## Supported operations

| Operation | Forward | Local gradient |
|-----------|---------|----------------|
| `+`  | `a + b` | `1` w.r.t. each operand |
| `*`  | `a * b` | `b` w.r.t. `a`, `a` w.r.t. `b` |
| `**` | `a ** n` | `n * a**(n-1)` (constant exponent) |
| `/`  | `a * b**-1` | composed from `*` and `**` |
| `tanh` | `tanh(x)` | `1 - tanh(x)**2` |
| `exp`  | `e**x` | `e**x` |
| `relu` | `max(0, x)` | `1` if `x > 0` else `0` |

## Design notes

- Every value is a **scalar** — the library is intentionally tiny and slow, built for understanding, not performance.
- Backprop uses a **topological sort** of the graph, then applies each node's local backward in reverse order. Gradients are **accumulated** (`+=`), which correctly handles a node used multiple times in the graph.
- `MLP.parameters()` flattens all weights and biases so an optimizer can update them.

## Requirements

Python 3.8+. Only the standard library (`math`, `random`).

## License

MIT