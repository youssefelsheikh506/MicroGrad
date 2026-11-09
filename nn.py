import random
from engine import Value

class Module:
    def zero_grad(self):
       for p in self.parameters():
          p.grad = 0.0

    def paramters(self):
        return []

class Neuron(Module):
    def __init__(self, nin):
      self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
      self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
      act = sum((x1*w1 for x1, w1 in zip(self.w,x)), self.b)
      return act.tanh()

    def paramters(self):
      return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout):
      self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
      outs = [n(x) for n in self.neurons]
      return outs[0] if len(outs) == 1 else outs

    def paramters(self):
      params = []
      for neuron in self.neurons:
         ps = neuron.paramters()
         params.extend(ps)
      return params

class MLP(Module):
    def __init__(self, nin, nouts):
      # 4 3 3 1
      # (4,3) (3,3) (3,1)
      sz = [nin] + nouts
      self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]
    
    def __call__(self, x):
      for layer in self.layers:
        x = layer(x)
      return x

    def paramters(self):
      params = []
      for layers in self.layers:
        ps = layers.paramters()
        params.extend(ps)
      return params