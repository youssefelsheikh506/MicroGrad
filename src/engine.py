import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.prev = set(_children)
        self.op = _op
        self.label = label
        self.grad = 0.0
        self._backword = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backword():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backword = backword
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other): # 2 * Value(2.0)  --> won't work
        return self * other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backword():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backword = backword
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports int/float"
        out = Value(self.data**other, (self,), f'**{other}')
        
        def backword():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backword = backword
        return out

    def __truediv__(self, other): # a/b = a * b^-1
        return self * other**-1
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def backword():
            self.grad += (1 - t**2) * out.grad
        out._backword = backword
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def backword():
            self.grad += out.data * out.grad
        out._backword = backword
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backword():
            self.grad += (out.data > 0) * out.grad
        out._backword = _backword

        return out

    def backword(self):
        topo = []
        visited = set()
        def build_topo(cur):
          if cur not in visited:
            visited.add(cur)
            for child in cur.prev:
              build_topo(child)
            topo.append(cur)
            
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
          node._backword()