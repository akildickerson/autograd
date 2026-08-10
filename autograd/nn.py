import random

from autograd.engine import Value


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):  # nout number of neurons in a layer
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(self.neurons) == 1 else out

    def paramaters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts):
        # nouts sizes of each layer
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))] 
        # all but the last layer have non linearities

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def paramaters(self):
        return [p for l in self.layers for p in l.paramaters()]
