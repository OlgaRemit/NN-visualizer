from torch import nn


class Model(nn.Module):
    def __init__(
        self,
        layers: list,  # list with number of neurons for each layer, first item - input layer, last - output(usually 1)
        activation_type: str="Tanh"
    ):
        super().__init__()
        self._layers = []
        self.activation = ACTIVATIONS.get(activation_type, nn.Tanh)
        for i, n in enumerate(layers[1:]):
            if i > 0:
                self._layers.append(self.activation())
            self._layers.append(nn.Linear(layers[i], n))
        self.model = nn.Sequential(*self._layers)


    def forward(self, x):
        return self.model(x)




ACTIVATIONS = {
    "Tanh": nn.Tanh
}
