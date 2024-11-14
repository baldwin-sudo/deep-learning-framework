

class Network :
    def __init__(self) -> None:
        pass
    def set_layers(self,layers_list):

        self.layers=layers_list
        self.numlayers=len(layers_list)
    def forward(self,x):
        for layer in self.layers :
            x=layer.forward(x)
        return x
    def save_model(self):
        pass
    def load_model(self):
        pass
    def reset_network(self):
        for layer in self.layers :
            layer.reset_layer()
        
    