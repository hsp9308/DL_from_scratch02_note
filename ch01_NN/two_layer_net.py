# Two layer net : Input - Hidden - Output
import sys, os 
sys.path.append(os.pardir)
from common.layers import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        W1 = np.random.randn(I, H)*0.01
        b1 = np.zeros(H)
        W2 = np.random.randn(H, O)*0.01
        b2 = np.zeros(O)

        self.layers = [Affine(W1, b1), 
                        Sigmoid(),
                        Affine(W2,b2)]

        self.loss_layer = SoftmaxWithLoss()


        # Weight and bias for each layer storaged in params LIST. 
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout