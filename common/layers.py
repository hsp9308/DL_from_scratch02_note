import numpy as np
import sys
sys.path.append("..")
from common.functions import softmax, cross_entropy

class MatMul:
    def __init__(self, W):
        self.params = [ W ]
        self.grads = [np.zeros_like(W)]
        self.x = None
        
    def forward(self, x):
        W, = self.params
        out = np.matmul(x, W)
        self.x = x
        return out
    
    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        # ... <- deepcopy
        self.grads[0][...] = dW
        return dx

class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self,dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
class Affine:
    '''
    Affine : FNN, (input) * (weight) + (bias)  = (output)
                    X     *    W     +  b      =   Z
                  (N x M) * (M * H)  + (N, H)  = (N x H)
    '''
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
        
    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        self.x = x
        return out
    
    def backward(self, dout):
        W, _ = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T,dout)
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx
        
class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax output
        self.t = None  # Answer label

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx
