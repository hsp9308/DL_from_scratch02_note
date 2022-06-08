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


# Word2vec의 연산을 간략화하기 위한 Embedding 계층
class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout) # 중복문제 해결을 위해 해당 행에 할당이 아닌 더하기. dout를 dW의 idx번째 행에 더해줌
        # 아래와 동일한 연산을 수행함.
        # for i, word_id in enumerate(self.idx):
        #   dW[word_id] += dout[i]
        # 일반적으로 파이썬에서 for문 보다는 넘파이의 내장 메서드를 사용하는 편이 더 빠름. 
        
        return None

# 은닉 - 출력 연산 Layer.
# Embedding과는 별개로 짬 (구조가 반대라서)

class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx): # 은닉층 뉴런(h)과 단어 ID의 넘파이 배열(idx) (배열인 이유는 미니배치 처리 가정)
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1) # 내적

        self.cache = (h, target_W)
        return out

    def backward(self, dout): # 순전파의 반대순서로 기울기를 전달해 구현
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


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

# Sigmoid 함수를 Loss function으로.
# Softmax 함수를 조금 바꾸는 것으로 쉽게 구현 가능.

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None # loss function storaged.
        self.y = None  # sigmoid output
        self.t = None  # Answer label

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size

        return dx


