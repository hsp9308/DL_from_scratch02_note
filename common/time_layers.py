from common.np import *
from common.layers import *
from common.functions import sigmoid

# 단일 RNN class : 이전 time의 값을 받아 1회 순전파 및 역전파를 하는 class.
class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
        
    # Forward propagation
    # h_prev, 전 시간대의 입력을 받는다.
    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.matmul(x,Wx) + np.matmul(h_prev,Wh) + b
        h_next = np.tanh(t)
        
        self.cache = (x, h_prev, h_next)
        return h_next
    
    # backward propagation, 이후 시간대에 대한 기울기를 입력받음.
    # Output : Input 및 h_prev에 대한 기울기를 반환 (이전 layers로 backpropagation을 위해 넘겨줌)
    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache
        
        dt = dh_next * (1 - h_next ** 2) # tanh의 미분
        db = np.sum(dt, axis=0)
        dWh = np.matmul(h_prev.T, dt)
        dh_prev = np.matmul(dt, Wh.T)
        dWx = np.matmul(x.T, dt)
        dx = np.matmul(dt, Wx.T)
        
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        
        return dx, dh_prev
