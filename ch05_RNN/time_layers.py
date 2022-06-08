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

# TimeRNN 계층 구현
# T개의 RNN 계층으로 구성됨. (T : 임의의 값. 예시에서는 10)
# 다음 T개 sequence에 넘겨줄 (그리고 역전파에서 넘겨받을) 은닉 상태 h를 인스턴스 변수로 저장.

class TimeRNN:
    '''
    parameters
    
    stateful : 은닉상태 저장 여부. 
        -True:  T번의 시간축 전파 후 은닉상태를 저장함.
        -False : 은닉상태를 저장하지 않음. (h=0 초기화됨.)
    
    '''
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        # self.layers : 다수의 RNN Layer를 저장하기 위한 인스턴스 변수
        self.layers = None
        
        # Hidden state and its gradient
        # Truncated BPTT이기에, 여기서 이전시간 역전파에 해당되는 dh는 쓰이지 않지만,
        # 7장 seq2seq에 활용하기 위해 여기서는 일단 저장한다.
        self.h, self.dh = None, None
        self.stateful = stateful
        
    def set_state(self, h):
        self.h = h
        
    def reset_state(self):
        self.h = None
        
    def forward(self, xs):
        Wx, Wh, b = self.params
        # N : mini-batch size
        # T : Number of RNN layers.
        # D : Input data vector dimension
        # H : Number of hidden nodes
        N, T, D = xs.shape
        D, H = Wx.shape
        
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')
        
        # 처음 전파되는 경우 또는 stateful=False일 경우,
        # h를 0으로 초기화.
        if not self.stateful or self.h is None:
            self.h = np.zeros((N,H), dtype='f')
            
        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:,t,:], self.h)
            hs[:,t,:] = self.h
            self.layers.append(layer)
            
        return hs
    
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape
        
        dxs = np.empty((N,T,D), dtype='f')
        # Truncated BPTT이기에 미래계층에서 역전파한 기울기 dh는 0으로 초기화된다.
        dh = 0
        grads = [0,0,0]
        
        # 시간에 역순으로 역전파가 이뤄짐에 주의.
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh) # 미래 계층에서의 기울기와 위(뒤) 계층에서의 기울기가 합산되어 전파됨.
            dxs[:, t, :] = dx
            
            # 각 RNN 계층에서의 gradient 합산
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
            
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        
        return dxs

# ============================================================
# 아래는 시계열 데이터를 한꺼번에 처리하는 layer.
# ============================================================

class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N,T,D),dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:,t,:] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:,t,:])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None

class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        # Reshaping x to rx:  to perform matrix product of rx*W
        rx = x.reshape(N*T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        # Re-reshaping it
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 정답 레이블이 원핫 벡터인 경우
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # 배치용과 시계열용을 정리(reshape)
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_label에 해당하는 데이터는 손실을 0으로 설정
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_label에 해당하는 데이터는 기울기를 0으로 설정

        dx = dx.reshape((N, T, V))

        return dx
