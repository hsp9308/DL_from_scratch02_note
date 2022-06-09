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
        t = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    # backward propagation, 이후 시간대에 대한 기울기를 입력받음.
    # Output : Input 및 h_prev에 대한 기울기를 반환 (이전 layers로 backpropagation을 위해 넘겨줌)
    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)  # tanh의 미분
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
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        # Truncated BPTT이기에 미래계층에서 역전파한 기울기 dh는 0으로 초기화된다.
        dh = 0
        grads = [0, 0, 0]

        # 시간에 역순으로 역전파가 이뤄짐에 주의.
        for t in reversed(range(T)):
            layer = self.layers[t]
            # 미래 계층에서의 기울기와 위(뒤) 계층에서의 기울기가 합산되어 전파됨.
            dx, dh = layer.backward(dhs[:, t, :] + dh)
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

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
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


# LSTM 계층
class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    # forward propagation : Input (x), memory cell (c_prev) 와 hidden state (h_prev) 가 인자로 전달됨.
    def forward(self, x, h_prev, c_prev):
        '''
        parameters
        N : number of samples in the mini-batch
        H : number of hidden nodes
        '''
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        # Affine transformation (= affine layer 와 똑같은 순전파 연산 시행함.)
        A = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b

        # stack의 순서는 f,g,i,o. (note의 수식과 같음.)
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        # Element-wise multiplication and additions
        c_next = f * c_prev + g*i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        # ds : dc_next 와 dh_t에서 온 역전파 값을 더함.
        # h_t = o * tanh(c_next) 이므로,
        # ds = dc_next + dh_next * dh_next/dtanh(c_next) * dtanh(c_next)/dc_next
        # ds = dc_next + dh_next * o * (1- tanh**2).
        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)

        dc_prev = ds * f

        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g ** 2)

        # 각 게이트 및 출력에 대한 slice node를 역전파에서 옆으로 쌓기.
        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev


class TimeLSTM:
    '''
    parameters

    -stateful : 메모리 셀 상태 및 은닉상태 저장 여부. 
        -True:  T번의 시간축 전파 후 메모리 셀 상태, 은닉상태를 저장함.
        -False : 메모리 셀 및 은닉상태를 저장하지 않음. (h=0, c=0 초기화됨.)
        **** 메모리 셀 상태 또한 시간축을 따라 전파되면서, Truncated BPTT 때에도 
        **** 메모리 셀 상태가 역전파되기 때문에, h_t와 함께 c_t 또한 인스턴트 변수로 저장한다.

    Note : TimeRNN 을 약간 변형시켜서 구현함.

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
        self.c = None   # LSTM에서 추가된 메모리셀 항목
        self.stateful = stateful

    def set_state(self, h, c=None):  # 메모리 셀 상태 추가
        self.h = h
        self.c = c  # 메모리셀 추가

    def reset_state(self):
        self.h = None
        self.c = None   # 메모리셀 추가

    def forward(self, xs):
        Wx, Wh, b = self.params
        # N : mini-batch size
        # T : Number of RNN layers.
        # D : Input data vector dimension
        # H : Number of hidden nodes
        N, T, D = xs.shape
        H = Wx.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        # 처음 전파되는 경우 또는 stateful=False일 경우,
        # h를 0으로 초기화.
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:  # 메모리 셀 추가
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs   # c는 다음 레이어로 전달되지 않고, 다음 시간대의 같은 LSTM 레이어로 전파되기에 반환되지 않음.

    # backward: 윗 계층에서 역전파해온 기울기 dhs를 입력받는다.
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        # Truncated BPTT이기에 시간 구간 중 가장 마지막 시간의 직후 미래계층에서 역전파한 기울기 dh, dc는 0으로 초기화된다.
        dh = 0
        dc = 0  # 메모리 셀에 대한 기울기도 초기화.

        grads = [0, 0, 0]

        # 시간에 역순으로 역전파가 이뤄짐에 주의.
        for t in reversed(range(T)):
            layer = self.layers[t]
            # 미래 계층에서의 기울기와 위(뒤) 계층에서의 기울기가 합산되어 전파됨.
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx

            # 각 RNN 계층에서의 gradient 합산
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

# TimeDropout 계층 : 시계열 데이터에 대한 Dropout 처리


# Dropout ratio = 1 - p
# p : probability of a node to retain its value.
# 1-p : probability of a note to drop its value (value=0).
class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask
