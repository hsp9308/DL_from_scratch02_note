from common.base_model import BaseModel
from common.np import *
from common.time_layers import *
import sys
sys.path.append('..')

# BaseModel을 상속받음.
# SimpleRnnlm 을 약간 변형하고, 기능이 몇 가지 추가됨.
# 추가된 메소드 : predict, [save_params, load_params]
# [] 내의 메소드들은 BaseModel로부터 상속받음에 유의.
#
# BetterRnnlm : 2개의 LSTM, 3개의 dropout layer가 특징.
# embedding과 affine의 weight matrix가 공유됨.


class BetterRnnlm(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100,
                 dropout_ratio=0.5):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # Initialization : Xavier
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4 * H).astype('f')
        # <- 1개의 원소 갯수가 바뀜에 주의.
        lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')
# Weight가 공유되기에, 불필요한 변수는 주석처리함.
#        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b)  # Embedding과 가중치 공유!
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    # predict : 주어진 시계열 시퀀스에 대해 각각의 시간 데이터의 바로 다음에 올 결과를 예측.
    def predict(self, xs, train_flg=False):
        # If train_flg is True, dropout is activated.
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts, train_flg=True):
        scroe = self.predict(xs, train_flg)  # 1권에서처럼, predict를 구현하여, 해당 메소드의 코드 일부를 대체함.
        loss = self.loss_layer.forward(scroe, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()
