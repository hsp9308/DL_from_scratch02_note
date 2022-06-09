import sys
sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel

# BaseModel을 상속받음.
# SimpleRnnlm 을 약간 변형하고, 기능이 몇 가지 추가됨.
# 추가된 메소드 : predict, [save_params, load_params]
# [] 내의 메소드들은 BaseModel로부터 상속받음에 유의.
class Rnnlm(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # Initialization : Xavier
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx,lstm_Wh,lstm_b,stateful=True),
            TimeAffine(affine_W,affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    # predict : 주어진 시계열 시퀀스에 대해 각각의 시간 데이터의 바로 다음에 올 결과를 예측.
    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        scroe = self.predict(xs) # 1권에서처럼, predict를 구현하여, 해당 메소드의 코드 일부를 대체함.
        loss = self.loss_layer.forward(scroe, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()
