import sys
sys.path.append('..')
import os
import pickle
from common.np import *
from common.util import to_gpu, to_cpu

# 모델 클래스 상속의 기반이 되는 기본 클래스 (base class)
# BaseModel 구현
# 각 모델에서 사용할 공통 기능인 파라미터 저장 및 불러오기 기능을 구현함.

class BaseModel:
    def __init__(self):
        self.params, self.grads = None, None

    def forward(self, *args):
        raise NotImplementedError
    
    def backward(self, *args):
        raise NotImplementedError

    def save_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'
        
        # 각 파라미터들은 16비트 부동소수점 수 (half precision)으로 저장.
        params = [p.astype(np.float16) for p in self.params]
        if GPU:
            params = [to_cpu(p) for p in params]
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'
        
        if '/' in file_name:
            file_name = file_name.replace('/',os.sep)

        if not os.path.exists(file_name):
            raise IOError('No file: ' + file_name)

        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        params = [p.astype('f') for p in params]
        if GPU:
            params = [to_gpu(p) for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]