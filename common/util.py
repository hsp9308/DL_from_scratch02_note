import numpy as np

def preprocess(text):
    text = text.lower()    
    text = text.replace('.', ' .')
    words = text.split(' ')
    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix =  np.zeros((corpus_size,corpus_size), dtype=np.int32)

    for ind, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_ind = ind - i
            right_ind = ind + i

            if left_ind >= 0:
                left_word_id = corpus[left_ind]
                co_matrix[word_id, left_word_id] += 1
            if right_ind < corpus_size:
                right_word_id = corpus[right_ind]
                co_matrix[word_id,right_word_id] += 1
        
    return co_matrix

def cos_similarity(x, y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)

    return np.dot(nx,ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print("no word %s found." % query )
        return
    
    print('\n[query] ' + query)
    query_id = word_to_id[query]
    # query word에 해당하는 행 (= vector) 가져오기
    query_vec = word_matrix[query_id]
    
    # Cosine similarity
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    
    # top 값 만큼 상위 단어들을 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print('%s: %s'%(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return

def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i,j] * N / (S[j]*S[i]) +eps)
            M[i,j] = max(0,pmi)

            if verbose:
                cnt += 1
                if cnt % (total/100 + 1) == 0:
                    print('%.1f%% 완료' % (100*cnt/total))

    return M

def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size+1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)

def convert_one_hot(corpus, vocab_size):
    '''원핫 표현으로 변환
    :param corpus: 단어 ID 목록(1차원 또는 2차원 넘파이 배열)
    :param vocab_size: 어휘 수
    :return: 원핫 표현(2차원 또는 3차원 넘파이 배열)
    '''
    N = corpus.shape[0]

    one_hot = []

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    # 맥락의 경우, 2차원이기 때문에 2차원 원소를 각각 변환해야 함.
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot