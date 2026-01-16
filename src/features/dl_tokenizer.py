from collections import Counter

class SimpleTokenizer:
    def __init__(self, max_vocab=20000, pad_token="<PAD>", unk_token="<UNK>"):
        self.max_vocab = max_vocab
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {pad_token: 0, unk_token: 1}
        self.idx2word = {0: pad_token, 1: unk_token}

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        for word, _ in counter.most_common(self.max_vocab - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text):
        return [self.word2idx.get(w, 1) for w in text.split()]  # 1 = <UNK>
