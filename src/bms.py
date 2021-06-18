import pickle
import numpy as np
import Levenshtein


data_dir = '../input/'


def read_pickle_from_file(pickle_file):
    with open(pickle_file,'rb') as f:
        x = pickle.load(f)
    return x


def write_pickle_to_file(pickle_file, x):
    with open(pickle_file, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)


class YNakamaTokenizer(object):

    def __init__(self, is_load=True):
        self.stoi = {}
        self.itos = {}

        if is_load:
            self.stoi = read_pickle_from_file(data_dir+'/tokenizer.stoi.pickle')
            self.itos = {k: v for v, k in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def build_vocab(self, text):
        vocab = set()
        for t in text:
            vocab.update(t.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {k: v for v, k in self.stoi.items()}

    def one_text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def one_sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def one_predict_to_inchi(self, predict):
        inchi = 'InChI=1S/'
        for p in predict:
            if p == self.stoi['<eos>'] or p == self.stoi['<pad>']:
                break
            inchi += self.itos[p]
        return inchi

    def text_to_sequence(self, text):
        sequence = [
            self.one_text_to_sequence(t)
            for t in text
        ]
        return sequence

    def sequence_to_text(self, sequence):
        text = [
            self.one_sequence_to_text(s)
            for s in sequence
        ]
        return text

    def predict_to_inchi(self, predict):
        inchi = [
            self.one_predict_to_inchi(p)
            for p in predict
        ]
        return inchi


def compute_lb_score(predict, truth):
    score = []
    for p, t in zip(predict, truth):
        s = Levenshtein.distance(p, t)
        score.append(s)
    score = np.array(score)
    return score