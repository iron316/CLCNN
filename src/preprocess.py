from torch.utils import data
import random
import numpy as np
import torch
import emoji


class PrepreprocessData(data.Dataset):
    def __init__(self, pairs, char_len, char2idx, ignore_idx):
        self.pairs = pairs
        self.char_len = char_len
        self.char2idx = char2idx
        self.ignore_idx = ignore_idx
        punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + \
            '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
        self.clean_dict = {c: "" for c in punct}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        sentence, label = self.pairs[i]
        sentence = sentence.lower().strip()
        sentence = self.cut_sentence(sentence)
        sentence = self.clean_sentence(sentence)
        sentence = self.prepro_sentence(sentence)
        idxs = torch.LongTensor(
            [self.char2idx.get(c, self.ignore_idx) for c in sentence])
        label = np.array(label).astype(np.float32)
        return idxs, label

    def clean_sentence(self, sentence):
        # sentence = emoji.demojize(sentence)
        sentence = sentence.translate(str.maketrans(self.clean_dict))
        return sentence

    def cut_sentence(self, sentence):
        if len(sentence) < 200:
            return sentence
        else:
            k = random.randint(0, len(sentence)-200)
            return sentence[k:k+200]

    def prepro_sentence(self, sentence):
        while len(sentence) < self.char_len:
            sentence += ' '
        k = random.randint(0, len(sentence)-self.char_len)
        return sentence[k:k+self.char_len]


def make_char2idx(x):
    chars = ''
    for sentence in x:
        chars = sentence.lower().strip()
    char_list = list(set(chars))
    if ' ' in char_list:
        char_list.append(' ')
    char_list.sort()
    char2idx = {c: i for i, c in enumerate(char_list)}
    return char2idx, char2idx[' ']
