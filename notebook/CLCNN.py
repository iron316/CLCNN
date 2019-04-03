import os
import random

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
char_len = 150
encode_dim = 128
batch = 1028
max_epoch = 50
device = torch.device('cuda')


train_df = pd.read_csv('data/train.csv')
train_df['comment_text'] = train_df['comment_text'].astype(str)
train_df['target'] = [int(target >= 0.5) for target in train_df['target']]
#test_df = pd.read_csv('data/test.csv')
#test_df['comment_text'] = test_df['comment_text'].astype(str)
#sub = pd.read_csv('data/sample_submission.csv')


punct_mapping = {"_": " ", "`": " "}
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + \
    '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, f' {p} ')
    return text


train_df['comment_text'] = train_df['comment_text'].apply(
    lambda x: clean_special_chars(x, punct, punct_mapping))
# test_df['comment_text'] = test_df['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))


class CLCNN(nn.Module):
    def __init__(self, encode_dim, char_len,
                 n_vocab, ignore_idx, feat_num=256):
        self.encode_dim = encode_dim
        self.char_len = char_len
        super(CLCNN, self).__init__()
        self.embed = nn.Embedding(n_vocab+1, encode_dim, ignore_idx)
        self.conv1 = nn.Conv1d(in_channels=encode_dim,
                               out_channels=feat_num, kernel_size=7)
        self.conv2 = nn.Conv1d(in_channels=feat_num,
                               out_channels=feat_num, kernel_size=7)
        self.conv3 = nn.Conv1d(in_channels=feat_num,
                               out_channels=feat_num, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=feat_num,
                               out_channels=feat_num, kernel_size=3)
        self.conv5 = nn.Conv1d(in_channels=feat_num,
                               out_channels=feat_num, kernel_size=3)
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs = x.size(0)
        h = self.embed(x)
        h = h.view(bs, self.encode_dim, self.char_len)
        h = self.maxpool1(self.relu(self.conv1(h)))
        h = self.maxpool2(self.relu(self.conv2(h)))
        h = self.maxpool3(self.relu(self.conv3(h)))
        h = self.relu(self.conv4(h))
        h = self.relu(self.conv5(h))
        h = h.view(bs, 1280)
        h = self.relu(self.fc1(h))
        h = self.sigmoid(self.fc2(h))
        return h




class PrepreprocessData(data.Dataset):
    def __init__(self, pairs, char_len, char2idx, ignore_idx):
        self.pairs = pairs
        self.char_len = char_len
        self.char2idx = char2idx
        self.ignore_idx = ignore_idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        sentence, label = self.pairs[i]
        sentence = sentence.lower().strip()
        sentence = self.prepro_sentence(sentence)
        idxs = torch.LongTensor(
            [self.char2idx.get(c, self.ignore_idx) for c in sentence])
        label = np.array(label).astype(np.float32)
        return idxs, label

    def prepro_sentence(self, sentence):
        while len(sentence) < self.char_len:
            sentence += ' '
        k = random.randint(0, len(sentence)-self.char_len)
        return sentence[k:k+self.char_len]


def make_char2idx(x):
    chars = []
    for sentence in x:
        sentence = sentence.lower()
        chars += [c for c in sentence]
    char_list = list(set(chars))
    if ' ' in char_list:
        char_list.append(' ')
    char_list.sort()
    char2idx = {c: i for i, c in enumerate(char_list)}
    return char2idx, char2idx[' ']


# In[6]:


char2idx, ignore_idx = make_char2idx(train_df.comment_text.values.tolist())
train = [(x, y) for x, y in zip(train_df.comment_text, train_df.target)]
# test = test_df.comment_text.values.tolist()


# In[7]:


train_data = PrepreprocessData(train, char_len, char2idx, ignore_idx)
train_loader = data.DataLoader(dataset=train_data,
                               batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)


# In[8]:


model = CLCNN(encode_dim, char_len, len(char2idx), ignore_idx).to(device)
print(model)
loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())


for epoch in range(max_epoch):
    epoch_loss = 0.0
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device).view(-1, 1)
        predict = model(x)
        loss = loss_func(predict, y)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(
        f'epoch is {epoch+1}/{max_epoch} loss : {epoch_loss/len(train_loader)}')
