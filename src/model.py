import torch.nn as nn


class CLCNN(nn.Module):
    def __init__(self, encode_dim, char_len,
                 n_vocab, ignore_idx, feat_num=256):
        self.encode_dim = encode_dim
        self.char_len = char_len
        super(CLCNN, self).__init__()
        self.embed = nn.Embedding(n_vocab+1, encode_dim, ignore_idx)
        self.conv1 = nn.Conv2d(in_channels=encode_dim,
                               out_channels=feat_num, kernel_size=(1, 7))
        self.conv2 = nn.Conv2d(in_channels=feat_num,
                               out_channels=feat_num, kernel_size=(1, 7))
        self.conv3 = nn.Conv2d(in_channels=feat_num,
                               out_channels=feat_num, kernel_size=(1, 3))
        self.conv4 = nn.Conv2d(in_channels=feat_num,
                               out_channels=feat_num, kernel_size=(1, 3))
        self.conv5 = nn.Conv2d(in_channels=feat_num,
                               out_channels=feat_num, kernel_size=(1, 3))
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        h = self.embed(x)
        h = self.dropout(h)
        h = h.view(-1, self.encode_dim, 1, self.char_len)
        h = self.maxpool1(self.relu(self.conv1(h)))
        h = self.maxpool2(self.relu(self.conv2(h)))
        h = self.maxpool3(self.relu(self.conv3(h)))
        h = self.relu(self.conv4(h))
        h = self.relu(self.conv5(h))
        h = h.view(-1, 1280)
        h = self.relu(self.fc1(h))
        h = self.fc2(h)
        return h
