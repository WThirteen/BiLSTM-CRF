# encoding:utf-8
from TorchCRF import CRF
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from utils import load_data
import config

# 命名体识别数据
class NERDataset(Dataset):
    def __init__(self, X, Y, *args, **kwargs):
        self.data = [{'x': X[i], 'y': Y[i]} for i in range(X.shape[0])]
    # 通过索引获得数据
    def __getitem__(self, index):
        return self.data[index]
    # 返回数据的长度
    def __len__(self):
        return len(self.data)


# LSTM_CRF模型
class NERLSTM_CRF(nn.Module):
    def __init__(self,config):
        super(NERLSTM_CRF, self).__init__()
        word2id = load_data()[0]
        tag2id = load_data()[1]

        vocab_size = len(word2id)
        num_tags = len(tag2id)

        self.embedding_dim = config.embedding_dim
        self.num_tags = num_tags
        self.hidden_dim = config.hidden_dim

        self.embeds = nn.Embedding(vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # 该属性设置后，需要特别注意数据的形状
        )

        self.linear = nn.Linear(self.hidden_dim, self.num_tags)

        # CRF 层
        self.crf = CRF(self.num_tags)

    def forward(self, x, mask):
        embeddings = self.embeds(x)
        feats, hidden = self.lstm(embeddings)
        emissions = self.linear(self.dropout(feats))
        outputs = self.crf.viterbi_decode(emissions, mask)
        return outputs

    def log_likelihood(self, x, labels, mask):
        embeddings = self.embeds(x)
        feats, hidden = self.lstm(embeddings)
        emissions = self.linear(self.dropout(feats))
        loss = -self.crf.forward(emissions, labels, mask)
        return torch.sum(loss)

