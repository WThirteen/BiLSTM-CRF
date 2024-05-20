# encoding:utf-8

import torch
import torch.nn as nn
import torch.optim as op
from torch.utils.data import DataLoader
from read_file_txt import load_data
from bilstm_crf_model import NERDataset
from bilstm_crf_model import NERLSTM_CRF
import config

word2id, tag2id, x_train, x_test, x_valid, y_train, y_test, y_valid, id2tag = load_data()


# 用于将实体类别解码，单字组合成单词
def parse_tags(text, path):
    tags = [id2tag[idx] for idx in path]

    begin = 0
    end = 0

    res = []
    for idx, tag in enumerate(tags):
        # 将连续的 同类型 的字连接起来
        if tag.startswith("B"):
            begin = idx
        elif tag.startswith("I"):
            end = idx
            word = text[begin:end + 1]
            label = tag[2:]
            res.append((word, label))
        elif tag=='O':
            res.append((text[idx], tag))
    return res




def utils_to_train():

    device = torch.device('cpu')
    train_dataset = NERDataset(x_train, y_train)
    valid_dataset = NERDataset(x_valid, y_valid)
    test_dataset = NERDataset(x_test, y_test)

    train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_data_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    model = NERLSTM_CRF(config).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = op.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    max_epoch = config.max_epoch

    return max_epoch, device, train_data_loader, valid_data_loader, test_data_loader, optimizer, model
