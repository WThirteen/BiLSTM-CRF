from sklearn.model_selection import train_test_split
import numpy as np
import config
tag2id = {"PAD":0, "B-LOC":1,"I-LOC":2, "B-PER":3, "I-PER":4, "B-ORG":5, "I-ORG":6, "O":7}
id2tag = {v:k for k, v in tag2id.items()}


def load_text(name, word2id=None):
    lines = []
    if not word2id:
        word2id, flag = {"pad":0, "unk":1}, True
    else:
        flag = False

    with open(name, encoding="utf-8") as fp:
        buff = []
        for line in fp:
            if not line.strip():
                lines.append(buff)
                buff = []
            else:
                buff.append(line.strip())

    x_train, y_train, maxlen= [],[], 120

    for line in lines:
        x, y = [], []
        for v in line:
            w,t = v.split()
            if w not in word2id and flag:
                word2id[w]=len(word2id)

            if t not in tag2id:
                tag2id[t]=(len(tag2id))

            x.append(word2id.get(w, 1))
            y.append(tag2id[t])

        if len(x)<maxlen:
            x = x+[0]*(maxlen-len(x))
            y = y+[0]*(maxlen-len(y))

        x = x[:maxlen]
        y = y[:maxlen]
        x_train.append(x)
        y_train.append(y)

    return np.array(x_train), np.array(y_train), word2id


def load_data():

    x_train, y_train, word2id = load_text(config.path_train)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
    # print("train:", x_train.shape, y_train.shape)
    # print("valid:", x_valid.shape, y_valid.shape)
    x_test, y_test, _ = load_text(config.path_test, word2id)

    # print("test len:", len(x_test))
    print(id2tag)

    return word2id, tag2id, x_train, x_test, x_valid, y_train, y_test, y_valid, id2tag


def main():
    word = load_data()
    print(len(word))

def part_word():
    line_word=[]




if __name__ == '__main__':
    main()