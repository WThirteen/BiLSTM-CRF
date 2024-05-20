# encoding:utf-8
import torch
from utils import load_data
from utils import parse_tags
from utils import utils_to_train
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

word2id = load_data()[0]
max_epoch, device, train_data_loader, valid_data_loader, test_data_loader, optimizer, model = utils_to_train()


# 中文命名体识别
class ChineseNER(object):
    def train(self):
        for epoch in range(max_epoch):
            # 训练模式
            model.train()
            for index, batch in enumerate(train_data_loader):
                # 梯度归零
                optimizer.zero_grad()

                # 训练数据-->gpu
                x = batch['x'].to(device)
                mask = (x > 0).to(device)
                y = batch['y'].to(device)

                # 前向计算计算损失
                loss = model.log_likelihood(x, y, mask)

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                               max_norm=10)

                # 更新参数
                optimizer.step()
                if index % 200 == 0:
                    print('epoch:%5d,------------loss:%f' %
                          (epoch, loss.item()))

            # 验证损失和精度
            aver_loss = 0
            preds, labels = [], []
            for index, batch in enumerate(valid_data_loader):

                # 验证模式
                model.eval()

                # 验证数据-->gpu
                val_x, val_y = batch['x'].to(device), batch['y'].to(device)
                val_mask = (val_x > 0).to(device)
                predict = model(val_x, val_mask)

                # 前向计算损失
                loss = model.log_likelihood(val_x, val_y, val_mask)
                aver_loss += loss.item()

                # 统计非0的，也就是真实标签的长度
                leng = []
                # res = val_y.cpu()
                for i in val_y.cpu():
                    tmp = []
                    for j in i:
                        if j.item() > 0:
                            tmp.append(j.item())
                    leng.append(tmp)

                for index, i in enumerate(predict):
                    preds += i[:len(leng[index])]

                for index, i in enumerate(val_y.tolist()):
                    labels += i[:len(leng[index])]

            # 损失值与评测指标
            aver_loss /= (len(valid_data_loader) * 64)
            precision = precision_score(labels, preds, average='macro')
            recall = recall_score(labels, preds, average='macro')
            f1 = f1_score(labels, preds, average='macro')
            report = classification_report(labels, preds)
            print(report)
            torch.save(model.state_dict(), 'params1.data_target_pkl')
            torch.save(model,'my_model')

    # 预测，输入为单句，输出为对应的单词和标签
    def predict(self, input_str=""):
        model.load_state_dict(torch.load("params1.data_target_pkl"))
        model.eval()
        if not input_str:
            input_str = input("请输入文本: ")

        input_vec = []
        for char in input_str:
            if char not in word2id:
                input_vec.append(word2id['[unknown]'])
            else:
                input_vec.append(word2id[char])

        # convert to tensor
        sentences = torch.tensor(input_vec).view(1, -1).to(device)
        mask = sentences > 0
        paths = model(sentences, mask)

        res = parse_tags(input_str, paths[0])
        return res

    # 在测试集上评判性能
    def test(self, test_dataloader):
        model.load_state_dict(torch.load("params1.data_target_pkl"))

        aver_loss = 0
        preds, labels = [], []
        for index, batch in enumerate(test_dataloader):

            # 验证模式
            model.eval()

            # 验证数据-->gpu
            val_x, val_y = batch['x'].to(device), batch['y'].to(device)
            val_mask = (val_x > 0).to(device)
            predict = model(val_x, val_mask)

            # 前向计算损失
            loss = model.log_likelihood(val_x, val_y, val_mask)
            aver_loss += loss.item()

            # 统计非0的，也就是真实标签的长度
            leng = []
            for i in val_y.cpu():
                tmp = []
                for j in i:
                    if j.item() > 0:
                        tmp.append(j.item())
                leng.append(tmp)

            for index, i in enumerate(predict):
                preds += i[:len(leng[index])]

            for index, i in enumerate(val_y.tolist()):
                labels += i[:len(leng[index])]

        # 损失值与评测指标
        aver_loss /= len(test_dataloader)
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        f1 = f1_score(labels, preds, average='macro')
        report = classification_report(labels, preds)
        print(report)


if __name__ == '__main__':
    cn = ChineseNER()
    cn.train()
    cn.predict("寒武纪在北京举办2022春季专场招聘会")
    # cn.test()


