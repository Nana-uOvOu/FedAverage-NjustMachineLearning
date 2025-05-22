import time

import torch
from torchvision import models
import torch.nn as nn
import Models.wzj_net as wzj
import utils.read_data as read_data\

class BaseClient():
    def __init__(self,
                 args,  # 需要输入参数
                 ):
        self.epoches = args.client_epoches
        self.batch_size = args.batch_size
        self.learn_rate = args.learn_rate
        self.server_ip = args.server_ip
        self.server_port = args.server_port

        # 设置设备和设备号
        if args.device == "cuda":
            idx = args.device_id
            self.device = f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        print(f"Using device: {self.device}")
        # 获取数据集
        self.dataloader = read_data.get_sub_dataset(args)

        data_iter = iter(self.dataloader)
        images, labels = next(data_iter)

        # print(images.shape)
        in_channels = images.shape[1]
        image_size = images.shape[2]
        # 创建模型
        if args.model == "wzj_net":
            self.model = wzj.wzJNet(in_channels, image_size, len(self.dataloader.dataset.dataset.classes))
        elif args.model == "resnet18":
            self.model = models.resnet18(pretrained=False)
            # 修改最后分类层
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, len(self.dataloader.dataset.dataset.classes))
        elif args.model == "VGG":
            self.model = models.vgg16(pretrained=False)
            # 修改最后的分类层（classifier 最后一层）
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, len(self.dataloader.dataset.dataset.classes))
        else:
            raise RuntimeError(f"{args.model} is not supported!!")

        # 将模型to cuda
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)

    def train(self):
        self.model.train()  # 设置为训练模式

        for epoch in range(1, self.epoches + 1):
            total_loss = 0.0
            correct = 0
            total = 0
            time_1 = time.time()
            for images, labels in self.dataloader:
                # 将数据to cuda
                images, labels = images.to(self.device), labels.to(self.device)
                # 清空上一步的梯度
                self.optimizer.zero_grad()
                # 前向传播
                y_pred = self.model(images)
                # 计算损失
                loss = self.criterion(y_pred, labels)
                total_loss += loss.item()
                # 反向传播 + 参数更新
                loss.backward()
                self.optimizer.step()
                # 计算准确率
                preds = y_pred.argmax(dim=1)  # 预测的类别索引
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            time_2 = time.time()
            # 每个 epoch 输出日志
            acc = 100 * correct / total
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch}/{self.epoches} - Loss: {avg_loss:.4f} - Acc: {acc:.2f}% Time cost: {time_2 - time_1:.2f}s")








