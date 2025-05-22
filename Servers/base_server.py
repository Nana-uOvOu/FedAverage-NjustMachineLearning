import socket
import threading

import torch
from torchvision import models
import torch.nn as nn
import Models.wzj_net as wzj
import utils.read_data as data_reader
class BaseServer():
    def __init__(self,
                 args  # 需要输入参数
                 ):
        self.epochs = args.global_epoches
        self.batch_size = args.batch_size
        self.learn_rate = args.learn_rate

        self.port = args.server_port
        self.max_client_num = args.max_client_num
        self.save_path = args.save_path
        self.model_name = args.model

        # 设置设备和设备号
        if args.device == "cuda":
            idx = args.device_id
            self.device = f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        print(f"Server Using device: {self.device}")
        # 获取数据集
        self.dataloader = data_reader.get_sub_dataset(args)

        data_iter = iter(self.dataloader)
        images, labels = next(data_iter)

        # print(images.shape)
        in_channels = images.shape[1]
        image_size = images.shape[2]
        # 创建模型
        if args.model == "wzj_net":
            self.model = wzj.wzJNet(in_channels, image_size, len(self.dataloader.dataset.classes))
        elif args.model == "resnet18":
            self.model = models.resnet18(pretrained=False)
            # 修改最后分类层
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, len(self.dataloader.dataset.classes))
        elif args.model == "VGG":
            self.model = models.vgg16(pretrained=False)
            # 修改最后的分类层（classifier 最后一层）
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, len(self.dataloader.dataset.classes))
        else:
            raise RuntimeError(f"{args.model} is not supported!!")

        # 加载已有模型
        if args.load_model_path is not None:
            self.model.load_state_dict(torch.load(args.load_model_path))

        # 将模型to cuda
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()














