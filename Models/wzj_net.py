import torch.nn as nn
import torch.nn.functional as F


class wzJNet(nn.Module):
    def __init__(self, in_channel, in_size, class_num, *args, **kwargs):
        # 存储原始图像大小和类别数
        super().__init__(*args, **kwargs)
        self.in_channel = in_channel
        self.in_size = in_size
        self.class_num = class_num
        self.proto_dim = 512


        # 设计一个网络
        # 这3层，特征图尺寸不变，通道数改变，提取基本特征。
        self.layer1 = nn.Sequential(
            # 3通道→32;
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32→64;图像尺寸不变
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64→128;图像尺寸不变
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )


        # 这3层，特征图通道数不变，尺寸改变，提取深层特征
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        in_size /= 2

        # 最后3层，混合MaxPool和Conv2d，提取高层语义特征
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 计算当前图片尺寸
        in_size /= 2
        in_size /= 2
        # 全连接分类
        self.fully_connected = nn.Linear(in_features=int(512 * in_size * in_size), out_features=class_num)


    def forward(self, x, proto = False):
        x = self.layer1(x)
        x = self.layer2(x) # 加上残差连接玩玩看
        x = self.layer3(x)
        # 忘了展开报错了 记得展开
        if not proto:
            x = x.reshape(x.shape[0], -1)
            x = self.fully_connected(x)
        else:
            # 使用平均池化并展平
            x = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
            x = x.view(x.size(0), -1)  # (B, C)
        return x




