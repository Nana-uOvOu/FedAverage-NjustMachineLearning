import datetime
import os
import socket
import threading
import pickle
import struct
import time

import torch
import gzip

from .base_server import BaseServer
from collections import OrderedDict
import matplotlib.pyplot as plt

class FedAverageServer(BaseServer):
    def __init__(self, args):
        super().__init__(args)


    # 处理每一个客户端的连接
    def client_handler(self, client_socket, address, client_models, client_idx):
        # 首先给服务器分发初始模型
        print("---------------------------------------------Client Connected---------------------------------------------")
        print(f"Client {address} is connected! Index: {client_idx}")

        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024 * 512)  # 发送缓冲区512MB
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 512)  # 接收缓冲区512MB
        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 关闭 Nagle 算法

        # 通知客户端的idx
        client_socket.sendall(struct.pack("!I", client_idx))

        # 向客户端分发全局模型
        # 将float32转换为float16，降低通信开销
        state_dict = {
            k: v.half() if v.dtype == torch.float32 else v
            for k, v in self.model.state_dict().items()
        }
        model_stream = gzip.compress(pickle.dumps(state_dict))
        model_len = struct.pack("!I", len(model_stream)) # 强制发送4字节长度
        client_socket.sendall(model_len)
        client_socket.sendall(model_stream)

        # 接受客户端训练后的模型
        while True:
            data = client_socket.recv(13)
            if data == b"Client Model:":
                break
        model_length = client_socket.recv(4)
        model_length = struct.unpack("!I", model_length)[0]  # 解包4字节长度字段
        model_data = b""
        while len(model_data) < model_length:
            packet = client_socket.recv(65536)
            model_data += packet

        client_model = pickle.loads(gzip.decompress(model_data))
        # 防止出现不同设备的bug，当仅使用一张显卡时，可以忽略
        for key in client_model.keys():
            client_model[key] = client_model[key].to(self.device)
        client_models[client_idx-1] = client_model
        client_socket.close()

        print(f"[Index: {client_idx}] Client {address} has done!")

    def start(self):
        # 监听localhost:port
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", self.port))
        server.listen(self.max_client_num)
        print(f"---------------------------------------------Server Listening on port {self.port}---------------------------------------------\n")

        # 进行global_epoch次全局训练
        accuracy_list = []
        loss_list = []
        min_loss = 1e10
        for global_epoch in range(self.epochs):
            print(f"---------------------------------------------Global Epoch: {global_epoch}---------------------------------------------")
            time_1 = time.time()
            client_models = [0] * self.max_client_num  # 存储所有客户端的模型
            threads = []    # 存储所有客户端的Thread

            for i in range(1,1 + self.max_client_num):
                # print(f"Server waiting for [Client {i}]...")
                client_socket, client_address = server.accept()
                thread = threading.Thread(
                    target=self.client_handler,
                    args=(client_socket, client_address, client_models, i))
                thread.start()
                threads.append(thread)

            # 主进程等待所有客户端结束
            for thread in threads:
                thread.join()

            # 执行参数聚合
            self.aggregate(client_models)

            # 参数聚合后，执行模型测试
            print("[Testing model...]")
            total_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.dataloader:
                    # 将数据to cuda
                    images, labels = images.to(self.device), labels.to(self.device)
                    # 切换到eval模式
                    self.model.eval()
                    y_pred = self.model(images)
                    # 计算损失
                    loss = self.criterion(y_pred, labels)
                    total_loss += loss.item()

                    preds = y_pred.argmax(dim=1)  # 预测的类别索引
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            # 输出日志
            acc = 100 * correct / total
            avg_loss = total_loss / len(self.dataloader)
            time_2 = time.time()
            print(f"Epoch {global_epoch+1}/{self.epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.2f}% - Total time: {time_2 - time_1}")

            accuracy_list.append(acc)
            loss_list.append(avg_loss)

            # 早停：记录最低Loss值，连续early_stop_rounds次超过最低Loss就停止
            if self.early_stop:
                up_count = 0
                stop_flag = False
                for i in range(len(loss_list)-1,-1,-1):
                    if min_loss >= loss_list[i]:
                        min_loss = loss_list[i]
                        break
                    else:
                        up_count += 1
                    if up_count >= self.early_stop_rounds:
                        stop_flag = True
                        break
                if stop_flag:
                    break
        # 结束，保存数据并绘制图像
        path = self.save_path + "/model/" + self.model_name
        os.makedirs(path, exist_ok=True)
        path += f"/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.pth"
        print(f"Saving model to path [{path}]...")
        torch.save(self.model.state_dict(), path)

        # 绘制精准度图像
        # Loss 曲线
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o')
        plt.title("Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        # Accuracy 曲线
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, marker='o')
        plt.title("Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")

        plt.tight_layout()
        path = self.save_path + "/result_img/"
        os.makedirs(path, exist_ok=True)
        path += self.model_name + f"_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
        plt.savefig(path)





    def aggregate(self,client_models):
            # 直接求平均，默认所有Client的样本数量一样。也可以使用加权平均。
            avg_model_dict = OrderedDict()

            for key in client_models[0].keys():
                # 所有客户端对应 key 的张量相加求平均
                avg_model_dict[key] = sum(d[key] for d in client_models) / self.max_client_num


            # 加载到全局模型中
            self.model.load_state_dict(avg_model_dict)

