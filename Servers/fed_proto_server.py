import datetime
import os
import socket
import threading
import pickle
import struct
import time

import numpy as np
import torch
import gzip

from .base_server import BaseServer
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt

class FedProtoServer(BaseServer):
    def __init__(self, args):
        super().__init__(args)
        self.global_proto = {}


    # 处理每一个客户端的连接
    def client_handler(self, client_socket, address, client_protos, client_cnts, client_idx, client_acc, client_loss):
        # 首先给服务器分发初始模型
        print("---------------------------------------------Client Connected---------------------------------------------")
        print(f"Client {address} is connected! Index: {client_idx}")

        # 通知客户端的idx
        client_socket.sendall(struct.pack("!I", client_idx))

        # 发送全局原型
        proto_stream = gzip.compress(pickle.dumps(self.global_proto))
        client_socket.sendall(struct.pack("!I", len(proto_stream)))
        client_socket.sendall(proto_stream)
        # 接收客户端原型
        while client_socket.recv(14) != b"Client Result:":
            pass
        # 先发acc再发loss
        client_acc[client_idx-1] = struct.unpack("!f", client_socket.recv(4))[0]
        client_loss[client_idx-1] = struct.unpack("!f", client_socket.recv(4))[0]
        while client_socket.recv(13) != b"Client Proto:":
            pass
        length = struct.unpack("!I", client_socket.recv(4))[0]
        data = b""
        while len(data) < length:
            data += client_socket.recv(length - len(data))
        local_proto, local_cnt = pickle.loads(gzip.decompress(data))
        # 防止出现不同设备的bug，当仅使用一张显卡时，可以忽略
        for key in local_proto.keys():
            local_proto[key] = local_proto[key].to(self.device)

        client_protos[client_idx-1] = local_proto
        client_cnts[client_idx-1] = local_cnt

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
            clients_proto, clients_cnt = [None]*self.max_client_num, [None]*self.max_client_num
            client_acc, client_loss = [None]*self.max_client_num, [None]*self.max_client_num
            threads = []    # 存储所有客户端的Thread
            client_sockets = []

            for i in range(1,1 + self.max_client_num):
                # print(f"Server waiting for [Client {i}]...")
                client_socket, client_address = server.accept()
                client_sockets.append(client_socket)
                thread = threading.Thread(
                    target=self.client_handler,
                    args=(client_socket, client_address, clients_proto, clients_cnt, i, client_acc, client_loss))
                thread.start()
                threads.append(thread)

            print("Waiting for clients...")
            # 主进程等待所有客户端结束
            for thread in threads:
                thread.join()

            # 向客户端发送Over指令
            for client_socket in client_sockets:
                client_socket.sendall(b"Over")
                client_socket.close()

            # 执行参数聚合
            self.aggregate(clients_proto, clients_cnt)

            # 参数聚合后，执行模型测试
            acc = np.mean(client_acc)
            avg_loss = np.mean(client_loss)
            print(f"[Server]Epoch {global_epoch+1}/{self.epochs} Average Loss: {avg_loss} - Accuracy: {acc}")

            accuracy_list.append(acc)
            loss_list.append(avg_loss)



            # 早停：记录最低Loss值，连续early_stop_rounds超过最低Loss就停止
            up_count = 0
            stop_flag = False
            for i in range(len(loss_list)-1,-1,-1):
                if min_loss >= loss_list[i]:
                    min_loss = loss_list[i]
                    break
                else:
                    up_count += 1
                # 连续early_stop_rounds轮loss不下降
                if up_count >= self.early_stop_rounds:
                    stop_flag = True
                    break
            if stop_flag:
                print("[Server] Stopped!!")
                break
        # 结束，保存数据并绘制图像
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

    # 原型聚合
    def aggregate(self, clients_proto, clients_cnt):
        sum_dict = defaultdict(lambda: 0)
        cnt_dict = defaultdict(lambda: 0)

        for p, c in zip(clients_proto, clients_cnt):
            for cls in p:
                sum_dict[cls] += p[cls] * c[cls]
                cnt_dict[cls] += c[cls]
        self.global_proto = {cls: (sum_dict[cls] / cnt_dict[cls]) for cls in sum_dict}

