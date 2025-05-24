import time
from copy import deepcopy

from .base_client import BaseClient
import socket
import pickle
import struct
import gzip
import torch
class FedAverageClient(BaseClient):
    def __init__(self, args):
        super().__init__(args)

    def start(self):
        global_epoch = 0
        try:
            while True:
                # 首先尝试和服务器建立连接
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.connect((self.server_ip, self.server_port))

                client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024 * 512)  # 发送缓冲区512MB
                client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 512)  # 接收缓冲区512MB
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 关闭 Nagle 算法

                # 读取分配到的客户端序号
                client_idx = client.recv(4)
                self.index = struct.unpack("!I", client_idx)[0]  # 解包4字节长度字段
                print(f"---------------------Global Epoch: {global_epoch},Assigned Index: {self.index}---------------------")

                # 读取分发的原始model
                print(f"[Client {self.index}]Receiving global model...")
                time_1 = time.time()
                model_length = client.recv(4)
                model_length = struct.unpack("!I", model_length)[0] # 解包4字节长度字段
                model_data = b""
                while len(model_data) < model_length:
                    packet = client.recv(65536)
                    model_data += packet

                global_model = pickle.loads(gzip.decompress(model_data))
                self.model.load_state_dict(global_model)

                time_2 = time.time()
                print(f"[Client {self.index}]Global model received!!!Model length: {model_length} - Time cost: {time_2 - time_1}")

                # 训练模型
                self.train()

                # 训练结束后，向服务器发送模型
                print(f"[Client {self.index}]Finished!Sending model to server...")
                # 将float32转换为float16，降低通信开销
                state_dict = {
                    k: v.half() if v.dtype == torch.float32 else v
                    for k, v in self.model.state_dict().items()
                }
                model_stream = gzip.compress(pickle.dumps(state_dict))
                model_len = struct.pack("!I", len(model_stream))  # 强制发送4字节长度
                client.sendall(b"Client Model:")
                client.sendall(model_len)
                client.sendall(model_stream)

                global_epoch += 1
        except ConnectionResetError as e:
            print(f"[Client {self.index}]Task finished!!")
            exit(0)


