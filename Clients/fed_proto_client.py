import time

from torch.utils.data import DataLoader

from .base_client import BaseClient
import socket
import pickle
import struct
import gzip
import torch
import utils.read_data as read_data
class FedProtoClient(BaseClient):
    def __init__(self, args, dataset):
        super().__init__(args)
        if args.model == "wzj_net":
            self.proto_dim = 512
        else:
            raise RuntimeError("原型学习只支持wzj_net")

        self.test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    # 计算本地类别→原型、样本数
    def compute_local_proto(self):
        self.model.eval()
        feat_sum, feat_cnt = {}, {}
        with torch.no_grad():
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                f = self.model.forward(x, proto=True)  # [b, d]
                for cls in y.unique():
                    mask = (y == cls)
                    feat_sum[int(cls)] = feat_sum.get(int(cls), 0) + f[mask].sum(0)
                    feat_cnt[int(cls)] = feat_cnt.get(int(cls), 0) + mask.sum().item()
        # 平均
        local_proto = {c: (feat_sum[c] / feat_cnt[c]) for c in feat_sum}
        return local_proto, feat_cnt

    # 自定义训练：加原型正则 (server_proto 在上一轮收到)
    def train(self, server_proto=None, lam=0.3):
        total_loss = 0.0
        correct = 0
        total = 0
        time_1 = time.time()
        self.model.train()
        for epoch in range(self.epoches):
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                # 原型正则
                if server_proto:  # 字典非空才能用
                    f = self.model.forward(x, proto=True)  # [B, d]
                    mask = torch.tensor(
                        [int(lbl.item()) in server_proto for lbl in y],
                        device=self.device
                    )  # True=有原型
                    if mask.any():  # 至少有 1 个样本能用
                        idx = mask.nonzero(as_tuple=True)[0]
                        f_sel = f[idx]  # 取可用特征
                        g_sel = torch.stack(
                            [server_proto[int(y[i].item())] for i in idx]
                        ).to(self.device)
                        loss = loss + lam * torch.mean((f_sel - g_sel).pow(2))
                loss.backward()
                self.optimizer.step()
                preds = logits.argmax(dim=1)  # 预测的类别索引
                correct += (preds == y).sum().item()
                total += y.size(0)
                total_loss += loss.item()
            time_2 = time.time()
            # 每个 epoch 输出日志
            acc = 100 * correct / total
            avg_loss = total_loss / len(self.dataloader)
            print(
                f"[Client {self.index}]Epoch {epoch + 1}/{self.epoches} - Loss: {avg_loss:.4f} - Acc: {acc:.2f}% Time cost: {time_2 - time_1:.2f}s")
        # 进行全局测试
        total_loss = 0.0
        correct = 0
        total = 0
        time_1 = time.time()
        self.model.eval()
        for x, y in self.test_dataloader:
            x, y = x.to(self.device), y.to(self.device)
            y_pre = self.model(x)
            loss = self.criterion(y_pre, y)
            preds = y_pre.argmax(dim=1)  # 预测的类别索引
            correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()
        time_2 = time.time()
        acc = 100 * correct / total
        avg_loss = total_loss / len(self.test_dataloader)
        print(
            f"[Client {self.index}]Test: Loss: {avg_loss:.4f} - Acc: {acc:.2f}% Time cost: {time_2 - time_1:.2f}s")


        return acc, avg_loss

    # 通信流程
    def start(self):
        while True:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.server_ip, self.server_port))

            # 领取 idx, 接收全局原型
            client_idx = sock.recv(4)
            self.index = struct.unpack("!I", client_idx)[0]  # 解包4字节长度字段
            print(f"Client {self.index}:Start Receiving Global Proto!!!")

            proto_len = struct.unpack("!I", sock.recv(4))[0]
            data = b""
            while len(data) < proto_len:
                data += sock.recv(proto_len - len(data))
            global_proto = pickle.loads(gzip.decompress(data))  # dict{c:tensor}
            # 本地训练
            acc,loss = self.train(server_proto=global_proto)
            # 计算并发送本地原型
            print(f"[Client {self.index}]: Finished, Sending proto to server")
            local_proto, cnt = self.compute_local_proto()
            payload = gzip.compress(pickle.dumps((local_proto, cnt)))
            sock.sendall(b"Client Result:")
            sock.sendall(struct.pack("!f", float(acc)))
            sock.sendall(struct.pack("!f", float(loss)))
            sock.sendall(b"Client Proto:")
            sock.sendall(struct.pack("!I", len(payload)))
            sock.sendall(payload)

            # 当接收到Over后，才能停机
            while sock.recv(4) != b"Over":
                pass

            sock.close()


