import time

from torch.utils.data import DataLoader

from .base_client import BaseClient
import socket
import pickle
import struct
import gzip
import torch
import torch.nn.functional as F
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

                targets = 0
                # 原型最近邻 logits
                f = self.model(x, proto=True)  # (B, d)
                if server_proto:  # dict{cls:tensor}
                    classes = sorted(server_proto.keys())
                    proto_mat = torch.stack([server_proto[c] for c in classes]
                                            ).to(self.device).detach()  # (C, d)
                    logits = - torch.cdist(f, proto_mat) ** 2  # (B, C)

                    idx_map = {c: i for i, c in enumerate(classes)}
                    targets = torch.tensor([idx_map[int(t)] for t in y],
                                           device=self.device)
                    ce_loss = F.cross_entropy(logits, targets)


                else:  # 首轮无原型时仍用本地 fc
                    logits = self.model(x)  # 常规 fc
                    ce_loss = self.criterion(logits, y)
                    targets = y
                # 原型对齐正则
                mse_loss = 0.0
                if server_proto:
                    idx = [i for i, lbl in enumerate(y) if int(lbl) in server_proto]
                    if idx:
                        f_sel = f[idx]
                        g_sel = torch.stack([server_proto[int(y[i])]
                                             for i in idx]).to(self.device)
                        mse_loss = torch.mean((f_sel - g_sel).abs())
                loss = ce_loss + lam * mse_loss
                # 反向与统计
                loss.backward()
                self.optimizer.step()

                preds = logits.argmax(dim=1)  # 预测的类别索引
                correct += (preds == targets).sum().item()
                total += y.size(0)
                total_loss += loss.item()
            time_2 = time.time()
            # 每个 epoch 输出日志
            acc = 100 * correct / total
            avg_loss = total_loss / len(self.dataloader)
            print(
                f"[Client {self.index}]Epoch {epoch + 1}/{self.epoches} - Loss: {avg_loss:.4f} - Acc: {acc:.2f}% Time cost: {time_2 - time_1:.2f}s")
            correct, total = 0, 0
            total_loss = 0
        # 进行全局测试
        total_loss = 0.0
        correct = 0
        total = 0
        time_1 = time.time()
        self.model.eval()
        for x, y in self.test_dataloader:
            x, y = x.to(self.device), y.to(self.device)

            targets = 0
            # 原型最近邻 logits
            f = self.model(x, proto=True)  # (B, d)
            if server_proto:  # dict{cls:tensor}
                classes = sorted(server_proto.keys())
                idx_map = {c: i for i, c in enumerate(classes)}
                targets = torch.tensor([idx_map[int(t)] for t in y],
                                       device=self.device)
                proto_mat = torch.stack([server_proto[c] for c in classes]
                                        ).to(self.device)  # (C, d)
                logits = - torch.cdist(f, proto_mat) ** 2  # (B, C)
                ce_loss = F.cross_entropy(logits, targets)
            else:
                logits = self.model(x)
                ce_loss = self.criterion(logits, y)
                targets = y
            # 原型对齐正则
            mse_loss = 0.0
            if server_proto:
                idx = [i for i, lbl in enumerate(y) if int(lbl) in server_proto]
                if idx:
                    f_sel = f[idx]
                    g_sel = torch.stack([server_proto[int(y[i])]
                                         for i in idx]).to(self.device)
                    mse_loss = torch.mean((f_sel - g_sel).pow(2))
            loss = ce_loss + lam * mse_loss

            preds = logits.argmax(dim=1)  # 预测的类别索引
            correct += (preds == targets).sum().item()
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


