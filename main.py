import argparse
import torch
import numpy as np
import os
from Clients.fed_average_client import FedAverageClient
from Clients.fed_proto_client import FedProtoClient
from Servers.fed_average_server import FedAverageServer
import threading
import utils.read_data as read_data

from Servers.fed_proto_server import FedProtoServer


# 设置随机数种子
# 该函数作者：22大数据冯梓原
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_args():
    # 使用argparse库来进行参数解析
    parser = argparse.ArgumentParser()
    # 设置随机数种子
    parser.add_argument('--seed', type=int, default=666)
    # 设置设备和设备号
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--device_id', type=int, default=0)
    # 设置基本客户端训练参数
    parser.add_argument('--client_epoches', type=int, default=2)    # 客户端epoch
    parser.add_argument('--global_epoches', type=int, default=100)
    # 客户端训练的batch_size, 为了模拟边缘设备，不应该设置太大，否则占用过多内存和算力，不符合实际
    # 但实际训练时为了速度可以适当增加
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--server_batch_size', type=int, default=64)
    parser.add_argument('--learn_rate', type=float, default=0.001)
    parser.add_argument('--model', type=str, default="wzj_net") # 可以尝试用我的wzjnet，虽然是我随便设计的
    # 设置FedAvg需要的参数
    # 在FedAvg模式下，Client需要获取中心服务器发放的模型，使用自己的数据进行训练，结束后上传给服务器进行聚合
    # Server需要向所有Client发放模型，等待客户端全部训练结束后聚合整个模型。若有几个客户端过慢，则放弃。
    parser.add_argument('--role', type=str, default="client")
    parser.add_argument('--max_client_num', type=int, default=10)  # 服务器最多需要的客户端数量
    parser.add_argument('--server_port', type=int, default=8888) # 设置服务器监听的端口号a
    parser.add_argument('--server_ip', type=str, default="localhost") # 设置客户端连接的ip地址
    # 设置数据集信息
    parser.add_argument('--create_client_count', type=int, default=5) # 一次创建多少个client
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument('--data_split_ratio', type=float, default=0.2)  # 每个客户端随机选取的data数
    parser.add_argument('--noniid', type=bool, default=False)
    parser.add_argument('--noniid_classes', type=int, default=3)
    # 设置保存信息
    parser.add_argument('--save_path', type=str, default="./result")
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--early_stop', type=bool, default=True)
    parser.add_argument('--early_stop_rounds', type=int, default=5)
    # 原型学习
    parser.add_argument('--proto', type=bool, default=False)



    args = parser.parse_args()
    return args

def init_role(args,dataset):
    # 设置随机数种子
    # set_seed(args.seed)

    # 对于客户端，创建对应客户端实例
    if args.role == "client":
        if not args.proto:
            role = FedAverageClient(args)
        else:
            # proto模式下，需要一个dataset
            role = FedProtoClient(args,dataset)
        role.start()
    elif args.role == "server":
        if not args.proto:
            role = FedAverageServer(args)
        else:
            role = FedProtoServer(args)
        role.start()

    else:
        raise Exception(f"{args.role} is an invalid role!!!")


if __name__ == '__main__':
    args =  init_args()
    if args.proto:
        dataset = read_data.get_test_dataset(args)
    else:
        dataset = None
    if args.role == "client":
        threads = []
        for i in range(args.create_client_count):
            thread = threading.Thread(
                target=init_role,
                args=(args,dataset))
            thread.start()
            threads.append(thread)
        for t in threads:
            t.join()
    else:
        init_role(args,dataset)



'''
我去掉了seed的设置，否则所有Client的训练结果和模型将会完全一模一样，FedAvg就没有意义了
注意：一次开太多线程(4个以上)会导致通信卡死,某几个线程永远无法继续运行！内存会溢出！如果要测试，请设置服务器max_client_num = 1，并设置客户端create_client_count=1

CIFAR10
建立服务器:
使用CIFAR10, 需要10个客户端，自动监听8888端口
记得修改device id
python main.py --role server --dataset CIFAR10 --server_port 8888 --global_epoches 100 --device_id 1  

建立客户端：
使用服务器对应的CIFAR10，一次建立5个客户端，共4次，服务器IP默认为127.0.0.1
经过测试，CIFAR10大约在60轮左右过拟合
需要修改device id
python main.py --role client --dataset CIFAR10 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 1 --create_client_count 5 --device_id 1
python main.py --role client --dataset CIFAR10 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 1 --create_client_count 5 --device_id 2

CIFAR100只需要修改--dataset 为CIFAR100即可

noniid数据集：
增加一个参数：--noniid True, 还要设置每个客户端的noniid类别数--noniid_classes 3(病理性采样，只选取noniid_classes个类别)
对于CIFAR10: 
python main.py --role server --dataset CIFAR10 --server_port 8888 --global_epoches 200 --early_stop_rounds 20 --device_id 1 
python main.py --role client --dataset CIFAR10 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 1 --create_client_count 5 --noniid True --noniid_classes 3 --device_id 1
python main.py --role client --dataset CIFAR10 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 1 --create_client_count 5 --noniid True --noniid_classes 3 --device_id 2

对于CIFAR100：
python main.py --role server --dataset CIFAR100 --server_port 8888 --global_epoches 200 --early_stop_rounds 10 --max_client_num 20 --device_id 1 
python main.py --role client --dataset CIFAR100 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 3 --create_client_count 5 --noniid True --noniid_classes 50 --data_split_ratio 0.4 --device_id 1
python main.py --role client --dataset CIFAR100 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 3 --create_client_count 5 --noniid True --noniid_classes 50 --data_split_ratio 0.4 --device_id 2
python main.py --role client --dataset CIFAR100 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 3 --create_client_count 5 --noniid True --noniid_classes 50 --data_split_ratio 0.4 --device_id 3
python main.py --role client --dataset CIFAR100 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 3 --create_client_count 5 --noniid True --noniid_classes 50 --data_split_ratio 0.4 --device_id 4

'''

'''
原型学习
不知道为什么，客户端连接时总会有几个线程连不上，所以10个客户端，建立3*4 = 12个，多两个更安全。
CIFAR10 noniid
服务器：
python main.py --role server --dataset CIFAR10 --server_port 8888 --global_epoches 200 --early_stop_rounds 20 --proto True --device_id 1
客户端
python main.py --role client --dataset CIFAR10 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 1 --create_client_count 4 --noniid True --noniid_classes 3 --proto True --device_id 1

python main.py --role client --dataset CIFAR10 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 1 --create_client_count 4 --noniid True --noniid_classes 3 --proto True --device_id 2

python main.py --role client --dataset CIFAR10 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 1 --create_client_count 4 --noniid True --noniid_classes 3 --proto True --device_id 3
'''

'''
我发现在我本机上跑这个代码，内存很容易溢出，当我把代码搬到科研训练导师的服务器上之后，tcp通信速度及其慢，现在是真正体会到了什么叫做“联邦学习目前瓶颈在于通信”
之后可以尝试增加一些压缩算法，可能能够更快

我发现了，如果我的wzj_net使用两层全连接，那参数量会到5亿字节以上，一个模型500MB。loopback的tcp传输都需要一分钟以上
然而，当我去掉第一层全连接，参数量瞬间掉到700万，tcp传输时间1秒
将传输float32改为传输float16，参数量变为500万，进一步压缩了通信量
'''


'''
原型学习、知识蒸馏
'''

