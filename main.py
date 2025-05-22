import argparse
import torch
import numpy as np
import os
from Clients.fed_average_client import FedAverageClient
from Servers.fed_average_server import FedAverageServer
import threading
import utils.read_data as read_data

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
    parser.add_argument('--client_epoches', type=int, default=5)
    parser.add_argument('--global_epoches', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learn_rate', type=float, default=0.005)
    parser.add_argument('--model', type=str, default="wzj_net") # 可以尝试用我的wzjnet，虽然是我随便设计的
    # 设置FedAvg需要的参数
    # 在FedAvg模式下，Client需要获取中心服务器发放的模型，使用自己的数据进行训练，结束后上传给服务器进行聚合
    # Server需要向所有Client发放模型，等待客户端全部训练结束后聚合整个模型。若有几个客户端过慢，则放弃。
    parser.add_argument('--role', type=str, default="client")
    parser.add_argument('--max_client_num', type=int, default=3)  # 服务器需要的客户端数量
    parser.add_argument('--server_port', type=int, default=8888) # 设置服务器监听的端口号a
    parser.add_argument('--server_ip', type=str, default="localhost") # 设置客户端连接的ip地址
    # 设置数据集信息
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument('--data_split_ratio', type=float, default=0.5)  # 每个客户端随机选取的data数
    # 设置保存信息
    parser.add_argument('--save_path', type=str, default="./result")
    parser.add_argument('--load_model_path', type=str, default=None)


    args = parser.parse_args()
    return args

def init_role(args):
    # 设置随机数种子
    set_seed(args.seed)

    # 对于客户端，创建对应客户端实例
    if args.role == "client":
        role = FedAverageClient(args)
        role.start()
    elif args.role == "server":
        role = FedAverageServer(args)
        role.start()

    else:
        raise Exception(f"{args.role} is an invalid role!!!")


if __name__ == '__main__':
    args =  init_args()
    init_role(args)



'''
注意，seed必须要设置，否则所有Client的训练结果和模型将会完全一模一样，FedAvg就没有意义了
CIFAR10比较简单，容易过拟合

建立服务器:
使用CIFAR10, 需要3个客户端，自动监听8888端口
记得修改device id
python main.py --role server --dataset CIFAR10 --max_client_num 3 --server_port 8888 --global_epoches 15 --seed 666 --device_id 1  

建立客户端：
使用服务器对应的CIFAR10，需要建立3个，执行3次下列指令，服务器IP默认为127.0.0.1
python main.py --role client --dataset CIFAR10 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 5 --seed 6666 --device_id 1 
python main.py --role client --dataset CIFAR10 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 5 --seed 66666 --device_id 1 
python main.py --role client --dataset CIFAR10 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 5 --seed 666666 --device_id 1 
'''


'''
我发现在我本机上跑这个代码，内存很容易溢出，但至少tcp通信时，本地回环loopback模型通信速度很快
然而当我把代码搬到科研训练导师的服务器上之后，tcp通信速度及其慢，现在是真正体会到了什么叫做“联邦学习目前瓶颈在于通信”
之后可以尝试增加一些压缩算法，可能能够更快

我发现了，如果我的wzj_net使用两层全连接，那参数量会到5亿字节以上，一个模型500MB。loopback的tcp传输都需要一分钟以上
然而，当我去掉第一层全连接，参数量瞬间掉到700万，tcp传输时间1秒
全连接层确实复杂，我服了
'''


