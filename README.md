# 基于联邦学习的图像分类
922127970152 王子骏

## 安装需求库
`pip install -r requirements.txt`
## 开始训练:IID数据
使用下面指令启动服务器(注意：一次开太多线程(4个以上)会导致通信卡死,某几个线程永远无法继续运行！内存会溢出！如果要测试，请设置服务器max_client_num = 1，并设置客户端create_client_count=1)
1. IID数据，建立服务器:
   1. 使用CIFAR10, 需要10个客户端，自动监听8888端口
   2. 记得修改device id
   3. `python main.py --role server --dataset CIFAR10 --server_port 8888 --global_epoches 100 --device_id 1`
2. IID数据，建立客户端：
   1. 使用服务器对应的CIFAR10，一次建立5个客户端，共2次，服务器IP默认为127.0.0.1
   2. 经过测试，CIFAR10大约在60轮左右过拟合
   3. 注意需要修改device id
   4. `python main.py --role client --dataset CIFAR10 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 1 --create_client_count 5 --device_id 1`
CIFAR100只需要修改--dataset 为CIFAR100即可

## nonIID数据：
1. 增加一个参数：`--noniid True`, 还要设置每个客户端的noniid类别数`--noniid_classes 3`(病理性采样，只选取noniid_classes个类别)

2. 对于CIFAR10: 
   1. 服务器：`python main.py --role server --dataset CIFAR10 --server_port 8888 --global_epoches 200 --early_stop_rounds 30 --device_id 1 `
   2. 客户端`python main.py --role client --dataset CIFAR10 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 1 --create_client_count 5 --noniid True --noniid_classes 5 --device_id 1`
3. 对于CIFAR100：
   1. 服务器：`python main.py --role server --dataset CIFAR100 --server_port 8888 --global_epoches 200 --early_stop_rounds 10 --max_client_num 20 --device_id 1 `
   2. 客户端：`python main.py --role client --dataset CIFAR100 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 3 --create_client_count 5 --noniid True --noniid_classes 50 --data_split_ratio 0.4 --device_id 1`
3. 具体参数可以在main.py中查看
## 原型学习方法
不知道为什么，客户端连接时总会有几个线程连不上，所以10个客户端，建立3*4 = 12个，多两个更安全。
1. CIFAR10 noniid:
    1. 服务器： `python main.py --role server --dataset CIFAR10 --server_port 8888 --global_epoches 200 --early_stop_rounds 20 --proto True --device_id 1`
    2. 客户端:`python main.py --role client --dataset CIFAR10 --server_ip 127.0.0.1 --server_port 8888 --client_epoches 1 --create_client_count 4 --noniid True --noniid_classes 3 --proto True --device_id 1`