from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np
from torch.utils.data import TensorDataset
import torch

def get_sub_dataset(args):
    dataset_name = args.dataset
    data_split_ratio = args.data_split_ratio
    role = args.role
    batch_size = args.batch_size

    if dataset_name == 'CIFAR10':
        # 将数据集toTensor后进行标准化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])
        # 随机抽取
        # 客户端随机抽取train数据
        if role == 'client':
            train_dataset = datasets.CIFAR10(root='./Datasets/', train=True, download=True, transform=transform)
            # 随机抽样数据集
            # 做原型学习时发现，这个方法是为了增加noniid数据的训练效果，所以必须要选取出noniid数据w
            if not args.noniid:
                n = len(train_dataset)
                k = int(n * data_split_ratio)
                indices = np.random.permutation(n)[:k]
                subset = Subset(train_dataset, indices)
            else:
                # non-IID 抽样：只取指定类的数据
                num_classes_per_client = args.noniid_classes
                total_classes = 10  # CIFAR-10 是 10 类
                all_targets = np.array(train_dataset.targets)

                # 随机选 num_classes_per_client 个类
                chosen_classes = np.random.choice(total_classes, num_classes_per_client, replace=False)

                indices = []
                for cls in chosen_classes:
                    cls_indices = np.where(all_targets == cls)[0]
                    k_cls = int(len(cls_indices) * data_split_ratio)
                    sampled = np.random.choice(cls_indices, k_cls, replace=False)
                    indices.extend(sampled)

                subset = Subset(train_dataset, indices)

            all_data = [subset[i] for i in range(len(subset))]  # 列表，每个元素是 (image, label)

            # 拆开 image 和 label，拼成 Tensor
            images, labels = zip(*all_data)
            images = torch.stack(images)
            labels = torch.tensor(labels)

            # 构造独立 Dataset
            new_dataset = TensorDataset(images, labels)
            new_dataset.classes = train_dataset.classes

            # 构建 DataLoader
            train_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            return train_loader
        # 服务器无需随机抽取
        elif role == 'server':
            test_dataset = datasets.CIFAR10(root='./Datasets/', train=False, download=True, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            return test_loader
        else:
            raise RuntimeError(f'{role} is an invalid role!!')
    elif dataset_name == 'CIFAR100':
        # 将数据集toTensor后进行标准化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])
        # 随机抽取
        # 客户端随机抽取train数据
        if role == 'client':
            train_dataset = datasets.CIFAR100(root='./Datasets/', train=True, download=True, transform=transform)
            # 随机抽样数据集
            if not args.noniid:
                n = len(train_dataset)
                k = int(n * data_split_ratio)
                indices = np.random.permutation(n)[:k]
                subset = Subset(train_dataset, indices)
            else:
                # non-IID 抽样：只取指定类的数据
                num_classes_per_client = args.noniid_classes
                total_classes = 100  # CIFAR-100 是 100 类
                all_targets = np.array(train_dataset.targets)

                # 随机选 num_classes_per_client 个类
                chosen_classes = np.random.choice(total_classes, num_classes_per_client, replace=False)

                indices = []
                for cls in chosen_classes:
                    cls_indices = np.where(all_targets == cls)[0]
                    k_cls = int(len(cls_indices) * data_split_ratio)
                    sampled = np.random.choice(cls_indices, k_cls, replace=False)
                    indices.extend(sampled)

                subset = Subset(train_dataset, indices)
            all_data = [subset[i] for i in range(len(subset))]  # 列表，每个元素是 (image, label)

            # 拆开 image 和 label，拼成 Tensor
            images, labels = zip(*all_data)
            images = torch.stack(images)
            labels = torch.tensor(labels)

            # 构造独立 Dataset
            new_dataset = TensorDataset(images, labels)
            new_dataset.classes = train_dataset.classes

            # 构建 DataLoader
            train_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            return train_loader
        # 服务器无需随机抽取
        elif role == 'server':
            test_dataset = datasets.CIFAR100(root='./Datasets/', train=False, download=True, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            return test_loader
        else:
            raise RuntimeError(f'{role} is an invalid role!!')
    else:
        raise RuntimeError(f'{dataset_name} is invalid dataset!!!')

def get_test_dataset(args):
    dataset_name = args.dataset
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])

        dataset = datasets.CIFAR10(root='./Datasets/', train=False, download=True, transform=transform)
    else:
        # 将数据集toTensor后进行标准化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])
        # 随机抽取
        # 客户端随机抽取train数据
        dataset = datasets.CIFAR100(root='./Datasets/', train=False, download=True, transform=transform)

    return dataset






