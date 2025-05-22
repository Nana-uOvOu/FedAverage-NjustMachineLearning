from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np

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
            n = len(train_dataset)
            k = int(n * data_split_ratio)
            indices = np.random.permutation(n)[:k]
            train_dataset = Subset(train_dataset, indices)

            # 构建 DataLoader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            return train_loader
        # 服务器无需随机抽取
        elif role == 'server':
            test_dataset = datasets.CIFAR10(root='./Datasets/', train=False, download=True, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            return test_loader
        else:
            raise RuntimeError(f'{role} is an invalid role!!')
    elif dataset_name == 'ImageNet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        if role == 'client':
            imagenet_train = datasets.ImageFolder(root='./Datasets/ImageNet/train', transform=transform)
            # 随机抽样数据集
            n = len(imagenet_train)
            k = int(n * data_split_ratio)
            indices = np.random.permutation(n)[:k]
            imagenet_train = Subset(imagenet_train, indices)

            train_loader = DataLoader(imagenet_train, batch_size=batch_size, shuffle=True, num_workers=2)
            return train_loader

        elif role == 'server':
            imagenet_val = datasets.ImageFolder(root='./Datasets/ImageNet/val', transform=transform)
            imagenet_test = datasets.ImageFolder(root='./Datasets/ImageNet/test', transform=transform)


            val_loader = DataLoader(imagenet_val, batch_size=batch_size, shuffle=True, num_workers=4)
            test_loader = DataLoader(imagenet_test, batch_size=batch_size, shuffle=True, num_workers=4)
            return val_loader, test_loader
        else:
            raise RuntimeError(f'{role} is an invalid role!!')
    else:
        raise RuntimeError(f'{dataset_name} is invalid dataset!!!')




