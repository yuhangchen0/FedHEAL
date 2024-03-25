import torchvision.transforms as transforms
import numpy as np

from utils.conf import data_path
from PIL import Image
from datasets.utils.federated_dataset import FederatedDataset, partition_digits_domain_skew_loaders, partition_digits_label_skew_loaders
import torch.utils.data as data
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from backbone.ResNet import resnet10, resnet12
from torchvision.datasets import MNIST, SVHN, ImageFolder, DatasetFolder, USPS


class MyDigits(data.Dataset):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=True, data_name=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.data_name == 'mnist':
            dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)
        elif self.data_name == 'usps':
            dataobj = USPS(self.root, self.train, self.transform, self.target_transform, self.download)
        elif self.data_name == 'svhn':
            if self.train:
                dataobj = SVHN(self.root, 'train', self.transform, self.target_transform, self.download)
            else:
                dataobj = SVHN(self.root, 'test', self.transform, self.target_transform, self.download)
        return dataobj

    def __getitem__(self, index: int):
        img, target = self.dataset[index]
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class ImageFolder_Custom(DatasetFolder):
    def __init__(self, data_name, root, train=True, transform=None, target_transform=None):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.imagefolder_obj = ImageFolder(self.root + self.data_name + '/train/', self.transform, self.target_transform)
        else:
            self.imagefolder_obj = ImageFolder(self.root + self.data_name + '/val/', self.transform, self.target_transform)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class FedLeaDigits(FederatedDataset):
    NAME = 'fl_digits'
    SETTING = 'domain_skew'
    DOMAINS_LIST = ['mnist', 'usps', 'svhn', 'syn']
    
    train_percent_dict = {'mnist': 0.01, 'usps': 0.01, 'svhn': 0.01, 'syn': 0.01}
    
    N_CLASS = 10
    Nor_TRANSFORM = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))])

    Singel_Channel_Nor_TRANSFORM = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))])

    def get_data_loaders(self, selected_domain_list=[]):

        using_list = self.DOMAINS_LIST if len(selected_domain_list) == 0 else selected_domain_list

        nor_transform = self.Nor_TRANSFORM
        sin_chan_nor_transform = self.Singel_Channel_Nor_TRANSFORM

        train_dataset_list = []
        test_dataset_list = []

        test_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             self.get_normalization_transform()])

        sin_chan_test_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             self.get_normalization_transform()])

        for _, domain in enumerate(using_list):
            if domain in ['syn', 'mnistm']:
                train_dataset = ImageFolder_Custom(data_name=domain, root=data_path(), train=True,
                                                   transform=nor_transform)
            elif domain in ['mnist', 'usps']:
                train_dataset = MyDigits(data_path(), train=True,
                                         download=True, transform=sin_chan_nor_transform, data_name=domain)
            else:
                train_dataset = MyDigits(data_path(), train=True,
                                         download=False, transform=nor_transform, data_name=domain)
            train_dataset_list.append(train_dataset)

        for _, domain in enumerate(self.DOMAINS_LIST):
            if domain in ['syn', 'mnistm']:
                test_dataset = ImageFolder_Custom(data_name=domain, root=data_path(), train=False,
                                                  transform=test_transform)
            elif domain in ['mnist', 'usps']:
                test_dataset = MyDigits(data_path(), train=False,
                                        download=True, transform=sin_chan_test_transform, data_name=domain)
            else:
                test_dataset = MyDigits(data_path(), train=False,
                                        download=False, transform=test_transform, data_name=domain)

            test_dataset_list.append(test_dataset)
            
        
        traindls, testdls = partition_digits_domain_skew_loaders(train_dataset_list, test_dataset_list, self)
        return traindls, testdls

    def get_fedce_val_dataset(self, selected_domain_list=[]):
        using_list = selected_domain_list

        local_val_dataset_list = []
        
        test_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             self.get_normalization_transform()])

        sin_chan_test_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             self.get_normalization_transform()])

        for _, domain in enumerate(using_list):
            if domain == 'syn':
                local_test_dataset = ImageFolder_Custom(data_name=domain, root=data_path(), train=True,
                                                   transform=test_transform).imagefolder_obj
            else:
                if domain in ['mnist', 'usps']:
                    local_test_dataset = MyDigits(data_path(), train=True,
                                             download=False, transform=sin_chan_test_transform, data_name=domain).dataset
                else:
                    local_test_dataset = MyDigits(data_path(), train=True,
                                             download=False, transform=test_transform, data_name=domain).dataset
            
            num_samples = min(20, len(local_test_dataset))
            indices = np.random.choice(len(local_test_dataset), num_samples, replace=False)
            sampler = data.sampler.SubsetRandomSampler(indices)
            
            val_loader = data.DataLoader(local_test_dataset,
                                batch_size=self.args.local_batch_size, sampler=sampler)
            local_val_dataset_list.append(val_loader)

        return local_val_dataset_list
    
    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedLeaDigits.Nor_TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(parti_num, names_list):
        nets_dict = {'resnet10': resnet10, 'resnet12': resnet12}
        nets_list = []
        if names_list == None:
            for j in range(parti_num):
                nets_list.append(resnet10(FedLeaDigits.N_CLASS))
        else:
            for j in range(parti_num):
                net_name = names_list[j]
                nets_list.append(nets_dict[net_name](FedLeaDigits.N_CLASS))
        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))
        return transform
