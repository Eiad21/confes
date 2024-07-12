import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from math import inf

import torch.nn.functional as F
from numpy.testing import assert_array_almost_equal
from torchvision import datasets, transforms

from data_tissue import TissueMNISTDataset
from data_chex import CheXpertDataset
from data_chex_i import CheXpertDatasetI
from data_rsna import RSNAPneumoniaDataset


class Dataset:

    def __init__(self,
                 dataset_name,
                 data_dir='./data',
                 num_samples=0,
                 noise_type=None,
                 noise_rate=None,
                 random_seed=1,
                 device=torch.device('cuda')
                 ):
        self.dataset_name = dataset_name
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.device = device
        self.random_seed = random_seed
        self.train_sampler = None
        self.test_sampler = None

        if self.dataset_name == "cifar10":
            cifar_mean = [0.4914, 0.4822, 0.4465]
            cifar_std = [0.2023, 0.1994, 0.2010]
            transform_pipe = [transforms.RandomHorizontalFlip(),
                              transforms.RandomCrop(32, 4),
                              transforms.ToTensor(),
                              transforms.Normalize(cifar_mean, cifar_std)]

            self.train_set = Cifar10(root=data_dir,
                                     train=True,
                                     transform=transforms.Compose(transform_pipe),
                                     download=True)

            self.test_set = Cifar10(root=data_dir,
                                    train=False,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(),
                                         transforms.Normalize(cifar_mean, cifar_std)]),
                                    download=True)
            self.num_classes = 10
            self.input_size = 32 * 32 * 3
            self.is_noisy = []
            if noise_type is not None:
                self.clean_labels = torch.tensor(self.train_set.cifar10.targets)
                train_noisy_labels_tensor, is_noisy_labels = self.make_labels_noisy()
                self.is_noisy = is_noisy_labels[:]
                self.train_set.cifar10.targets = train_noisy_labels_tensor.detach()

        elif self.dataset_name == "cifar100":
            cifar_mean = [0.507, 0.487, 0.441]
            cifar_std = [0.267, 0.256, 0.276]
            transform_pipe = [transforms.RandomHorizontalFlip(),
                              transforms.RandomCrop(32, 4),
                              transforms.ToTensor(),
                              transforms.Normalize(cifar_mean, cifar_std), ]
            self.train_set = Cifar100(root=data_dir,
                                      train=True,
                                      transform=transforms.Compose(transform_pipe),
                                      download=True)
            self.test_set = Cifar100(root=data_dir,
                                     train=False,
                                     transform=transforms.Compose([transforms.ToTensor(),
                                                                   transforms.Normalize(cifar_mean,
                                                                                        cifar_std)]),
                                     download=True)
            self.num_classes = 100
            self.input_size = 32 * 32 * 3
            if noise_type is not None:
                self.clean_labels = torch.tensor(self.train_set.cifar100.targets)
                train_noisy_labels_tensor, is_noisy_labels = self.make_labels_noisy()
                self.is_noisy = is_noisy_labels[:]
                self.train_set.cifar100.targets = train_noisy_labels_tensor.detach()

        elif self.dataset_name == "tissue":
            # Define transformations for the dataset

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
])
            s = 28
            self.num_classes = 8
            self.input_size = s * s * 3
            self.train_set = TissueMNISTDataset(split='train', size=s, transform=transform, download=True)
            self.test_set = TissueMNISTDataset(split='val', size=s, transform=transform, download=True)


            if noise_type is not None:
                self.clean_labels = torch.tensor(self.train_set.targets)
                train_noisy_labels_tensor, is_noisy_labels = self.make_labels_noisy()
                self.is_noisy = is_noisy_labels[:]
                self.train_set.targets = train_noisy_labels_tensor.detach()

        elif self.dataset_name == 'clothing1m':
            self.num_classes = 14
            self.input_size = 224 * 224 * 3
            self.num_samples = num_samples
            # self.num_samples = 250000
            c1m_mean = [0.6959, 0.6537, 0.6371]
            c1m_std = [0.3113, 0.3192, 0.3214]

            self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
            self.train_set = Clothing1M(data_dir, num_samples=self.num_samples, mode='train',
                                        transform=self.transform_train)
            self.train_sampler = None

            self.test_set = Clothing1M(data_dir, num_samples=self.num_samples, mode='test',
                                       transform=self.transform_test)

        elif self.dataset_name == 'isic':
            s = 32
            self.num_classes = 2
            self.input_size = s * s * 3
            self.num_samples = num_samples
            isic_mean = [0.7251, 0.6000, 0.5407]
            isic_std = [0.0939, 0.1310, 0.1540]

            self.transform_train = transforms.Compose([
                # transforms.Resize(256),
                transforms.Resize(32),
                transforms.RandomCrop(s),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(isic_mean, isic_std),
            ])
            self.transform_test = transforms.Compose([
                # transforms.Resize(256),
                transforms.Resize(32),
                transforms.CenterCrop(s),
                transforms.ToTensor(),
                transforms.Normalize(isic_mean, isic_std),
            ])
            self.train_set = ISIC(data_dir, mode='train', transform=self.transform_train)
            self.train_sampler = None

            self.test_set = ISIC(data_dir, mode='test', transform=self.transform_test)

            if noise_type is not None:
                self.clean_labels = torch.tensor(self.train_set.train_labels)
                train_noisy_labels_tensor, is_noisy_labels = self.make_labels_noisy()
                self.is_noisy = is_noisy_labels[:]
                self.train_set.train_labels = train_noisy_labels_tensor.detach()

        elif self.dataset_name == "chex":
            self.num_classes = 5
            self.input_size = 180 * 180 * 3
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.Resize((180, 180)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ])

            self.train_set = CheXpertDataset(transform, mode="train")
            self.train_sampler = None

            self.test_set = CheXpertDataset(transform, mode="valid")

            if noise_type is not None:
                self.clean_labels = torch.tensor(self.train_set.train_labels)
                train_noisy_labels_tensor, is_noisy_labels = self.make_labels_noisy(multi=True)
                self.is_noisy = is_noisy_labels[:]
                self.train_set.train_labels = train_noisy_labels_tensor.detach()

        elif "chex" in self.dataset_name:
            label_index = int(self.dataset_name.split("_")[-1])
            
            self.num_classes = 2
            self.input_size = 180 * 180 * 3
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.Resize((180, 180)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ])

            self.train_set = CheXpertDatasetI(transform, mode="train", label_index=label_index, oversample=True)
            self.train_sampler = None

            self.test_set = CheXpertDatasetI(transform, mode="valid", label_index=label_index)

            if noise_type is not None:
                self.clean_labels = torch.tensor(self.train_set.labels)
                train_noisy_labels_tensor, is_noisy_labels = self.make_labels_noisy()
                self.is_noisy = is_noisy_labels[:]
                self.train_set.train_labels = train_noisy_labels_tensor.detach()

        elif self.dataset_name == "rsna":
            self.num_classes = 2
            self.input_size = 224 * 224 * 3
            test_size = 0.3
            self.num_samples = 8000
            
            transform_pipe_train = [transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()]
            
            self.transform_train = transforms.Compose(transform_pipe_train)

            transform_pipe_test = [transforms.ToPILImage(),
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor()]

            self.transform_test = transforms.Compose(transform_pipe_test)
            
            self.dataset = RSNAPneumoniaDataset(transform=self.transform_train,
                                     views=["PA", "AP"],
                                     unique_patients=False)
            
            dataset_test = RSNAPneumoniaDataset(transform=self.transform_test,
                                     views=["PA", "AP"],
                                     unique_patients=False)
            
            indices = list(range(len(self.dataset)))

            split = int(np.floor(test_size * len(self.dataset)))
            np.random.shuffle(indices)
            self.train_idx, self.test_idx = indices[split:], indices[:split]

            # clean test targets
            test_targets = self.dataset.labels[self.test_idx]
            class_count_test = np.unique(test_targets, return_counts=True)[1]
            weight_test = 1. / class_count_test
            self.test_weights = torch.tensor(weight_test, dtype=torch.float)

            if noise_type is not None:
                self.clean_labels = torch.tensor(self.dataset.labels)
                self.make_xray_labels_noisy()
            
            # making noisy train dataset balanced
            class_num = torch.zeros(self.num_classes)
            new_train_idx = []
            for i in self.train_idx:
                label = self.dataset.labels[i]
                if class_num[label] < (self.num_samples / self.num_classes) and len(new_train_idx) < self.num_samples:
                    new_train_idx.append(i)
                    class_num[label] += 1
            random.shuffle(new_train_idx)
            self.train_idx = new_train_idx
            print('Training dist. : ' + str(class_num))

            # making test dataset balanced
            class_num = torch.zeros(self.num_classes)
            new_test_idx = []
            for i in self.test_idx:
                label = self.dataset.labels[i]
                if class_num[label] < (self.num_samples / self.num_classes) and len(new_test_idx) < self.num_samples:
                    new_test_idx.append(i)
                    class_num[label] += 1
            random.shuffle(new_test_idx)
            self.test_idx = new_test_idx
            print('Testing dist. : ' + str(class_num))
            print('***********************************************************************')

            self.train_sampler = None
            
            self.dataset.csv = self.dataset.csv.iloc[self.train_idx].reset_index(drop=True)
            self.dataset.labels = self.dataset.labels[self.train_idx]
            self.train_set = self.dataset
            self.clean_labels = self.clean_labels[self.train_idx]
            dataset_test.csv = dataset_test.csv.iloc[self.test_idx].reset_index(drop=True)
            dataset_test.labels = dataset_test.labels[self.test_idx]
            self.test_set = dataset_test

    def set_target(self, key, target):
        if self.dataset_name == "cifar10":
            self.train_set.cifar10.targets[key] = target
        elif self.dataset_name == "cifar100":
            self.train_set.cifar100.targets[key] = target
        elif self.dataset_name == "tissue":
            self.train_set.targets[key] = target
        elif self.dataset_name == "clothing1m":
            pass # TODO: to be implemented
        elif self.dataset_name == "isic" or "chex" in self.dataset_name:
            self.train_set.train_labels[key] = target
        elif self.dataset_name == "rsna":
            self.train_set.labels[key] = target
        else:
            raise Exception("Not handled")

    def get_targets(self):
        if self.dataset_name == "cifar10":
            return self.train_set.cifar10.targets
        elif self.dataset_name == "cifar100":
            return self.train_set.cifar100.targets
        elif self.dataset_name == "tissue":
            return self.train_set.targets
        elif self.dataset_name == "clothing1m":
            return None # TODO: to be implemented
        elif self.dataset_name == "isic" or "chex" in self.dataset_name:
            return self.train_set.train_labels
        elif self.dataset_name == "rsna":
            return self.train_set.labels
        else:
            raise Exception("Not handled")      
         
    # ------------------------------------------------------------------------------------------------------------------
    # Taken from https://github.com/xiaoboxia/CDR/blob/6665a8ba265f0f60291ed7775042575db05bed61/utils.py
    def make_labels_noisy(self):
        clean_labels_np = self.clean_labels.detach().numpy()
        clean_labels_np = clean_labels_np[:, np.newaxis]
        m = clean_labels_np.shape[0]
        noisy_labels = clean_labels_np.copy()

        is_noisy = m * [None]
        if self.noise_rate is None:
            raise ValueError("Noise rate needs to be specified ....")

        if self.noise_type == "symmetric":
            noise_matrix = self.compute_noise_transition_symmetric()

        elif self.noise_type == "instance":
            noise_matrix = self.compute_noise_transition_instance()
        
        elif self.noise_type == "pairflip":
            noise_matrix = self.compute_noise_transition_pairflip()


        print('Size of noise transition matrix: {}'.format(noise_matrix.shape))

        if self.noise_type == "symmetric" or self.noise_type == "pairflip":
            assert noise_matrix.shape[0] == noise_matrix.shape[1]
            assert np.max(clean_labels_np) < noise_matrix.shape[0]
            assert_array_almost_equal(noise_matrix.sum(axis=1), np.ones(noise_matrix.shape[1]))
            assert (noise_matrix >= 0.0).all()

            flipper = np.random.RandomState(self.random_seed)
            for idx in np.arange(m):
                i = clean_labels_np[idx]
                flipped = flipper.multinomial(1, noise_matrix[i, :][0], 1)[0]
                noisy_labels[idx] = np.where(flipped == 1)[0]
                is_noisy[idx] = (noisy_labels[idx] != i)[0]
        elif self.noise_type == "instance":
            l = [i for i in range(self.num_classes)]
            for idx in np.arange(m):
                noisy_labels[idx] = np.random.choice(l, p=noise_matrix[idx])
                is_noisy[idx] = (noisy_labels[idx] != clean_labels_np[idx])[0]

        # noise_or_not = (noisy_labels != clean_labels_np)
        actual_noise_rate = (noisy_labels != clean_labels_np).mean()
        assert actual_noise_rate > 0.0
        print('Actual_noise_rate : {}'.format(actual_noise_rate))
        return torch.tensor(np.squeeze(noisy_labels)), is_noisy


    def make_xray_labels_noisy(self):
        map_file_path = '../rsna/rsna_to_nih.csv'
        map_df = pd.read_csv(map_file_path)
        for idx in self.train_idx:
            patient_id = self.dataset.csv['patientid'][idx]
            matched_row = map_df.loc[map_df['patientId'] == patient_id].iloc[0]
            orig_label_arr = matched_row['orig_labels'].split(';')
            if self.dataset.labels[idx] == 1:
                if 'Pneumonia' in orig_label_arr:
                    pass
                elif (
                        'Consolidation' in orig_label_arr or
                        'Infiltration' in orig_label_arr) and \
                        'Pneumonia' not in orig_label_arr:
                    if torch.rand(1) <= self.noise_rate:
                        self.dataset.labels[idx] = 0
                elif 'No Finding' in orig_label_arr:
                    self.dataset.labels[idx] = 0
                else:
                    self.dataset.labels[idx] = 0
            elif self.dataset.labels[idx] == 0:
                if 'Pneumonia' in orig_label_arr:
                    self.dataset.labels[idx] = 1

                elif ('Consolidation' in orig_label_arr or 'Infiltration' in orig_label_arr) and \
                        ('Pneumonia' not in orig_label_arr):
                    if torch.rand(1) <= self.noise_rate:
                        self.dataset.labels[idx] = 1
                else:
                    pass
    
    # ------------------------------------------------------------------------------------------------------------------
    # Taken from https://github.com/xiaoboxia/CDR/blob/6665a8ba265f0f60291ed7775042575db05bed61/utils.py
    def compute_noise_transition_symmetric(self):
        noise_matrix = np.ones((self.num_classes, self.num_classes))
        noise_matrix = (self.noise_rate / (self.num_classes - 1)) * noise_matrix

        if self.noise_rate > 0.0:
            # 0 -> 1
            noise_matrix[0, 0] = 1. - self.noise_rate
            for i in range(1, self.num_classes - 1):
                noise_matrix[i, i] = 1. - self.noise_rate
            noise_matrix[self.num_classes - 1, self.num_classes - 1] = 1. - self.noise_rate
            # print(noise_matrix)
        return noise_matrix

    # ------------------------------------------------------------------------------------------------------------------
    # Taken from https://github.com/xiaoboxia/CDR/blob/6665a8ba265f0f60291ed7775042575db05bed61/tools.py
    def compute_noise_transition_instance(self):
        clean_labels = self.clean_labels
        norm_std = 0.1
        np.random.seed(int(self.random_seed))
        torch.manual_seed(int(self.random_seed))
        torch.cuda.manual_seed(int(self.random_seed))

        noise_matrix = []
        flip_distribution = stats.truncnorm((0 - self.noise_rate) / norm_std,
                                            (1 - self.noise_rate) / norm_std,
                                            loc=self.noise_rate,
                                            scale=norm_std)
        flip_rate = flip_distribution.rvs(clean_labels.shape[0])

        W = np.random.randn(self.num_classes, self.input_size, self.num_classes)
        W = torch.FloatTensor(W).to(self.device)
        for i, (image, label, _) in enumerate(self.train_set):
            # 1*m *  m*10 = 1*10 = A.size()
            image = image.detach().to(self.device)
            A = image.view(1, -1).mm(W[label]).squeeze(0)
            A[label] = -inf
            A = flip_rate[i] * F.softmax(A, dim=0)
            A[label] += 1 - flip_rate[i]
            noise_matrix.append(A)
        noise_matrix = torch.stack(noise_matrix, 0).cpu().numpy()
        return noise_matrix

    # ------------------------------------------------------------------------------------------------------------------
    #https://github.com/tmllab/PES/blob/54662382dca22f314911488d79711cffa7fbf1a0/common/NoisyUtil.py
    def compute_noise_transition_pairflip(self):

        noise_matrix = np.eye(self.num_classes)

        if self.noise_rate > 0.0:
            # 0 -> 1
            noise_matrix[0, 0], noise_matrix[0,1] = 1. - self.noise_rate, self.noise_rate
            for i in range(1, self.num_classes - 1):
                noise_matrix[i, i], noise_matrix[i, i+1] = 1. - self.noise_rate, self.noise_rate
            noise_matrix[self.num_classes - 1, self.num_classes - 1], noise_matrix[self.num_classes-1, 0] = 1. - self.noise_rate, self.noise_rate
            # print(noise_matrix)
        return noise_matrix


    # ------------------------------------------------------------------------------------------------------------------
    def split_into_batches(self, batch_size, train_sampler=None):
        if train_sampler is not None:
            self.train_dataloader = torch.utils.data.DataLoader(dataset=self.train_set,
                                                                batch_size=batch_size,
                                                                sampler=train_sampler,
                                                                drop_last=False)

        else:
            self.train_dataloader = torch.utils.data.DataLoader(dataset=self.train_set,
                                                                batch_size=batch_size,
                                                                shuffle=True,
                                                                drop_last=False)

        self.test_dataloader = torch.utils.data.DataLoader(dataset=self.test_set,
                                                           batch_size=batch_size,
                                                           shuffle=False)
        self.sample_rate = batch_size / len(self.train_set)
        return self.train_dataloader, self.test_dataloader

# -------------------------------------------------------------------------------------------------------------
# Taken from https://github.com/chenpf1025/IDN/blob/master/dataset.py
class Clothing1M(torch.utils.data.Dataset):
    def __init__(self, root, mode='train',
                 soft=False, target_prob=None,
                 transform=None, target_transform=None, num_classes=14, num_samples=0):
        self.root = root
        self.mode = mode
        self.soft = soft
        self.target_prob = target_prob
        self.transform = transform
        self.target_transform = target_transform

        train_labels_path = os.path.join(root, 'annotations/noisy_label_kv.txt')
        self.train_labels = self.file_reader(train_labels_path, labels=True)
        test_labels_path = os.path.join(root, 'annotations/clean_label_kv.txt')
        self.test_labels = self.file_reader(test_labels_path, labels=True)

        if self.mode == 'train':
            file_path = os.path.join(root, "annotations/noisy_train_key_list.txt")
            imgs = self.file_reader(file_path, labels=False)
            random.shuffle(imgs)
            class_num = torch.zeros(num_classes)
            self.imgs = []
            for impath in imgs:
                label = self.train_labels[impath]
                if class_num[label] < (num_samples / 14) and len(self.imgs) < num_samples:
                    self.imgs.append(impath)
                    class_num[label] += 1
            random.shuffle(self.imgs)
            print('Number of training samples : ' + str(class_num))

        if self.mode == 'val':
            file_path = os.path.join(root, "annotations/clean_val_key_list.txt")
            self.imgs = self.file_reader(file_path, labels=False)

        if self.mode == 'test':
            file_path = os.path.join(root, 'annotations/clean_test_key_list.txt')
            self.imgs = self.file_reader(file_path, labels=False)

        if not os.path.exists(file_path):
            raise RuntimeError('Dataset not found or not extracted.' +
                               'You can contact the author of Clothing1M for the download link. <Xiao, Tong, '
                               'et al. (2015). Learning from massive noisy labeled data for image classification>')

    def __getitem__(self, index):
        impath = self.imgs[index]
        if self.mode == 'train':
            target = self.train_labels[impath]
        else:
            target = self.test_labels[impath]
        img = Image.open(impath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.soft:
            target_soft = self.target_prob[index]
            return img, target_soft, target
        else:
            return img, target, index

    def __len__(self):
        return len(self.imgs)

    def file_reader(self, path, labels=False):
        if labels:
            file_data = {}
        else:
            file_data = []
        with open(path, 'r') as rf:
            for line in rf.readlines():
                row = line.strip().split()
                img_path = self.root + '/' + row[0][7:]
                if labels:
                    file_data[img_path] = int(row[1])
                else:
                    file_data.append(img_path)

        return file_data

# ----------------------------------------------------------------------------------------------------------------
class Cifar10(Dataset):
    def __init__(self, root, train, transform, download):
        self.cifar10 = datasets.CIFAR10(root=root,
                                        download=download,
                                        train=train,
                                        transform=transform)
        self.is_noisy = []

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)


class Cifar100(Dataset):
    def __init__(self, root, train, transform, download):
        self.cifar100 = datasets.CIFAR100(root=root,
                                          download=download,
                                          train=train,
                                          transform=transform)

    def __getitem__(self, index):
        data, target = self.cifar100[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar100)

# ----------------------------------------------------------------------------------------------------------------
class ISIC(torch.utils.data.Dataset):
    def __init__(self, root, mode='train',
                 soft=False, target_prob=None,
                 transform=None, target_transform=None, num_classes=14, num_samples=0):
        self.root = root
        self.mode = mode
        self.soft = soft
        self.target_prob = target_prob
        self.transform = transform
        self.target_transform = target_transform

        train_labels_path = os.path.join(root, 'ISBI2016_ISIC_Part3_Training_GroundTruth.csv')
        self.train_labels = self.file_reader(train_labels_path, mode="train")
        test_labels_path = os.path.join(root, 'ISBI2016_ISIC_Part3_Test_GroundTruth.csv')
        self.test_labels = self.file_reader(test_labels_path, mode="test")

        if self.mode == 'train':
            file_path = os.path.join(root, "ISBI2016_ISIC_Part3_Training_Data")
            self.imgs = [os.path.join(file_path, img_name) for img_name in os.listdir(file_path)]
            print('Number of training samples : ' + str(len(self.imgs)))

        if self.mode == 'test':
            file_path = os.path.join(root, "ISBI2016_ISIC_Part3_Test_Data")
            self.imgs = [os.path.join(file_path, img_name) for img_name in os.listdir(file_path)]

    def __getitem__(self, index):
        impath = self.imgs[index]
        if self.mode == 'train':
            target = self.train_labels[index]
        else:
            target = self.test_labels[index]
        img = Image.open(impath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.imgs)

    def file_reader(self, path, mode):
        file_data = []
        with open(path, 'r') as rf:
            for line in rf.readlines():
                row = line.strip().split(',')
                # Convert 'benign' to 0 and 'malignant' to 1
                if mode == "train":
                    label = 0 if row[1] == 'benign' else 1
                else:
                    label = 0 if row[1] == '0.0' else 1
                file_data.append(label)
        return file_data

    def set_corrected_labels_(self, corrected_labels):
        for key, value in corrected_labels.items():
            self.train_labels[key] = value
            

if __name__ == "__main__":

    ci = Cifar100(root='./data',
                  train=True,
                  transform=None,
                  download=True)
    print(ci.cifar100.targets)
