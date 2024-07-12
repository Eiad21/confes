# import os
# import json
# import torch
# import random
# import numpy as np
# import pandas as pd
# from PIL import Image
# import pydicom
# from skimage.io import imread
# from skimage.color import gray2rgb
# from scipy import stats
# from math import inf
# import torch.nn.functional as F
# from numpy.testing import assert_array_almost_equal
# from torchvision import datasets, transforms
# from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
# import csv

# class Dataset:

#     # ------------------------------------------------------------------------------------------------------------------
#     def __init__(self,
#                  dataset_name,
#                  data_dir='./data',
#                  num_samples=0,
#                  noise_type=None,
#                  noise_rate=None,
#                  random_seed=1,
#                  device=torch.device('cuda')
#                  ):
#         self.dataset_name = dataset_name
#         self.noise_type = noise_type
#         self.noise_rate = noise_rate
#         #self.random_seed = random_seed
#         self.device = device
#         self.train_sampler = None
#         self.test_sampler = None

#         if self.dataset_name == "paediatric-pneumonia":
#             train_dir = data_dir + '/train'
#             test_dir = data_dir + '/test'
#             # pp_mean_primia = [0.4814, 0.4814, 0.4814]
#             # pp_std_primia = [0.2377, 0.2377, 0.2377]
#             pp_mean = [0.485, 0.456, 0.406]
#             pp_std = [0.229, 0.224, 0.225]
#             transform_pipe_train = [transforms.RandomResizedCrop(224),
#                                     transforms.RandomRotation(15),
#                                     transforms.RandomHorizontalFlip(),
#                                     transforms.CenterCrop(224),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(pp_mean, pp_std), ]
#             self.train_set = datasets.ImageFolder(train_dir,
#                                                   transform=transforms.Compose(transform_pipe_train))
#             transform_pipe_test = [transforms.Resize(224),
#                                    transforms.CenterCrop(224),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize(pp_mean, pp_std), ]

#             self.test_set = datasets.ImageFolder(test_dir,
#                                                  transform=transforms.Compose(transform_pipe_test))
#             # print(self.test_set.imgs)
#             self.num_classes = 3
#             self.input_size = 224 * 224 * 3
#             if noise_type is not None:
#                 # print(self.train_set.targets[0:100])
#                 train_noisy_labels_tensor = self.make_labels_noisy()
#                 self.train_set.targets = train_noisy_labels_tensor.detach()
#                 # print(self.train_set.targets[0:100])

#         elif self.dataset_name == "rsna":
#             self.num_classes = 2
#             self.input_size = 224 * 224 * 3
#             test_size = 0.3
#             self.num_samples = 8000
#             transform_pipe_train = [transforms.ToPILImage(),
#                                     transforms.Resize(256),
#                                     transforms.CenterCrop(224),
#                                     transforms.ToTensor()]
#             self.transform_train = transforms.Compose(transform_pipe_train)

#             transform_pipe_test = [transforms.ToPILImage(),
#                                    transforms.Resize(256),
#                                    transforms.CenterCrop(224),
#                                    transforms.ToTensor()]

#             self.transform_test = transforms.Compose(transform_pipe_test)

#             csvpath = "preprocess/kaggle_stage_2_train_labels.csv.zip"
#             dicomcsvpath = "preprocess/kaggle_stage_2_train_images_dicom_headers.csv.gz"
#             imgpath = os.path.join(data_dir, "stage_2_train_images_jpg")

#             self.dataset = RSNAPneumoniaDataset(imgpath=imgpath,
#                                                 csvpath=csvpath,
#                                                 dicomcsvpath=dicomcsvpath,
#                                                 transform=self.transform_train,
#                                                 views=["PA", "AP"],
#                                                 unique_patients=False)
#             dataset_test = RSNAPneumoniaDataset(imgpath=imgpath,
#                                                 csvpath=csvpath,
#                                                 dicomcsvpath=dicomcsvpath,
#                                                 transform=self.transform_test,
#                                                 views=["PA", "AP"],
#                                                 unique_patients=False)

#             # (len(self.train_set)=26684) = 6012 pneumonia_like + 20672 no_pneumonia_like
#             indices = list(range(len(self.dataset)))

#             split = int(np.floor(test_size * len(self.dataset)))
#             np.random.shuffle(indices)
#             self.train_idx, self.test_idx = indices[split:], indices[:split]
#             # clean test targets
#             test_targets = self.dataset.labels[self.test_idx]
#             class_count_test = np.unique(test_targets, return_counts=True)[1]
#             weight_test = 1. / class_count_test
#             self.test_weights = torch.tensor(weight_test, dtype=torch.float)
#             if noise_type is not None:
#                 self.make_xray_labels_noisy()
#                 # noisy train dataset from here forward

#             # making noisy train dataset balanced
#             class_num = torch.zeros(self.num_classes)
#             new_train_idx = []
#             for i in self.train_idx:
#                 label = self.dataset.labels[i]
#                 if class_num[label] < (self.num_samples / self.num_classes) and len(new_train_idx) < self.num_samples:
#                     new_train_idx.append(i)
#                     class_num[label] += 1
#             random.shuffle(new_train_idx)
#             self.train_idx = new_train_idx
#             print('Training dist. : ' + str(class_num))

#             # # # oversampling
#             # train_targets = self.dataset.labels[self.train_idx]
#             # class_count_train = np.unique(train_targets, return_counts=True)[1]
#             # weight_train = 1. / class_count_train
#             # train_samples_weight = weight_train[train_targets]
#             # self.train_samples_weight = torch.from_numpy(train_samples_weight)
#             # self.train_sampler = WeightedRandomSampler(self.train_samples_weight, len(self.train_samples_weight))

#             # making test dataset balanced
#             class_num = torch.zeros(self.num_classes)
#             new_test_idx = []
#             for i in self.test_idx:
#                 label = self.dataset.labels[i]
#                 if class_num[label] < (self.num_samples / self.num_classes) and len(new_test_idx) < self.num_samples:
#                     new_test_idx.append(i)
#                     class_num[label] += 1
#             random.shuffle(new_test_idx)
#             self.test_idx = new_test_idx
#             print('Testing dist. : ' + str(class_num))
#             print('***********************************************************************')

#             self.train_sampler = None
#             self.train_set = torch.utils.data.Subset(self.dataset, self.train_idx)
#             self.test_set = torch.utils.data.Subset(dataset_test, self.test_idx)

#         elif self.dataset_name == 'clothing1m':
#             self.num_classes = 14
#             self.input_size = 224 * 224 * 3
#             self.num_samples = num_samples
#             # self.num_samples = 250000
#             c1m_mean = [0.6959, 0.6537, 0.6371]
#             c1m_std = [0.3113, 0.3192, 0.3214]

#             self.transform_train = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.RandomCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#             ])
#             self.transform_test = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#             ])
#             self.train_set = Clothing1M(data_dir, num_samples=self.num_samples, mode='train',
#                                         transform=self.transform_train)
#             self.train_sampler = None

#             # print(len(self.train_set))
#             # selected_idx = list(range(0, 3000))
#             # self.train_set = torch.utils.data.Subset(self.train_set, selected_idx)

#             self.test_set = Clothing1M(data_dir, num_samples=self.num_samples, mode='test',
#                                        transform=self.transform_test)

#             # self.val_set = Clothing1M(data_dir,  num_samples=self.num_samples, train=True,
#             #                            transform=self.transform_test)

#     # ------------------------------------------------------------------------------------------------------------------
#     # https://github.com/xiaoboxia/CDR/blob/6665a8ba265f0f60291ed7775042575db05bed61/utils.py
#     def make_labels_noisy(self):

#         # clean_labels = torch.tensor(self.train_set.targets)
#         clean_labels_np = self.clean_labels.detach().numpy()
#         clean_labels_np = clean_labels_np[:, np.newaxis]
#         m = clean_labels_np.shape[0]
#         noisy_labels = clean_labels_np.copy()

#         is_noisy = m * [None]
#         if self.noise_rate is None:
#             raise ValueError("Noise rate needs to be specified ....")

#         if self.noise_type == "symmetric":
#             noise_matrix = self.compute_noise_transition_symmetric()
#             print(noise_matrix)
#             print('Size of noise transition matrix: {}'.format(noise_matrix.shape))

#         elif self.noise_type == "instance":
#             noise_matrix = self.compute_noise_transition_instance()
#             print('Size of noise transition matrix: {}'.format(noise_matrix.shape))
#             print(noise_matrix)

#         elif self.noise_type == "asymmetric":
#             if self.dataset_name == 'cifar100':
#                 noisy_labels, is_noisy, noise_matrix = self.compute_apply_noise_transition_asymmetric_cifar100(clean_labels_np)
#                 print('Size of noise transition matrix: {}'.format(noise_matrix.shape))
#             elif self.dataset_name == 'cifar10':
#                 noisy_labels, is_noisy = self.compute_apply_noise_transition_asymmetric_cifar10(clean_labels_np)
#                 print(len(is_noisy))
#         elif self.noise_type == 'pairflip':
#             noise_matrix = self.compute_noise_transition_pairflip()
#             print(noise_matrix)
#             print('Size of noise transition matrix: {}'.format(noise_matrix.shape))

#         elif self.noise_type == "label_dp":
#             noise_matrix = self.compute_noise_transition_label_dp()
#             print(noise_matrix)
#             print('Size of noise transition matrix: {}'.format(noise_matrix.shape))
#         else:
#             print('Noise type is not specified ')
#             exit()

#         if self.noise_type == 'symmetric' or self.noise_type == 'pairflip' or  self.noise_type == 'label_dp':
#             assert noise_matrix.shape[0] == noise_matrix.shape[1]
#             assert np.max(clean_labels_np) < noise_matrix.shape[0]
#             assert_array_almost_equal(noise_matrix.sum(axis=1), np.ones(noise_matrix.shape[1]))
#             assert (noise_matrix >= 0.0).all()

#             flipper = np.random.RandomState()
#             for idx in np.arange(m):
#                 i = clean_labels_np[idx]
#                 flipped = flipper.multinomial(1, noise_matrix[i, :][0], 1)[0]
#                 noisy_labels[idx] = np.where(flipped == 1)[0]
#                 is_noisy[idx] = (noisy_labels[idx] != i)[0]

#             # noise_or_not = (noisy_labels != clean_labels_np)
#             actual_noise_rate = (noisy_labels != clean_labels_np).mean()
#             assert actual_noise_rate > 0.0
#             print('Actual_noise_rate : {}'.format(actual_noise_rate))

#         elif self.noise_type == "instance":
#             l = [i for i in range(self.num_classes)]
#             for idx in np.arange(m):
#                 noisy_labels[idx] = np.random.choice(l, p=noise_matrix[idx])
#                 is_noisy[idx] = (noisy_labels[idx] != clean_labels_np[idx])[0]
#             # noise_or_not = (noisy_labels != clean_labels_np)
#             actual_noise_rate = (noisy_labels != clean_labels_np).mean()
#             assert actual_noise_rate > 0.0
#             print('Actual_noise_rate : {}'.format(actual_noise_rate))
        
#         root_dir = './data/'
#         csv_path=os.path.join(root_dir, self.noise_type+str(self.noise_rate)+'.csv')
#         df = pd.DataFrame({"label" : clean_labels_np.flatten(), "label_noisy" : noisy_labels.flatten()})
#         df.to_csv(csv_path, index=False)
#         print('Wrote noisy labels to file ...')

#         return torch.tensor(np.squeeze(noisy_labels)), is_noisy

#     # ------------------------------------------------------------------------------------------------------------------
#     def make_xray_labels_noisy(self):
#         map_file_path = 'preprocess/rsna_to_nih.csv'
#         map_df = pd.read_csv(map_file_path)
#         for idx in self.train_idx:
#             patient_id = self.dataset.csv['patientid'][idx]
#             matched_row = map_df.loc[map_df['patientId'] == patient_id].iloc[0]
#             orig_label_arr = matched_row['orig_labels'].split(';')
#             if self.dataset.labels[idx] == 1:
#                 if 'Pneumonia' in orig_label_arr:
#                     pass
#                 elif (
#                         'Consolidation' in orig_label_arr or
#                         'Infiltration' in orig_label_arr) and \
#                         'Pneumonia' not in orig_label_arr:
#                     if torch.rand(1) <= self.noise_rate:
#                         self.dataset.labels[idx] = 0
#                 elif 'No Finding' in orig_label_arr:
#                     self.dataset.labels[idx] = 0
#                 else:
#                     self.dataset.labels[idx] = 0
#             elif self.dataset.labels[idx] == 0:
#                 if 'Pneumonia' in orig_label_arr:
#                     self.dataset.labels[idx] = 1

#                 elif ('Consolidation' in orig_label_arr or 'Infiltration' in orig_label_arr) and \
#                         ('Pneumonia' not in orig_label_arr):
#                     if torch.rand(1) <= self.noise_rate:
#                         self.dataset.labels[idx] = 1
#                 else:
#                     pass

#     # ------------------------------------------------------------------------------------------------------------------
#     # https://github.com/xiaoboxia/CDR/blob/6665a8ba265f0f60291ed7775042575db05bed61/utils.py 
#     def compute_noise_transition_symmetric(self):

#         noise_matrix = np.ones((self.num_classes, self.num_classes))
#         noise_matrix = (self.noise_rate / (self.num_classes - 1)) * noise_matrix

#         if self.noise_rate > 0.0:
#             # 0 -> 1
#             noise_matrix[0, 0] = 1. - self.noise_rate
#             for i in range(1, self.num_classes - 1):
#                 noise_matrix[i, i] = 1. - self.noise_rate
#             noise_matrix[self.num_classes - 1, self.num_classes - 1] = 1. - self.noise_rate
#             # print(noise_matrix)
#         return noise_matrix

#     def compute_noise_transition_label_dp(self):
#         import math
#         epsilon = self.noise_rate
#         self.noise_rate = math.pow(math.e, epsilon) / (math.pow(math.e, epsilon) + self.num_classes -1)
        
#         noise_matrix = np.ones((self.num_classes, self.num_classes))
#         noise_matrix = ((1-self.noise_rate) / (self.num_classes - 1)) * noise_matrix

#         if self.noise_rate > 0.0:
#             # 0 -> 1
#             noise_matrix[0, 0] = self.noise_rate
#             for i in range(1, self.num_classes - 1):
#                 noise_matrix[i, i] = self.noise_rate
#             noise_matrix[self.num_classes - 1, self.num_classes - 1] = self.noise_rate
#             # print(noise_matrix)
#         return noise_matrix
#     # ------------------------------------------------------------------------------------------------------------------
#     #https://github.com/tmllab/PES/blob/54662382dca22f314911488d79711cffa7fbf1a0/common/NoisyUtil.py
#     def compute_noise_transition_pairflip(self):

#         noise_matrix = np.eye(self.num_classes)

#         if self.noise_rate > 0.0:
#             # 0 -> 1
#             noise_matrix[0, 0], noise_matrix[0,1] = 1. - self.noise_rate, self.noise_rate
#             for i in range(1, self.num_classes - 1):
#                 noise_matrix[i, i], noise_matrix[i, i+1] = 1. - self.noise_rate, self.noise_rate
#             noise_matrix[self.num_classes - 1, self.num_classes - 1], noise_matrix[self.num_classes-1, 0] = 1. - self.noise_rate, self.noise_rate
#             # print(noise_matrix)
#         return noise_matrix


#     # ------------------------------------------------------------------------------------------------------------------
#     # https://github.com/xiaoboxia/CDR/blob/6665a8ba265f0f60291ed7775042575db05bed61/utils.py
#     def build_for_cifar100(self,size_subclass, noise):
#         """ random flip between two random classes.
#         """
#         assert (noise >= 0.) and (noise <= 1.)

#         P = (1. - noise) * np.eye(size_subclass)
#         for i in np.arange(size_subclass - 1):
#             P[i, i + 1] = noise

#         # adjust last row
#         P[size_subclass - 1, 0] = noise

#         assert_array_almost_equal(P.sum(axis=1), 1, 1)
#         return P

#     def compute_apply_noise_transition_asymmetric_cifar100(self, clean_labels_np):

#         nb_superclasses = 20
#         nb_subclasses = 5
#         noise_matrix = np.eye(self.num_classes)
#         m = clean_labels_np.shape[0]
#         noisy_labels = clean_labels_np.copy()
#         is_noisy = m * [None]
#         if self.noise_rate  > 0.0:
#             for i in np.arange(nb_superclasses):
#                 init, end = i * nb_subclasses, (i + 1) * nb_subclasses
#                 noise_matrix[init:end, init:end] = self.build_for_cifar100(nb_subclasses, self.noise_rate)


#                 assert noise_matrix.shape[0] == noise_matrix.shape[1]
#                 assert np.max(clean_labels_np) < noise_matrix.shape[0]
#                 assert_array_almost_equal(noise_matrix.sum(axis=1), np.ones(noise_matrix.shape[1]))
#                 assert (noise_matrix >= 0.0).all()
#                 flipper = np.random.RandomState()
#                 for idx in np.arange(m):
#                     i = clean_labels_np[idx]
#                     flipped = flipper.multinomial(1, noise_matrix[i, :][0], 1)[0]
#                     noisy_labels[idx] = np.where(flipped == 1)[0]
#                     is_noisy[idx] = (noisy_labels[idx] != i)[0]
#                 actual_noise_rate = (noisy_labels != clean_labels_np).mean()
#             assert actual_noise_rate > 0.0
#             print('Actual_noise_rate : {}'.format(actual_noise_rate))

#             return torch.tensor(np.squeeze(noisy_labels)), is_noisy, noise_matrix

#     # ------------------------------------------------------------------------------------------------------------------
#     def compute_apply_noise_transition_asymmetric_cifar10(self, clean_labels_np):
#         m = clean_labels_np.shape[0]
#         is_noisy = m * [None]
#         source_class = [9, 2, 3, 5, 4]
#         target_class = [1, 0, 5, 3, 7]
#         noisy_labels= clean_labels_np.copy()
#         for s, t in zip(source_class, target_class):
#             cls_idx = np.where(np.array(clean_labels_np) == s)[0]
#             n_noisy = int(self.noise_rate * cls_idx.shape[0])
#             noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
#             for idx in noisy_sample_index:
#                 i = clean_labels_np[idx]
#                 noisy_labels[idx] = t
#                 is_noisy[idx] = (noisy_labels[idx] != i)[0]

#             actual_noise_rate = (noisy_labels != clean_labels_np).mean()
#         assert actual_noise_rate > 0.0
#         print('Actual_noise_rate : {}'.format(actual_noise_rate))
#         return torch.tensor(np.squeeze(noisy_labels)), is_noisy
#     # ------------------------------------------------------------------------------------------------------------------

#     # https://github.com/xiaoboxia/CDR/blob/6665a8ba265f0f60291ed7775042575db05bed61/tools.py
#     def compute_noise_transition_instance(self):
#         clean_labels = self.clean_labels
#         norm_std = 0.1
#         #np.random.seed(int(self.random_seed))
#         #torch.manual_seed(int(self.random_seed))
#         #torch.cuda.manual_seed(int(self.random_seed))

#         noise_matrix = []
#         flip_distribution = stats.truncnorm((0 - self.noise_rate) / norm_std,
#                                             (1 - self.noise_rate) / norm_std,
#                                             loc=self.noise_rate,
#                                             scale=norm_std)
#         flip_rate = flip_distribution.rvs(clean_labels.shape[0])

#         W = np.random.randn(self.num_classes, self.input_size, self.num_classes)
#         W = torch.FloatTensor(W).to(self.device)
#         for i, (image, label, _) in enumerate(self.train_set):
#             # 1*m *  m*10 = 1*10 = A.size()
#             image = image.detach().to(self.device)
#             A = image.view(1, -1).mm(W[label]).squeeze(0)
#             A[label] = -inf
#             A = flip_rate[i] * F.softmax(A, dim=0)
#             A[label] += 1 - flip_rate[i]
#             noise_matrix.append(A)
#         noise_matrix = torch.stack(noise_matrix, 0).cpu().numpy()
#         # print(noise_matrix)
#         return noise_matrix

#     # ------------------------------------------------------------------------------------------------------------------
#     def split_into_batches(self, batch_size, train_sampler=None):
#         if train_sampler is not None:
#             print("...................................")
#             print(train_sampler)
#             self.train_dataloader = torch.utils.data.DataLoader(dataset=self.train_set,
#                                                                 batch_size=batch_size,
#                                                                 sampler=train_sampler,
#                                                                 drop_last=False)

#         else:
#             self.train_dataloader = torch.utils.data.DataLoader(dataset=self.train_set,
#                                                                 batch_size=batch_size,
#                                                                 shuffle=True,
#                                                                 drop_last=False)

#         self.test_dataloader = torch.utils.data.DataLoader(dataset=self.test_set,
#                                                            batch_size=batch_size,
#                                                            shuffle=False)
#         self.sample_rate = batch_size / len(self.train_set)
#         return self.train_dataloader, self.test_dataloader


# # ---------------------------------------------------------------------------------------------------------------------------
# # https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py
# class RSNAPneumoniaDataset:
#     """RSNA Pneumonia Detection Challenge
    
#     Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert 
#     Annotations of Possible Pneumonia.
#     Shih, George, Wu, Carol C., Halabi, Safwan S., Kohli, Marc D., Prevedello, Luciano M., 
#     Cook, Tessa S., Sharma, Arjun, Amorosa, Judith K., Arteaga, Veronica, Galperin-Aizenberg, 
#     Maya, Gill, Ritu R., Godoy, Myrna C.B., Hobbs, Stephen, Jeudy, Jean, Laroia, Archana, 
#     Shah, Palmi N., Vummidi, Dharshan, Yaddanapudi, Kavitha, and Stein, Anouk.  
#     Radiology: Artificial Intelligence, 1 2019. doi: 10.1148/ryai.2019180041.
    
#     More info: https://www.rsna.org/en/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018
    
#     Challenge site:
#     https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
    
#     JPG files stored here:
#     https://academictorrents.com/details/95588a735c9ae4d123f3ca408e56570409bcf2a9
#     """
#     datapath = './data'

#     def __init__(self,
#                  imgpath,
#                  csvpath=os.path.join(datapath, "kaggle_stage_2_train_labels.csv.zip"),
#                  dicomcsvpath=os.path.join(datapath, "kaggle_stage_2_train_images_dicom_headers.csv.gz"),
#                  views=["PA"],
#                  transform=None,
#                  data_aug=None,
#                  nrows=None,
#                  seed=0,
#                  pure_labels=False,
#                  unique_patients=True,
#                  pathology_masks=False,
#                  extension=".jpg"):

#         super(RSNAPneumoniaDataset, self).__init__()
#         #np.random.seed(seed)  # Reset the seed so all runs are the same.
#         self.imgpath = imgpath
#         self.transform = transform
#         self.data_aug = data_aug
#         self.pathology_masks = pathology_masks

#         self.pathologies = ["Pneumonia", "Lung Opacity"]

#         self.pathologies = sorted(self.pathologies)

#         self.extension = extension
#         self.use_pydicom = (extension == ".dcm")

#         # Load data
#         self.csvpath = csvpath
#         self.raw_csv = pd.read_csv(self.csvpath, nrows=nrows)

#         # The labels have multiple instances for each mask
#         # So we just need one to get the target label
#         self.csv = self.raw_csv.groupby("patientId").first()

#         self.dicomcsvpath = dicomcsvpath
#         self.dicomcsv = pd.read_csv(self.dicomcsvpath, nrows=nrows, index_col="PatientID")

#         self.csv = self.csv.join(self.dicomcsv, on="patientId")

#         # Remove images with view position other than specified
#         self.csv["view"] = self.csv['ViewPosition']
#         self.limit_to_selected_views(views)

#         self.csv = self.csv.reset_index()

#         # Get our classes.
#         self.labels = self.csv["Target"].values.flatten()  # same labels for both

#         # set if we have masks
#         self.csv["has_masks"] = ~np.isnan(self.csv["x"])

#         self.labels = np.asarray(self.labels).T
#         self.labels = self.labels.astype(np.int)

#         # patientid
#         self.csv["patientid"] = self.csv["patientId"].astype(str)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         sample = {}
#         sample["idx"] = idx
#         sample["lab"] = self.labels[idx]
#         imgid = self.csv['patientId'].iloc[idx]
#         img_path = os.path.join(self.imgpath, imgid + self.extension)

#         if self.use_pydicom:
#             img = pydicom.filereader.dcmread(img_path).pixel_array
#         else:
#             img = gray2rgb(imread(img_path))

#         # save original image
#         # imsave('img_saved.jpg', img)

#         sample["img"] = img
#         #transform_seed = np.random.randint(2147483647)

#         if self.pathology_masks:
#             sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])

#         if self.transform is not None:
#             #random.seed(transform_seed)

#             sample["img"] = self.transform(sample["img"])
#             if self.pathology_masks:
#                 for i in sample["pathology_masks"].keys():
#                     ##random.seed(transform_seed)
#                     sample["pathology_masks"][i] = self.transform(sample["pathology_masks"][i])

#         # save_image(sample["img"], 'input.jpg')
#         return sample["img"], sample["lab"]

#     def get_mask_dict(self, image_name, this_size):

#         base_size = 1024
#         scale = this_size / base_size

#         images_with_masks = self.raw_csv[self.raw_csv["patientId"] == image_name]
#         path_mask = {}

#         # All masks are for both pathologies
#         for patho in ["Pneumonia", "Lung Opacity"]:
#             mask = np.zeros([this_size, this_size])

#             # Don't add masks for labels we don't have
#             if patho in self.pathologies:

#                 for i in range(len(images_with_masks)):
#                     row = images_with_masks.iloc[i]
#                     xywh = np.asarray([row.x, row.y, row.width, row.height])
#                     xywh = xywh * scale
#                     xywh = xywh.astype(int)
#                     mask[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]] = 1

#             # Resize so image resizing works
#             mask = mask[None, :, :]

#             path_mask[self.pathologies.index(patho)] = mask
#         return path_mask

#     def limit_to_selected_views(self, views):
#         """This function is called by subclasses to filter the
#         images by view based on the values in .csv['view']
#         """
#         if type(views) is not list:
#             views = [views]
#         if '*' in views:
#             # if you have the wildcard, the rest are irrelevant
#             views = ["*"]
#         self.views = views

#         # missing data is unknown
#         self.csv.view.fillna("UNKNOWN", inplace=True)

#         if "*" not in views:
#             self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view
