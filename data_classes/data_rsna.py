import os
import numpy as np
import pandas as pd
import pydicom
import random
from skimage.io import imread
from skimage.color import gray2rgb

from torchvision import transforms

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

# ---------------------------------------------------------------------------------------------------------------------------
# https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py
class RSNAPneumoniaDataset(Dataset):
    """RSNA Pneumonia Detection Challenge
    
    Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert 
    Annotations of Possible Pneumonia.
    Shih, George, Wu, Carol C., Halabi, Safwan S., Kohli, Marc D., Prevedello, Luciano M., 
    Cook, Tessa S., Sharma, Arjun, Amorosa, Judith K., Arteaga, Veronica, Galperin-Aizenberg, 
    Maya, Gill, Ritu R., Godoy, Myrna C.B., Hobbs, Stephen, Jeudy, Jean, Laroia, Archana, 
    Shah, Palmi N., Vummidi, Dharshan, Yaddanapudi, Kavitha, and Stein, Anouk.  
    Radiology: Artificial Intelligence, 1 2019. doi: 10.1148/ryai.2019180041.
    
    More info: https://www.rsna.org/en/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018
    
    Challenge site:
    https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
    
    JPG files stored here:
    https://academictorrents.com/details/95588a735c9ae4d123f3ca408e56570409bcf2a9
    """
    datapath = '../rsna'

    def __init__(self,
                 imgpath=os.path.join(datapath, "stage_2_train_images"),
                 csvpath=os.path.join(datapath, "stage_2_train_labels.csv"),
                 dicomcsvpath=os.path.join(datapath, "kaggle_stage_2_train_images_dicom_headers.csv.gz"),
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 pure_labels=False,
                 unique_patients=True,
                 pathology_masks=False,
                 extension=".dcm"):

        super(RSNAPneumoniaDataset, self).__init__()
        #np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks

        self.pathologies = ["Pneumonia", "Lung Opacity"]

        self.pathologies = sorted(self.pathologies)

        self.extension = extension
        self.use_pydicom = (extension == ".dcm")

        # Load data
        self.csvpath = csvpath
        self.raw_csv = pd.read_csv(self.csvpath, nrows=nrows)

        # The labels have multiple instances for each mask
        # So we just need one to get the target label
        self.csv = self.raw_csv.groupby("patientId").first()

        self.dicomcsvpath = dicomcsvpath
        self.dicomcsv = pd.read_csv(self.dicomcsvpath, nrows=nrows, index_col="PatientID")

        self.csv = self.csv.join(self.dicomcsv, on="patientId")

        # Remove images with view position other than specified
        self.csv["view"] = self.csv['ViewPosition']
        self.limit_to_selected_views(views)

        self.csv = self.csv.reset_index()

        # Get our classes.
        self.labels = self.csv["Target"].values.flatten()  # same labels for both

        # set if we have masks
        self.csv["has_masks"] = ~np.isnan(self.csv["x"])

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.int64)

        # patientid
        self.csv["patientid"] = self.csv["patientId"].astype(str)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        
        imgid = self.csv['patientId'].iloc[int(idx)]
        
        img_path = os.path.join(self.imgpath, imgid + self.extension)

        if self.use_pydicom:
            img = pydicom.filereader.dcmread(img_path).pixel_array
            if img.ndim == 2:  # If the image is grayscale, convert to RGB
                img = gray2rgb(img)
        else:
            img = gray2rgb(imread(img_path))

        sample["img"] = img

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])

        if self.transform is not None:

            sample["img"] = self.transform(sample["img"])
            if self.pathology_masks:
                for i in sample["pathology_masks"].keys():
                    ##random.seed(transform_seed)
                    sample["pathology_masks"][i] = self.transform(sample["pathology_masks"][i])

        return sample["img"], sample["lab"], idx
    
    def calculate_class_weights(self):
        class_sample_count = np.array([len(np.where(self.labels == t)[0]) for t in np.unique(self.labels)])
        weight = 1. / class_sample_count
        weight = weight / np.sum(weight) * len(class_sample_count)  # Normalize weights
        return torch.from_numpy(weight).float()
    
    def get_mask_dict(self, image_name, this_size):

        base_size = 1024
        scale = this_size / base_size

        images_with_masks = self.raw_csv[self.raw_csv["patientId"] == image_name]
        path_mask = {}

        # All masks are for both pathologies
        for patho in ["Pneumonia", "Lung Opacity"]:
            mask = np.zeros([this_size, this_size])

            # Don't add masks for labels we don't have
            if patho in self.pathologies:

                for i in range(len(images_with_masks)):
                    row = images_with_masks.iloc[i]
                    xywh = np.asarray([row.x, row.y, row.width, row.height])
                    xywh = xywh * scale
                    xywh = xywh.astype(int)
                    mask[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]] = 1

            # Resize so image resizing works
            mask = mask[None, :, :]

            path_mask[self.pathologies.index(patho)] = mask
        return path_mask

    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown
        self.csv.view.fillna("UNKNOWN", inplace=True)

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view


# class Dataset:

#     # ------------------------------------------------------------------------------------------------------------------
#     def __init__(self):
#         #self.random_seed = random_seed
#         self.dataset_name = "rsna"
#         self.train_sampler = None
#         self.test_sampler = None
#         self.noise_rate = 0.1

#         if self.dataset_name == "rsna":
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


#             self.dataset = RSNAPneumoniaDataset(transform=self.transform_train,
#                                      views=["PA", "AP"],
#                                      unique_patients=False)
            
#             dataset_test = RSNAPneumoniaDataset(transform=self.transform_test,
#                                      views=["PA", "AP"],
#                                      unique_patients=False)

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
            
#             if True:
#                 self.clean_labels = torch.tensor(self.dataset.labels)
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
#             # self.clean_labels = self.clean_labels[self.train_idx]
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

#             self.dataset.csv = self.dataset.csv.iloc[self.train_idx].reset_index(drop=True)
#             self.dataset.labels = self.dataset.labels[self.train_idx]
#             self.train_set = self.dataset


#             dataset_test.csv = dataset_test.csv.iloc[self.test_idx].reset_index(drop=True)
#             dataset_test.labels = dataset_test.labels[self.test_idx]
#             self.test_set = dataset_test
            

#     def make_xray_labels_noisy(self):
#         map_file_path = '../rsna/rsna_to_nih.csv'
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


# if __name__ == "__main__":
#     d = Dataset()
#     print(len(d.train_set))

#     label_counts = {0: 0, 1: 0}

#     # # Iterate through the dataset
#     for i, (image, label, _) in enumerate(d.train_set):
#         # Get the label for the current sample
#         label_counts[label] += 1

#     print(label_counts)
