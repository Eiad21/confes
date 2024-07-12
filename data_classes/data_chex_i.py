from collections import Counter

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from torchvision import transforms

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def multihot(y):
    Y = []
    for i in range(y.shape[0]):
        mhe = []
        for j in range(y.shape[1]):
            # if(j != 0):
            if(y[i,j] == 0):
                mhe += [1, 0, 0]
            elif(y[i, j] == 1):
                mhe += [0, 1, 0]
            else:
                mhe += [0, 0, 1]
            # elif(j == 0):
            #     if(y[i,j] == 0):
            #         mhe += [1, 0]
            #     elif(y[i, j] == 1):
            #         mhe += [0, 1]
        Y.append(mhe)
    return np.asarray(Y)


def clahe(image_name):
    image = cv2.imread(image_name, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    he_img = clahe.apply(image)
    image = cv2.cvtColor(he_img, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(image)


class CheXpertDatasetI(Dataset):
    def __init__(self, transform=None, mode="train", label_index=0, oversample=True):
        """
        image_list: list of paths containing images
        labels: corresponding multi-class labels of image_list
        transform: optional transform to be applied on a sample.
        """
        self.mode = mode
        if self.mode == "train":
            csv = "../../../../../vol/aimspace/projects/CheXpert/CheXpert-small/train.csv"
        else:
            csv = "../../../../../vol/aimspace/projects/CheXpert/CheXpert-small/valid.csv"
        self.label_index = label_index
        image_list, labels = self.getImagesLabels(csv, "both")
        image_list = image_list[:8000]
        self.labels = np.array(labels[:8000, label_index], dtype=np.int64)
        self.image_names = image_list
        

        if self.mode == "train" and oversample:
            self.oversample()

        self.transform = transform


    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""

        dir = "../../../../../vol/aimspace/projects/CheXpert/CheXpert-small/"
        
        image_name = self.image_names[index]
        # Split the path by "/"
        image_parts = image_name.split("/")

        # Join the parts starting from the second element
        image_name = "/".join(image_parts[1:])
        image_name = dir + image_name
        image = Image.open(image_name).convert('RGB')

        label = self.labels[index]


        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, index
        

    def __len__(self):
        return len(self.image_names)
    
    def oversample(self):
        # Count the occurrences of each class
        class_counts = Counter(self.labels)
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        majority_count = class_counts[majority_class]
        minority_count = class_counts[minority_class]

        # Calculate the imbalance ratio
        imbalance_ratio = majority_count / minority_count
        # Oversample the minority class
        if imbalance_ratio > 1:
            # Calculate the number of samples needed to balance the classes
            oversample_count = majority_count - minority_count

            # Find the indices of the minority class
            minority_indices = [i for i, label in enumerate(self.labels) if label == minority_class]

            # Randomly sample indices from the minority class to oversample
            oversample_indices = np.random.choice(minority_indices, size=oversample_count, replace=True)

            # Add the oversampled indices to the dataset
            self.image_names = np.concatenate((self.image_names, [self.image_names[i] for i in oversample_indices]))
            # self.image_names.extend([self.image_names[i] for i in oversample_indices])
            self.labels = np.concatenate((self.labels, [self.labels[i] for i in oversample_indices]))
            # self.train_labels.extend([self.train_labels[i] for i in oversample_indices])


    def getImagesLabels(self, filename, policy):
        """
        filename: path to the csv file containing all the imagepaths and associated labels
        """
        df = pd.read_csv(filename)
        relevant_cols = ['Path', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

        df = df[relevant_cols]
        df = df.replace(np.nan, 0.0)

        if policy == 'zeros':
            df = df.replace(-1.0, 0.0)
        elif policy == 'ones':
            df = df.replace(-1.0, 1.0)
        elif policy == 'both':
            df['Cardiomegaly'] = df['Cardiomegaly'].replace(-1.0, 0.0)
            df['Consolidation'] = df['Consolidation'].replace(-1.0, 0.0)

            df['Atelectasis'] = df['Atelectasis'].replace(-1.0, 1.0)
            df['Edema'] = df['Edema'].replace(-1.0, 1.0)
            df['Pleural Effusion'] = df['Pleural Effusion'].replace(-1.0, 1.0)

        elif policy == 'multi' or policy == 'ignore':
            df = df.replace(-1.0, 2.0)


        X = df['Path']
        y = df[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']]

        return np.asarray(X), np.asarray(y)
    

if __name__ == "__main__":
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.Resize((180, 180)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ])
    
    # transform = transforms.Compose([
    #     transforms.Resize((48, 48)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(32, 4),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ])

    train_dataset = CheXpertDatasetI(transform, mode="train", label_index=0, oversample=True)

    # Initialize counts dictionary to store label counts
    label_counts = {0: 0, 1: 0}

    # Iterate through the dataset
    for i in range(len(train_dataset)):
        # Get the label for the current sample
        label = train_dataset.labels[i].item()
        label_counts[label] += 1
    print(label_counts)
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # print(len(train_dataset))
    # for img, label, idx in train_loader:
    #     print(img.shape)
