import torch
from tqdm import tqdm
import numpy as np
from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
import sys

import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix


# Set optimizer according to the training strategy and
# the type of the optimizer (return optimizer)
def set_optimizer(optimizer_name,
                  dataset_name,
                  model,
                  learning_rate,
                  all_epochs,
                  gamma,
                  momentum=0.9,
                  weight_decay=1e-3):
    model_params = model.parameters()
    if optimizer_name == "adam":
        optimizer = optim.Adam(model_params, learning_rate)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model_params, learning_rate, weight_decay=weight_decay, momentum=momentum)

    if dataset_name == 'cifar10' or dataset_name == 'cifar100' or dataset_name == 'tissue':
        scheduler = CosineAnnealingLR(optimizer, all_epochs, learning_rate * gamma, verbose = True)
        print('Set learning rate scheduler to CosineAnnealingLR')

    elif dataset_name == 'clothing1m':
        scheduler = MultiStepLR(optimizer, milestones=40, gamma=gamma, verbose=True)
        print('Set learning rate scheduler to MultiStepLR')
    
    elif dataset_name == 'isic':
        scheduler = MultiStepLR(optimizer, milestones=[20], gamma=gamma, verbose=True)
        print('Set learning rate scheduler to MultiStepLR')

    elif 'chex' in dataset_name:
        scheduler = CosineAnnealingLR(optimizer, all_epochs, learning_rate * gamma, verbose = True)
        print('Set learning rate scheduler to CosineAnnealingLR')

    elif dataset_name == 'rsna':
        scheduler = CosineAnnealingLR(optimizer, all_epochs, learning_rate * gamma, verbose = True)
        print('Set learning rate scheduler to CosineAnnealingLR')
    return optimizer, scheduler


# ---------------------------------------------------------------------------------------------------------------
# Training on batches for only one epoch
# Return(training loss and accuracy)
def train_one_epoch(model,
                    data_loader,
                    num_batches,
                    optimizer,
                    device,
                    epoch,
                    class_weights=None):
    losses = []
    all_target_labels = np.array([], dtype=np.int64)
    all_pred_labels = np.array([], dtype=np.int64)

    print(f'Epoch # {epoch}')

    model.train()

    print("Start train")
    sys.stdout.flush()
    for batch_idx, (image, label, index) in enumerate(data_loader):
        all_target_labels = np.append(all_target_labels, label.detach().data)

        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)

        all_pred_labels = np.append(all_pred_labels, output.argmax(dim=1, keepdim=True).detach().cpu().data)
        if class_weights:
            loss = F.cross_entropy(output, label.to(device), weight=class_weights.to(device))
        else:
            loss = F.cross_entropy(output, label.to(device))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        # for clothing1m
        if batch_idx == num_batches:
            print('Reached maximum number of mini batches ..... ')
            break
    print("End train")
    sys.stdout.flush()
    loss_epoch_train = np.mean(losses)
    accuracy_epoch_train = metrics.accuracy_score(all_target_labels, all_pred_labels)
    print("Training loss: {}, Training accuracy: {} ".format(loss_epoch_train, accuracy_epoch_train))

    return loss_epoch_train, accuracy_epoch_train


# ----------------------------------------------------------------------------------------------------------------
def test_one_epoch(model,
                   data_loader,
                   device):
    test_metrics = {'test_acc': [], 'test_lss': [], 'cmatrix': [], 'f1': [], 'auc': [], 'fpr': [], 'tpr': [], 'thr': []}

    loss_test = 0
    n_test_samples = 0

    all_target_labels = np.array([], dtype=np.int64)
    all_pred_labels = np.array([], dtype=np.int64)
    all_prob_labels = np.array([])

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        print("Start test")
        sys.stdout.flush()
        for image, label, _ in data_loader:
            all_target_labels = np.append(all_target_labels, label.detach().data)
            n_test_samples += len(label)

            n_test_samples += len(label)
            image, label = image.detach().to(device), label.detach().to(device)
            output = model(image)
            loss_fn = nn.CrossEntropyLoss()

            loss_test = loss_test + (len(label) * (loss_fn(output, label).item()))

            all_pred_labels = np.append(all_pred_labels, output.argmax(dim=1, keepdim=True).detach().cpu().data)
            if len(np.unique(label.detach().cpu().data)) <= 2:
                all_prob_labels = np.append(all_prob_labels, output[::, 1].detach().cpu().data)
        print("End test")
        sys.stdout.flush()
    loss_test /= n_test_samples
    test_metrics['test_lss'].append(loss_test)

    accuracy_test = metrics.accuracy_score(all_target_labels, all_pred_labels)
    test_metrics['test_acc'].append(accuracy_test)
    print("Average test loss(weighted): {: .4f}, test accuracy: {: .4f}".format(loss_test, accuracy_test))

    cf_matrix = confusion_matrix(all_target_labels, all_pred_labels)
    test_metrics['cmatrix'].append(cf_matrix)

    if len(np.unique(label.detach().cpu().data)) == 2:
        fpr, tpr, thr = metrics.roc_curve(all_target_labels, all_prob_labels)
        auc = metrics.roc_auc_score(all_target_labels, all_prob_labels)
        f1 = metrics.f1_score(all_target_labels, all_pred_labels, average='weighted')

        test_metrics['f1'].append(f1)
        test_metrics['auc'].append(auc)
        test_metrics['fpr'].append(fpr)
        test_metrics['tpr'].append(tpr)
        test_metrics['thr'].append(thr)
        print("F1 score (weighted): {: .4f}, Area under ROC: {: .4f}".format(f1, auc))
    return test_metrics

def select_probability(probs, top5=False):
    if top5:
        # Get the indices of the top 5 probabilities
        top_5_indices = np.argsort(probs)[-3:]
        # Randomly select one of the top 5 indices
        new_label = np.random.choice(top_5_indices)
    else:
        new_label = probs.argmax()
    return new_label
# ----------------------------------------------------------------------------------------------------------------
def eval_train(model: torch.nn.Module,
               data_loader: DataLoader,
               sample_size: int,
               alpha: float,
               num_classes: int,
               device: str,
               label_correction_active: bool = False,
               clustering = None):
    """
    Evaluate the model on the training data loader and identify clean and noisy samples.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the training dataset.
        sample_size (int): Total number of samples in the dataset.
        alpha (float): Threshold for confidence error to classify samples as clean or noisy.
        num_classes (int): Number of classes in the dataset.
        label_correction_active (bool, optional): Whether to perform label correction. Defaults to False.
        correction_threshold (float, optional): Confidence threshold for label correction.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
            - clean_labels_epoch (torch.Tensor): Tensor containing indices of clean samples.
            - noisy_labels_epoch (torch.Tensor): Tensor containing indices of noisy samples.
            - corrected_labels
            - label_pred (np.ndarray): Predicted labels for all samples.
            - probs (torch.Tensor): Predicted probabilities for all samples.
            - per_sample_ce (np.ndarray): Confidence error for each sample.
    """
    label_pred = np.empty(sample_size)
    label_pred[:] = np.nan
    probs = np.empty((sample_size, num_classes))
    probs[:] = np.nan
    per_sample_ce = np.empty(sample_size)
    per_sample_ce[:] = np.nan
    small_ce = []
    large_ce = []

    var_output_dataset = np.ones(sample_size).tolist()
    corrected_labels = {}

    with torch.no_grad():
        print("Start eval")
        sys.stdout.flush()
        for image, label, index in data_loader:

            output = model(image.to(device))
                    
            y_pred = F.softmax(output, dim=1)
            y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
            loss_ce = F.cross_entropy(output.cpu(), label, reduction='none')

            var_output_batch = [(float(y_pred[index_local].max() - y_pred[index_local][label[index_local]])) for
                            index_local in range(len(y_pred))]
            
            for g_i, l_i in zip(index, np.arange(0, len(var_output_batch))):
                var_output_dataset[g_i] = var_output_batch[l_i]
                if clustering != "none":
                    continue
                label_pred[g_i] = output.argmax(dim=1, keepdim=True)[l_i]


                probs[g_i] = y_pred[l_i].cpu().detach()
                per_sample_ce[g_i] = loss_ce[l_i].cpu().detach()
                
                if var_output_dataset[g_i] <= alpha:
                    small_ce.append(g_i)
                elif label_correction_active and probs[g_i].max() >= 0.99:
                    small_ce.append(g_i)
                    new_label = select_probability(probs[g_i], top5=False)
                    corrected_labels[g_i.item()] = new_label
                else:
                    large_ce.append(g_i)
        
        if clustering == "kmeans":
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=2)
            var_output_dataset_np = np.array(var_output_dataset)
            clusters = kmeans.fit_predict(var_output_dataset_np.reshape(-1, 1))

            cluster_centers = kmeans.cluster_centers_
            clean_cluster = np.argmin(cluster_centers)

            # Assign samples based on cluster center proximity
            small_ce = np.where(clusters == clean_cluster)[0]
            large_ce = np.where(clusters != clean_cluster)[0]
        elif clustering == "agglo":
            # Perform Agglomerative Clustering
            agglo = AgglomerativeClustering(n_clusters=2)
            clusters = agglo.fit_predict(var_output_dataset_np.reshape(-1, 1))

            # Assign samples based on cluster labels
            clean_cluster = np.argmin([var_output_dataset_np[clusters == i].mean() for i in range(2)])
            small_ce = np.where(clusters == clean_cluster)[0]
            large_ce = np.where(clusters != clean_cluster)[0]
        elif clustering == "gmm":
            # Perform GMM clustering
            gmm = GaussianMixture(n_components=2)
            var_output_dataset_np = np.array(var_output_dataset)
            clusters = gmm.fit_predict(var_output_dataset_np.reshape(-1, 1))

            # Assign samples based on cluster center proximity
            cluster_centers = gmm.means_
            clean_cluster = np.argmin(cluster_centers)
            small_ce = np.where(clusters == clean_cluster)[0]
            large_ce = np.where(clusters != clean_cluster)[0]
        elif clustering == "dbscan":
            # Perform DBSCAN clustering
            var_output_dataset_np = np.array(var_output_dataset)
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(var_output_dataset_np.reshape(-1, 1))

            # Assign samples based on cluster labels
            clean_cluster = np.argmin(np.bincount(clusters[clusters != -1]))  # Ignore noise points (-1)
            small_ce = np.where(clusters == clean_cluster)[0]
            large_ce = np.where(clusters != clean_cluster)[0]
        elif clustering == "meanshift":
            # Perform Mean Shift Clustering
            mean_shift = MeanShift()
            clusters = mean_shift.fit_predict(var_output_dataset_np.reshape(-1, 1))

            # Assign samples based on cluster labels
            clean_cluster = np.argmin(np.bincount(clusters))
            small_ce = np.where(clusters == clean_cluster)[0]
            large_ce = np.where(clusters != clean_cluster)[0]


        print("End eval")
        sys.stdout.flush()
       
        clean_labels_epoch = torch.LongTensor(small_ce)
        noisy_labels_epoch = torch.LongTensor(large_ce)

        return clean_labels_epoch, noisy_labels_epoch, corrected_labels, label_pred, probs, per_sample_ce

