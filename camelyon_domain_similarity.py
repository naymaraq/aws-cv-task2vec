from task2vec import Task2Vec
from models import get_model
from datasets import get_dataset
from utils import read_yaml
import task_similarity
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import pandas as pd
import matplotlib.pyplot as plt


def main(dataset_list, dataset_names):

    _distances = []
    for _i in range(10):
        print("Run {}".format(_i))
        embeddings = []
        for name, dataset in list(zip(dataset_names, dataset_list)):
            probe_network = get_model('resnet34', pretrained=True, num_classes=2).cuda()
            embeddings.append(Task2Vec(probe_network,
                                    method = "variational",
                                    max_samples=25000,
                                    method_opts={"epochs": 3}).embed(dataset))
        ds_matrix = task_similarity.get_distance_matrix(embeddings, dataset_names)
        _distances.append(ds_matrix)

    mean = np.mean(_distances, axis=0)
    std = np.std(_distances, axis=0)

    np.save("mean.npy", mean)
    np.save("std.npy", std)

def plot_distance_matrix(distance_matrix_mean, distance_matrix_std, labels=None, save_to=None):
    distance_matrix = distance_matrix_mean
    cond_distance_matrix = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(cond_distance_matrix, method='complete', optimal_ordering=True)
    if labels is not None:
        distance_matrix = pd.DataFrame(distance_matrix, index=labels, columns=labels)
        distance_matrix_std = pd.DataFrame(distance_matrix_std, index=labels, columns=labels)

    sns.clustermap(distance_matrix, 
                  row_linkage=linkage_matrix, 
                  col_linkage=linkage_matrix, 
                  annot = distance_matrix,
                  cmap='viridis_r')
    if save_to:
        plt.savefig(save_to)

if __name__=="__main__":

    config = read_yaml("conf/config.yaml")

    (
        zero_domain_train, 
        third_domain_train, 
        forth_domain_train,
        zero_domain_train_aug,
        third_domain_train_aug,
        forth_domain_train_aug,
        second_domain_test, 
        first_domain_val
    ) = get_dataset(root=config.dataset.root, config=config.dataset)

    dataset_list = [zero_domain_train, 
                    third_domain_train,
                    forth_domain_train,
                    second_domain_test, 
                    first_domain_val]
    dataset_names = ["train_0", 
                    "train_3",
                    "train_4",
                    "test_2", 
                    "val_1"]

    #main(dataset_list, dataset_names)
    distance_matrix_mean, distance_matrix_std = np.load("mean.npy"), np.load("std.npy")
    plot_distance_matrix(distance_matrix_mean, 
                         distance_matrix_std, 
                         labels=dataset_names,
                         save_to="resnet34-epoch3-s25000-10runs.png")
