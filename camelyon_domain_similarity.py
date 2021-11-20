from task2vec import Task2Vec
from models import get_model
from datasets import get_dataset
from utils import read_yaml
import task_similarity

config = read_yaml("conf/config.yaml")

zero_domain_train, third_domain_train, forth_domain_train, second_domain_test, first_domain_val = get_dataset(root=config.dataset.root, config=config.dataset)

# probe_network = get_model('resnet34', pretrained=True, num_classes=10)
# probe_network.to(config.device)

dataset_list = [zero_domain_train, third_domain_train, forth_domain_train, second_domain_test, first_domain_val]
dataset_names = ["train_0", "train_3", "train_4", "test_2", "val_1"]

embeddings = []
for name, dataset in zip(dataset_names, dataset_list):
    print(f"Embedding {name}")
    probe_network = get_model('resnet34', pretrained=True, num_classes=2).cuda()
    embeddings.append( Task2Vec(probe_network, max_samples=1000, skip_layers=6).embed(dataset) )
task_similarity.plot_distance_matrix(embeddings, dataset_names)