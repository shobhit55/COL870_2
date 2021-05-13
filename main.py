from models import RRN, Resnet, BasicBlockBN
from datasets import sudoku_char_data, sudoku_data_8, sudoku_data_train
from train_loops import train_classifier, train_rrn
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import distance
import gc
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description = 'Argument parser to automate experiments-running process.')
parser.add_argument('-train_dir', '--train_data_dir', type = str, default = '', action = 'store')
parser.add_argument('-test_dir', '--test_data_dir', type = str, default = '', action = 'store')
parser.add_argument('-sam', '--sample_file', type = str, default = '', action = 'store')
parser.add_argument('-out', '--output_file', type = str, default = '', action = 'store')
args = parser.parse_args()

output_file = args.output_file
train_data_dir = args.train_data_dir
test_data_dir = args.test_data_dir
sample_file = args.sample_file

#Clustering
num_clusters = 8
dataset = sudoku_char_data(train_data_dir+'/target')
true_x = torch.tensor(np.load(sample_file), dtype=torch.float32)
true_x = true_x.view(true_x.shape[0], -1)

x_all_0 = np.zeros((len(dataset), 784))
labels_all_0 = np.zeros(len(dataset))
dataloader = DataLoader(dataset, batch_size=int(len(dataset)/8), num_workers=0 if device==torch.device('cpu') else 2, drop_last=True)

k=0
for i, x in enumerate(dataloader):
    x = x.view(x.shape[0], -1)
    x_new = x
    cluster = KMeans(n_clusters=num_clusters, random_state=0, verbose=False, max_iter=300, tol=1e-6)
    cluster.fit(x_new)
    labels = list(cluster.labels_)
    true_labels = list(cluster.predict(true_x[1:-1]))

    try:
        labels = list(map(lambda x: true_labels.index(x), labels))
        x_all_0[k:k+len(x),:] = x
        labels_all_0[k:k+len(x)] = np.array(labels)+1
        k = k+len(x)
    except:
        print('continued')
x_all_0 = x_all_0[:k]
labels_all_0 = labels_all_0[:k]

x_7_9 = x_all_0[(labels_all_0==4) | (labels_all_0==7)]
cluster = KMeans(n_clusters=2, random_state=0, verbose=False, max_iter=300, tol=1e-6)
cluster.fit(x_7_9[:,28*4:17*28])
labels = list(cluster.labels_)
true_labels = list(cluster.predict(true_x[[4,7],28*4:17*28]))
labels = list(map(lambda x: true_labels.index(x), labels))
la = [4,7]
labels = list(map(lambda x: la[x], labels))
labels_all_0[(labels_all_0==4) | (labels_all_0==7)] = labels

dataset = sudoku_char_data(train_data_dir+'/query')
dataloader = DataLoader(dataset, batch_size=int(len(dataset)/8), num_workers=0 if device==torch.device('cpu') else 2)

x = next(iter(dataloader))
x = x.view(x.shape[0], -1)
pca = PCA(n_components=6) #6
x_new = pca.fit_transform(x)
cluster = KMeans(n_clusters=num_clusters, random_state=0, tol=1e-6)
cluster.fit(x_new)
labels = list(cluster.labels_)
true_label = cluster.predict(pca.transform(true_x[0:1]))
dist = []
def myfunc(e):
    return e[1]
for j in range(num_clusters):
    dist.append([j, distance.cosine(pca.transform(true_x[0:1]), cluster.cluster_centers_[j])])
dist.sort(key=myfunc)
labels = np.array(labels)
x_0 = x[(labels==dist[0][0]) | (labels==dist[1][0]) | (labels==dist[2][0])]
labels_0 = np.zeros(len(x_0))
x_all_0 = np.concatenate((x_all_0, x_0), axis=0)
labels_all_0 = np.concatenate((labels_all_0, labels_0))
labels_all_0 = np.expand_dims(labels_all_0, 1)
sudoku_digits_data = np.concatenate((labels_all_0, x_all_0), axis=1)
np.random.shuffle(sudoku_digits_data)

#Classifier------------------------------------------------------------------------------------------------------------
optim_key = 'SGD'
train_data = sudoku_data_train(sudoku_digits_data, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=1024, num_workers=0 if device==torch.device('cpu') else 2, shuffle=True, drop_last=True)

torch.cuda.empty_cache()
class_model = Resnet(BasicBlockBN, n_layers=1, num_classes=9, input_dim=1).to(device)
class_model = train_classifier(class_model, train_loader, optim_key=optim_key, device=device)

#RRN------------------------------------------------------------------------------------------------------------

grid_size = 8
batch_size = 64
class_model.eval()
train_data = sudoku_data_8(train_data_dir+'/query', train_data_dir+'/target', typ='train')
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2 if device==torch.device('cuda') else 0, pin_memory=False)

del sudoku_digits_data
gc.collect()
torch.cuda.empty_cache()
model = RRN(hidden_dim=96, embed_size=16, n_steps=32, grid_size=grid_size).to(device)
model = train_rrn(model, train_loader, class_model, device=device)