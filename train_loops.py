import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from functools import partial
from datasets import sudoku_char_data
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import distance
tqdm = partial(tqdm, position=0, leave=True)

def clustering(train_data_dir, sample_file, device=torch.device('cuda')):
    print('Clustering started')
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
    print('Clustering done...')
    return sudoku_digits_data

def train_rrn(model, train_loader, class_model, device=torch.device('cuda')):
    print("RRN Training Started...")
    wd = 1e-4
    epochs = 53 #-------------------------------------------
    lr = 2e-4
    optim_key = 'Adam'
    optimizer_dict = {"AdaDel": torch.optim.Adadelta(model.parameters(), weight_decay=wd), "Adam": torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd), "SGD": torch.optim.SGD(model.parameters(), lr = lr, weight_decay=wd)}
    optimizer = optimizer_dict[optim_key]
    if device==torch.device('cuda'):
        torch.cuda.synchronize()
    start_epoch = 1
    model.train()
    for epoch in tqdm(range(start_epoch, epochs+1)):
        for i, (input, target) in enumerate(train_loader):
            optimizer.zero_grad()
            input, target = input.to(device), target.to(device)
            with torch.no_grad():
                pred = torch.argmax(class_model(torch.cat((input, target), dim=0).view(-1,1,28,28)), dim=1).reshape(2,-1,8,8)
            input, target = pred[0], pred[1]

            loss, _, _ = model(input, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5) #all
            optimizer.step()
    print("Training Done...")
    return  model

def train_classifier(model, train_loader, optim_key, device=torch.device('cuda')):
    epochs = 7
    lr = 0.001
    lr_ad = 0.005
    wd = 5e-4
    criterion = nn.CrossEntropyLoss()
    print("Training Started...")
    optimizer_dict = { "Adam": torch.optim.Adam(model.parameters(), lr = lr, weight_decay=wd), "SGD": torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9), "AdaDel": torch.optim.Adadelta(model.parameters(), lr=lr_ad, weight_decay=wd) }
    optimizer = optimizer_dict[optim_key]
    if device==torch.device('cuda'):
        torch.cuda.synchronize()
    start_epoch = 1
    for epoch in tqdm(range(start_epoch, epochs+1)):
        model.train()
        for batch, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device) 
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    print("Training Done...")
    return  model

def sudoku_check(grid):
    for i in range(8):
        if sum(grid[i])!=sum(set(grid[i])):
            return 0
        if sum(grid[:,i])!=sum(set(grid[:,i])):
            return 0
        # pdb.set_trace()
        if np.sum(grid[int(i/2)*2:int(i/2)*2+2, (i%2)*4:(i%2)*4+4])!=sum(list(set(grid[int(i/2)*2:int(i/2)*2+2, (i%2)*4:(i%2)*4+4].flatten()))):
            return 0
    return 1

def test_rrn(model, class_model, test_loader, output_file, device=torch.device('cuda')):
    model.eval()
    output_full = torch.empty(0, device=device)
    for i, input in enumerate(test_loader):
        input = input.to(device)
        with torch.no_grad():
            pred = torch.argmax(class_model(input.view(-1,1,28,28)), dim=1).reshape(-1,8,8)
        input = pred
        with torch.no_grad():
            _, _, output = model(input, input) #batch, 64
        output_full = torch.cat((output_full, output))
    
    output_full+=1
    output_full = output_full.cpu().to(dtype = torch.int).numpy()
    out_list = []
    correct=0
    for i in range(len(output_full)):
        correct+=sudoku_check(output_full[i].reshape(8,8))
        txt = ','.join(list(map(str, output_full[i])))
        out_list.append(str(i)+'.png,'+txt)
    print(correct, len(output_full))
    print('Proportion of valid sudoku boards: ', correct/len(output_full))
    output_str = '\n'.join(out_list)
    f = open(output_file,"w")
    f.write(output_str)
    f.close()