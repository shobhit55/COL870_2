train_data_dir

#Clustering
class sudoku_char_data(Dataset):
    def __init__(self, dir):
        self.data_dir = dir
        self.img = None
    
    def __getitem__(self, index):
        if index%64==0:
            self.img = cv2.imread(self.data_dir+'/'+str(int(index/64))+'.png', 0) #224x224 array
        # print(img.shape)
        dign = index%64
        row = int(dign/8)
        col = dign%8
        return torch.tensor(self.img[row*28:(row+1)*28, col*28:(col+1)*28], dtype=torch.float32)
    
    def __len__(self):
        onlyfiles = next(os.walk(self.data_dir))[2]
        return len(onlyfiles)*64

num_clusters = 8
dataset = sudoku_char_data(train_data_dir+'/target')
true_x = torch.tensor(np.load('data/sample_images_sudoku.npy'), dtype=torch.float32)
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

class BasicBlockBN(nn.Module):
    def __init__(self, input_dim, output_dim, identity_downsample = None, stride=1, norm_layer = 'torch_bn'):
        super(BasicBlockBN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias = False)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias = False)
        self.norm_layer = norm_layer
        self.bn1 = nn.BatchNorm2d(output_dim).to(device)
        self.bn2 = nn.BatchNorm2d(output_dim).to(device)    
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.init_weights()

    def init_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          nn.init.xavier_uniform_(m.weight, gain = 2)
        if isinstance(m, nn.Linear):
          nn.init.kaiming_uniform_(m.weight)
      
    def forward(self, x):
        input = x.to(device)
        x = self.conv1(x)
        x = self.bn1(x).to(device)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x).to(device)
        if self.identity_downsample is not None:
            input = self.identity_downsample(input).to(device)
        x = x + input
        x = self.relu(x)
        return x

class Resnet(nn.Module):
    def __init__(self, block, n_layers, num_classes = 10, input_dim = 3, norm_layer = 'torch_bn'):
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 16, kernel_size = 3, stride=1, padding=1, bias = False) #28x28 output # Random crop left
        self.norm_layer = norm_layer
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace = True)
        self.layer1 = self.layer(16, 16, block, n_layers, stride=1) #28x28 output, 16channels
        self.layer2 = self.layer(16, 32, block, n_layers) #14x14 output, 32channels
        self.layer3 = self.layer(32, 64, block, n_layers) #7x7 output, 64channels
        self.pool_out = nn.AvgPool2d(kernel_size=7) #1x1 output, 64 channels
        self.fc_out_layer = nn.Linear(64,num_classes) # fully connected output layer
        self.init_weights()
        self.fea = None

    def layer(self, input_dim, output_dim, block, num_blocks, stride=2):
        bn = nn.BatchNorm2d(output_dim).to(device)
        if stride!=1:
          cov = nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=2, bias =False)
          nn.init.kaiming_uniform_(cov.weight)
          identity_downsample_1 = nn.Sequential(
                                                    cov,
                                                    bn
                                                  ).to(device)
        else:
            identity_downsample_1 = None

        layers = []
        layers.append(block(input_dim, output_dim, identity_downsample_1, stride, norm_layer = self.norm_layer ).to(device)) #increases channels and downsamples feature map
        for i in range(num_blocks-1):
            layers.append(block(output_dim, output_dim, norm_layer = self.norm_layer).to(device))
        return nn.Sequential(*layers)

    def init_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          nn.init.xavier_uniform_(m.weight, gain = 2)

    def pr(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          print(m.weight)
        if isinstance(m, nn.Linear):
          print(m.weight)
    
    def get_fea(self):
      return self.fea

    def sudoku_check(grid):
        for i in range(8):
            if sum(grid[i])!=sum(set(grid[i])):
                return 1
            if sum(grid[:,i])!=sum(set(grid[:,i])):
                return 1
            if sum(grid[int(i/2)*2:int(i/2)*2+1, (i%2)*4:(i%2)*4+4])!=sum(set(grid[int(i/2)*2:int(i/2)*2+1, (i%2)*4:(i%2)*4+4])):
                return 1
        return 0

    def forward(self, x, typ='sudoku'): #batch, 64, 1, 28, 28
        if typ == 'sudoku':
            x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.bn1(x) #.to(device)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool_out(x).to(device)
        x = x.view(-1,64)
        self.fea = x.clone().detach().cpu()
        x = self.fc_out_layer(x)

        x_ad = x.view(-1, 64, )
        return x

class sudoku_data_train(Dataset):
    def __init__(self, sudoku_digits_data, transform=None):
        self.data_file = sudoku_digits_data[:int(0.8*len(sudoku_digits_data))]
        self.transform = transform

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, index):
        x = self.data_file[index]
        y = int(x[0])
        x = torch.tensor(np.reshape(x[1:], (28,28) ), dtype=torch.float32).view(1,28,28)
        return x, y

def train_classifier(model, train_loader, optim_key):
    epochs = 14
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
    for epoch in range(start_epoch, epochs+1):
      model.train()
      for batch, (input, target) in enumerate(train_loader):
          input, target = input.to(device), target.to(device) 
          optimizer.zero_grad()
          output = model(input) #.to(device)
          loss = criterion(output, target)
          loss.backward()
          optimizer.step()
    print("Training Done...")
    return  model

optim_key = 'AdaDel'
train_data = sudoku_data_train(sudoku_digits_data, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=1024, num_workers=0 if device==torch.device('cpu') else 2, shuffle=True, drop_last=True)

torch.cuda.empty_cache()
class_model = Resnet(BasicBlockBN, n_layers=2, num_classes=9, input_dim=1).to(device)
class_model = train_classifier(class_model, train_loader, optim_key=optim_key)

#RRN------------------------------------------------------------------------------------------------------------
class sudoku_data_8(Dataset):
    def __init__(self, folderq, foldert, typ):
        self.data_dirq = folderq
        self.data_dirt = foldert
        self.typ = typ
    
    def __getitem__(self, index):
        if self.typ=='val':
            maxlen = 8000
        else:
            maxlen = 0
        imgq = torch.tensor(cv2.imread(self.data_dirq+'/'+str(index+maxlen)+'.png', 0), dtype=torch.float32)#, device=device) #224x224 array
        imgt = torch.tensor(cv2.imread(self.data_dirt+'/'+str(index+maxlen)+'.png', 0), dtype=torch.float32)#, device=device) #224x224 array
        imgq = torch.transpose(imgq.reshape(8,28,8,28), 1,2).reshape(64,1,28,28)
        imgt = torch.transpose(imgt.reshape(8,28,8,28), 1,2).reshape(64,1,28,28)
        return imgq, imgt
    
    def __len__(self):
        onlyfiles = next(os.walk(self.data_dirq))[2]
        if self.typ=='train':
            return int(len(onlyfiles)*0.8)
        elif self.typ=='val':
            return int(len(onlyfiles)*0.2)

class def_MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(def_MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(self, x):
        return self.layer(x)

class RRN(nn.Module):
    def __init__(self, hidden_dim, embed_size, n_steps = 32, grid_size=8):
        super(RRN, self).__init__()

        self.grid_size = grid_size
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim
        self.sudoku_edges = self.sudoku_edges()

        self.msg_func = def_MLP(2*hidden_dim, hidden_dim)   #f
        self.input_func = def_MLP(3*embed_size, hidden_dim) #xj
        self.pred = nn.Linear(hidden_dim, grid_size)          #r
        self.lstmcell = nn.LSTMCell(2*hidden_dim, hidden_dim) #LSTMG

        self.digit_embed = nn.Embedding(grid_size+1, embed_size)
        self.row_embed = nn.Embedding(grid_size, embed_size)
        self.col_embed = nn.Embedding(grid_size, embed_size)
    
    def compute_acc(self, pred, target): #batch, grid*grid; batch, grid, grid
        target = target.view(-1, self.grid_size*self.grid_size)-1
        c = pred==target
        c = torch.sum(c, dim=1)
        return len(c[c==self.grid_size*self.grid_size])/len(c)

    def sudoku_edges(self): #returns a list where list[j] is a list of cells in N(j)
        def cross(a):
            return np.array([[i for i in a.flatten() if not i == j] for j in a.flatten()])

        idx = np.arange(self.grid_size*self.grid_size).reshape(self.grid_size, self.grid_size)
        rows, columns, squares = -np.ones((self.grid_size*self.grid_size, self.grid_size-1), dtype=np.int), -np.ones((self.grid_size*self.grid_size, self.grid_size-1), dtype=np.int), -np.ones((self.grid_size*self.grid_size, self.grid_size-1), dtype=np.int)
        edges = []
        
        for i in range(self.grid_size):
            rows[idx[i, :].flatten()] = cross(idx[i, :])
            columns[idx[:, i].flatten()] = cross(idx[:, i])
        if self.grid_size==9:
            for i in range(3):
                for j in range(3):
                    squares[idx[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3].flatten()] = cross(idx[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3])
        elif self.grid_size==8:
            for i in range(4):
                for j in range(2):
                    squares[idx[i * 2:(i + 1) * 2, j * 4:(j + 1) * 4].flatten()] = cross(idx[i * 2:(i + 1) * 2, j * 4:(j + 1) * 4])

        for j in range(self.grid_size*self.grid_size):
            edges.append(list(set(list(rows[j])+list(columns[j])+list(squares[j]))))
        
        return edges #grid*grid, n_edges - 17 for 8x8 and 20 for 9x9

    def sudoku_check(grid):
        for i in range(8):
            if sum(grid[i])!=sum(set(grid[i])):
                return 1
            if sum(grid[:,i])!=sum(set(grid[:,i])):
                return 1
            if sum(grid[int(i/2)*2:int(i/2)*2+1, (i%2)*4:(i%2)*4+4])!=sum(set(grid[int(i/2)*2:int(i/2)*2+1, (i%2)*4:(i%2)*4+4])):
                return 1
        return 0

    def msg_passing(self, h_t, edges):
        h_edges = h_t[:,edges]
        h_t = h_t.unsqueeze(2).expand(-1,-1,len(edges[0]),-1) #batch_size, grid*grid, n_edges, hidden_dim 
        msg_func_tens = torch.cat((h_edges, h_t), dim=3) #batch_size, grid*grid, n_edges, 2*hidden_dim        
        msg_func_tens = self.msg_func(msg_func_tens) #batch_size, grid*grid, n_edges, hidden_dim
        msg_func_tens = torch.sum(msg_func_tens, dim=2) #batch_size, grid*grid, hidden_dim
        return msg_func_tens

    def node_update_func(self, x, msg_func_tens, h_t, s_t):
        lstm_in = torch.cat((msg_func_tens,x), dim=2).view(-1,2*self.hidden_dim) #batch_size*grid*grid, 2*hidden_dim
        h_t = h_t.view(-1,self.hidden_dim)
        h_t, s_t = self.lstmcell(lstm_in, (h_t, s_t))
        h_t = h_t.view(-1, self.grid_size*self.grid_size, self.hidden_dim) #batch_size, grid*grid, hidden_dim
        return h_t, s_t

    def forward(self, x, target):
        # x - shape(batch,grid,grid)
        x = torch.tensor(x, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        batch_size = x.shape[0]
        
        x = self.digit_embed(x.reshape(batch_size, -1)) #batch, grid*grid, emb_size
        row_embed = self.row_embed(torch.tensor(np.arange(self.grid_size), device=x.device).unsqueeze(1).expand(-1,self.grid_size)).view(self.grid_size*self.grid_size,self.embed_size).unsqueeze(0).expand(batch_size,-1,-1) #batch, grid*grid, emb_size
        col_embed = self.col_embed(torch.tensor(np.arange(self.grid_size), device=x.device).unsqueeze(0).expand(self.grid_size,-1)).view(self.grid_size*self.grid_size,self.embed_size).unsqueeze(0).expand(batch_size,-1,-1) #batch, grid*grid, emb_size
        x = self.input_func(torch.cat((x,row_embed,col_embed), dim=2)) #batch, grid*grid, hidden_dim

        h_t = torch.zeros(batch_size, self.grid_size*self.grid_size, self.hidden_dim).to(x.device) #shape(batch, grid*grid, hidden_dim)
        s_t = torch.zeros(batch_size*self.grid_size*self.grid_size, self.hidden_dim).to(x.device)
        
        edges = self.sudoku_edges

        loss=torch.tensor(0, dtype=torch.float32, device=x.device)
        acc=[]
        for i in range(self.n_steps):            
            msg_func_tens = self.msg_passing(h_t, edges) #batch_size, grid*grid, hidden_dim
            h_t, s_t = self.node_update_func(x, msg_func_tens, h_t, s_t)
            out_t = self.pred(h_t) #batch_size, grid*grid, grid_size
            out_t = nn.functional.log_softmax(out_t, dim=2) #batch_size, grid*grid, grid_size
            acc.append(self.compute_acc(torch.argmax(out_t, dim=2), target))
            out_t = out_t.view(-1,self.grid_size)
            loss_n = (-torch.sum(out_t[range(out_t.shape[0]),target.view(-1)-1]))/(batch_size*self.grid_size*self.grid_size)
            loss += loss_n
        loss = loss/(self.n_steps)
        return loss, acc, torch.argmax(out_t.view(-1, self.grid_size*self.grid_size, self.grid_size), dim=2)

def train_rrn(model, train_loader, class_model):
    print("RRN Training Started...")
    wd = 1e-4
    epochs = 30
    lr = 2e-4
    optim_key = 'Adam'
    optimizer_dict = { "AdaDel": torch.optim.Adadelta(model.parameters(), weight_decay=wd), "Adam": torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd), "SGD": torch.optim.SGD(model.parameters(), lr = lr, weight_decay=wd) }
    optimizer = optimizer_dict[optim_key]
    if device==torch.device('cuda'):
        torch.cuda.synchronize()
    start_epoch = 1
    model.train()
    for epoch in range(start_epoch, epochs+1):
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

grid_size = 8
batch_size = 64
class_model.eval()
train_data = sudoku_data_8(train_data_dir+'/query', train_data_dir+'/target', typ='train')
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2 if device==torch.device('cuda') else 0, pin_memory=False)

gc.collect()
torch.cuda.empty_cache()
model = RRN(hidden_dim=96, embed_size=16, n_steps=32, grid_size=grid_size).to(device)
model = train_rrn(model, train_loader, class_model)