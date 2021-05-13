import torch
import torch.nn as nn
import numpy as np

# device = torch.device('cuda')

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

class BasicBlockBN(nn.Module):
    def __init__(self, input_dim, output_dim, identity_downsample = None, stride=1, norm_layer = 'torch_bn'):
        super(BasicBlockBN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias = False)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias = False)
        self.norm_layer = norm_layer
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.bn2 = nn.BatchNorm2d(output_dim)
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
        input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            input = self.identity_downsample(input)
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
        bn = nn.BatchNorm2d(output_dim)#.to(device)
        if stride!=1:
          cov = nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=2, bias =False)
          nn.init.kaiming_uniform_(cov.weight)
          identity_downsample_1 = nn.Sequential(
                                                    cov,
                                                    bn
                                                  )#.to(device)
        else:
            identity_downsample_1 = None

        layers = []
        layers.append(block(input_dim, output_dim, identity_downsample_1, stride, norm_layer = self.norm_layer ))#.to(device)) #increases channels and downsamples feature map
        for i in range(num_blocks-1):
            layers.append(block(output_dim, output_dim, norm_layer = self.norm_layer))#.to(device))
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
        x = self.pool_out(x)#.to(device)
        x = x.view(-1,64)
        self.fea = x.clone().detach().cpu()
        x = self.fc_out_layer(x)

        x_ad = x.view(-1, 64, )
        return x
