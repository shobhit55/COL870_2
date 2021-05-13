from torch.utils.data import Dataset
import os
import torch
import cv2
import numpy as np

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

class sudoku_data_8_test(Dataset):
    def __init__(self, folderq):
        self.data_dirq = folderq
    
    def __getitem__(self, index):
        imgq = torch.tensor(cv2.imread(self.data_dirq+'/'+str(index)+'.png', 0), dtype=torch.float32)#, device=device) #224x224 array
        # imgt = torch.tensor(cv2.imread(self.data_dirt+'/'+str(index)+'.png', 0), dtype=torch.float32)#, device=device) #224x224 array
        imgq = torch.transpose(imgq.reshape(8,28,8,28), 1,2).reshape(64,1,28,28)
        # imgt = torch.transpose(imgt.reshape(8,28,8,28), 1,2).reshape(64,1,28,28)
        return imgq
    
    def __len__(self):
        onlyfiles = next(os.walk(self.data_dirq))[2]
        return int(len(onlyfiles))

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