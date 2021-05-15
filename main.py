from models import RRN, Resnet, BasicBlockBN
from datasets import sudoku_data_8, sudoku_data_train, sudoku_data_8_test
from train_loops import train_classifier, train_rrn, test_rrn, clustering
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

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

#Clustering----------------------------------------------------------------------------------
sudoku_digits_data = clustering(train_data_dir, sample_file, device=device)

#Classifier------------------------------------------------------------------------------------------------------------
optim_key = 'SGD'
train_data = sudoku_data_train(sudoku_digits_data, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=1024, num_workers=0 if device==torch.device('cpu') else 2, shuffle=True, drop_last=True)

del sudoku_digits_data
gc.collect()
torch.cuda.empty_cache()
class_model = Resnet(BasicBlockBN, n_layers=1, num_classes=9, input_dim=1).to(device)
print('resnet defined')
class_model = train_classifier(class_model, train_loader, optim_key=optim_key, device=device)

#RRN------------------------------------------------------------------------------------------------------------

grid_size = 8
batch_size = 64
class_model.eval()
train_data = sudoku_data_8(train_data_dir+'/query', train_data_dir+'/target', typ='train')
test_data = sudoku_data_8_test(test_data_dir)
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2 if device==torch.device('cuda') else 0)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2 if device==torch.device('cuda') else 0)

torch.cuda.empty_cache()
model = RRN(hidden_dim=96, embed_size=16, n_steps=32, grid_size=grid_size).to(device)
model = train_rrn(model, train_loader, class_model, device=device)
test_rrn(model, class_model, test_loader, output_file, device=device)