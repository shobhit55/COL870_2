import torch
import torch.nn as nn
import numpy as np

def train_rrn(model, train_loader, class_model, device=torch.device('cuda')):
    print("RRN Training Started...")
    wd = 1e-4
    epochs = 53
    lr = 2e-4
    optim_key = 'Adam'
    optimizer_dict = {"AdaDel": torch.optim.Adadelta(model.parameters(), weight_decay=wd), "Adam": torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd), "SGD": torch.optim.SGD(model.parameters(), lr = lr, weight_decay=wd)}
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
    for epoch in range(start_epoch, epochs+1):
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
    def sudoku_check(grid):
        for i in range(8):
            if sum(grid[i])!=sum(set(grid[i])):
                return 0
            if sum(grid[:,i])!=sum(set(grid[:,i])):
                return 0
            if np.sum(grid[int(i/2)*2:int(i/2)*2+2, (i%2)*4:(i%2)*4+4])!=sum(list(set(grid[int(i/2)*2:int(i/2)*2+2, (i%2)*4:(i%2)*4+4].flatten()))):
                return 0
        return 1
    for i in range(len(output_full)):
        correct+=sudoku_check(output_full[i].reshape(8,8))
        txt = ','.join(list(map(str, output_full[i])))
        out_list.append(str(i)+'.png,'+txt)
    print('Proportion of valid sudoku boards: ', correct/len(output_full))
    output_str = '\n'.join(out_list)
    f = open(output_file,"w")
    f.write(output_str)
    f.close()