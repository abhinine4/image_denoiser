import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from dncnn import Dncnn
from dataloader import Residual_data
from utils import AverageMeter
from config import *

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    if gnl is not None:
        gnl = list(map(lambda x: int(x), gnl.split(',')))
    if df is not None:
        df = list(map(lambda x: int(x), df.split(',')))
    if jq is not None:
        jq = list(map(lambda x: int(x), jq.split(',')))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.manual_seed(seed)

    if arch == 'Dncnn-S':
        model = Dncnn(num_layers=17)
    elif arch == 'Dncnn-B':
        model = Dncnn(num_layers=20)
    elif arch == 'Dncnn-3':
        model = Dncnn(num_layers=20)

      
    model = model.cuda()
    # print(model.parameters)

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = Residual_data(image_dir, patch_size, gnl, df, jq, False) 
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=threads,
                            pin_memory=True,
                            drop_last=True)

    for epoch in range(num_epochs):
        epoch_losses = AverageMeter()

        with tqdm(total=(len(dataset) - len(dataset)%batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch+1, num_epochs))
            for data in dataloader:
                inputs, labels = data

                inputs = inputs.cuda()
                labels = labels.cuda()

                preds = model(inputs)

                loss = criterion(preds, labels)/(2*len(inputs))

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs))        
    torch.save(model.state_dict(),os.path.join(save_weights, '{}_epoch_{}.pth'.format(arch, epoch+1)))

