# Client.py
#      - local training
#      - local evaluation

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import EEGNet
from utils import weights_init
from SAM import SAM

class Client(object):
    def __init__(self, args, train_dataset, eval_dataset, id=-1):
        self.args = args
        self.client_id = id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # FedBS disables batch norm tracking
        bn_track = False if args.fedbs else True
        self.local_model = EEGNet(
            sample_rate=args.sample_rate, channels=args.channels, F1=args.F1, 
            D=args.D, F2=args.F2, time=args.samples, class_num=args.class_num, 
            drop_out=args.dropout, bn_track=bn_track
        ).to(self.device)
        
        self.local_model.apply(weights_init) 
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    def local_train(self, global_model):
        # 1. Sync local model with the server's global model
        for name, param in global_model.state_dict().items():
            if self.args.fedbs and 'bn' in name:
                continue # FedBS skips BatchNorm layers during sync
            self.local_model.state_dict()[name].copy_(param.clone())

        self.local_model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.lr, weight_decay=1e-4, momentum=0.9)

        # 2. Train the model on the local client data
        for epoch in range(self.args.local_epochs):
            for X, y in self.train_dataloader:
                X, y = X.to(self.device), y.to(self.device)

                if self.args.fedbs:
                    # FedBS uses SAM (Sharpness-Aware Minimization)
                    minimizer = SAM(optimizer, self.local_model, self.args.rho)
                    
                    # Ascent Step (Find the sharpest loss)
                    criterion(self.local_model(X), y).backward()
                    minimizer.ascent_step()
                    
                    # Descent Step (Minimize it)
                    criterion(self.local_model(X), y).backward()
                    minimizer.descent_step()
                else:
                    # Standard FedAvg uses basic SGD
                    optimizer.zero_grad()
                    loss = criterion(self.local_model(X), y)
                    loss.backward()
                    optimizer.step()

        # 3. Calculate the weight difference to send back to the server
        weight_diff = {}
        for name, data in self.local_model.state_dict().items():
            weight_diff[name] = (data - global_model.state_dict()[name])

        return weight_diff

    def local_eval(self, global_model):
        # Sync weights before testing
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
            
        self.local_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        eval_acc, eval_loss = 0, 0
        with torch.no_grad():
            for X, y in self.eval_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.local_model(X)
                
                eval_loss += criterion(y_hat, y).item()
                pred = y_hat.max(1)[1]
                y_true = y.max(1)[1]
                eval_acc += pred.eq(y_true).sum().item()

        eval_loss /= len(self.eval_dataloader.dataset)
        eval_acc /= len(self.eval_dataloader.dataset)

        return eval_loss, eval_acc