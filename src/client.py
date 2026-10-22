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
        for name, param in global_model.state_dict().items():
            if self.args.fedbs and 'bn' in name:
                continue 
            self.local_model.state_dict()[name].copy_(param.clone())

        self.local_model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.lr, weight_decay=1e-4, momentum=0.9)

        for epoch in range(self.args.local_epochs):
            for X, y in self.train_dataloader:
                # DATA IS ALREADY ON GPU - NO NEED TO MOVE IT!

                if self.args.fedbs:
                    minimizer = SAM(optimizer, self.local_model, self.args.rho)
                    
                    criterion(self.local_model(X), y).backward()
                    minimizer.ascent_step()
                    
                    criterion(self.local_model(X), y).backward()
                    minimizer.descent_step()
                else:
                    optimizer.zero_grad()
                    loss = criterion(self.local_model(X), y)
                    loss.backward()
                    optimizer.step()
                
                # Apply the MaxNormConstraint after the step
                self.local_model.MaxNormConstraint()

        weight_diff = {}
        
        # --- DIFFERENTIAL PRIVACY ENGINE ---
        # These parameters will eventually be passed from train.py via self.args
        apply_dp = getattr(self.args, 'dp', False)
        clip_bound = getattr(self.args, 'clip_bound', 1.0)
        noise_multiplier = getattr(self.args, 'noise_multiplier', 0.1)

        # 1. Calculate raw weight differences
        for name, data in self.local_model.state_dict().items():
            weight_diff[name] = (data - global_model.state_dict()[name])

        if apply_dp:
            # 2. Calculate the global L2 norm of the entire weight update
            total_norm = 0.0
            for name, diff in weight_diff.items():
                if diff.dtype in [torch.float32, torch.float64]:
                    total_norm += diff.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            # 3. Calculate scaling factor for clipping
            clip_factor = clip_bound / max(1e-6, total_norm)
            clip_factor = min(1.0, clip_factor) # Only scale down, never up

            # 4. Apply clipping and inject Gaussian noise
            for name, diff in weight_diff.items():
                if diff.dtype in [torch.float32, torch.float64]:
                    # Clip
                    clipped_diff = diff * clip_factor
                    # Add Noise: std_dev = clip_bound * noise_multiplier
                    noise = torch.normal(
                        mean=0.0, 
                        std=clip_bound * noise_multiplier, 
                        size=clipped_diff.size(), 
                        device=clipped_diff.device
                    )
                    weight_diff[name] = clipped_diff + noise
        # -----------------------------------

        return weight_diff

    def local_eval(self, global_model):
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
            
        self.local_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        eval_acc, eval_loss = 0, 0
        with torch.no_grad():
            for X, y in self.eval_dataloader:
                # DATA IS ALREADY ON GPU - NO NEED TO MOVE IT!

                y_hat = self.local_model(X)
                
                eval_loss += criterion(y_hat, y).item()
                pred = y_hat.max(1)[1]
                eval_acc += pred.eq(y).sum().item()

        eval_loss /= len(self.eval_dataloader.dataset)
        eval_acc /= len(self.eval_dataloader.dataset)

        return eval_loss, eval_acc