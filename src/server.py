# Server.py:
#       - model aggregate
#       - model test
#       - model save

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import EEGNet
from utils import weights_init

class Server(object):
    def __init__(self, args, test_dataset):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        bn_track = False if args.fedbs else True
        self.global_model = EEGNet(
            sample_rate=args.sample_rate, channels=args.channels, F1=args.F1, 
            D=args.D, F2=args.F2, time=args.samples, class_num=args.class_num, 
            drop_out=args.dropout, bn_track=bn_track
        ).to(self.device)

        self.global_model.apply(weights_init) 
        self.test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    def model_aggregate(self, client_weight_dict, avg_weight_dict, candidates_id_list):
        for id in candidates_id_list:
            for name, data in self.global_model.state_dict().items():
                update_per_layer = client_weight_dict[id][name] * avg_weight_dict[id]
                
                if data.type() != update_per_layer.type():
                    data.add_(update_per_layer.to(torch.int64))  
                else:
                    data.add_(update_per_layer)

    def model_test(self):
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_acc, test_loss = 0, 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                # DATA IS ALREADY ON GPU - NO NEED TO MOVE IT!
                y_hat = self.global_model(X)
                
                test_loss += criterion(y_hat, y).item()
                pred = y_hat.max(1)[1]
                y_true = y.max(1)[1]
                test_acc += pred.eq(y_true).sum().item()

        test_loss /= len(self.test_dataloader.dataset)
        test_acc /= len(self.test_dataloader.dataset)

        return test_loss, test_acc

    def model_save(self, path):
        torch.save(self.global_model.state_dict(), path)