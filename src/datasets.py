import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import scipy.io as scio
import torchvision.transforms as transforms
from scipy.linalg import fractional_matrix_power
import os

class EA(object):
    """Euclidean Alignment (EA) transformation for EEG data."""
    def __call__(self, x):
        new_x = np.zeros_like(x)
        for i in range(x.shape[0]):
            cov = np.zeros((x.shape[1], x.shape[2], x.shape[2]))
            for j in range(x.shape[1]):
                cov[j] = np.cov(x[i, j])
            refEA = np.mean(cov, axis=0)
            sqrtRefEA = fractional_matrix_power(refEA, -0.5)
            new_x[i] = np.matmul(sqrtRefEA, x[i])
        return new_x

class ArrayToTensor(object):
    """Converts numpy arrays to PyTorch FloatTensors."""
    def __call__(self, x):
        return torch.from_numpy(x).type(torch.FloatTensor)

class ZScoreNorm(object):
    """Z-Score Normalization for EEG data."""
    def __call__(self, x):       
        new_x = np.zeros_like(x)
        for i in range(x.shape[0]):
            temp_x = x[i, 0] 
            for j in range(1, x.shape[1]):
                temp_x = np.concatenate((temp_x, x[i, j]), axis=1) 
            mean_c = np.mean(temp_x, axis=1, keepdims=True) 
            std_c = np.std(temp_x, axis=1, keepdims=True) 
            new_x[i] = (x[i] - mean_c) / std_c
        return new_x

class MIDataset(Dataset):
    """Motor Imagery Dataset Loader for .mat files."""
    def __init__(self, random_state, subject_id: list, root='./data/BNCI2014001', mode='train', test_size=0.2, data_transform=None, label_transform=None):
        self.mode = mode
        
        data_transform = [t for t in data_transform if t is not None] if data_transform else []
        label_transform = [t for t in label_transform if t is not None] if label_transform else []
        self.data_transform = transforms.Compose(data_transform)
        self.label_transform = transforms.Compose(label_transform)

        X, y = [], []
        for i in subject_id:
            mat_path = os.path.join(root, f"{i}.mat")
            data = scio.loadmat(mat_path)
            
            if 'BNCI2015001' in root:
                data['X'] = data['X'][:400]
                data['y'] = data['y'][:400]
            
            N, C, T = data['X'].shape
            X.append(data['X'])
            df = pd.get_dummies(data['y'])
            y.append(df.to_numpy()) 

        self.data = np.array(X)
        self.label = np.array(y).reshape(N * len(subject_id), -1)
        
        self.data = self.data_transform(self.data)
        self.label = self.label_transform(self.label)
        self.data = self.data.reshape(N * len(subject_id), 1, C, T)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.data, self.label, test_size=test_size, random_state=random_state
        )

        combined = list(zip(self.data, self.label))
        np.random.seed(random_state)
        np.random.shuffle(combined)
        self.data, self.label = zip(*combined)

        # ---------------------------------------------------------
        # THE GPU FIX: PRE-LOAD ALL DATA DIRECTLY TO VRAM
        # ---------------------------------------------------------
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.X_train = self.X_train.to(device)
        self.y_train = self.y_train.to(device)
        self.X_val = self.X_val.to(device)
        self.y_val = self.y_val.to(device)
        
        # Convert the shuffled tuples back into stacked GPU tensors
        self.data = torch.stack(self.data).to(device)
        self.label = torch.stack(self.label).to(device)

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.X_train[index], self.y_train[index]
        elif self.mode == 'val':
            return self.X_val[index], self.y_val[index]
        elif self.mode == 'all':
            return self.data[index], self.label[index]

    def __len__(self):
        if self.mode == 'train':
            return len(self.X_train)
        elif self.mode == 'val':
            return len(self.X_val)
        elif self.mode == 'all':
            return len(self.data)