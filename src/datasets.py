import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import scipy.io as scio
import torchvision.transforms as transforms
from scipy.linalg import fractional_matrix_power
import os
import pickle  # <-- Added for DEAP .dat files


class ArrayToTensor(object):
    """Converts numpy arrays to PyTorch FloatTensors."""
    def __call__(self, x):
        return torch.from_numpy(x).type(torch.FloatTensor)

class EA(object):
    """Euclidean Alignment (EA) transformation for EEG data."""
    def __call__(self, x):
        new_x = np.zeros_like(x)
        for i in range(x.shape[0]):
            cov = np.zeros((x.shape[1], x.shape[2], x.shape[2]))
            for j in range(x.shape[1]):
                cov[j] = np.cov(x[i, j])
            refEA = np.mean(cov, axis=0)

            sqrtRefEA = np.real(fractional_matrix_power(refEA, -0.5))
            
            new_x[i] = np.nan_to_num(np.matmul(sqrtRefEA, x[i]))
            
        return new_x


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
            
            std_c[std_c == 0] = 1e-8 
            
            new_x[i] = np.nan_to_num((x[i] - mean_c) / std_c)
            
        return new_x
    

class MIDataset(Dataset):
    """Dataset Loader for Motor Imagery (.mat) and Emotion Decoding (.dat) files."""
    def __init__(self, random_state, subject_id: list, root='./data/BNCI2014001', mode='train', test_size=0.2, data_transform=None, label_transform=None):
        self.mode = mode
        
        data_transform = [t for t in data_transform if t is not None] if data_transform else []
        label_transform = [t for t in label_transform if t is not None] if label_transform else []
        self.data_transform = transforms.Compose(data_transform)
        self.label_transform = transforms.Compose(label_transform)

        X, y = [], []
        
        for i in subject_id:
            
            # ==========================================
            # DEAP DATASET ROUTING
            # ==========================================
            if 'DEAP' in root.upper():
                # Load DEAP pickle file (s1.dat, s2.dat, etc.)
                dat_path = os.path.join(root, f"s{i}.dat")
                with open(dat_path, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                
                # 1. Extract EEG Data: shape is (40 trials, 40 channels, 8064 time steps)
                # We slice [:32] to drop the 8 peripheral channels and keep only the brainwaves.
                eeg_data = data['data'][:, :32, :]
                
                # 2. Extract Emotion Labels: shape is (40 trials, 4 labels)
                # Index 0 is Valence (Happiness). Index 1 is Arousal (Excitement).
                # We threshold the 1-9 rating at 5.0 to create a binary 0/1 classification.
                valence_ratings = data['labels'][:, 0]
                binary_labels = (valence_ratings > 5.0).astype(int)
                
                N, C, T = eeg_data.shape
                X.append(eeg_data)
                y.append(binary_labels)
                
            # ==========================================
            # BNCI MOTOR IMAGERY ROUTING
            # ==========================================
            else:
                mat_path = os.path.join(root, f"{i}.mat")
                data = scio.loadmat(mat_path)
                
                if 'BNCI2015001' in root:
                    data['X'] = data['X'][:400]
                    data['y'] = data['y'][:400]
                
                N, C, T = data['X'].shape
                X.append(data['X'])
                
                clean_labels = [str(item[0]) if isinstance(item, (np.ndarray, list)) else str(item) for item in data['y'].flatten()]
                df = pd.get_dummies(clean_labels)
                labels = df.to_numpy().argmax(axis=1)
                y.append(labels)

        # 1. Convert lists to numpy arrays and flatten the labels
        self.data = np.array(X)
        self.label = np.array(y).flatten() 

        # 2. Apply transformations (like Euclidean Alignment)
        self.data = self.data_transform(self.data)
        self.label = self.label_transform(self.label)
        
        # CRITICAL SAFEGUARD: CrossEntropyLoss requires labels to be LongTensors
        if isinstance(self.label, torch.Tensor):
            self.label = self.label.long()

        # 3. Reshape data to (Total_Trials, 1, Channels, Time)
        self.data = self.data.reshape(-1, 1, C, T)

        # 4. Split into train and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.data, self.label, test_size=test_size, random_state=random_state
        )

        combined = list(zip(self.data, self.label))
        np.random.seed(random_state)
        np.random.shuffle(combined)
        self.data, self.label = zip(*combined)

        # ---------------------------------------------------------
        # GPU VRAM PRE-LOADER
        # ---------------------------------------------------------
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.X_train = self.X_train.to(device)
        self.y_train = self.y_train.to(device)
        self.X_val = self.X_val.to(device)
        self.y_val = self.y_val.to(device)
        
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