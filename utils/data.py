import numpy as np

import torch
from torch.utils.data import Dataset

#Класс для трансформации временного ряда. В качестве трансофрмации используется логарифт относительного прироста
class DataTransform:
    
    def __init__(self, const):
        #Значение, относительного которого будет произведена трасформация
        self.const = const
    
    def transform(self, data):
        
        result = []
        for idx, value in enumerate(data):
            if idx == 0:
                result.append(np.log(value / self.const))
            else:
                result.append(np.log(value / data[idx - 1]))               
        return result
    
    def inverse_transform(self, data):
        
        result = []
        for idx, value in enumerate(data):
            if idx == 0:
                result.append(self.const)
            else:
                result.append(result[idx - 1] * np.exp(value))
        return result
    
#Класс для создания структуры, необходимой для нейронной сети    
class TSDataset(Dataset):
    
    def __init__(self, time_series, history_size=30):
        """
        time_series - временной ряд
        history_size - количество предыдущих значений, которые учитываются при построении модели
        """
        
        self.history_size = history_size
        self.len = len(time_series) - history_size
        
        self.time_series = []
        self.targets = []
        for idx, ts in enumerate(time_series[self.history_size:], start=self.history_size):
            self.time_series.append(time_series[idx - self.history_size: idx])
            self.targets.append(ts)
            
        self.time_series = torch.tensor(self.time_series).float()
        self.targets = torch.tensor(self.targets).float()
        
    def __getitem__(self, idx):
        
        if self.targets is not None:
            return self.time_series[idx], self.targets[idx]

    def __len__(self):
        
        return self.len