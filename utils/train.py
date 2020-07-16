import numpy as np
import pandas as pd

import tqdm

import torch
from torch.utils.data import DataLoader

#Класс для обучения модели
class Trainer:
    def __init__(self, model, criterion, device, learning_rate):
        
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.device = device
        self.history_buffer = None
        self.history_size = None
        
    def train_step(self, dataloader):
        
        self.model.train()
        total_loss = 0.
        
        for X_batch, y_batch in dataloader:
            
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            X_batch = X_batch.unsqueeze(dim=-1)
            
            pred = self.model(X_batch)
            
            loss = self.criterion(pred, y_batch)
            loss.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item() * len(X_batch)
                    
        total_loss = total_loss / len(dataloader.dataset)
        return total_loss
    
    def eval_step(self, dataloader):
        
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            
            for X_batch, y_batch in dataloader:
                
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)   
                X_batch = X_batch.unsqueeze(dim=-1)

                pred = self.model(X_batch)

                loss = self.criterion(pred, y_batch)
                total_loss += loss.item() * len(X_batch)

        total_loss = total_loss / len(dataloader.dataset)
        return total_loss 
    
    def fit(self, train_dataloader, val_dataloader=None, gap=None, n_epochs=10, verbose=False):
        
            self.history_size = train_dataloader.dataset.time_series.shape[1]
            self.history_buffer = list(train_dataloader.dataset.targets.detach().cpu().numpy()[::-1][:self.history_size][::-1])
            
            if val_dataloader is not None:
                val=True
            else:
                val=False                 
                
            best_score = np.inf
            
            for epoch in range(n_epochs):
                
                loss_train = self.train_step(train_dataloader)
                loss = '{:>3}/{} train: {:.4f} |'.format(epoch, n_epochs, loss_train)
                
                if val:
                    loss_val = self.eval_step(val_dataloader)
                    loss = ' {} val: {:.4f}'.format(loss, loss_val)    
                    
                    if gap is not None:
                        if loss_val < best_score:
                            best_score = loss_val
                            patience = 0
                        else:
                            patience += 1
                            if patience == gap:
                                break
                        
                if (epoch % 10 == 0) & verbose:
                    print(loss)
                   
            self.model.cpu()  

        
    def predict(self, n_pred=30):
    
        predict = self.history_buffer
        
        self.model.eval()
        with torch.no_grad():
            for i in range(n_pred):
                value = predict[-self.history_size:]
                x = torch.tensor(value).unsqueeze(0).unsqueeze(-1).float()
                predict.append(self.model(x).detach().cpu().numpy()[0])

        return predict[::-1][:n_pred][::-1]