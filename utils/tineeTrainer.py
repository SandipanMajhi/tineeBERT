from Models.bert import TineeBERT
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm



class TineeBERTTrainer:
    def __init__(self, task = "mlm", lr = 1e-4, num_epochs = 10000, device = "cpu"):
        self.task = task


    def pretrain(self, model, train_loader, optimizer, loss_fn, epochs):
        pass


