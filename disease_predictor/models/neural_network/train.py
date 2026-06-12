import argparse
import torch as tc
from datetime import datetime
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from .dataset import Dataset
from .model import DiseasePredictor
# python -m disease_predictor.models.neural_network.train -ep 5 -lr 0.1

parser = argparse.ArgumentParser()
parser.add_argument("--lr", "-lr", type=float, default=0.01)
parser.add_argument("--epoch", "-ep", type=int, default=10)
parser.add_argument("--device", "-d", type=str, default="cpu")
parser.add_argument("--load_state", "-ls", action="store_true")
args = parser.parse_args()

def train(
    epoch = 10,
    lr = 0.1
):
    model = DiseasePredictor()
    if(args.load_state):
        model.load_state()
    model.to(args.device)
    criterion = CrossEntropyLoss()
    optimizer = SGD(params=model.parameters(), lr=lr)
    train_batches = DataLoader(
        Dataset(),
        batch_size=256,
        shuffle=True
    )
    test_batches = DataLoader(
        Dataset(train=False),
        batch_size=256,
        shuffle=True
    )
    
    for ep in range(epoch):
        start = datetime.now()
        
        total_loss = 0
        train_count = 0
        train_accuracy = 0
        for X, Y in train_batches:
            X = X.to(args.device)
            Y = Y.to(args.device)
            
            Y_hat = model(X)
            pred = tc.argmax(Y_hat, dim=1)
            loss = criterion(Y_hat, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_count += 1
            pred = tc.argmax(Y_hat, dim=1)
            train_accuracy += (pred == Y).float().mean().item()
        
        test_count = 0
        test_accuracy = 0
        test_pos_percent_avg = 0
        for X, Y in test_batches:
            X = X.to(args.device)
            Y = Y.to(args.device)
            
            Y_hat = model(X)
            test_pos_percent = model.predict_proba(X)
            test_pos_percent = test_pos_percent[tc.arange(0, Y.shape[0]), Y].mean()
            test_pos_percent_avg += test_pos_percent.item()
            pred = tc.argmax(Y_hat, dim=1)
            
            test_count += 1
            pred = tc.argmax(Y_hat, dim=1)
            test_accuracy += (pred == Y).float().mean().item()
        
        end = datetime.now()    
        
        print(f"Ep: {ep} - T: {end - start} - L: {total_loss/train_count:.5f} - Train Acc: {train_accuracy/train_count:.5f} - Test Acc: {test_accuracy/test_count:.5f} - Test Pos Percent : {test_pos_percent_avg / test_count:.5f}")
        model.save_state()
    
train(args.epoch, args.lr)