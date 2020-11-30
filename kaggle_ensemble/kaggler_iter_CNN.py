import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch import FloatTensor, LongTensor
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import CriterionCallback, EarlyStoppingCallback, AccuracyCallback
from catalyst.contrib.nn.optimizers import RAdam, Lookahead
import collections
import numpy as np

'''Code found here https://www.kaggle.com/yakuben/crgl2020-iterative-cnn-approach/notebook'''

class OneIterationReverseNet(nn.Module):
    def __init__(self, info_ch, ch):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(info_ch, ch, 5, padding=4, padding_mode='circular')
        self.conv2 = nn.Conv2d(ch, ch, 3, )
        self.conv3 = nn.Conv2d(ch, info_ch, 3)
        
        
    def forward(self, inp):
        x = self.relu(self.conv1(inp))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x
    
class ReverseModel(nn.Module):
    def __init__(self, info_ch=64, ch=128):
        super().__init__()
        self.relu = nn.ReLU()
        self.encoder = nn.Conv2d(1, info_ch, 7, padding=3, padding_mode='circular')# you can use other model
        self.reverse_one_iter = OneIterationReverseNet(info_ch, ch)# you can use other model
        self.decoder = nn.Conv2d(info_ch, 1, 3, padding=1, padding_mode='circular')# you can use other model
        
    
    def forward(self, stop, delta):
        x = self.relu(self.encoder(stop-0.5))
        
        for i in range(delta.max().item()):
            y = self.reverse_one_iter(x)
            
            # this 2 lines allow use samples with different delta in one batch
            mask = (delta > i).reshape(-1,1,1,1)
            x = x*(~mask).float() + y*mask.float()
        
        x = self.decoder(x)
        
        return x 
    
train_val = pd.read_csv('../data/train.csv', index_col='id')
test = pd.read_csv('../data/test.csv', index_col='id')

train, val = train_test_split(train_val, test_size=0.2, shuffle=True, random_state=42, stratify=train_val['delta'])


def line2grid_tensor(data, device='cuda'):
    grid = data.to_numpy().reshape((data.shape[0], 1, 25, 25))
    return FloatTensor(grid).to(device)



class TaskDataset(Dataset):
    def __init__(self, data, device='cuda'):
        self.delta = LongTensor(data['delta'].to_numpy()).to(device)
        if data.shape[1] == 1251: 
            self.start = line2grid_tensor(data.iloc[:,1:626], device)
            self.stop = line2grid_tensor(data.iloc[:,626:], device)
        else:
            self.start = None
            self.stop = line2grid_tensor(data.iloc[:,1:], device)
        
    def __len__(self):
        return len(self.delta)

    def __getitem__(self, idx):
        if self.start is None:
            return {'stop': self.stop[idx], 'delta': self.delta[idx]}
        return {'start': self.start[idx], 'stop': self.stop[idx], 'delta': self.delta[idx]}



dataset_train = TaskDataset(train)
dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)

dataset_val = TaskDataset(val)
dataloader_val = DataLoader(dataset_val, batch_size=128, shuffle=False)

dataset_test = TaskDataset(test)
dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False)




runner = SupervisedRunner(device='cuda', input_key=['stop', 'delta'], )

loaders = {'train': dataloader_train, 'valid': dataloader_val}#collections.OrderedDict({'train': dataloader_train, 'valid': dataloader_val})

model = ReverseModel()

optimizer = Lookahead(RAdam(params=model.parameters(), lr=1e-3))

criterion = {"bce": nn.BCEWithLogitsLoss()}

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

callbacks = [
        CriterionCallback(input_key='start', prefix="loss", criterion_key="bce"),
        EarlyStoppingCallback(patience=5),
    ]

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=callbacks,
    logdir="./logs",
    num_epochs=5,#TODO 
    main_metric="loss",
    minimize_metric=True,
    verbose=True,
)


best_model = ReverseModel().to('cuda')
best_model.load_state_dict(torch.load('logs/checkpoints/best.pth')['model_state_dict'])



def predict_batch(model, batch):
    model.eval()
    with torch.no_grad():
        prediction = model(batch['stop'], batch['delta'])
        prediction = torch.sigmoid(prediction).detach().cpu().numpy()
        return prediction

    
def predict_loader(model, loader):
    predict = [predict_batch(model, batch) for batch in loader]
    predict = np.concatenate(predict)
    return predict


def validate_loader(model, loader, lb_delta=None, threshold=0.5):
    prediction_val = predict_loader(best_model, loader)
    y_val = loader.dataset.start.detach().cpu().numpy()
    delta_val = loader.dataset.delta.detach().cpu().numpy()

    score = ((prediction_val > threshold) == y_val).mean(axis=(1,2,3))
    print(f'All data accuracy: {score.mean()}')
        
    delta_socre = {}
    for i in range(1, 6):
        delta_socre[i] = score[delta_val==i].mean()#print(f'delta={i} accuracy: {score[delta_val==i].mean()}')
        print(f'delta={i} accuracy: {delta_socre[i]}')
        
    if lb_delta is not None:
        lb_delta = lb_delta.value_counts(normalize=True)
        test_score = sum([lb_delta[i]*delta_socre[i] for i in range(1,6)])
        print(f'VAL score         : {1-score.mean()}')
        print(f'LB  score estimate: {1-test_score}')
    
    
def make_submission(prediction, threshold=0.5, sample_submission_path='./sample_submission.csv'):
    prediction = (prediction > threshold).astype(int).reshape(-1, 625)
    
    sample_submission = pd.read_csv(sample_submission_path, index_col='id')
    sample_submission.iloc[:] = prediction
    return sample_submission


validate_loader(best_model, dataloader_val, test['delta'])

prediction_test = predict_loader(best_model, dataloader_test)


submission = make_submission(prediction_test)
submission.to_csv('iter_CNN_pred.csv')






