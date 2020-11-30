'''Code found on this kaggle submission
https://www.kaggle.com/yakuben/crgl-probability-extension-true-target-problem'''

import pandas as pd
import torch
import numpy as np
from torch import FloatTensor
from torch.utils.data import IterableDataset, DataLoader
import torch.nn as nn
from torch.nn import BCELoss
from torch.optim import Adam
from tqdm.notebook import trange, tqdm

torch.cuda.empty_cache()

neighbors_roll_axes = [(i,j) for i in range(-1,2) for j in range(-1, 2) if not (i==0 and j==0)]


def binary_forward_iteration(grid, delta=1):
    for _ in range(delta):
        neighbor_sum = torch.cat([torch.roll(torch.roll(grid, i, 2), j, 3) for i,j  in neighbors_roll_axes], dim=1)
        neighbor_sum = neighbor_sum.sum(dim=1, keepdim=True)
        grid = ((neighbor_sum == 3) | ((grid==1)  & (neighbor_sum == 2)))
    return grid


neighbors_roll_axes = [(i,j) for i in range(-1,2) for j in range(-1, 2) if not (i==0 and j==0)]


combination_alive2 = [(i,j) for i in range(8) for j in range(i)]
combination_alive2_dead6 = [([i,j]+[8+k for k in range(8) if (k!=i and k!=j)]) for i,j in combination_alive2]

combination_alive3 = [(i,j,k) for i in range(8) for j in range(i) for k in range(j)]
combination_alive3_dead5 = [([i,j,k]+[8+l for l in range(8) if (l!=i and l!=j and l!=k)]) for i,j,k in combination_alive3]


def get_neighbors(grid):
    return torch.stack([torch.roll(torch.roll(grid, i, 2), j, 3) for i,j  in neighbors_roll_axes])

def n_neigbors_nearby_prob(neighbors, neighbor_nearby=2):
    if neighbor_nearby==2:
        combination = combination_alive2_dead6
    else:
        combination = combination_alive3_dead5
    neighbors = torch.cat([neighbors, 1 - neighbors])
    return torch.stack([neighbors[c].prod(dim=0) for c in combination]).sum(dim=0)


def probabilistic_forward_iteration_autograd(grid):
    neighbors = get_neighbors(grid)

    neighbors_p2 = n_neigbors_nearby_prob(neighbors, 2)
    neighbors_p3 = n_neigbors_nearby_prob(neighbors, 3)

    alive_prob = neighbors_p3 + neighbors_p2*grid
    return alive_prob


neighbor_alive2_cell_alive = {}
neighbor_alive2_cell_dead = {}

neighbor_alive3_cell_alive = {}
neighbor_alive3_cell_dead = {}

for cell in range(8):
    neighbor_alive2_cell_alive[cell] = [(cell,j) for j in range(8) if j!=cell]
    neighbor_alive2_cell_alive[cell] = [([j]+[8+k for k in range(8) if (k!=i and k!=j)]) for i,j in neighbor_alive2_cell_alive[cell]]
    
    neighbor_alive2_cell_dead[cell] = [(i,j) for i in range(8) for j in range(i) if i!=cell and j!=cell]
    neighbor_alive2_cell_dead[cell] = [([i,j]+[8+k for k in range(8) if (k!=i and k!=j and k!=cell)]) for i,j in neighbor_alive2_cell_dead[cell]]
    
    neighbor_alive3_cell_alive[cell] = [(i,j,cell) for i in range(8) for j in range(i) if i!=cell and j!=cell]
    neighbor_alive3_cell_alive[cell] = [([i,j]+[8+l for l in range(8) if (l!=i and l!=j and l!=k)]) for i,j,k in neighbor_alive3_cell_alive[cell]]

    neighbor_alive3_cell_dead[cell] = [(i,j,k) for i in range(8) for j in range(i) for k in range(j) if i!=cell and j!=cell and k!=cell]
    neighbor_alive3_cell_dead[cell] = [([i,j,k]+[8+l for l in range(8) if (l!=i and l!=j and l!=k and l!=cell)]) for i,j,k in neighbor_alive3_cell_dead[cell]]


def get_neighbors_backward(grad_output):
    return torch.stack([torch.roll(torch.roll(grad_output[idx], -i, 2), -j, 3) for idx, (i,j)  in enumerate(neighbors_roll_axes)]).sum(dim=0)


def n_neigbors_nearby_prob_backward(grad_output, neighbors, neighbor_nearby=2):
    if neighbor_nearby==2:
        combination_cell_alive = neighbor_alive2_cell_alive
        combination_cell_dead = neighbor_alive2_cell_dead
    else:
        combination_cell_alive = neighbor_alive3_cell_alive
        combination_cell_dead = neighbor_alive3_cell_dead
    
    neighbors = torch.cat([neighbors, 1 - neighbors])
    coef = []
    for cell in range(8):
        cell_live_coef = torch.stack([neighbors[l].prod(dim=0) for l in combination_cell_alive[cell]]).sum(dim=0)
        cell_dead_coef = torch.stack([neighbors[d].prod(dim=0) for d in combination_cell_dead[cell]]).sum(dim=0)
        coef.append(cell_live_coef-cell_dead_coef)
    coef = torch.stack(coef)
    return coef*grad_output


class ProbabilisticForwardIteration(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, delta=1):
        ctx.grid = grid
        return probabilistic_forward_iteration_autograd(grid)
    

    @staticmethod
    def backward(ctx, grad_out):
        grid = ctx.grid
        neighbors = get_neighbors(grid)
        neighbors_p2 = n_neigbors_nearby_prob(neighbors, neighbor_nearby=2)     
        
        grad_n2_out = grad_out*grid
        grad_n3_out = grad_out
        
        grad_n2_inp = n_neigbors_nearby_prob_backward(grad_n2_out, neighbors, neighbor_nearby=2)
        grad_n3_inp = n_neigbors_nearby_prob_backward(grad_n3_out, neighbors, neighbor_nearby=3)
        
        grad_neighbors_out = grad_n2_inp + grad_n3_inp
        
        grad_neighbors_inp = get_neighbors_backward(grad_neighbors_out)
        
        grad_inp = grad_neighbors_inp + neighbors_p2*grad_out
        return grad_inp, None
    
    
def probabilistic_forward_iteration(grid, delta=1, autograd=True):
    """autograd=False slower but use less memory"""
    if autograd:
        for _ in range(delta):
            grid = probabilistic_forward_iteration_autograd(grid)
    else:
        for _ in range(delta):
            grid = ProbabilisticForwardIteration.apply(grid)
    return grid


    


neighbors_roll_axes = [(i,j) for i in range(-1,2) for j in range(-1, 2) if not (i==0 and j==0)]


def generate_random_start_batch(batch_size):
    return np.random.randint(low=0, high=2, size=(batch_size, 1, 25, 25), dtype=bool)

def straight_iter_binary_numpy(grid, delta=1):
    for _ in range(delta):
        neighbor_sum = np.concatenate([np.roll(np.roll(grid, i, 2), j, 3) for i,j  in neighbors_roll_axes], axis=1)
        neighbor_sum = neighbor_sum.sum(axis=1, keepdims=True)
        grid = ((neighbor_sum == 3) | ((grid==1)  & (neighbor_sum == 2)))
    return grid


class DataStream():
    def __init__(self, delta=None, batch_size=128, drop_empty=False, drop_ch_dim=False):
        self.init_delta = delta
        self.batch_size = batch_size
        self.drop_empty= drop_empty
        self.drop_ch_dim = drop_ch_dim
        
    def __iter__(self):
        while True:
            x = generate_random_start_batch(self.batch_size)
            delta = self.init_delta if self.init_delta else np.random.randint(1,6)
            x = straight_iter_binary_numpy(x, 5+delta)
            
            if self.drop_empty:
                x = x[x.any(axis=2).any(axis=2).reshape(-1)]
                
            if self.drop_ch_dim:
                x = x[:,0,:,:]
            
            yield x.astype(float), delta    
    
class DataStreamTorch(IterableDataset):
    def __init__(self, delta=None, batch_size=128, drop_empty=False, drop_ch_dim=False):
        self.ds = DataStream(delta, batch_size, drop_empty, drop_ch_dim)
        
    def __iter__(self):
        for x, delta in self.ds:
            yield FloatTensor(x), delta
            

def pass_collate(batch):
    return batch[0]


def get_datastream_loader(delta=None, batch_size=128, drop_empty=False, drop_ch_dim=False, num_workers=0):
    dataset = DataStreamTorch(delta, batch_size, drop_empty, drop_ch_dim)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=pass_collate, num_workers=num_workers)
    return dataloader   


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 512, 7, padding=3, padding_mode='circular')
        self.conv2 = nn.Conv2d(512, 256, 5, padding=2, padding_mode='circular')
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1, padding_mode='circular')
        self.conv4 = nn.Conv2d(256, 1, 1)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x
    
    
class FixPredictBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(5, 256, 5, padding=2, padding_mode='circular')
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1, padding_mode='circular')
        self.conv3 = nn.Conv2d(256, 256, 1)
        self.conv4 = nn.Conv2d(256, 1, 3, padding=1, padding_mode='circular')
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x, x_prev_pred):
        with torch.no_grad():
            x_prev_pred_bin = x_prev_pred>0.5
            x_pred_bin = binary_forward_iteration(x_prev_pred_bin)
            x_pred = probabilistic_forward_iteration(x_prev_pred)
        x = torch.cat([x, x_prev_pred, x_prev_pred_bin.float(), x_pred, x_pred_bin.float()], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fix_pred = FixPredictBlock()
    
    def forward(self, x, n_it=5):
        x_prev_pred = x
        for i in range(n_it):
            x_prev_pred = self.fix_pred(x, x_prev_pred)
        return x_prev_pred   
    
    


N_iter = 2#2000
device = 'cpu' #'cuda'
loader = get_datastream_loader(batch_size=128, num_workers=0, drop_empty=True, delta=1)
model = Model().to(device)
criterion = BCELoss()
optimizer = Adam(model.parameters(), lr=1e-3)

tqdm_loader = tqdm(loader)
for i, (stop_state, _) in enumerate(tqdm_loader):
    print(i)
    stop_state = stop_state.to(device)
    
    optimizer.zero_grad()
    start_state_prediction = model(stop_state)
    stop_state_prediction = probabilistic_forward_iteration(start_state_prediction)
    loss = criterion(stop_state_prediction, stop_state)
    
    loss.backward()
    optimizer.step()
        
    with torch.no_grad():
        bce = loss.item()
        start_state_alive = (start_state_prediction>0.5).float().mean().item()
        accuracy = ((stop_state_prediction > 0.5) == (stop_state>0.5)).float().mean().item()
        accuracy_true = (binary_forward_iteration(start_state_prediction>0.5)==(stop_state>0.5)).float().mean().item()
    
    tqdm_loader.postfix = 'bce: {:0.10f} | start_state_alive: {:0.5f} | accuracy: {:0.10f} | accuracy_true: {:0.10f}'\
    .format(bce, start_state_alive, accuracy, accuracy_true)
    
    if i > N_iter:
        tqdm_loader.close()
        break

for param in model.parameters():
    param.requires_grad = False
    
model.eval()


for batch in loader:
    stop_state = batch[0]#.cuda()#TODO enable this when using cuda instead of cpu
    break
    
for n_iter in [1,10,100]:
    acc = (stop_state == binary_forward_iteration(model(stop_state, n_iter) > 0.5)).float().mean().item()
    print(f'model n_iter={n_iter} accuracy: {acc}')
    
    
    
    


def direct_gradient_optimization(batch, n_iter, lr, device='cuda', reduse_alife=False):
    stop_state = batch
    start_state = nn.Parameter(torch.rand(stop_state.shape).to(device)-1)
    criterion = BCELoss()
    optimizer = Adam([start_state], lr=lr,)
    tqdm_loader = trange(n_iter)
    for _ in tqdm_loader:
        optimizer.zero_grad()
        start_state_prob = torch.sigmoid(start_state)
        stop_state_prediction = probabilistic_forward_iteration(start_state_prob, autograd=False)
        
        bce_loss = criterion(stop_state_prediction, stop_state)
        start_state_alive = start_state_prob.mean()
        if reduse_alife and start_state_alive.item() > 0:
            loss = bce_loss + start_state_alive
        else:
            loss = bce_loss
            
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            bce = bce_loss.item()
            alive_cells = start_state_alive.item()
            accuracy = ((stop_state_prediction > 0.5) == (stop_state>0.5)).float().mean().item()
            accuracy_true = (binary_forward_iteration(start_state_prob>0.5)==(stop_state>0.5)).float().mean().item()

        tqdm_loader.postfix = 'bce: {:0.10f} | start_state_alive: {:0.5f} | accuracy: {:0.10f} | accuracy_true: {:0.10f}'.format(bce, alive_cells, accuracy, accuracy_true)
    
    return torch.sigmoid(start_state.detach())#.cpu().reshape(-1,625)


def direct_gradient_optimization_predict(data, delta, n_iter=100, lr=1, device='cuda'):
    data = FloatTensor(np.array(data)).reshape((-1, 1, 25, 25)).to(device)
    for i in range(delta-1):
        data = direct_gradient_optimization(data, n_iter, lr, reduse_alife=True, device=device)
        data = (data>0.5).float()
    
    data = direct_gradient_optimization(data, n_iter, 1, reduse_alife=False, device=device)
    return (data>0.5).detach().cpu().int().reshape(-1,625).numpy()    
    
    
    

test = pd.read_csv('../data/test.csv', index_col='id')
submission = pd.read_csv('./sample_submission.csv', index_col='id')
    
    
for delta in range(1,6):
    mask = test['delta']==delta
    data = test[mask].iloc[:,1:]
    submission[mask] = direct_gradient_optimization_predict(data, delta, n_iter=4, lr=1,device=device)
    
    
submission.to_csv('prob_extend_kaggle.csv')
   
    
    
    
    
    
    
    
    
    
