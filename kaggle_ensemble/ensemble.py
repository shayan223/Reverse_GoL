import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

#Function to check if starting frame is a solution        
def checkStart(start, end, iterCount, shaped=True, extra_dim=False):
    #if shaped = true -> 2d input expected
    #if shaped = false -> 1d input expected
    #start = initial GoL configuration (our solution)
    #end = ending configuration (what we start wtih)
    
    #reshape if necesary 
    if(shaped == False):
        start_2d = np.reshape(start, (25,25))
    else: 
        start_2d = start
    
    if(extra_dim == True):
        start_2d = start[0,:,:]
    
    for k in range(iterCount):
        #compute number of neighbors per cell
        nbrCount = testfilter(start_2d)
        
        #change every element in start_2d according to its neighbors
        for i in range(start_2d.shape[0]):
            for j in range(start_2d.shape[1]):
                start_2d[i][j] = gameLogic(start_2d[i][j],nbrCount[i][j])
        
        #if the new game state matches our end result, return true
        if(np.array_equal(start_2d.flatten(),end)):
            return True
    
    #if no match occurs after all iterations, start and end do not match
    return False 
        
#uses a convolution to compute the next GoL step 
def testfilter(x):
    filterWindow = np.array([[1,1,1],
                            [1,0,1],
                            [1,1,1]])
    #the filter represents the weight the convolution will multiply by
    #every element becomes the sum of its neighboors * 1 + itself*0
    result = ndimage.convolve(x,filterWindow,mode='wrap')
    return result
    
def gameLogic(x, nbr):
    
    #Any live cell with two or three live neighbours survives.
    if(x == 1 and (nbr == 2 or nbr == 3)):
        return 1
    #Any dead cell with three live neighbours becomes a live cell.
    if(x == 0 and (nbr == 3)):
        return 1
    #All other live cells die in the next generation. Similarly, all other dead cells stay dead.
    elif(x == 1):
        return 0
    #empty cells with no neighbors remain empty
    else:
        return 0;
    
    
'''######################     Validate simulation     ####################'''
'''
data = pd.read_csv('../data/train.csv')
print(data)
data.drop('id', axis='columns', inplace=True)
deltas = data['delta'].to_numpy()

starts = data.loc[:, 'start_0':'start_624'].to_numpy()
stops = data.loc[:, 'stop_0':'stop_624'].to_numpy()

ensemble_solution = starts
print(ensemble_solution.shape)
total_score = 0;
max_score = starts.shape[0]

for i in tqdm(range(ensemble_solution.shape[0])):
    if(checkStart(ensemble_solution[i], stops[i], deltas[i]+10,shaped=False)):
        total_score += 1

print('Accuracy: ', total_score/max_score)
'''

'''######################     Validation Predictions     ####################'''

test_data = pd.read_csv('../data/test.csv')
#print(test_data)
test_data = test_data.drop('id', axis='columns')
test_deltas = test_data['delta']
test_data = test_data.drop('delta', axis='columns')
test_data = test_data.to_numpy()


########  Create ensemble prediction  ############

model1 = pd.read_csv('./hashmap__solv_kaggle.csv')
model2 = pd.read_csv('./iter_CNN_pred.csv')
model3 = pd.read_csv('./prob_extend_kaggle.csv')
#print(model1)
model1.drop('id', axis='columns', inplace=True)
model2.drop('id', axis='columns', inplace=True)
model3.drop('id', axis='columns', inplace=True)

model1 = model1.to_numpy()
model2 = model2.to_numpy()
model3 = model3.to_numpy()

#print(model1)
#print(model2)
#print(model3)

## Basic estimation with simple average ##

#average all values, rounding to nearest number (1 or 0)
ensemble_solution = np.around(np.mean([model1, model2, model3], axis=0))
#counts = np.count_nonzero(ensemble_solution == 1)
#print(ensemble_solution)
#print(ensemble_solution.shape)
#print(counts)



total_score = 0;
max_score = test_data.shape[0]

for i in tqdm(range(ensemble_solution.shape[0])):
    if(checkStart(ensemble_solution[i], test_data[i], test_deltas[i]+10,shaped=False)):
        total_score += 1

print('Accuracy: ', total_score/max_score)












