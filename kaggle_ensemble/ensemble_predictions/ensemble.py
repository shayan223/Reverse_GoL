import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm
import os
import csv

### GLOBAL CONSTANTS ###
SIM_ALL_MODELS = False
SAVE_ACCURACIES = False
SAVE_PATH = './' 
OUTPUT_FILE_NAME = 'Extra_Pruned_Ensemble_Accuracies.csv'
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
#### UNCOMMENT TO CHECK GAME LOGIC. should get 100% accuracy when running on solutions

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

test_data = pd.read_csv('../../data/test.csv')
#print(test_data)
test_data = test_data.drop('id', axis='columns')
test_deltas = test_data['delta']
test_data = test_data.drop('delta', axis='columns')
test_data = test_data.to_numpy()


########  Create list of models to ensemble  #######

pathList = ['./Iter_CNN.csv',
            './forward_loss_iter_CNN.csv',
            './prob_extension.csv',
            './quick_neighborhood.csv',
            './neural_CNN.csv',
            './z3_constraint.csv',
            './Iter_CNN_With_Post.csv',
            './rand_forest.csv',
            './GAN.csv'
            ]

modelList = []

########  Create ensemble prediction  ############
iters = test_data.shape[0]

for path in pathList:
    #read in every listed model output
    newModel = pd.read_csv(path)
    #trim excess information to format for simulation/validation
    newModel.drop('id', axis='columns', inplace=True)
    newModel = newModel.to_numpy()
    #after model has been prepped for simulation, add it to list
    modelList.append(newModel)
    
    
### Basic estimation with simple average ###
#use the first for all even, second for weights.
predictions = np.around(np.mean(modelList, axis=0))
#predictions = np.around(np.average(modelList, weights=[1,1,1,1], axis=0))

if(predictions.shape != test_data.shape):
    print("WARNING: solutions and pradiction shapes do not match!")
    print("Predictions: ",predictions.shape)
    print("Solutions: ",test_data.shape)

if(np.array_equal(predictions, predictions.astype(bool)) == False):
    print("WARNING: Non-binary solutions, this is probably wrong!")

### Run simulation on each model, finding individual accuracy ###
accuracies = {}

if(SIM_ALL_MODELS == True):
    for i, model in enumerate(modelList):
        total_score = 0
        max_score = test_data.shape[0]
        
        for j in tqdm(range(iters)):
            if(checkStart(model[j], test_data[j], test_deltas[j],shaped=False)):
                total_score += 1
        
        modelString = os.path.basename(pathList[i])
        acc = total_score/max_score
        print(modelString, '  Accuracy: ', acc)
        accuracies[modelString] = acc
    


####  Run simulation on predictions  ######

total_score = 0
max_score = test_data.shape[0]

for i in tqdm(range(iters)):
    if(checkStart(predictions[i], test_data[i], test_deltas[i],shaped=False)):
        total_score += 1

acc = total_score/max_score
print('Accuracy: ', acc)

accuracies['Ensemble'] = acc

if(SAVE_ACCURACIES == True):
    '''
    with open('accuracies.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Model','Accuracy'])
        writer.writeheader()
        print(accuracies)
        print(type(accuracies))
        writer.writerows(accuracies)'''
    df = pd.DataFrame.from_dict(accuracies, orient="index")
    df.columns = ['Accuracy']
    df.to_csv(OUTPUT_FILE_NAME)






