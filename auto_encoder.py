import numpy as np
import pandas as pd
import keras
from keras import layers
from scipy import ndimage
import matplotlib as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import encoders

'''********************
Data being used is based on a 25 x 25 game of life box
that has been flattened to a list of length 625
********************'''

#Function to check if starting frame is a solution
def checkStart(start, end, iterCount):
    #BOTH AS 1D ARRAYS
    #start = initial GoL configuration (our solution)
    #end = ending configuration (what we start wtih)
    
    #convert from flattened to 2d matrix
    start_2d = np.reshape(start, (25,25))
    
    
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
    
    

''''##################    DATA INIT     ##################################'''

#TODO check to see whether adding deltas to training data is a good idea or not

#read data
data = pd.read_csv('./data/train.csv')
#print(data)


#number of steps from start to end
deltas = data['delta'].to_numpy()#.to_frame()
#print(deltas)

#y_train data
#list of starting data frames (our solutions)
starts = data.loc[:, 'start_0':'start_624'].to_numpy()
#starts = deltas.join(starts)
#print(starts.shape)

#x_train data
#list of ending data frames (our starting points, ya that sounds confusing)
#as well as deltas (number of steps between start and stop)
stops = data.loc[:, 'stop_0':'stop_624'].to_numpy()
#stops = deltas.join(stops)
#print(stops)

x_train, x_test, y_train, y_test = train_test_split(
    stops, starts, test_size=.2, random_state=10)

'''reshape to 25x25 if using a conv net, otherwise comment out the following'''
reshaping_dim = (-1,1,25,25)

#for i in range(x_train.shape[0]):
x_train = np.reshape(x_train,reshaping_dim)
y_train = np.reshape(y_train, reshaping_dim)
x_test = np.reshape(x_test, reshaping_dim)
y_test = np.reshape(y_test, reshaping_dim)

'''Verify that the GoL simulation is working correctly'''
'''
total_score = 0;
#max_score = starts.shape[0]
max_score = 500
test_starts = starts.to_numpy()
test_stops = stops.to_numpy()
test_deltas = deltas.to_numpy()
for i in tqdm(range(max_score)):
    if(checkStart(test_starts[i], test_stops[i], test_deltas[i][0])):
        total_score += 1
'''


'''#############      Create auto encoder model       #################'''

##########   Basic encoder   ##########
'''
auto_encoder, encoder, decoder = encoders.basic_encoder()

auto_encoder.fit(x_train, y_train,
                epochs=15,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, y_test))
'''
###########   Convolutional encoder   #########

'''TODO: get this working

auto_encoder, encoder, decoder = encoders.conv_encoder()

auto_encoder.fit(x_train, y_train,
                epochs=15,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, y_test))
'''

'''###########   Convolutional Neural network   ##########'''

######   Basic convolutional network prediction   ########

#auto_encoder = encoders.basic_conv_sequential_1D()


auto_encoder = encoders.deep_conv(x_train.shape[0],x_train.shape[1])

auto_encoder.summary()
auto_encoder.fit(x_train, y_train,
                epochs=15,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, y_test))

'''######################     Test Predictions     ####################'''

test_data = pd.read_csv('./data/test.csv')
#print(test_data)
test_data = test_data.drop('id', axis='columns')
test_deltas = test_data['delta']
test_data = test_data.drop('delta', axis='columns')
test_data = test_data.to_numpy()

#NOTE: only use below on CNN (to reshape)
test_data = np.reshape(test_data, reshaping_dim)

''' ###  Use if using encoder and not CNN  ###
#use encoder to convert end frame to start
encoded_solution = encoder.predict(test_data)
'''

###  Use for CNN  ###
encoded_solution = auto_encoder.predict(test_data)

#round all predictions to nearest whole number
encoded_solution = np.rint(encoded_solution)
#round negatives to zero to get GoL frame
encoded_solution[encoded_solution < 0] = 0

#print(encoded_solution)
#print(type(encoded_solution))
total_score = 0;
max_score = test_data.shape[0]

#for i in tqdm(range(encoded_solution.shape[0])):
for i in tqdm(range(1000)):
    if(checkStart(encoded_solution[i], test_data[i], test_deltas[i]+10)):
        total_score += 1

print('Accuracy: ', total_score/max_score)






















