import numpy as np
import pandas as pd
import keras
from keras import layers
from scipy import ndimage
import matplotlib as plt
import seaborn as sb
from sklearn.model_selection import train_test_split


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
        if(np.equal(start_2d.flatten(),end)):
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
    
    
    
''''##################    DATA INIT     ##################################'''

#TODO check to see whether adding deltas to training data is a good idea or not

#read data
data = pd.read_csv('./data/train.csv')
#print(data)


#number of steps from start to end
deltas = data['delta'].to_frame()
#print(deltas)

#y_train data
#list of starting data frames (our solutions)
starts = data.loc[:, 'start_0':'start_624']
starts = deltas.join(starts)
#print(starts.shape)

#x_train data
#list of ending data frames (our starting points, ya that sounds confusing)
#as well as deltas (number of steps between start and stop)
stops = data.loc[:, 'stop_0':'stop_624']
stops = deltas.join(stops)
#print(stops)

x_train, x_test, y_train, y_test = train_test_split(
    stops, starts, test_size=.2, random_state=10)
'''
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
'''
'''#############      Create auto encoder model       #################'''
'''basic auto encoder model found at 
https://blog.keras.io/building-autoencoders-in-keras.html'''

enc_dim = 32
end_frame = keras.Input(shape=(626,))
encoded = layers.Dense(enc_dim, activation='relu')(end_frame)
decoded = layers.Dense(626, activation='sigmoid')(encoded)

auto_encoder = keras.Model(end_frame, decoded)

#encoder model
encoder = keras.Model(end_frame, encoded)

#decoder model
# This is our encoded (32-dimensional) input
encoded_endFrame = keras.Input(shape=(enc_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = auto_encoder.layers[-1]
# Create the decoder model
decoder = keras.Model(encoded_endFrame, decoder_layer(encoded_endFrame))



auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')

auto_encoder.fit(x_train, y_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, y_test))



'''######################     Test Predictions     ####################'''

test_data = pd.read_csv('./data/test.csv')
print(test_data)
test_data = test_data.drop('id', axis='columns')
test_data = test_data.to_numpy()

encoded_solution = encoder.predict(test_data)
print(encoded_solution)
#checkStart(start, end, iterCount)























