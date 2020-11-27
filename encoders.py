import numpy as np
import keras
from keras import layers
from keras import activations
import keras.backend as K
from scipy import ndimage
import tensorflow as tf

def basic_encoder(): 
    
    '''basic auto encoder model found at 
    https://blog.keras.io/building-autoencoders-in-keras.html'''
    
    #The encoding dim is the same as the input because we want to
    # map the end frame to an equally sized start frame
    enc_dim = 625
    end_frame = keras.Input(shape=(enc_dim,))
    encoded = layers.Dense(enc_dim, activation='relu')(end_frame)
    decoded = layers.Dense(enc_dim, activation='sigmoid')(encoded)
    
    auto_encoder = keras.Model(end_frame, decoded)
    
    #encoder model
    encoder = keras.Model(end_frame, encoded)
    
    #decoder model
    # This is our encoded input
    encoded_endFrame = keras.Input(shape=(enc_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = auto_encoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_endFrame, decoder_layer(encoded_endFrame))
    
    auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return auto_encoder, encoder, decoder


def conv_encoder():
    enc_dim = 625
    end_frame = keras.Input(shape=(enc_dim,))
    input_frame = layers.Reshape((25,25,1))(end_frame)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(25,25))(input_frame)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    
    encoder = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    decoder = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    auto_encoder = keras.Model(input_frame, decoder)
    auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return auto_encoder, encoder, decoder

    

#TODO data fed into this needs to be reshaped as (num_of_data,dim,1)
def basic_conv_sequential_1D():
    enc_dim=625
    
    model = keras.Sequential()
    #model.add(keras.layers.MaxPooling1D(pool_size=64))

    model.add(keras.layers.Dense(enc_dim,
                                 input_shape=(enc_dim,)
                                  ))
    model.add(keras.layers.Conv1D(filters=32, 
                                  kernel_size=4, 
                                  activation='relu',
                                  #input_shape=(enc_dim,),
                                  data_format='channels_last'
                                  ))
    #model.add(keras.layers.MaxPooling1D(pool_size=64))
    model.add(keras.layers.Conv1D(filters=32, 
                                  kernel_size=4, 
                                  activation='relu',
                                  data_format='channels_last'
                                  ))
    model.add(keras.layers.MaxPooling1D(pool_size=64))
    model.add(keras.layers.Dropout(0.5))
    #model.add(keras.layers.MaxPooling1D(pool_size=64))
    model.add(keras.layers.Conv1D(filters=32,
                                  kernel_size=4, 
                                  activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=64))
    model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(enc_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])

    return model



def basic_conv(data_count,enc_dim):
    #enc_dim = 625
 
    input_frame = keras.Input(shape=(enc_dim,))
    
    x = keras.layers.Dense(enc_dim)(input_frame)
    
    out_layer = keras.layers.Dense(enc_dim, activation='sigmoid')(x)    
    
    model = keras.Model(inputs=input_frame,outputs=out_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def deep_conv(data_count,enc_dim):
    
    input_dim = (1,25,25)
    input_frame = keras.Input(shape=input_dim)
    
    #x = keras.layers.Dense(enc_dim)(input_frame)
    
    x = layers.Conv2D(filters=32,kernel_size=(3,3),padding='same')(input_frame)#(x)
    #x = keras.layers.BatchNormalization()(x)
    
    x = layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same')(x) 
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same')(x) 
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same')(x) 
    out_layer = layers.Dense(25, activation='relu')(x)
    
    model = keras.Model(inputs=input_frame,outputs=out_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model



'''Based on the following algorithm
 https://www.kaggle.com/yakuben/crgl2020-iterative-cnn-approach
 
 Creating custom keras training
 https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
 
 '''
 
''''##################    Custom training functions     ###################'''

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

#Simulate a GoL for N steps
#Function to check if starting frame is a solution        
def Gol_sim(start, end, iterCount, shaped=True):
    #if shaped = true -> 2d input expected
    #if shaped = false -> 1d input expected
    #start = initial GoL configuration (our solution)
    #end = ending configuration (what we start wtih)
    
    #reshape if necesary 
    if(shaped == False):
        start_2d = np.reshape(start, (25,25))
    else: 
        start_2d = start
    
    
    for k in range(iterCount):
        #compute number of neighbors per cell
        nbrCount = testfilter(start_2d)
        
        #change every element in start_2d according to its neighbors
        for i in range(start_2d.shape[0]):
            for j in range(start_2d.shape[1]):
                start_2d[i][j] = gameLogic(start_2d[i][j],nbrCount[i][j])
        
    return start_2d
    
def custom_loss(delta=1):
    def loss_func(y_true, y_pred):
        #TODO switch from using 1 to delta
        pred_frame = Gol_sim(y_pred, y_true, 1)
        #counts number of non matching bits in frame
        loss_val = np.count_nonzero(pred_frame==y_true)
        return loss_val
    
    return loss_func

#def custom_loss2(y_true, y_pred)

def iter_conv(data_count,enc_dim,x_train,y_train,train_deltas,epochs=1):
    
    #resets backend variables
    K.clear_session()
    
###################    define model    #####################
    input_dim = (1,25,25)
    input_frame = keras.Input(shape=input_dim)
    
    #x = keras.layers.Dense(enc_dim)(input_frame)
    x = layers.Conv2D(filters=32,kernel_size=(3,3),padding='same')(input_frame)#(x)
    #x = keras.layers.BatchNormalization()(x)
    
    x = layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same')(x)
    out_layer = layers.Dense(25, activation='relu')(x)
    
    model = keras.models.Model(input_frame,out_layer)
    
    model.summary()
    
#####################    begin training    #####################
#   good resource for creating a custom training routine (what this is based on)
#   https://gist.github.com/JVGD/2add7789ab83588a397bbae6ff614dbf 

    optimizer = keras.optimizers.Adam(lr=0.001)
    #loss = custom_loss(5)#5 assumes a constant delta of 5 across all data
    loss = keras.losses.categorical_crossentropy(keras.Input(shape=input_dim),
                                                 model(keras.Input(shape=input_dim)))
    update_op = optimizer.get_updates(params=model.trainable_weights,loss=loss)
    '''loss(keras.Input(shape=input_dim),
    model(keras.Input(shape=input_dim))
    ))#converts input from numpy to tensor'''
    
    train = K.function(inputs=[x_train,y_train],
                       outputs=loss,#outputs=[loss,model.layer[-1].output],
                       updates=update_op)
    
    test = K.function(inputs=[x_train,y_train],
                      outputs=[loss])
    
    for epoch in range(epochs):
        
        training_losses = []
        
        for cur_sample in range(data_count):
            #TODO apply train_deltas to loop rather than constant 5
            sample_delta = 5 #train_deltas[cur_sample]
            
            #loop to feedback output for delta time steps of prediction
            sample = x_train[cur_sample]
            target = y_train[cur_sample]
            
            #add batch size as dimension
            sample = np.expand_dims(sample, axis=0)
            target = np.expand_dims(target, axis=0)
            
            #convert to tensors
            sample = K.constant(sample)
            target = K.constant(target)
            
            cur_input = cur_sample
            #target = tf.convert_to_tensor(target)
            for i in range(sample_delta):
                #cur_input = tf.convert_to_tensor(cur_input)
                #calculate loss, running a training iteration
                loss_train = train([tf.convert_to_tensor(cur_input), tf.convert_to_tensor(target)])
                training_losses.append(loss_train[0])
                
                #set next input to current output (out_layer.output)
                cur_input = model.predict(cur_input)
                
        train_loss_mean = np.mean(training_losses)
       
        print("Epoch ",epoch,";  Current mean training loss: ",train_loss_mean)
        
        
        '''
        #Now compute test values (no training)
        losses_test = []
        for cur_sample in range(data_count)'''
        
        
        
        
    '''
    model = keras.Model(inputs=input_frame,outputs=out_layer)
    model.compile(loss=Loss,
                  optimizer='adam',
                  metrics=['accuracy'])
    '''
    return model



'''#############   Another more educated try at a basic encoder ############'''


def basic_encoder_mk2(data_count,enc_dim,x_train,y_train,train_deltas,epochs=1): 
    
    '''applying iterative model to the basic encoder'''
    
    enc_dim = (1,25,25)
    
    end_frame = keras.Input(shape=(enc_dim))
    encoded = layers.Conv2D(filters=32,kernel_size=(3,3), padding='same')(end_frame)
    x = layers.Conv2D(filters=64,kernel_size=(3,3), padding='same')(encoded)
    x = layers.Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu')(x)
    decoded = layers.Dense(25, activation='relu')(x)
    #decoded = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(x)
    
    
    auto_encoder = keras.Model(end_frame, decoded)
    
    #encoder model
    encoder = keras.Model(end_frame, encoded)
    
    #decoder model
    # This is our encoded input
    encoded_endFrame = keras.Input(shape=(25,25))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = auto_encoder.layers[-1]
    # Create the decoder model
    #decoder = keras.Model(encoded_endFrame, decoder_layer(encoded_endFrame))
    
    auto_encoder.compile(optimizer='adam', loss='categorical_crossentropy')
    
    return auto_encoder, encoder#, decoder






















