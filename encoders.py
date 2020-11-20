import keras
from keras import layers
from keras import activations

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


















