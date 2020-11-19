import keras
from keras import layers

def basic_encoder(): 
    
    '''basic auto encoder model found at 
    https://blog.keras.io/building-autoencoders-in-keras.html'''
    
    #The encoding dim is the same as the input because we want to
    # map the end frame to an equally sized start frame
    enc_dim = 625
    end_frame = keras.Input(shape=(625,))
    encoded = layers.Dense(enc_dim, activation='relu')(end_frame)
    decoded = layers.Dense(625, activation='sigmoid')(encoded)
    
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