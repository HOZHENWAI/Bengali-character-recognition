from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, AveragePooling2D, Dropout, BatchNormalization, Activation, Reshape, Add
from keras.initializers import glorot_uniform

def identity_block(X,f,filters):
    """

    Inputs:

    Outputs:
    """

    #
    F1,F2,F3 = filters

    #
    X_saved = X

    # Layer 1
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Layer 2
    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='SAME', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Final layer
    X = Conv2D(filters = F3, kernel_size=(1,1), strides = (1,1), padding = 'valid',kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)

    X = Add()([X, X_saved])
    X = Activation('relu')(X)

    return X

def short_conv_block(X,f,filters,s):
    """
    Inputs:

    Outputs:

    """

    #
    F1,F2,F3 = filters
    X_shortcut = X
    #
    X = Conv2D(F1, (1, 1), strides = (s,s), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    X = Conv2D(F2, (f,f), strides = (1,1), padding = 'SAME', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    X = Conv2D(F3, (1, 1), strides = (1,1), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)

    X_shortcut = Conv2D(F3, (1,1), strides = (s,s), padding='valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)

    X = Add()([ X,X_shortcut])
    X = Activation('relu')(X)

    return X

def conv_block(X,f,filters, s):
    """
    Inputs:

    Outputs:

    """

    #
    F1,F2,F3 = filters

    #
    X = Conv2D(F1, (1, 1), strides = (s,s), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    X = Conv2D(F2, (f,f), strides = (1,1), padding = 'SAME', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    X = Conv2D(F3, (1, 1), strides = (1,1), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    return X

def dense_block(X,filters):
    """
    Inputs:

    Outputs:


    """

    #
    F1,F2,F3 = filters

    #
    X = Dense(F1, activation='relu', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dense(F2, activation='relu', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dense(F3, activation='relu', kernel_initializer = glorot_uniform(seed=0))(X)

    return X
