from keras.models import Model
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Dense, Activation, MaxPool2D,AveragePooling2D,Flatten
from keras.initializers import glorot_uniform

from models.modules import identity_block, short_conv_block



def ResNet30M(input_shape, classes):
    """

    """

    #
    X_input = Input(input_shape)

    #
    X = ZeroPadding2D(1)(X_input)

    #
    X = Conv2D(64, (3, 3), strides = (1, 1), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPool2D((3, 3), strides=(2, 2))(X)

    #
    X = short_conv_block(X,f=3,filters=[64,64,256], s=1)
    X = identity_block(X,3,[64,64,256])
    X = identity_block(X,3,[64,64,256])

    #
    X = short_conv_block(X,f=3,filters=[128,128,512],s=2)
    X = identity_block(X,3,[128,128,512])
    X = identity_block(X,3,[128,128,512])
    X = identity_block(X,3,[128,128,512])

    #
    X = short_conv_block(X,f=3,filters=[256,256,1024],s=2)
    X = identity_block(X,3,[256,256,1024])
    X = identity_block(X,3,[256,256,1024])

    #
    X = AveragePooling2D(pool_size=(2,2))(X)
    X = Flatten()(X)
    X = Dense(1024, activation='relu', kernel_initializer = glorot_uniform(seed=0))(X)

    #
    X_g = Dense(1024 , activation = 'relu', kernel_initializer = glorot_uniform(seed=0))(X)
    X_g = Dense(512 , activation = 'relu', kernel_initializer = glorot_uniform(seed=0))(X_g)
    X_g = Dense(classes[0], activation = 'softmax', name = 'graph_g'+str(classes[0]))(X_g)

    X_v = Dense(classes[1], activation='softmax', name='graph_v' + str(classes[1]))(X)

    X_c = Dense(classes[2], activation='softmax', name='graph_c' + str(classes[2]))(X)

    model = Model(inputs = X_input, outputs = [X_g, X_v, X_c], name = 'ResNet30M')
    return model
