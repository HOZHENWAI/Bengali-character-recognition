from keras.models import Model
from keras.layers import Input, ZeroPadding2D, Dense, Conv2D, BatchNormalization, Activation, MaxPool2D,AveragePooling2D,Flatten
from keras.initializers import glorot_uniform

from models.modules import conv_block

def ConvNet10M(input_shape, classes):
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
    X = conv_block(X,f=3,filters=[64,64,256], s=1)
    X = conv_block(X,3,[64,64,256],s=1)
    X = conv_block(X,3,[64,64,256],s=1)

    #
    X = AveragePooling2D(pool_size=(2,2))(X)
    X = Flatten()(X)
    X = Dense(256, activation='relu', kernel_initializer = glorot_uniform(seed=0))(X)

    #
    X_g = Dense(256 , activation = 'relu', kernel_initializer = glorot_uniform(seed=0))(X)
    X_g = Dense(256 , activation = 'relu', kernel_initializer = glorot_uniform(seed=0))(X_g)
    X_g = Dense(classes[0], activation = 'softmax', name = 'graph_g'+str(classes[0]))(X_g)

    X_v = Dense(classes[1], activation='softmax', name='graph_v' + str(classes[1]))(X)

    X_c = Dense(classes[2], activation='softmax', name='graph_c' + str(classes[2]))(X)

    model = Model(inputs = X_input, outputs = [X_g, X_v, X_c], name = 'ConvNet10M')
    return model
