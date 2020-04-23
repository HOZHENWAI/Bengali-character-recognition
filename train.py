import argparse
from os import path, listdir
import pandas as pd
import gc
from helper_functions.utils import resize, rescale, threshold, build_callback, load_parquet
from helper_functions.preprocess import MultiOutputDataGenerator
from helper_functions.preprocess.process import one_hot_encoding, retrieve_y
from helper_functions.visualisation.loss import *
from models import *
from sklearn.model_selection import train_test_split



parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ResNet55M', choices = ['ResNet55M', 'ResNet55O','ResNet30M','ResNet30O', 'ConvNet10M'],help='ResNet55M|ResNet55O|ResNet30M|ResNet30O|ConvNet10M')
parser.add_argument('--preview', action = 'store_true', help='True|False')
parser.add_argument('--update', default = 0, type = int, help='Set True if updating the weights of a pretrained model')
parser.add_argument('--batch_size', default = 256, type = int, help = 'batch size for training, DEFAULT= 256')
parser.add_argument('--epochs', default = 25, type = int, help = 'number of epochs, default = 25')
parser.add_argument('--data_format', default = 'parquet', choices = ['parquet'], help = 'for now, the only training implementation use parquet datatype')
parser.add_argument('--metadata_filename', default='train.csv', help = 'labels should be in a csv')
parameters = parser.parse_args()

    #get the current dir
here = path.abspath(path.dirname(__file__))
data_path = here+'/data/'

# temporary variable, will change this once i get more time
parquet_size = [137,236]
optimizer = 'adam'
loss = 'categorical_crossentropy'
metrics = ['accuracy']

# load and extract the different classes

class_map = pd.read_csv(here+'/class_map.csv')
grapheme_root_class_n = class_map.loc[class_map.component_type=='grapheme_root'].shape[0]
vowel_diacritic_class_n = class_map.loc[class_map.component_type=='vowel_diacritic'].shape[0]
consonant_diacritic_class_n = class_map.loc[class_map.component_type=='consonant_diacritic'].shape[0]

# reduce the data dimension
RATIO = 0.4
SIZE = [int(parquet_size[0]*RATIO), int(parquet_size[1]*RATIO)]
N_CHANNELS = 1 # Black and White

#load the models
model = {
    'ResNet55M' : ResNet55M((SIZE[0],SIZE[1],1), classes=[grapheme_root_class_n,vowel_diacritic_class_n,consonant_diacritic_class_n]),
    'ResNet55O' : ResNet55O((SIZE[0],SIZE[1],1), grapheme_root_class_n*vowel_diacritic_class_n*consonant_diacritic_class_n),
    'ResNet30M' : ResNet30M((SIZE[0],SIZE[1],1), classes=[grapheme_root_class_n,vowel_diacritic_class_n,consonant_diacritic_class_n]),
    'ResNet30O' : ResNet30O((SIZE[0],SIZE[1],1), grapheme_root_class_n*vowel_diacritic_class_n*consonant_diacritic_class_n),
    'ConvNet10M' : ConvNet10M((SIZE[0],SIZE[1],1), classes=[grapheme_root_class_n,vowel_diacritic_class_n,consonant_diacritic_class_n])
    }[parameters.model]

#Compilation time
model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

# set the weights directory
weights_dir = {
    'ResNet55M' : here+'/models/resnet55_multiple/weights/',
    'ResNet55O' : here+'/models/resnet55_one/weights/',
    'ResNet30M' : here+'/models/resnet30_multiple/weights/',
    'ResNet30O' : here+'/models/resnet30_one/weights/',
    'ConvNet10M' : here+'/models/convnet10_multiple/weights/'
    }[parameters.model]

#if update then load the weights
if parameters.update > 0 :
    model.load_weights(weights_dir+'weights-'+str(parameters.update)+'.hdf5')

callbacks = build_callback(parameters.model, weights_dir)

# Now, this part is specific to the data format I used: parquet files
train = pd.read_csv(data_path+parameters.metadata_filename) # the training metadata
for file in listdir(data_path):
    im_ids, X_train = load_parquet(file, data_path)
    Y_train = retrieve_y(parameters.model, im_ids, train)
    Y_train = one_hot_encoding(parameters.model,Y_train)
    X_train = threshold(X_train)
    X_train = resize(X_train, parquet_size, (SIZE[0],SIZE[1]))
    X_train = rescale(X_train)
    datagen = MultiOutputDataGenerator(
        featurewise_center = False,
        samplewise_center = False,
        featurewise_std_normalization = False,
        samplewise_std_normalization = False,
        zca_whitening=False,
        rotation_range=15,
        zoom_range = 0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False,
        vertical_flip=False)
    if parameters.model in ['ResNet55M', 'ResNet30M', 'ConvNet10M']:
        x_train, x_test, y_train_g, y_test_g, y_train_v, y_test_v, y_train_c, y_test_c = train_test_split(X_train, Y_train[0], Y_train[1], Y_train[2], test_size=0.05, random_state=2020)
        y_gen = {'graph_g168' : y_train_g, 'graph_v11': y_train_v, 'graph_c7':y_train_c}
        val_data = [y_test_g, y_test_v, y_test_c]
    else:
        x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.05, random_state=2020)
        y_gen = {'graph_out : y_train'}
        val_data = [y_test]
    datagen.fit(x_train)
    gc.collect()
    history = model.fit_generator(datagen.flow(x_train, y_gen, batch_size = parameters.batch_size),
    steps_per_epoch = x_train.shape[0]//parameters.batch_size,epochs = parameters.epochs, verbose = 2
    , callbacks = callbacks, validation_data = (x_test,val_data))

# Add visualisation Later
