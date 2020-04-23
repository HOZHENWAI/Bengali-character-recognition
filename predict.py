import argparse
from os import path, listdir, mkdir
import pandas as pd
import numpy as np
from helper_functions.utils import resize, rescale, threshold, build_callback, load_resize
from helper_functions.visualisation.preview import image_from_char
from models import *
from matplotlib.pyplot import imread

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ResNet55M', choices = ['ResNet55M', 'ResNet55O','ResNet30M','ResNet30O', 'ConvNet10M'],help='ResNet55M|ResNet55O|ResNet30M|ResNet30O|ConvNet10M')

parameters = parser.parse_args()

    #get the current dir
here = path.abspath(path.dirname(__file__))

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

model.load_weights(weights_dir+'weights.hdf5') #pretrained

#load all images
test = []
for file in listdir(here+'/examples/'):
    image = load_resize(here+'/examples/'+file, (parquet_size[0], parquet_size[1]))
    test.append(image.reshape(-1))
#preprocess
test = np.array(test)
test = threshold(test)
test = resize(test, parquet_size, (SIZE[0],SIZE[1]))
test = rescale(test)
test = test.values.reshape(-1, SIZE[0], SIZE[1], N_CHANNELS)

#predict
prediction = model.predict(test)

#show the results
for n_ex,file in enumerate(listdir(here+'/examples/')):
    example_dir = here+'/pred/'+file[0:-4]
    if parameters.model in ['ResNet55M', 'ResNet30M', 'ConvNet10M']:
        for n_type, type in enumerate(['grapheme_root', 'vowel_diacritic','consonant_diacritic']):
            pred, prob = np.argmax(prediction[n_type][n_ex]),np.max(prediction[n_type][n_ex])
            image_from_char(class_map.loc[(class_map.component_type == type) & (class_map.label == pred), 'component'].values[0], parquet_size, example_dir + f' {type } -  {prob }')
    # else:
    #     pred,prob = np.argmax([0][n_ex]), np.max(prediction[0][n_ex])
    #     image_from_char(class_map.iloc[pred].component,parquet_size, f'grapheme_prob')
