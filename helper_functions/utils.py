import pandas as pd
import numpy as np
import cv2
import PIL.Image as Image
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

def load_parquet(file_name, data_path):
    '''
    Later, really specific usage
    '''
    df = pd.read_parquet(data_path+name)
    image_id = df['image_id']
    df.drop('image_id', axis = 1, inplace=True)

    return image_id,df

def threshold(df,limit=175):
    '''
    Later,...
    '''
    _,th = cv2.threshold(np.array(df), limit,255, cv2.THRESH_BINARY_INV)

    return pd.DataFrame(th)

def load_resize(path ,size):
    """
    Later,...
    """
    out = Image.open(path)
    out = out.resize(size, box=(10,10,out.width-10, out.height-10))
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2GRAY)
def resize(df,size_pre, size_post):
    '''
    Later,...
    '''
    resized = {}
    for i in range(df.shape[0]):
        resized[df.index[i]]=cv2.resize(df.iloc[i].values.reshape(size_pre[0],size_pre[1]),size_post,interpolation=cv2.INTER_AREA).reshape(-1)
    resized = pd.DataFrame(resized).T

    return resized

def rescale(df, scale=255.):
    """
    Later,
    """

    return df/scale

def build_callback(model, weights_dir):
    """
    Later,
    """
    if model in ['ResNet55M', 'ResNet30M', 'ConvNet10M']:
        learning_rate_reduction_root = ReduceLROnPlateau(monitor='val_graph_g168_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
        learning_rate_reduction_vowel = ReduceLROnPlateau(monitor='val_graph_v11_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
        learning_rate_reduction_consonant = ReduceLROnPlateau(monitor='val_graph_c7_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

        #model_save = ModelCheckpoint('weights.hdf5', monitor='val_graph_g168_accuracy', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
        model_save2 = ModelCheckpoint(weights_dir+'weights-{epoch}.hdf5', monitor='val_graph_g168_accuracy', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        #early_stop = EarlyStopping(monitor='val_graph_g168_accuracy', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

        callbacks = [learning_rate_reduction_root, learning_rate_reduction_vowel, learning_rate_reduction_consonant, model_save2]
    else:
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
        model_save2 = ModelCheckpoint(weights_dir+'weights-{epoch}.hdf5', monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        callbacks = [learning_rate_reduction, model_save2]

    return callbacks
