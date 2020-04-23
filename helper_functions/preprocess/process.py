import pandas as pd

def retrieve_y(model,im_ids,df):
    """
    Later,
    """
    if model in ['ResNet55M', 'ResNet30M', 'ConvNet10M']:
        return df.loc[im_ids, ['grapheme_root','vowel_diacritic', 'consonant_diacritic']]
    else:
        return df.loc[im_ids, 'grapheme']
def one_hot_encoding(model,df_Y):
    """
    Later,
    """
    if model in ['ResNet55M', 'ResNet30M', 'ConvNet10M']:
        Y_grapheme = pd.get_dummies(df_Y['grapheme_root'])
        Y_vowel = pd.get_dummies(df_Y['vowel_diacritic'])
        Y_consonant = pd.get_dummies(df_Y['consonant_diacritic'])
        return [Y_grapheme, Y_vowel, Y_consonant]
    else:
        Y = pd.get_dummies(df_Y)
        return Y
