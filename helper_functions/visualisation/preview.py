from matplotlib import pyplot as plt
from os import path
import cv2
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont



def show_image_from_df(index, df, size):
    '''
    Show image number 'index' from a pandas dataframe 'df' where each rows correspond to an image with dim 'size'
    This implementation assume B&W images.
    #Not really required but can be handy.
    '''
    plt.imshow(df.loc[index].values.reshape(size))

def image_from_char(char,size, title):
    '''
    Draw an image from the bengali dictionary.

    '''
    width, height = size[0],size[1]

    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)
    myfont = ImageFont.truetype(path.abspath(path.dirname(__file__))
                        +'/fonts/kalpurush-2.ttf',min(height,width))
    w,h = draw.textsize(char, font=myfont)
    draw.text(((width-w)/2, (height-h)/2), char, font=myfont)
    image.save(title+'.jpeg')
    return image
