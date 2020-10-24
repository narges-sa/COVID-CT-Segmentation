### Narges Saeidy
'''#########################################################################'''
import os
import sys
import  time
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from skimage.io import imread
from keras import backend as K
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.preprocessing.image import  img_to_array
from keras.models import Model
from keras.layers import  BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from TV_UNET import get_unet, TV_bin_loss
'''#########################################################################'''
im_width = 128
im_height = 128
EpoachesNo=100
BatcheSZE=32
num_class=2

'''#########################################################################'''
#loading Masks & images then save & labeling loop
path=''
ids1 = next(os.walk(path + 'LungInfection-train/Pseudo-label/Imgs/'))[2]

train = np.zeros((len(ids1), im_width, im_height,1), dtype=np.float32)

mask = np.zeros((len(ids1), im_height, im_width,num_class), dtype=np.float32)

print("No. of Images = ", len(ids1))

sys.stdout.flush()

for n, id_ in tqdm(enumerate(ids1), total=len(ids1)): 
            img = imread(path+'LungInfection-train/Pseudo-label/Imgs/'+id_)
            x_img = img_to_array(img)
            x_img = resize(x_img, (im_width, im_height, 1), mode = 'constant', preserve_range = True)
            train[n,:,:] =  x_img


            mask_ID, ext = os.path.splitext(path+'LungInfection-train/Pseudo-label/GT/'+id_)
            msk= imread(mask_ID+'.png')
            msk = resize(msk, (im_width, im_height), mode = 'constant',preserve_range = True, anti_aliasing=False)
            msk=np.round(msk/255)

            for i in range(0, num_class):
              mask[n,:,:,i] = np.where(msk ==i,1,0)
            
'''#########################################################################'''
#loading Masks & images then save & labeling loop
ids2 = next(os.walk(path + 'LungInfection-train/Doctor-label/Imgs/'))[2]

X_train = np.zeros((len(ids2), im_width, im_height,1), dtype=np.float32)

X_mask = np.zeros((len(ids2), im_height, im_width,num_class), dtype=np.float32)

print("No. of Images = ", len(ids2))

sys.stdout.flush()

for n, id_ in tqdm(enumerate(ids2), total=len(ids2)): 
            img = imread(path+'LungInfection-train/Doctor-label/Imgs/'+id_)
            x_img = img_to_array(img)
            x_img = resize(x_img, (im_width, im_height, 1), mode = 'constant', preserve_range = True)
            X_train[n,:,:] =  x_img


            mask_ID, ext = os.path.splitext(path+'LungInfection-train/Doctor-label/GT/'+id_)
            msk= imread(mask_ID+'.png')
            msk = resize(msk, (im_width, im_height), mode = 'constant',preserve_range = True, anti_aliasing=False)
            msk=np.round(msk/255)

            for i in range(0, num_class):
              X_mask[n,:,:,i] = np.where(msk ==i,1,0)

train=np.vstack((train,X_train))

mask=np.vstack((mask,X_mask))

'''#########################################################################'''

def dice_coef(y_pred, y_true):
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.0001) / (K.sum(y_true_f) + K.sum(y_pred_f) + 0.0001)


def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def my_loss(y_true, y_pred):
  layer_names=[layer.name for layer in model.layers]
  for l in layer_names:
    if l==layer_names[-1]:
      value = TV_bin_loss(y_true, y_pred)
    else:
      value = binary_crossentropy(K.flatten(y_true),K.flatten(y_pred))
  return value
'''#########################################################################'''
############ Applying TV_UNET Model
from tensorflow.keras.metrics import Recall, Precision

input_img = Input((im_height, im_width,1), name='img')

model = get_unet(input_img, n_filters=64, dropout=0.2, batchnorm=True)

model.compile(optimizer=Adam(learning_rate=0.001) , loss = [my_loss], metrics=['accuracy',dice_loss,Recall(name='recall_1'),
                                                            Precision(name='pre_1')])
model.summary()

'''#########################################################################'''
############ Split train and validation
X_train, X_valid, y_train, y_valid = train_test_split(train, mask, test_size=0.1, random_state=42)

callbacks = [
    EarlyStopping(patience=50, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-TV-UNet1.h5', verbose=1, save_best_only=True, save_weights_only=True)
]



results = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=callbacks,\
                    validation_data=(X_valid, y_valid))

'''#########################################################################'''
'''Test'''
#loading Masks & image-test then save & labeling loop
ids3 = next(os.walk(path + 'LungInfection-test/Imgs/'))[2]

y = np.zeros((len(ids3), im_width, im_height,1), dtype=np.float32)

label_test = np.zeros((len(ids3), im_height, im_width,num_class), dtype=np.float32)

print("No. of Images = ", len(ids3))

sys.stdout.flush()

for n, id_ in tqdm(enumerate(ids3), total=len(ids3)): 
            img = imread(path+'LungInfection-test/Imgs/'+id_)
            x_img = img_to_array(img)
            x_img = resize(x_img, (im_width, im_height, 1), mode = 'constant', preserve_range = True)
            y[n,:,:] =  x_img


            mask_ID, ext = os.path.splitext(path+'LungInfection-test/GT/'+id_)
            msk= imread(mask_ID+'.png')
            msk = resize(msk, (im_width, im_height), mode = 'constant',preserve_range = True)
            msk=np.round(msk/255)
            
            for i in range(0, num_class):
              label_test[n,:,:,i] = np.where(msk ==i,1,0)
'''#########################################################################'''           
########## load the best model
model.load_weights('model-TV-UNet1.h5')

preds_test1 = model.predict(y, verbose=1)

'''#########################################################################'''
############ calculate  recall or Sensitivity
tre=np.arange(0.1,1,0.1).tolist()

y_test_f=K.flatten(label_test[:,:,:,1])

preds_test_f=K.flatten(preds_test1[:,:,:,1])

m = tf.keras.metrics.Recall(thresholds=tre)

m.update_state(y_test_f, preds_test_f)

Recal_TV_unet=m.result().numpy()

print('Sensitivity_TV_unet=',Recal_TV_unet)

'''#########################################################################'''
############ calculate  Specificity
from sklearn.metrics import confusion_matrix

for i in tre:
  y_test_f=K.flatten(label_test[:,:,:,1]>i)
  preds_test_f=K.flatten(preds_test1[:,:,:,1]>i)

  tn, fp, fn, tp=confusion_matrix(y_test_f.numpy(), preds_test_f.numpy()).ravel()
  specificity = tn / (tn+fp)
  
  print('Specificity_TV_unet=', specificity,'\n')

'''#########################################################################'''
#calculate  Dice Score
def dice_coef(y_pred, y_true):
  for tr in np.arange(0.1,1,.1):
    y_pred_t = (y_pred[:,:,:,1] > tr).astype(np.uint8)
    y_true_f = np.array(y_true[:,:,:,1])
    y_pred_f = np.array(y_pred_t)
    intersection = np.sum(y_true_f * y_pred_f)
    sum_ = np.sum(np.abs(y_true_f) + np.abs(y_pred_f))
    print( 2*(intersection) /sum_ )

dice=dice_coef(preds_test1,label_test)

print('DSC_TV_unet=',dice)
'''#########################################################################'''
