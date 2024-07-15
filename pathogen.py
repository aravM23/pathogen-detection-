import numpy as np
import pandas as pd
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import random
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Activation,Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

train_image_path = '../input/plant-pathology-2021-fgvc8/train_images'
test_image_path = '../input/plant-pathology-2021-fgvc8/test_images'
train_df_path = '../input/plant-pathology-2021-fgvc8/train.csv'
test_df_path = '../input/plant-pathology-2021-fgvc8/sample_submission.csv'

df_train = pd.read_csv(train_df_path)
df_test=pd.read_csv(test_df_path)

df_test


df_train.labels.value_counts()

plt.figure(figsize=(15,12))
labels = sns.barplot(df_train.labels.value_counts().index,df_train.labels.value_counts())
for item in labels.get_xticklabels():
    item.set_rotation(45)

def batch_visualize(df,batch_size,path):
    sample_df = df_train.sample(9)
    image_names = sample_df["image"].values
    labels = sample_df["labels"].values
    plt.figure(figsize=(16, 12))
    
    for image_ind, (image_name, label) in enumerate(zip(image_names, labels)):
        plt.subplot(3, 3, image_ind + 1)
        image = cv2.imread(os.path.join(path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.title(f"{label}", fontsize=12)
        plt.axis("off")
    plt.show()
    
batch_visualize(df_train,9,train_image_path)

def batch_visualize_with_label(df,batch_size,path,label): 
    sample_df = df_train[df_train["labels"]==label].sample(9)
    image_names = sample_df["image"].values
    labels = sample_df["labels"].values
    plt.figure(figsize=(16, 12))
    
    for image_ind, (image_name, label) in enumerate(zip(image_names, labels)):
        plt.subplot(3, 3, image_ind + 1)
        image = cv2.imread(os.path.join(path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.axis("off")
    plt.show()

batch_visualize_with_label(df_train,9,train_image_path,'healthy')


batch_visualize_with_label(df_train,9,train_image_path,'scab')

batch_visualize_with_label(df_train,9,train_image_path,'frog_eye_leaf_spot')

batch_visualize_with_label(df_train,9,train_image_path,'rust')

batch_visualize_with_label(df_train,9,train_image_path,'complex')

HEIGHT = 128
WIDTH=128
SEED = 45
BATCH_SIZE= 64

train_datagen = ImageDataGenerator(rescale = 1/255.,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split = 0.2,
    zoom_range = 0.2,
    shear_range = 0.2,
    vertical_flip = False)

train_dataset = train_datagen.flow_from_dataframe(
    df_train,
    directory = train_image_path,
    x_col = "image",
    y_col = "labels",
    target_size = (HEIGHT,WIDTH),
    class_mode='categorical',
    batch_size = BATCH_SIZE,
    subset = "training",
    shuffle = True,
    seed = SEED,
    validate_filenames = False
)

validation_dataset = train_datagen.flow_from_dataframe(
    df_train,
    directory = train_image_path,
    x_col = "image",
    y_col = "labels",
    target_size = (HEIGHT,WIDTH),
    class_mode='categorical',
    batch_size = BATCH_SIZE,
    subset = "validation",
    shuffle = True,
    seed = SEED,
    validate_filenames = False
)

test_datagen = ImageDataGenerator(
    rescale = 1./255
)
INPUT_SIZE = (HEIGHT,WIDTH,3)
test_dataset=test_datagen.flow_from_dataframe(
    df_test,
    directory=test_image_path,
    x_col='image',
    y_col=None,
    class_mode=None,
    target_size=INPUT_SIZE[:2]
)

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(HEIGHT,WIDTH,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(12,activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.summary()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model_history=model.fit_generator(train_dataset,
                                  validation_data=validation_dataset,
                                  epochs=5,
                                  steps_per_epoch=train_dataset.samples//128,
                                 validation_steps=validation_dataset.samples//128,
                                 callbacks=[cp_callback]
                                 )


train_dataset.class_indices.items()

preds = model.predict(test_dataset)
print(preds)

preds_disease_ind=np.argmax(preds, axis=-1)

preds_disease_ind


