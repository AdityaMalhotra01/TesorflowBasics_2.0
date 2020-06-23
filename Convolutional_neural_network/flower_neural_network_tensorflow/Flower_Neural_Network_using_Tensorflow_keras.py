import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from tensorflow.keras import datasets, layers, models
data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                           fname='flower_photos', untar=True)
#data_dir = 'flower_photos'
print(data_dir)
data_dir = pathlib.Path(data_dir)
print(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
print(CLASS_NAMES)

roses = list(data_dir.glob('roses/*'))
print(roses[:3])
for image_path in roses[:3]:
    display.display(Image.open(str(image_path)))
    
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for i in range(25):
      plt.subplot(5,5,i+1)
      plt.imshow(image_batch[i])
      plt.title(CLASS_NAMES[label_batch[i]==1][0].title())
      plt.axis('off')
      
# print(train_data_gen)
# image_batch, label_batch = next(train_data_gen)
# show_batch(image_batch, label_batch)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# The 1./255 is to convert from uint8 to float32 in range [0,1].
train_data_gen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set= train_data_gen.flow_from_directory(directory=str(data_dir),
                                                     batch_size=32,
                                                     target_size=(64, 64),
                                                     classes = list(CLASS_NAMES))



# labal_batch = np.reshape(label_batch,(160,))

model = models.Sequential()
model.add(layers.Conv2D(32, 3, 3, activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64,3, 3, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5,activation = 'softmax')) 

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
 
history = model.fit_generator(training_set,
                              steps_per_epoch = 3670,
                              epochs = 50)





































