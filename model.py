import csv
import os
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Global Parameters :

batch_size = 100 # Batch Size for Generator
num_epoch = 3 # Epochs used for training
crop_top = 65 # Number of pixels cropped from top of image
crop_bottom = 25 # Number of pixels cropped from bottom of image
ch, row, col = 3, 160-crop_top-crop_bottom, 320  # Trimmed image format

lines = []

samples = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

# Generate training  & validation set's:
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Use a Generator to import images in batches
def generator(samples, batch_size):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			center_images = []
			left_images = []
			right_images = []
			center_angles = []

			for batch_sample in batch_samples:
				center_img = cv2.cvtColor(cv2.imread('data/IMG/' + batch_sample[0].split('/')[-1]),cv2.COLOR_BGR2RGB)
				left_img = cv2.cvtColor(cv2.imread('data/IMG/' + batch_sample[1].split('/')[-1]),cv2.COLOR_BGR2RGB)
				right_img = cv2.cvtColor(cv2.imread('data/IMG/' + batch_sample[2].split('/')[-1]),cv2.COLOR_BGR2RGB)

				# trim image to only see section with road
				center_img = center_img[crop_top : 160-crop_bottom, :]
				left_img = left_img[crop_top : 160-crop_bottom, :]
				right_img = right_img[crop_top : 160-crop_bottom, :]
				center_images.append(center_img)
				left_images.append(left_img)
				right_images.append(right_img)
				center_angles.append(float(line[3]))


			X_train = np.array(center_images)
			y_train = np.array(center_angles)
			yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

print ("\nTraining the Model..\n\nTraining Samples : {} \nValidation Samples : {} \n"\
	.format(len(train_samples),len(validation_samples)))

model = Sequential()

model.add(Lambda(lambda x: x/255- 0.5,input_shape=(row,col,ch),output_shape=(row,col,ch)))
model.add(Convolution2D(24,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(36,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(48,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Flatten())
model.add(Dense(2212))
model.add(Dense(1106))
model.add(Dense(100))
model.add(Dense(30))
model.add(Dense(1))
model.compile(loss = 'mse' , optimizer = 'adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= 
            len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=num_epoch)


#Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Mean Squared Error (MSE) Loss')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch #')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()


model.save('model.h5')
print ("Model Saved.")
exit()