import csv
import os
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers import Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Global Parameters :

batch_size = 100 # Batch Size for Generator
num_epoch = 5 # Epochs used for training
crop_top = 65 # Number of pixels cropped from top of image
crop_bottom = 25 # Number of pixels cropped from bottom of image
ch, row, col = 3, 160, 320  # Image size
side_offset = 0.2 # Turning angle offset for L/R images
train_val_split = 0.2

# Data Importing:


# List of sub-folders to import data from
data_folders = ["data_center_fwd1","data_center_fwd2", "data_center_bwd1",
	"data_weave_fwd1","data_weave_fwd2","data_turns_fwd1", "data_turns_bwd1"]
samples = []

# Import driving logs from each data set
print("\n\nReading in Data Files...\n")
for subfolder in data_folders:
	log_name = "data/"+subfolder + "/driving_log.csv"
	csvfile = open(log_name)
	inFile = csv.reader(csvfile)
	for line in inFile:
		samples.append(line)
	print("Read subfolder "+ subfolder + " Updated Length = " + str(len(samples)))

print ("Read all images. Total Length = " + str(len(samples)))

# Data Pre-Processing / Generation:

# Generate training  & validation set's:
train_samples, validation_samples = train_test_split(samples,test_size=train_val_split)

# Use a Generator to import images in batches
def generator(samples, batch_size):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		samples = shuffle(samples)
		for offset in range(0, num_samples, batch_size):

			# Capture a batch of frames
			batch_samples = samples[offset:offset+batch_size]

			center_images = []
			left_images = []
			right_images = []
			steering_angles = []

			# Loop through each frame, save all 3 images & steering angle
			for batch_sample in batch_samples:
				center_img = cv2.cvtColor(cv2.imread(batch_sample[0]),cv2.COLOR_BGR2RGB)
				left_img = cv2.cvtColor(cv2.imread(batch_sample[1]),cv2.COLOR_BGR2RGB)
				right_img = cv2.cvtColor(cv2.imread(batch_sample[2]),cv2.COLOR_BGR2RGB)
				center_images.append(center_img)
				left_images.append(left_img)
				right_images.append(right_img)
				steering_angles.append(float(batch_sample[3]))

			#Convert array's to numpy:
			images = np.array(center_images)
			left_images = np.array(left_images)
			right_images = np.array(right_images)
			steering_angles = np.array(steering_angles)

			#Add left and right images to data set
			images = np.concatenate((images,left_images,right_images))			
			steering_angles = np.concatenate((steering_angles,steering_angles + side_offset, steering_angles - side_offset))

			X_train = images
			y_train = steering_angles
			yield sklearn.utils.shuffle(X_train, y_train)


# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

print ("\nTraining the Model...")

# Model Architecture:

# Testing LeNet Architecture as a baseline:
'''
model = Sequential()
model.add(Lambda(lambda x: x/255. - 0.5,input_shape=(row,col,ch),output_shape=(row,col,ch)))
model.add(Cropping2D(cropping=((crop_top,crop_bottom), (0,0)), input_shape=(3,160,320)))
model.add(Convolution2D(6,5,5,activation = "relu"))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(6,5,5,activation = "relu"))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(16,5,5,activation = "relu"))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(16,5,5,activation = "relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(140))
model.add(Dense(1))
'''

# Final Architecture Imlpementation. Modelled after Nvidia paper

model = Sequential()
model.add(Lambda(lambda x: x/255. - 0.5,input_shape=(row,col,ch),output_shape=(row,col,ch)))
model.add(Cropping2D(cropping=((crop_top,crop_bottom), (0,0)), input_shape=(3,160,320)))
model.add(Convolution2D(24,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(36,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(48,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(2212))
model.add(Dense(1106))
model.add(Dense(100))
model.add(Dense(30))
model.add(Dense(1))

# Model Training & Validation: 

# Run the model using generators
model.compile(loss = 'mse' , optimizer = 'adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= 
            len(train_samples)*3, validation_data=validation_generator, 
            nb_val_samples=len(validation_samples)*3, nb_epoch=num_epoch)


#Save the model:
model.save('model.h5')
print ("Model Saved.")

#Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Mean Squared Error (MSE) Loss')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch #')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()


print ("Exiting...")
