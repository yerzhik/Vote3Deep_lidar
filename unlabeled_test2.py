import numpy as np
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Convolution3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.utils import np_utils
from sklearn.metrics import mean_squared_error
import glob
from keras import __version__ as keras_version

from keras import backend as K
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

avail_gpus = K.tensorflow_backend._get_available_gpus()
print(avail_gpus)

np.random.seed(420)

def read_data ():
	data = []
	files = glob.glob ('/home/root/host_home/projects/yerzhan/test_data/pcl_rs16Rec_20180822_0129/grid_files/*.csv')
	print("read_data: len of files: " + str(len(files)))
	for i in range (len(files)):
		files[i] = files [i].replace ('.csv', '')
		f = files[i]
		dt = np.loadtxt (f+'.csv', delimiter=',')
		#print(dt.shape)
		x = np.zeros (10*10*10*6)
		x = x.reshape ((6,10,10,10))
		for j in dt:
			for k in range (6):
				x[k][int(j[0]-1)][int(j[01]-1)][int(j[02]-1)] = j[k+3]
		data.append (x)

	data = np.array (data)
	arr = np.arange(len (data))
	np.random.shuffle(arr)
	tX = data[arr]
	return tX

def calc_err (a, b):
	s = np.zeros (len(a))
	for i in range (len (a)):
		s[i] = ((a[i] - b[i]) ** 2) 

def gen_model (input_shape=(6,10,10,10)):
	model = Sequential()
	inp = Input (input_shape)
	model.add(Convolution3D(8, 3, 3, 3, border_mode='same',
							input_shape=input_shape))
	X = Convolution3D(8, 3, 3, 3, border_mode='same')(inp)
	model.add(Activation('relu'))
	X = Activation('relu') (X)
	model.add(ZeroPadding3D())
	X = ZeroPadding3D()(X)
	model.add(Convolution3D(8, 3, 3, 3))
	X = Convolution3D(8, 3, 3, 3)(X)
	model.add(Activation('relu'))
	X = Activation('relu')(X)
	model.add(ZeroPadding3D())
	X = ZeroPadding3D()(X)

	model.add(Convolution3D(1, 3, 3, 3))
	X = Convolution3D(1, 3, 3, 3)(X)
	model.add(Activation('relu'))
	X = Activation('relu')(X)
	model.add(Flatten())
	X = Flatten()(inp)#(X)
	cm = model
	cm.compile(optimizer=Adam (1e-4), loss='mse'
		#, metrics = ['accuracy']
	)
	#print("the cm shape is:")
	#print(cm.get_output_at(0).get_shape().as_list())

	Z = X
	model.add(Dense(1024))
	X = Dense(1024)(X)
	model.add(Activation('relu'))
	X = Activation('relu')(X)
	model.add(Dense(1024))
	X = Dense(1024)(X)
	model.add(Activation('relu'))
	X = Activation('relu')(X)
	'''model.add(Dense(3))
	X1 = Dense(3)(X)
	model.add(Activation('softmax'))
	X1 = Activation('softmax', name='Class Act')(X1)
	'''
	model.add(Dense(400))
	X2 = Dense(400)(X)
	model.add(Activation('relu'))
	X2 = Activation('relu')(X2)
	model.add(Dense(4))
	X2 = Dense(4)(X2)
	model.add(Activation('relu'))

	model.compile(optimizer=Adam (1e-4), loss='mse'
		#, metrics = ['accuracy']
	)
	#return model,cm

	X2 = Activation('relu', name='Box_Act')(X2)
	model = Model (inp, [X1, X2])
	convModel = Model (inp, Z)
	sgd = SGD(lr=2e-3, decay=1e-4, momentum=0.9, nesterov=True)
	
	adm = Adam (lr = 1e-4)
	model.compile(optimizer=sgd, loss=['binary_crossentropy', 'mse']
		#, metrics = ['accuracy']
	)
	convModel.compile(optimizer=sgd, loss='mse'
		#, metrics = ['accuracy']
	)

	return model, convModel

if __name__ == '__main__':
	print('Keras version: {}'.format(keras_version))
	model, cm = gen_model ()
	model.summary ()
	print 'Reading data'
	tX = read_data ()
	model.load_weights('vote3deep.h5')
	print("let's predict tX of shape: " + str(tX.shape))
	y2 = model.predict(tX, batch_size=32, verbose=0)
	y = cm.predict (tX)
	np.savetxt ('tX_to_y2_predict.txt', y2)
	np.savetxt ('TX0-3_to_y_predict.txt', y)
	print 'DONE!'
