import os.path
import sys
import h5py
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D,add,normalization,Activation,concatenate,Dropout,LocallyConnected1D,LSTM,CuDNNLSTM
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
#from keras.applications.imagenet_utils import _obtain_input_shape
from keras.optimizers import RMSprop,Adamax,Adam,SGD
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.utils import to_categorical
from keras.models import load_model
import os

dpa_data_folder = "DPA_data/"
dpa_databases_folder = dpa_data_folder + "DPA_databases/"
dpa_trained_models_folder = dpa_data_folder + "DPA_trained_models/"


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def check_file_exists(file_path):
	if os.path.exists(file_path) == False:
		print("Error: provided file path '%s' does not exist!" % file_path)
		sys.exit(-1)
	return

def SCNet_v1(classes=256):

	input_shape = (500,1)
	img_input = Input(shape=input_shape)

	img_flatten=Flatten()(img_input)

 
	conv1 = Conv1D(32, 11, activation='elu',dilation_rate=13, padding='same')(img_input)
	conv1=normalization.BatchNormalization()(conv1)
	conv1 = Conv1D(48, 11, activation='elu', dilation_rate=11,padding='same')(conv1)
	conv1=normalization.BatchNormalization()(conv1)
	conv1 = Conv1D(64, 11, activation='elu', dilation_rate=9,padding='same')(conv1)
	conv1=normalization.BatchNormalization()(conv1)

	conv1 = AveragePooling1D(3, 3)(conv1)
 
	conv1 = CuDNNLSTM(64, return_sequences=True)(conv1)
	conv1=normalization.BatchNormalization()(conv1)

	conv1 = Conv1D(96, 11, activation='elu', dilation_rate=13,padding='same')(conv1)
	conv1=normalization.BatchNormalization()(conv1)
	conv1 = Conv1D(112, 11, activation='elu', dilation_rate=11,padding='same')(conv1)
	conv1=normalization.BatchNormalization()(conv1)
	conv1 = Conv1D(128, 11, activation='elu', dilation_rate=9,padding='same')(conv1)
	conv1=normalization.BatchNormalization()(conv1)
 
	conv1 = AveragePooling1D(4, 4)(conv1)
 
	conv1 = CuDNNLSTM(128, return_sequences=True)(conv1)
	conv1=normalization.BatchNormalization()(conv1)
 
	conv1 = Conv1D(192, 9, activation='elu',dilation_rate=13, padding='same')(conv1)
	conv1=normalization.BatchNormalization()(conv1)
	conv1 = Conv1D(224, 9, activation='elu', dilation_rate=11,padding='same')(conv1)
	conv1=normalization.BatchNormalization()(conv1)
	conv1 = Conv1D(256, 9, activation='elu', dilation_rate=9,padding='same')(conv1)
	conv1=normalization.BatchNormalization()(conv1)
 
	conv1 = AveragePooling1D(4, 4)(conv1)

	conv1 = CuDNNLSTM(512, return_sequences=False)(conv1)
	conv1=normalization.BatchNormalization()(conv1)

	x = Dense(256, activation='elu', name='fc1')(conv1)
	x= normalization.BatchNormalization()(x)

	x = Dense(classes, activation='softmax', name='predictions')(x)

	inputs = img_input
	# Create model.
	model = Model(inputs, x, name='SCNet_v1')
	optimizer = RMSprop(lr=5.5e-5)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model


def load_sca_model(model_file):
	check_file_exists(model_file)
	try:
		model = load_model(model_file)
	except:
		print("Error: can't load Keras model file '%s'" % model_file)
		sys.exit(-1)
	return model

#### dpa helper to load profiling and attack data (traces and labels)
# Loads the profiling and attack datasets from the dpa
# database
def load_dpa(dpa_database_file, load_metadata=False):
	check_file_exists(dpa_database_file)
	# Open the dpa database HDF5 for reading
	try:
		in_file  = h5py.File(dpa_database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % dpa_database_file)
		sys.exit(-1)
	# Load profiling traces
	X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
	# Load profiling labels
	Y_profiling = np.array(in_file['Profiling_traces/labels'])
	# Load attacking traces
	X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
	# Load attacking labels
	Y_attack = np.array(in_file['Attack_traces/labels'])
	if load_metadata == False:
		return (X_profiling, Y_profiling), (X_attack, Y_attack)
	else:
		return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])

#### Training high level function
def train_model(X_profiling, Y_profiling, model, save_file_name=None, epochs=150, batch_size=50,name=None):
	if save_file_name:
		check_file_exists(os.path.dirname(save_file_name))
		# Save model every epoch
		save_model = ModelCheckpoint(save_file_name, monitor='val_acc',save_best_only=True)
		es=EarlyStopping(monitor='val_acc',patience=15)
		decay=ReduceLROnPlateau(monitor='val_acc',patience=3,factor=0.4)
		callbacks = [save_model,es,decay]
		# Get the input layer shape
		input_layer_shape = model.get_layer(index=0).input_shape
		# Sanity check
		if input_layer_shape[1] != len(X_profiling[0]):
			print("Error: model input shape %d instead of %d is not expected ..." % (
			input_layer_shape[1], len(X_profiling[0])))
			sys.exit(-1)
		# Adapt the data shape according our model input
		if len(input_layer_shape) == 2:
			# This is a MLP
			Reshaped_X_profiling = X_profiling
		elif len(input_layer_shape) == 3:
			# This is a CNN: expand the dimensions
			Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
		else:
			print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
			sys.exit(-1)

		history = model.fit(x=Reshaped_X_profiling, y=to_categorical(Y_profiling, num_classes=256),
		                    batch_size=batch_size, verbose=1, epochs=epochs, callbacks=callbacks,validation_split=0.2)
	else:
		save_model = ModelCheckpoint(dpa_trained_models_folder+'best_model_'+name+'.h5', monitor='val_acc', save_best_only=True)
		es=EarlyStopping(monitor='val_acc',patience=15)
		decay = ReduceLROnPlateau(monitor='val_acc', patience=8,factor=0.8)
		callbacks = [save_model,es,decay]
		# Get the input layer shape
		input_layer_shape = model.get_layer(index=0).input_shape
		# Sanity check
		if input_layer_shape[1] != len(X_profiling[0]):
			print("Error: model input shape %d instead of %d is not expected ..." % (
			input_layer_shape[1], len(X_profiling[0])))
			sys.exit(-1)
		# Adapt the data shape according our model input
		if len(input_layer_shape) == 2:
			# This is a MLP
			Reshaped_X_profiling = X_profiling
		elif len(input_layer_shape) == 3:
			# This is a CNN: expand the dimensions
			Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
		else:
			print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
			sys.exit(-1)

		history = model.fit(x=Reshaped_X_profiling, y=to_categorical(Y_profiling, num_classes=256),
		                    batch_size=batch_size, verbose=1, epochs=epochs, callbacks=callbacks,validation_split=0.04)
	return history


# Our folders


# Load the profiling traces in the dpa database with no desync
(X_profiling, Y_profiling), (X_attack, Y_attack) = load_dpa(dpa_databases_folder + "DPA.h5")

cnn_best_model = SCNet_v1()
train_model(X_profiling, Y_profiling, cnn_best_model, epochs=75, batch_size=200,name='desync0_SCNet_v1')

