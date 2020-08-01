import os.path
import sys
import h5py
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D,add,normalization,Activation,concatenate,LocallyConnected1D,LSTM,CuDNNLSTM
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
#from keras.applications.imagenet_utils import _obtain_input_shape
from keras.optimizers import RMSprop,Adamax,Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.utils import to_categorical
from keras.models import load_model
import os

ascad_data_folder = "ASCAD_data/"
ascad_databases_folder = ascad_data_folder + "ASCAD_databases/"
ascad_trained_models_folder = ascad_data_folder + "ASCAD_trained_models/"


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def check_file_exists(file_path):
	if os.path.exists(file_path) == False:
		print("Error: provided file path '%s' does not exist!" % file_path)
		sys.exit(-1)
	return


def SCNet_v2(classes=256):
	input_shape = (700,1)
	img_input = Input(shape=input_shape)

#1-1
 

	conv1_4 = Conv1D(8, 7, activation='elu',dilation_rate=15, padding='same')(img_input)
	conv1_4=normalization.BatchNormalization()(conv1_4)
	conv4 = Conv1D(8, 7, activation='elu',dilation_rate=13, padding='same')(conv1_4)
	conv4=normalization.BatchNormalization()(conv4)
	conv4_1 = Conv1D(8, 7, activation='elu',dilation_rate=11, padding='same')(conv4)
	conv4_1=normalization.BatchNormalization()(conv4_1)
	conv1_2 = Conv1D(8, 7, activation='elu',dilation_rate=15, padding='same')(img_input)
	conv1_2=normalization.BatchNormalization()(conv1_2)
	conv2 = Conv1D(8, 7, activation='elu',dilation_rate=13, padding='same')(conv1_2)
	conv2=normalization.BatchNormalization()(conv2)
	conv2_1 = Conv1D(8, 7, activation='elu',dilation_rate=11, padding='same')(conv2)
	conv2_1=normalization.BatchNormalization()(conv2_1)


	conv1_41 = Conv1D(8, 7, activation='elu',dilation_rate=15, padding='same')(img_input)
	conv1_41=normalization.BatchNormalization()(conv1_41)
	conv41 = Conv1D(8, 7, activation='elu',dilation_rate=13, padding='same')(conv1_41)
	conv41=normalization.BatchNormalization()(conv41)
	conv4_11 = Conv1D(8, 7, activation='elu',dilation_rate=11, padding='same')(conv41)
	conv4_11=normalization.BatchNormalization()(conv4_11)
	conv1_21 = Conv1D(8, 7, activation='elu',dilation_rate=15, padding='same')(img_input)
	conv1_21=normalization.BatchNormalization()(conv1_21)
	conv21 = Conv1D(8, 7, activation='elu',dilation_rate=13, padding='same')(conv1_21)
	conv21=normalization.BatchNormalization()(conv21)
	conv2_11 = Conv1D(8, 7, activation='elu',dilation_rate=11, padding='same')(conv21)
	conv2_11=normalization.BatchNormalization()(conv2_11)
	conv11 = Conv1D(8, 7, activation='elu',dilation_rate=15, padding='same')(img_input)
	conv11=normalization.BatchNormalization()(conv11)
	conv11 = Conv1D(8, 7, activation='elu', dilation_rate=13,padding='same')(conv11)
	conv11=normalization.BatchNormalization()(conv11)
	conv11 = Conv1D(8, 7, activation='elu', dilation_rate=11,padding='same')(conv11)
	conv11=normalization.BatchNormalization()(conv11)

	conv1 = Conv1D(8, 7, activation='elu',dilation_rate=15, padding='same')(img_input)
	conv1=normalization.BatchNormalization()(conv1)
	conv1 = Conv1D(8, 7, activation='elu', dilation_rate=13,padding='same')(conv1)
	conv1=normalization.BatchNormalization()(conv1)
	conv1 = Conv1D(8, 7, activation='elu', dilation_rate=11,padding='same')(conv1)
	conv1=normalization.BatchNormalization()(conv1)

	conv1=concatenate([conv4_1,conv1,conv2_1,conv4_11,conv11,conv2_11],2)
	conv1 = AveragePooling1D(4, 4)(conv1)


	conv1 = CuDNNLSTM(16, return_sequences=True)(conv1)
	conv1=normalization.BatchNormalization()(conv1)



#2-1
	conv1_4 = Conv1D(32, 7, activation='elu', dilation_rate=9,padding='same')(conv1)
	conv1_4=normalization.BatchNormalization()(conv1_4)
	conv4 = Conv1D(32, 7, activation='elu',dilation_rate=7, padding='same')(conv1_4)
	conv4=normalization.BatchNormalization()(conv4)
	conv4_1 = Conv1D(32, 7, activation='elu', dilation_rate=5,padding='same')(conv4)
	conv4_1=normalization.BatchNormalization()(conv4_1)
	conv1_2 = Conv1D(32, 7, activation='elu',dilation_rate=9, padding='same')(conv1)
	conv1_2=normalization.BatchNormalization()(conv1_2)
	conv2 = Conv1D(32, 7, activation='elu', dilation_rate=7,padding='same')(conv1_2)
	conv2=normalization.BatchNormalization()(conv2)
	conv2_1 = Conv1D(32, 7, activation='elu',dilation_rate=5, padding='same')(conv2)
	conv2_1=normalization.BatchNormalization()(conv2_1)


	conv1_41 = Conv1D(32, 7, activation='elu', dilation_rate=9,padding='same')(conv1)
	conv1_41=normalization.BatchNormalization()(conv1_41)
	conv41 = Conv1D(32, 7, activation='elu',dilation_rate=7, padding='same')(conv1_41)
	conv41=normalization.BatchNormalization()(conv41)
	conv4_11 = Conv1D(32, 7, activation='elu', dilation_rate=5,padding='same')(conv41)
	conv4_11=normalization.BatchNormalization()(conv4_11)
	conv1_21 = Conv1D(32, 7, activation='elu',dilation_rate=9, padding='same')(conv1)
	conv1_21=normalization.BatchNormalization()(conv1_21)
	conv21 = Conv1D(32, 7, activation='elu', dilation_rate=7,padding='same')(conv1_21)
	conv21=normalization.BatchNormalization()(conv21)
	conv2_11 = Conv1D(32, 7, activation='elu',dilation_rate=5, padding='same')(conv21)
	conv2_11=normalization.BatchNormalization()(conv2_11)
	conv11 = Conv1D(32, 7, activation='elu', dilation_rate=9,padding='same')(conv1)
	conv11=normalization.BatchNormalization()(conv11)
	conv11 = Conv1D(32, 7, activation='elu', dilation_rate=7,padding='same')(conv11)
	conv11=normalization.BatchNormalization()(conv11)
	conv11 = Conv1D(32, 7, activation='elu', dilation_rate=5,padding='same')(conv11)
	conv11=normalization.BatchNormalization()(conv11)

	conv1 = Conv1D(32, 7, activation='elu', dilation_rate=9,padding='same')(conv1)
	conv1=normalization.BatchNormalization()(conv1)
	conv1 = Conv1D(32, 7, activation='elu', dilation_rate=7,padding='same')(conv1)
	conv1=normalization.BatchNormalization()(conv1)
	conv1 = Conv1D(32, 7, activation='elu', dilation_rate=5,padding='same')(conv1)
	conv1=normalization.BatchNormalization()(conv1)


	conv1=concatenate([conv4_1,conv1,conv2_1,conv4_11,conv11,conv2_11],2)
	conv1 = AveragePooling1D(4, 4)(conv1)

	conv1 = CuDNNLSTM(64, return_sequences=True)(conv1)
	conv1=normalization.BatchNormalization()(conv1)



#3-1
	conv1_4 = Conv1D(92, 7, activation='elu',dilation_rate=5, padding='same')(conv1)
	conv1_4=normalization.BatchNormalization()(conv1_4)
	conv4 = Conv1D(92, 7, activation='elu',dilation_rate=2, padding='same')(conv1_4)
	conv4=normalization.BatchNormalization()(conv4)
	conv4_1 = Conv1D(92, 7, activation='elu',dilation_rate=1, padding='same')(conv4)
	conv4_1=normalization.BatchNormalization()(conv4_1)
	conv1_2 = Conv1D(92, 7, activation='elu',dilation_rate=5, padding='same')(conv1)
	conv1_2=normalization.BatchNormalization()(conv1_2)
	conv2 = Conv1D(92, 7, activation='elu',dilation_rate=2, padding='same')(conv1_2)
	conv2=normalization.BatchNormalization()(conv2)
	conv2_1 = Conv1D(92, 7, activation='elu',dilation_rate=1, padding='same')(conv2)
	conv2_1=normalization.BatchNormalization()(conv2_1)

  
	conv1_41 = Conv1D(92, 7, activation='elu', dilation_rate=5,padding='same')(conv1)
	conv1_41=normalization.BatchNormalization()(conv1_41)
	conv41 = Conv1D(92, 7, activation='elu',dilation_rate=2, padding='same')(conv1_41)
	conv41=normalization.BatchNormalization()(conv41)
	conv4_11 = Conv1D(92, 7, activation='elu', dilation_rate=1,padding='same')(conv41)
	conv4_11=normalization.BatchNormalization()(conv4_11)
	conv1_21 = Conv1D(92, 7, activation='elu',dilation_rate=5, padding='same')(conv1)
	conv1_21=normalization.BatchNormalization()(conv1_21)
	conv21 = Conv1D(92, 7, activation='elu', dilation_rate=2,padding='same')(conv1_21)
	conv21=normalization.BatchNormalization()(conv21)
	conv2_11 = Conv1D(92, 7, activation='elu',dilation_rate=1, padding='same')(conv21)
	conv2_11=normalization.BatchNormalization()(conv2_11)
	conv11 = Conv1D(92, 7, activation='elu', dilation_rate=5,padding='same')(conv1)
	conv11=normalization.BatchNormalization()(conv11)
	conv11 = Conv1D(92, 7, activation='elu', dilation_rate=2,padding='same')(conv11)
	conv11=normalization.BatchNormalization()(conv11)
	conv11 = Conv1D(92, 7, activation='elu', dilation_rate=1,padding='same')(conv11)
	conv1=normalization.BatchNormalization()(conv1)

	conv1 = Conv1D(92, 7, activation='elu',dilation_rate=5, padding='same')(conv1)
	conv1=normalization.BatchNormalization()(conv1)
	conv1 = Conv1D(92, 7, activation='elu', dilation_rate=2,padding='same')(conv1)
	conv1=normalization.BatchNormalization()(conv1)
	conv1 = Conv1D(92, 7, activation='elu', dilation_rate=1,padding='same')(conv1)
	conv1=normalization.BatchNormalization()(conv1)

	conv1=concatenate([conv4_1,conv1,conv2_1,conv4_11,conv11,conv2_11],2)
	conv1 = AveragePooling1D(4, 4)(conv1)
 

	conv1 = CuDNNLSTM(256 , return_sequences=False)(conv1)
	conv1=normalization.BatchNormalization()(conv1)

	x = Dense(256, activation='elu')(conv1)
	x= normalization.BatchNormalization()(x)

	x = Dense(classes, activation='softmax')(x)

	inputs = img_input
	# Create model.
	model = Model(inputs, x, name='SCNet_v2')
	optimizer = RMSprop(lr=5e-5)
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

#### ASCAD helper to load profiling and attack data (traces and labels)
# Loads the profiling and attack datasets from the ASCAD
# database
def load_ascad(ascad_database_file, load_metadata=False):
	check_file_exists(ascad_database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file  = h5py.File(ascad_database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
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
def train_model(X_profiling, Y_profiling, model, save_file_name=None, epochs=150, batch_size=100,name=None):
	if save_file_name:
		check_file_exists(os.path.dirname(save_file_name))
		# Save model every epoch
		save_model = ModelCheckpoint(save_file_name, monitor='val_acc',save_best_only=True)
		es=EarlyStopping(monitor='val_acc',patience=10)
		decay=ReduceLROnPlateau(monitor='val_acc',patience=3,factor=0.5)
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
		                    batch_size=batch_size, verbose=1, epochs=epochs, callbacks=callbacks,sameation_split=0.2)
	else:
		save_model = ModelCheckpoint(ascad_trained_models_folder+'best_model_'+name+'.h5', monitor='val_acc', save_best_only=True)
		es=EarlyStopping(monitor='val_acc',patience=15)
		decay = ReduceLROnPlateau(monitor='val_loss', patience=6,factor=0.1)
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
	return history


# Our folders


# Load the profiling traces in the ASCAD database with no desync
(X_profiling, Y_profiling), (X_attack, Y_attack) = load_ascad(ascad_databases_folder + "ASCAD.h5")
(X_profiling_desync50, Y_profiling_desync50), (X_attack_desync50, Y_attack_desync50) = load_ascad(ascad_databases_folder + "ASCAD_desync50.h5")
(X_profiling_desync100, Y_profiling_desync100), (X_attack_desync100, Y_attack_desync100) = load_ascad(ascad_databases_folder + "ASCAD_desync100.h5")

#### No desync
cnn_best_model = SCNet_v2()
train_model(X_profiling, Y_profiling, cnn_best_model, epochs=75, batch_size=100,name='desync0_SCNet_v2')
#### Desync = 50
cnn_best_model = SCNet_v2()
train_model(X_profiling_desync50, Y_profiling_desync50, cnn_best_model,  epochs=75, batch_size=100,name='desync50_SCNet_v2')
#### Desync = 100
cnn_best_model = SCNet_v2()
train_model(X_profiling_desync100, Y_profiling_desync100, cnn_best_model,  epochs=75, batch_size=100,name='desync100_SCNet_v2')
