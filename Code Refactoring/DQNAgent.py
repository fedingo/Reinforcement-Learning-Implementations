from keras.layers import Input, Dense, Conv2D, Flatten, Dropout
from keras.models import Model, clone_model
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K

import random
import numpy as np
from abstractAgent import *

def huber_loss(a, b):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    use_linear_term = K.cast(use_linear_term, 'float32')

    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

class DQNAgent(abstractAgent):

	def __init__(self, state_size, action_size):
		print("Creating a DQN Agent")

		self.exploration_rate  = 1
		self.exploration_decay = 0.99
		self.exploration_min   = 0.02

		self.train_rate = 4
		self.update_frozen = 50
		self.batch_size = 16
		self.double = True

		learning_rate = 5e-4

		architecture = {'conv' : [4, 16],
						'fc'   : [64, 64]
						}

		super().__init__(state_size, action_size, architecture, learning_rate = learning_rate)

	def __del__(self):
		del self.model

	def __reshaped_state_size__(self, dim = 1):
		result = 0

		if type(self.state_size) == type(1):
			result = [dim, self.state_size]
		elif type(self.state_size) == type(list()):
			result = [dim] + self.state_size
		else:
			raise Exception("State_size type not supported")

		return result

	def build(self):

		# Model for the 1D state_size
		global image_input, out
		if type(self.state_size) == type(1):

			image_input = Input(shape = [self.state_size])
			x = image_input

			for layer_size in self.architecture['fc']:
				x = Dense(layer_size, activation='relu')(x)

			out = Dense(self.action_size, activation='linear')(x)

		# Model for the Multi-Dimensional state_size
		elif type(self.state_size) == type(list()):
			image_input = Input(shape = self.state_size)
			x = image_input

			for layer_filters in self.architecture['conv']:
				x = Conv2D(layer_filters, (4,4), (2,2), activation='relu')(x)

			x = Flatten()(x)

			for layer_size in self.architecture['fc']:
				x = Dense(layer_size, activation='relu')(x)

			out = Dense(self.action_size, activation='linear')(x)

		else:
			raise Exception("State_size type not supported")
			
		self.model = Model(inputs=image_input, outputs=out)
		self.model.compile(loss=huber_loss, optimizer=Adam(lr = self.learning_rate))

		self.frozen_model = clone_model(self.model)
		self.frozen_model.set_weights(self.model.get_weights())


	# function to define when to train the network
	def hasToTrain(self, step, done, episode):

		if episode % self.update_frozen == 0 and done:
			self.sync_models()

		if done:
			self.decay_explore()

		return step % self.train_rate == 0

	def train(self):

		if len(self.memory) < self.batch_size:
			return

		index_list = range(len(self.memory))
		minibatch = random.sample(index_list, self.batch_size)
	
		x_train = []
		y_train = []
		
		for index in minibatch:
			
			obj = self.memory[index]

			state  = obj['state']
			action = obj['action']
			reward = obj['reward']
			next_s = obj['next_state']
			done   = obj['done']

			stateSize = self.__reshaped_state_size__()

			data_train = np.reshape(state, stateSize)
			target = self.model.predict(data_train)[0]
				
			data_next = np.reshape(next_s, stateSize)
			if self.double:
				target_action = np.argmax(self.model.predict(data_next))
				max_next_target = self.frozen_model.predict(data_next)[0][target_action]
			else:
				max_next_target = np.max(self.frozen_model.predict(data_next)[0])
				
			target[action] = reward + self.gamma*max_next_target
			if done:
				target[action] = reward
				
			x_train.append(data_train)
			y_train.append(target)

		stateSizeBatch = self.__reshaped_state_size__(self.batch_size)
		x_train = np.array(x_train)
		x_train = np.reshape(x_train, stateSizeBatch)
		y_train = np.array(y_train)
		y_train = np.reshape(y_train, [self.batch_size, self.action_size])

		self.model.train_on_batch(x_train, y_train)

	def act(self, state, testing = False):
		if np.random.uniform(0,1) < self.exploration_rate and not testing:
			action = np.random.randint(self.action_size)
		else:
			stateSize = self.__reshaped_state_size__()

			state = np.reshape(state, stateSize)
			q_values = self.model.predict(state)[0]			
			action = np.argmax(q_values)
		
		return action

	def decay_explore (self):
		if self.exploration_rate > self.exploration_min:
			self.exploration_rate *= self.exploration_decay
		else:
			if self.exploration_rate != 0:
				print("\n No more Exploration!")
				self.exploration_rate = 0

	def sync_models(self):
		print("Model Sync")
		self.frozen_model.set_weights(self.model.get_weights())