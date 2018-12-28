from keras.layers import Input, Dense
from keras.models import Model, clone_model
from keras.optimizers import Adam

import random
import numpy as np
from abstractAgent import *

class DQNAgent(abstractAgent):

	def __init__(self, state_size, action_size, architecture):
		print("Creating a DQN Agent")

		self.exploration_rate  = 1
		self.exploration_decay = 0.995
		self.exploration_min   = 0.02

		self.train_rate = 32
		self.update_frozen = 40
		self.batch_size = 32
		self.double = True

		super().__init__(state_size, action_size, architecture)

	def build(self):
		print("Building model!")

		# Model for the 1D state_size
		if type(self.state_size) == type(1):

			image_input = Input(shape = [self.state_size])
			x = image_input

			for layer_size in self.architecture:
				x = Dense(layer_size, activation='relu')(x)

			out = Dense(self.action_size, activation='linear')(x)
			
			self.model = Model(inputs=image_input, outputs=out)
			self.model.compile(loss='mse', optimizer=Adam(lr = self.learning_rate))
			self.model.summary()

			self.frozen_model = clone_model(self.model)
			self.frozen_model_half = clone_model(self.model)
			self.pre_sync_models()
			self.sync_models()


	# function to define when to train the network
	def hasToTrain(self, step, done, episode):
		if episode % self.update_frozen == 0 and done:
			self.sync_models()
			self.clear_memory()

		if episode % (self.update_frozen//2) == 0 and done:
			self.pre_sync_models()

		if done:
			self.decay_explore()

		return ((step + 1) % (self.train_rate) == 0 or done)

	def train(self):
		current_batch = self.batch_size
		if len(self.memory) < self.batch_size:
			current_batch = len(self.memory)
		
		index_list = range(len(self.memory))
		minibatch = random.sample(index_list, current_batch)
	
		x_train = []
		y_train = []
		
		for index in minibatch:
			
			obj = self.memory[index]

			state  = obj['state']
			action = obj['action']
			reward = obj['reward']
			next_s = obj['next_state']

			data_train = np.reshape(state, [1, self.state_size])	
			target = self.model.predict(data_train)[0]
				
			data_next = np.reshape(next_s, [1, self.state_size])
			if self.double:
				target_action = np.argmax(self.model.predict(data_next))
				max_next_target = self.frozen_model.predict(data_next)[0][target_action]
			else:
				max_next_target = np.max(self.frozen_model.predict(data_next)[0])
				
			target[action] = reward + self.gamma*max_next_target
				
			x_train.append(data_train)
			y_train.append(target)
			
		x_train = np.array(x_train)
		x_train = np.reshape(x_train, [current_batch, self.state_size])
		y_train = np.array(y_train)
		y_train = np.reshape(y_train, [current_batch, self.action_size])
		
		self.model.train_on_batch(x_train, y_train)

	def act(self, state):
		if np.random.uniform(0,1) < self.exploration_rate:
			action = np.random.randint(self.action_size)
		else:
			state = np.reshape(state, [1, len(state)])
			q_values = self.model.predict(state)[0]			
			action = np.argmax(q_values)
		
		return action

	def decay_explore (self):
		if self.exploration_rate > self.exploration_min:
			#print("Exploration rate: %f" % self.exploration_rate)
			self.exploration_rate *= self.exploration_decay
		else:
			if self.exploration_rate != 0:
				self.exploration_rate = 0

	def pre_sync_models(self):
		self.frozen_model_half.set_weights(self.model.get_weights())

	def sync_models(self):
		self.frozen_model.set_weights(self.frozen_model_half.get_weights())