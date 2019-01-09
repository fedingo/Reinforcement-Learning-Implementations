import numpy as np
from abstractAgent import *
import random

def normalize (array):

	a_max = np.max(array)
	a_min = np.min(array)

	# Reward Min-Max Normalization (-1, +1)
	if a_max == a_min:
		return np.zeros(len(array))
	
	array -= a_min
	array /= (a_max - a_min)/2
	array -= 1

	return array

class PGAgent(abstractAgent):

	def __init__(self, state_size, action_size):
		print("Creating a Policy Gradient Agent")

		self.entropy_weight = 5e-5
		self.update_rate= 5
		self.batch_size = 2048	# Batch size for the training

		architecture = [64, 64]

		super().__init__(state_size, action_size, architecture)

	## Currently NOT allowing for any Image Input
	## Todo: add convolution 
	def build(self):
		
		with tf.name_scope("inputs"):
			self.input_net = tf.placeholder(tf.float32, [None, self.state_size], name="input_state")
			self.actions   = tf.placeholder(tf.int32, [None, 1], name="actions_performed")
			self.d_rewards = tf.placeholder(tf.float32, [None, ], name="discounted_epiosde_rewards")		
		
		with tf.name_scope("dense_layers_architecture"):
			self.dense1 = self.input_net

			for n_layer in self.architecture:
				self.dense1 = tf.contrib.layers.fully_connected(inputs = self.dense1,
			                                                num_outputs = n_layer,
			                                                activation_fn = tf.nn.relu,
			                                                weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False))

		with tf.name_scope("softmax_output"):
			self.output = tf.contrib.layers.fully_connected(inputs = self.dense1,
														num_outputs = self.action_size,
														activation_fn = None,
														weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False))  

			self.action_distribution = tf.nn.softmax(self.output)  

		with tf.name_scope("entropy"):
			self.entropy = self.entropy_weight * tf.reduce_sum(\
			tf.multiply(self.action_distribution, tf.log(self.action_distribution)))


		with tf.name_scope("loss"):
			log_prob = tf.log(self.action_distribution)
			indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self.actions
			act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)
			self.loss = -tf.reduce_mean(tf.multiply(act_prob, self.d_rewards)) \
			 + self.entropy

		with tf.name_scope("training_operation"):
			self.train_operation = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

	# Function to discount the past rewards on the last episode
	def discountRewards(self):

		if not self.memory[-1]["done"]:
			raise Exception("Trying to Discount not at the end of the episode")

		accumulator = 0
		for obj in reversed(self.memory):
			if obj["done"] and accumulator != 0:
				break
			obj["reward"] += self.gamma*accumulator
			accumulator = obj["reward"]

	# function to define when to train the network
	def hasToTrain(self, step, done, episode):

		if done:
			self.discountRewards()
		return done and (episode+1) % self.update_rate == 0

	# this function is not needed since we never have hasToTrain == True 
	def train(self):

		current_batch_size = min([len(self.memory), self.batch_size])
		index_list = range(len(self.memory))
		minibatch = random.sample(index_list, current_batch_size)
		
		states_train = []
		action_train = []
		reward_train = []

		for index in minibatch:
			
			obj = self.memory[index]

			state  = obj['state']
			action = obj['action']
			reward = obj['reward']
			
			states_train.append(state)	
			action_train.append(action)
			reward_train.append(reward)


		states_train = np.array(states_train)
		action_train = np.array(action_train)
		reward_train = normalize(np.array(reward_train))

		states_train = np.reshape(states_train, [current_batch_size, self.state_size])
		action_train = np.reshape(action_train, [current_batch_size, 1])
		
		self.tf_session.run([self.loss, self.train_operation], feed_dict={
											   self.input_net: states_train,
	                                           self.actions:   action_train,
	                                           self.d_rewards: reward_train 
                                                })

	def act(self, state):

		prob_distribution = self.tf_session.run(self.action_distribution, 
							feed_dict={self.input_net: state.reshape([1,self.state_size])})

		action = np.random.choice(self.action_size, 1, p=prob_distribution.ravel())[0]
		
		return action
