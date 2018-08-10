import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from collections import deque

import tensorflow as tf

#Define the Choosen game from the list below
choosen = 2
restart_bound = 1000 
episode_cap = 3000

render = False

# 1 Mountain Car
# 2 CartPole
# 3 Acrobot
# 4 Mountain Car Easy

class myPGAgent:
	
	def __init__(self, state_size, action_size):
		
		self.state_size = state_size
		self.action_size = action_size
		
		self.memory = deque( maxlen = 5000)
		
		self.learning_rate = 1e-3

		self.gamma = 0.99		# discount rate
		self.update_rate = 5 	# N° of episodes between updates
		self.batch_size = 1024	# Batch size for the training

		self.layers_architecture = [64, 64]
		
		self.create_model()

	def create_model(self):
		
		with tf.name_scope("inputs"):
			self.input_net = tf.placeholder(tf.float32, [None, self.state_size], name="input_state")
			self.actions   = tf.placeholder(tf.int32, [None, 1], name="actions_performed")
			self.d_rewards = tf.placeholder(tf.float32, [None, ], name="discounted_epiosde_rewards")		
		
		with tf.name_scope("dense_layers_architecture"):
			self.dense1 = self.input_net

			for n_layer in self.layers_architecture:
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

		with tf.name_scope("loss"):
			log_prob = tf.log(self.action_distribution)
			indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self.actions
			act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)
			self.loss = -tf.reduce_mean(tf.multiply(act_prob, self.d_rewards))

		with tf.name_scope("training_operation"):
			self.train_operation = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

	def store(self, state, action, reward, done):
		
		self.memory.append([state, action, reward, done])

	def clear_memory(self):

		self.memory.clear()

	def act(self, state, tf_session):

		prob_distribution = tf_session.run(self.action_distribution, 
							feed_dict={self.input_net: state.reshape([1,self.state_size])})

		action = np.random.choice(self.action_size, 1, p=prob_distribution.ravel())[0]
		
		return action
		
	def discount_rewards_last_episode (self):
		
		done_encountered = 0 #we need to go back until we find the end of the second-last episode
		cumulative = 0

		for x in reversed(self.memory):

			if x[3]: #done
				if done_encountered == 0:
					done_encountered += 1
					cumulative = x[2] #reward
				else:
					break
			else:
				x[2] += self.gamma*cumulative
				cumulative = x[2]

	def train (self, tf_session):
		
		current_batch_size = min([len(self.memory), self.batch_size])
		index_list = range(len(self.memory))
		minibatch = random.sample(index_list, current_batch_size)
		
		states_train = []
		action_train = []
		reward_train = []

		# Reward Min-Max Normalization (-1, +1)
		reward_list = np.array(self.memory)[:,2]
		reward_max = np.max(reward_list)
		reward_min = np.min(reward_list)
		if reward_max != reward_min:
			reward_list -= reward_min
			reward_list /= (reward_max - reward_min)/2
			reward_list -= 1
		
		for state, action, reward, done in np.array(self.memory)[minibatch]:
			states_train.append(state)

		states_train = np.array(states_train)
		action_train = np.array(self.memory)[minibatch,1]
		reward_train = reward_list[minibatch]

		states_train = np.reshape(states_train, [current_batch_size, self.state_size])
		action_train = np.reshape(action_train, [current_batch_size, 1])
		
		tf_session.run([self.loss, self.train_operation], feed_dict={self.input_net: states_train,
	                                           self.actions:   action_train,
	                                           self.d_rewards: reward_train 
                                                })
		


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate():
	x_array = range(len(mean_array))
	plt.plot(x_array, mean_array, 'C1')
	plt.plot(x_array, mean_array_100, 'C2')
	plt.draw()
	plt.pause(0.001)
	return  plt.fignum_exists(1)
	
plt.ion()
plt.show()

if choosen == 1:
	env = gym.make('MountainCar-v0')
	target_score = -150
	done_reward = +10
	cap_score = -200
	lowest_score = -200

elif choosen == 2:
	env = gym.make('CartPole-v0')
	target_score = 160
	done_reward = -10
	cap_score = 200
	lowest_score = 10

elif choosen == 3:
	env = gym.make('Acrobot-v1')
	target_score = -200
	done_reward = +10
	cap_score = -500
	lowest_score = -500

elif choosen == 4:
	gym.envs.register(
	    id='MountainCarMyEasyVersion-v0',
	    entry_point='gym.envs.classic_control:MountainCarEnv',
	    max_episode_steps= 500,
	    reward_threshold= -500.0,
	)
	env = gym.make('MountainCarMyEasyVersion-v0')
	target_score = -150
	done_reward = +10
	cap_score = -500
	lowest_score = -500

state = env.reset()
score = 0
score_mean = lowest_score
mean_array = []

score_mean_100 = lowest_score
mean_array_100 = []
episode = 0

timer_to_restart = restart_bound
testing = False

start_time = time.time()

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = myPGAgent(state_size, action_size)		

with tf.Session() as session:
	session.run(tf.global_variables_initializer())

	while True:
		
		action = agent.act(state, session)
		state, reward, done, info = env.step(action)
		score += reward

		if abs(score) < abs(cap_score) and done:
			reward = done_reward
			
		agent.store(state, action, reward, done)
		if done:

			# update the rewards of the current episode
			agent.discount_rewards_last_episode()

			if episode % agent.update_rate == 0:
				agent.train(session)
				agent.clear_memory()

			episode += 1

			score_mean = 0.9*score_mean + 0.1*score
			mean_array.append(score_mean)

			score_mean_100 = 0.99*score_mean_100 + 0.01*score
			mean_array_100.append(score_mean_100)
		
			if abs(score_mean - lowest_score) < 5:
				timer_to_restart -= 1
			else:
				timer_to_restart = restart_bound

			# if the network is stuck for too much, we re-initialize the weights
			if timer_to_restart <= 0:
				session.run(tf.global_variables_initializer())
				timer_to_restart = restart_bound
				print("Re-initialize the weights")

			if not animate():
				break

			state = env.reset()
			
			time_taken = (time.time() - start_time)
			print('Episode: %d - Score: %f - Time Elapsed: %d s' % (episode, score, time_taken))
			score = 0

			if score_mean > target_score:
				testing = True
				break

			if episode >= episode_cap:
				break

	if testing:
		print("Target score reached. Starting testing phase")

		# Test phase
		score = 0
		mean_score = 0
		episode = 0
		state = env.reset()
		while True:

			action = agent.act(state, session )
			state, reward, done, info = env.step(action)
			score += reward

			if done:
				episode += 1		
				state = env.reset()

				mean_score += score
				score = 0

				if episode >= 100:
					break

		print("Testing phase concluded: Final Model Score is %f" % (mean_score/100))

	animate()
	plt.savefig('prova.png')
