import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

from keras import backend as K
from keras.layers import *
from keras.models import Model, Sequential, clone_model
from keras.optimizers import SGD, Adam
from collections import deque

#Define the Choosen game from the list below
choosen = 2
train_rate = 4
cap_episodes = 3000

Double = True

# 1 Mountain Car
# 2 CartPole
# 3 Acrobot
# 4 Mountain Car Easy

def huber_loss(a, b):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    use_linear_term = K.cast(use_linear_term, 'float32')

    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

class myDQNAgent:
	
	def __init__(self, state_size, action_size):
		
		self.state_size = state_size
		self.action_size = action_size
		
		self.memory = deque( maxlen=10000)
		
		self.exploration_rate  = 1
		self.exploration_decay = 0.995
		self.exploration_min   = 0.02

		self.step_counter = 0
		
		self.learning_rate = 0.0008
		self.gamma = 0.99 			# discount rate
		self.update_frozen = 50		# N° of episodes between saves of model
		# self.act_freq = 3			# Action selection frequence
		self.batch_size = 32		# Batch size for the training
		
		self.model = self.create_model([state_size], action_size)
		
		self.frozen_model = clone_model(self.model)
		self.frozen_model.set_weights(self.model.get_weights())

	def create_model(self, input_dim, num_actions):
		
		image_input = Input(shape=input_dim)
		x = image_input
		
		x = Dense(64, activation='relu')(x)
		x = Dense(64, activation='relu')(x)
		out = Dense(num_actions, activation='linear')(x)
		
		model = Model(inputs=image_input, outputs=out)
		model.compile(loss=huber_loss, optimizer=Adam(lr = self.learning_rate))
		model.summary()
		
		return model
		
		
	def store(self, state, action, reward, next_state):
		
		self.memory.append([state, action, reward, next_state])

	def act(self, state):
		
		if np.random.uniform(0,1) < self.exploration_rate:
			action = np.random.randint(self.action_size)
		else:
			state = np.reshape(state, [1, self.state_size])
			q_values = self.model.predict(state)[0]			
			action = np.argmax(q_values)
		
		return action

	def clear_memory(self):

		self.memory.clear()
		
	def decay_explore (self):
		if self.exploration_rate > self.exploration_min:
			self.exploration_rate *= self.exploration_decay
		else:
			self.exploration_rate = 0
		
	def sync_models(self):
		print("Checkpoint!")
		self.frozen_model.set_weights(self.model.get_weights())
		
	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)	
		
	def train (self, double = False):

		if len(self.memory) < self.batch_size:
			return
		
		index_list = range(len(self.memory))
		minibatch = random.sample(index_list, self.batch_size)
	
		x_train = []
		y_train = []
		
		for x, action, reward, next_x in np.array(self.memory)[minibatch]:
			
			data_train = np.reshape(x, [1, state_size])	
			target = self.model.predict(data_train)[0]
				
			data_next = np.reshape(next_x, [1, state_size])
			if double:
				target_action = np.argmax(self.model.predict(data_next))
				max_next_target = self.frozen_model.predict(data_next)[0][target_action]
			else:
				max_next_target = np.max(self.frozen_model.predict(data_next)[0])
				
			target[action] = reward + self.gamma*max_next_target
				
			x_train.append(data_train)
			y_train.append(target)
			
		x_train = np.array(x_train)
		x_train = np.reshape(x_train, [self.batch_size, self.state_size])
		y_train = np.array(y_train)
		y_train = np.reshape(y_train, [self.batch_size, self.action_size])
		
		self.model.train_on_batch(x_train, y_train)



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

x = env.reset()
score = 0
score_mean = lowest_score
mean_array = []

score_mean_100 = lowest_score
mean_array_100 = []
episode = 0

start_time = time.time()

state_size = env.observation_space.shape[0]

action_size = env.action_space.n
agent = myDQNAgent(state_size, action_size)		

#agent.load('model.h5')
i = 0
testing = False

while True:

	action = agent.act(x)
	next_x, reward, done, info = env.step(action)

	score += reward

	if done and score != cap_score:
		reward = done_reward
		
	agent.store(x, action, reward, next_x)
	x = next_x
	
	i += 1
	if i % train_rate == 0:
		agent.train(Double)
	
	if done:
		episode += 1		
		x = env.reset()
		
		time_taken = (time.time() - start_time)
		print('Episode: %d - Score: %f - Time Elapsed: %d s - Exploration Rate: %f' % (episode, score, time_taken, agent.exploration_rate))
		
		score_mean = 0.9*score_mean + 0.1*score
		mean_array.append(score_mean)

		score_mean_100 = 0.99*score_mean_100 + 0.01*score
		mean_array_100.append(score_mean_100)
		score = 0
		
		if not animate():
			break		
		agent.decay_explore()

		if episode % 20 == 0:
			agent.clear_memory()
		
		if episode > 1 and episode % 100 == 0:
			agent.save('model.h5')

		if episode % agent.update_frozen == 0:
			agent.sync_models()

		# Stop condition
		if score_mean > target_score:
			testing = True
			break

		if episode >= cap_episodes:
			break

if testing:
	print("Target score reached. Starting testing phase")

	# Test phase
	score = 0
	mean_score = 0
	episode = 0
	while True:

		action = agent.act(x)
		x, reward, done, info = env.step(action)
		score += reward

		if done:
			episode += 1		
			x = env.reset()
			mean_score += score
			score = 0

			if episode >= 100:
				break

	print("Testing phase concluded: Final Model Score is %f" % (mean_score/100))

animate()
plt.savefig('prova.png')