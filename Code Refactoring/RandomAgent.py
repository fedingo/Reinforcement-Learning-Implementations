import numpy as np
from abstractAgent import *

class RandomAgent(abstractAgent):

	def __init__(self, state_size, action_size, architecture):
		super().__init__(state_size, action_size, architecture)
		print("Creating a Random Agent")

	def build(self):
		print("No model to build!")

	# function to define when to train the network
	def hasToTrain(self, step, done, episode):
		return False

	# this function is not needed since we never have hasToTrain == True 
	def train(self):
		raise NotImplementedError("Please implement the training method")

	def act(self, state):
		return np.random.randint(self.action_size)
