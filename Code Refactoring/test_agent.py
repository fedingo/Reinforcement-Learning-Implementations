import gym
import gym.spaces
import time
# Used to hide gym core warning
gym.logger.set_level(40)

import sys, inspect
import importlib
import scoreViewer as sv

# Used to hide the Tensorflow warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("\n======= RL Agent Testing with the OpenAI gym =======\n")

if len(sys.argv) < 3:
	print ("Usage: <agent_class> <environment>")
	print ("Optional: <target_score>")
	sys.exit(0)

agent_class = sys.argv[1]
environment = sys.argv[2]
target_score = 0

if len(sys.argv) == 4:
	target_score = int(sys.argv[3])

# Importing the given class
module = importlib.import_module(agent_class)
class_tuples = inspect.getmembers(module, inspect.isclass)
agents_list = []
for el in class_tuples:
	if 'Agent' in el[0]:
		agents_list += [el]

agent_tuple = agents_list[0]
agent_class = agent_tuple[1]
print('Loading class: ' + agent_tuple[0])

# Preparing image target
image_path  = "images/" + agent_tuple[0] + "_" + environment + ".png"

# Created user defined environment
print('Load gym environment: ' + environment + "\n")
env = gym.make(environment)

# Creating the Gym environment
state = env.reset()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

viewer = sv.scoreViewer(target_score = target_score)
agent = agent_class(state_size, action_size, [32, 32])

step = 0
score = 0
testing = False
start_time = time.time()

while True:

	action = agent.act(state)
	next_state, reward, done, info = env.step(action)
	score += reward

	obj = { \
		"state"  : state,
		"action" : action,
		"reward" : reward,
		"done"   : done,
		"next_state" : next_state
	}

	agent.store(obj)
	state = next_state

	if agent.hasToTrain(step, done, viewer.getEpisodeNumber()):
		agent.train()
	step += 1

	if done:
		if not viewer.addScore(score):
			break

		time_taken = (time.time() - start_time)
		print('Episode: %d - Score: %f - Time Elapsed: %d s' %\
				 (viewer.getEpisodeNumber(), score, time_taken), end="\r")

		if viewer.getEpisodeNumber() % 25 == 0:
			viewer.printMeans()

		step = 0; score = 0; state = env.reset()

		if viewer.isFinished():
			testing = True
			break

		if viewer.getEpisodeNumber() >= 1000:
			break

print("\nClosing...")

if testing:
	print("Target score reached. Starting testing phase")

	# Test phase
	total_score = 0
	episode = 0
	state = env.reset()

	while True:
		action = agent.act(state)
		state, reward, done, info = env.step(action)
		total_score += reward

		if done:
			episode += 1		
			state = env.reset()

			if episode >= 100:
				break

	print("Testing phase concluded: Final Model Score is %f" % (total_score/100))

viewer.saveToFile(image_path)