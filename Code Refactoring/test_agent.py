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

def test_agent(tested_agent, env):

	total_score = 0
	episode = 0
	test_episodes = 50
	state = env.reset()

	while episode < test_episodes:
		action = tested_agent.act(state, testing = True)
		state, reward, done, info = env.step(action)
		total_score += reward

		if done:
			episode += 1
			state = env.reset()

	print("\nModel Score is %.2f" % (total_score/test_episodes))

	return total_score/test_episodes


print("\n======= RL Agent Testing with the OpenAI gym =======\n")

if len(sys.argv) < 3:
	print ("Usage: <agent_class> <environment>")
	print ("Optional: <target_episodes>")
	sys.exit(0)

agent_class = sys.argv[1]
environment = sys.argv[2]
target_score = 0
target_episodes = 1000

if len(sys.argv) == 4:
	target_episodes = int(sys.argv[3])

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

step = 0; score = 0; episodes = 0
start_time = time.time()

agent = agent_class(state_size, action_size)
tested_performance = test_agent(agent, env)
viewer = sv.scoreViewer(target_score = target_score)

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

	if agent.hasToTrain(step, done, episodes):
		agent.train()
	step += 1

	if done:
		episodes += 1
		if not viewer.addScore(score):
			break

		time_taken = (time.time() - start_time)
		print('Episode: %d - Score: %f - Time Elapsed: %d s' %\
				 (viewer.getEpisodeNumber(), score, time_taken), end="\r")

		if episodes % 25 == 0:
			viewer.printMeans()

		score = 0; state = env.reset()

		if viewer.isFinished():
			break

		if episodes >= target_episodes:
			break


test_agent(agent, env)

print("Closing...")
viewer.saveToFile(image_path)