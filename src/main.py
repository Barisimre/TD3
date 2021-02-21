import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG
from generative_replay import GenerativeReplay
from datetime import datetime


if __name__ == "__main__":
	
	# Hyper parameters

	# General
	USE_GENERATIVE = True
	NO_REPLAY = False
	RECORD_TRAINING_TIMES = False
	ENV = "InvertedPendulum-v2"
	START_TIMESTEPS = 15e3
	END = START_TIMESTEPS + 50e3
	EVAL_FREQ = 5e3
	MAX_TIMESTEPS = 2e5
	SEED = 13
	# FILE_NAME = ENV + "_" + list(str(datetime.now()).split())[-1]
	FILE_NAME = "a"
	
	F_TIME = 5000
	VAE_F = 0
	TD3_F = 0
	MILESTONES = [8, 15, 20, 30, 40, 50, 60, 70, 80, 90]

	# TD3 parameters
	EXPL_NOISE = 0.1
	BATCH_SIZE = 256
	DISCOUNT = 0.99
	TAU = 0.005
	POLICY_NOISE = 0.2
	NOISE_CLIP = 0.5
	POLICY_FREQ = 2

	evaluations = []
	td3times = []
	vaetimes = []

	running_av = 0
	

	print(f"Start new process with {ENV} and file name {FILE_NAME}")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	env = gym.make(ENV)

	# Set seeds
	env.seed(SEED)
	torch.manual_seed(SEED)
	np.random.seed(SEED)
	
	# Some env dimentions
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	# Build TD3
	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": DISCOUNT,
		"tau": TAU,
		"policy_noise": POLICY_NOISE * max_action,
		"noise_clip": NOISE_CLIP * max_action,
		"policy_freq": POLICY_FREQ
	}

	policy = TD3.TD3(**kwargs)

	# Make the replay component
	replay_component = None
	if USE_GENERATIVE:
		replay_component = GenerativeReplay()
	elif NO_REPLAY:
		replay_component = utils.ReplayBuffer(state_dim, action_dim, BATCH_SIZE)
	else:
		replay_component = utils.ReplayBuffer(state_dim, action_dim)

	training_moments = []
	

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0


	for t in range(int(MAX_TIMESTEPS)):

		if TD3_F > 0:
			TD3_F -= 1
	
		episode_timesteps += 1

		if t >= END:
			raise ValueError

		# Select action randomly or according to policy based on the start timesteps
		if t < START_TIMESTEPS:
			action = env.action_space.sample()
			episode_num = 0
		else:
			replay_component.training = True
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * EXPL_NOISE, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay component

		VAE_training = replay_component.add(state, action, next_state, reward, done_bool)
		if VAE_training:
			training_moments.append(episode_num)

		state = next_state
		episode_reward += reward


		# Train agent after collecting sufficient data
		if t >= START_TIMESTEPS and TD3_F == 0:
			policy.train(replay_component, BATCH_SIZE)

		if done: 
			running_av = 0.4*running_av + 0.6*episode_reward

			if t >= START_TIMESTEPS:
				if running_av > MILESTONES[0] and TD3_F == 0:
					MILESTONES = MILESTONES[1:]
					TD3_F = F_TIME
					td3times.append(episode_num)
					np.save(f"./results/incoming/{FILE_NAME}_td3", td3times)

					VAE_F = 0


				if running_av < 4:
					MILESTONES = [8, 15, 20, 30, 40, 50, 60, 70, 80, 90]




			print(f"Episode {episode_num}, reward is {episode_reward}, running average {running_av}, TD3 {TD3_F}, VAE {VAE_F}, {MILESTONES}")
			if t >= START_TIMESTEPS:
				evaluations.append(episode_reward)
				np.save(f"./results/incoming/{FILE_NAME}", evaluations)
				if RECORD_TRAINING_TIMES:
					np.save(f"./results/incoming/{FILE_NAME}_times", training_moments)

			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 