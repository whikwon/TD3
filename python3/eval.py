import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
	avg_reward = 0.
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		while not done:
			action = policy.select_action(np.array(obs))
			obs, reward, done, _ = env.step(action)
			env.render()
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default="HalfCheetah-v2")			# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--weight_filename", default="TD3_HalfCheetah-v2_0", type=str)
	parser.add_argument("--weight_dir", default="./pytorch_models", type=str)
	args = parser.parse_args()

	weight_filename = args.weight_filename
	weight_dir = args.weight_dir
	env = gym.make(args.env_name)

	# Set seeds
	env.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	# Initialize policy
	policy = TD3.TD3(state_dim, action_dim, max_action)

    # Load weight
	policy.load(weight_filename, weight_dir)

	# Evaluate untrained policy
	evaluations = [evaluate_policy(policy)]
