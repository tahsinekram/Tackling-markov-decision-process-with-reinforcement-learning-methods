#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym
import numpy as np
import random
from timeit import default_timer as timer
from datetime import timedelta
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
import pickle
gym.envs.register('FrozenLake-v2',entry_point='WindyFrozen:FrozenLakeEnv')


from WindyFrozen import generate_random_map
file = open("wf-8.p","rb")
wf = pickle.load(file)
file.close()


# This function test_policy is used for testing a given policy on an environment for n_epoch number of times
# The function takes in three arguments:
#    env: an environment object, which can be used to interact with the environment the policy is being tested on
#    policy: a policy, represented as a mapping from states to actions
#    n_epoch: the number of times to run the test (default is 1000)
# It records the total reward and number of episodes for each run and stores them in the lists 'rewards' and 'episode_counts' respectively.
# After all the test runs are complete, the function calculates the mean reward and mean number of episodes per run, and returns them along with the 'rewards' and 'episode_counts' lists.

def test_policy(env, policy, n_epoch=1000):
    rewards = []
    episode_counts = []
    for i in range(n_epoch):
        current_state = env.reset()[0]
        ep = 0
        done = False
        episode_reward = 0
        while not done and ep < 10000:
            ep += 1
            act = int(policy[current_state])
            new_state, reward, done, _, info= env.step(act)
            episode_reward += reward
            current_state = new_state
        rewards.append(episode_reward)
        episode_counts.append(ep)
    
    # all done
    mean_reward = sum(rewards)/len(rewards)
    mean_eps = sum(episode_counts)/len(episode_counts)
    return mean_reward, mean_eps, rewards, episode_counts 


# This function q_learning implements Q-Learning algorithm to solve an environment
# The function takes in five optional arguments:
#    env: an environment object, which can be used to interact with the environment 
#    discount: the discount factor for future rewards, default is 0.9
#    total_episodes: the number of episodes to run the algorithm, default is 2
#    alpha: the learning rate of the algorithm, default is 0.1
#    decay_rate: the rate at which the exploration parameter epsilon decays, if not provided calculated as 1/total_episodes
#    min_epsilon: the minimum value of the exploration parameter epsilon, default is 0.01
# 
# The function initializes the Q-table with zeros and sets the initial exploration parameter epsilon to 1.0.
# For each episode, it starts by resetting the environment, and chooses actions based on the Q-table or with a probability epsilon,
# it chooses a random action. It updates the Q-table using Q-Learning update rule. 
# if the episode is completed, it breaks the loop. 
# It also decays the exploration parameter epsilon at each episode using the decay rate
# After all episodes are completed, the function prints the total time taken to solve the environment and return 
# the action with maximum Q value for each state, number of episodes, time spent, Q-table and rewards obtained in each episode.

def q_learning(env, discount=0.9, total_episodes=2, alpha=0.1, decay_rate=None,
               min_epsilon=0.01):
    
    start = timer()
    
    number_of_states = env.observation_space.n
    number_of_actions = env.action_space.n
    
    qtable = np.zeros((number_of_states, number_of_actions))
    learning_rate = alpha
    gamma = discount

    # exploration parameter
    epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = 0.01
    
    if not decay_rate:
        decay_rate = 1./total_episodes
    
    rewards = []
    for episode in range(int(total_episodes)):

        state = env.reset()[0]
        step = 0
        done = False
        total_reward = 0
        it = 0
        while True:
            it += 1

            exp_exp_tradeoff = random.uniform(0,1)


            if exp_exp_tradeoff > epsilon:
                b = qtable[state, :]
                action = np.random.choice(np.where(b == b.max())[0])

            else:
                action = env.action_space.sample()

            new_state, reward, done, _, info = env.step(action)
            total_reward += reward
            if not done:
                qtable[state, action] = qtable[state, action] + learning_rate*(reward + gamma*np.max(qtable[new_state, :]) - qtable[state, action])
            else:
                qtable[state, action] = qtable[state,action] + learning_rate*(reward - qtable[state,action])

            state = new_state

            if done:
                break
                 
        rewards.append(total_reward)
        epsilon = max(max_epsilon -  decay_rate * episode, min_epsilon) 
    
    end = timer()
    time_spent = timedelta(seconds=end-start)
    print("Solved in: {} episodes and {} seconds".format(total_episodes, time_spent))
    return np.argmax(qtable, axis=1), total_episodes, time_spent, qtable, rewards


# This function train_and_test_q_learning is used to train and test Q-Learning algorithm for different hyperparameters
# The function takes in six arguments:
#    env: an environment object, which can be used to interact with the environment 
#    discount: a list of discount factors for future rewards, default is [0.9]
#    total_episodes: a list of number of episodes to run the algorithm, default is [2]
#    alphas: a list of learning rates, default is [0.1]
#    decay_rates: a list of decay rates of the exploration parameter epsilon, default is [0.01]
#    mute: if True, the function will not print any output, default is False
# 
# The function creates a nested dictionary to store the results of different combinations of hyperparameters
# For each combination of hyperparameters, it runs the q_learning function and saves the results in the dictionary.
# The results include the policy, q-table, rewards, mean reward and mean episode per iteration, iteration and time spent to solve the environment.
# After all combinations are completed, the function returns the dictionary containing all the results.


def train_and_test_q_learning(env, discount=[0.9], total_episodes=[2], alphas=[0.1], decay_rates=[0.01], mute=False):
    
    min_epsilon = 0.01
    
    q_dict = {}
    for dis in discount:
        q_dict[dis] = {}
        for eps in total_episodes:
            q_dict[dis][eps] = {}
            for alpha in alphas:
                q_dict[dis][eps][alpha] = {}
                for dr in decay_rates:
                    q_dict[dis][eps][alpha][dr] = {}
                    
                    # run q_learning
                    q_policy, q_solve_iter, q_solve_time, q_table, rewards = q_learning(env, dis, eps, alpha, dr, min_epsilon)
                    q_mrews, q_meps, _, __ = test_policy(env, q_policy)
                    q_dict[dis][eps][alpha][dr]["mean_reward"] = q_mrews
                    q_dict[dis][eps][alpha][dr]["mean_eps"] = q_meps
                    q_dict[dis][eps][alpha][dr]["q-table"] = q_table
                    q_dict[dis][eps][alpha][dr]["rewards"] = rewards 
                    q_dict[dis][eps][alpha][dr]["iteration"] = q_solve_iter
                    q_dict[dis][eps][alpha][dr]["time_spent"] = q_solve_time
                    q_dict[dis][eps][alpha][dr]["policy"] = q_policy
                    if not mute:
                        print("gamma: {} total_eps: {} lr: {}, dr: {}".format(dis, eps, alpha, dr))
                        print("Iteration: {} time: {}".format(q_solve_iter, q_solve_time))
                        print("Mean reward: {} - mean eps: {}".format(q_mrews, q_meps))
    return q_dict

# This function map_discretize takes in a 2D map represented as a list of lists and converts it into a discretized version of the map.
# The function takes one argument:
#     the_map: the original map represented as a list of lists
# The function loops through the each element of the_map, checking the value of the element and assigns a numerical value to it.
# "S" is converted to 0, "F" to 0, "H" to -1, "G" to 1, "W" to 2 and assigns these values to the corresponding location in the new map.
# The function returns the discretized map.

def map_discretize(the_map):
    size = len(the_map)
    dis_map = np.zeros((size,size))
    for i, row in enumerate(the_map):
        for j, loc in enumerate(row):
            if loc == "S":
                dis_map[i, j] = 0
            elif loc == "F":
                dis_map[i, j] = 0
            elif loc == "H":
                dis_map[i, j] = -1
            elif loc == "G":
                dis_map[i, j] = 1
            elif loc == "W":
                dis_map[i, j] = 2
    return dis_map

'''
This function takes in a policy which is a list of 
integers representing actions at each state and converts it into a 2D numpy 
array where each element of the array is an action corresponding to the state 
at the same index.
'''

def policy_numpy(policy):
    size = int(np.sqrt(len(policy)))
    pol = np.asarray(policy)
    pol = pol.reshape((size, size))
    return pol


'''
This function takes in the map size and a policy and display 
the policy over the map in the form of arrows. It uses the map_discretize function 
to discretize the map, then uses the policy_numpy function to convert the policy 
into a 2D numpy array. The function then loops through the array and assigns an arrow 
corresponding to the action at that state. The function then uses the matplotlib 
library to display the map with the policy arrows.
'''

def see_policy(map_size, policy):
    map_name = str(map_size)+"x"+str(map_size)
    data = map_discretize(wf)
    np_pol = policy_numpy(policy)
    plt.imshow(data, interpolation="nearest")

    for i in range(np_pol[0].size):
        for j in range(np_pol[0].size):
            arrow = '\u2190'
            if np_pol[i, j] == 1:
                arrow = '\u2193'
            elif np_pol[i, j] == 2:
                arrow = '\u2192'
            elif np_pol[i, j] == 3:
                arrow = '\u2191'
            text = plt.text(j, i, arrow,
                           ha="center", va="center", color="w")
    plt.show()

'''
This function takes in a dictionary and converts it into a pandas dataframe with columns "Discount Rate", "Training Episodes", "Learning Rate", "Decay Rate", "Reward", "Time Spent". The function loops through the dictionary and appends the values of each key to the dataframe.
'''
def dict_to_df(the_dict):
    the_df = pd.DataFrame(columns=["Discount Rate", "Training Episodes", "Learning Rate", 
                                   "Decay Rate", "Reward", "Time Spent"])
    for dis in the_dict:
        for eps in the_dict[dis]:
            for lr in the_dict[dis][eps]:
                for dr in the_dict[dis][eps][lr]:
                    rew = the_dict[dis][eps][lr][dr]["mean_reward"]
                    time_spent = the_dict[dis][eps][lr][dr]["time_spent"].total_seconds()
                    dic = {"Discount Rate": dis, "Training Episodes": eps, "Learning Rate":lr, 
                           "Decay Rate":dr, "Reward": rew, "Time Spent": time_spent}
                    the_df = the_df.append(dic, ignore_index=True)
    return the_df


'''
This function takes in a list x and a window size N and calculates the running mean of the list. It uses numpy library to calculate the cumulative sum of the list and returns the running mean by dividing the cumulative sum by the window size.
'''

import numpy as np
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

'''
This of code is making use of the gym library and creating an environment with the "FrozenLake-v2" environment and a custom map. It then runs the train_and_test_q_learning function for different values of discount rate, total episodes, learning rate and decay rate. And saves the resulting dictionary in a pickle file.
'''

env = gym.make("FrozenLake-v2",desc=wf)
episodes = [1e5]
decays = [1e-3,1e-5]
print ("STARTING Q")
q_dict = train_and_test_q_learning(env, discount=[0.5,0.99], total_episodes=episodes,
                          alphas=[0.01,0.5], decay_rates=decays)
print ("COMPLETED Q")

pickle.dump(q_dict, open("windyfrozen8-q.p", "wb"))


