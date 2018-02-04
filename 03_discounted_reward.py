'''
Cartpole differs from FrozenLake because the state is not discrete.
This example discretizes states using buckets

* added discount factor

edited from
https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
'''
import gym
import numpy as np
import pdb

env = gym.make('CartPole-v0')

max_episodes = 100
max_timestep = 200

discount = 0.9

num_buckets = (1, 1, 6, 3)  # (x, x', theta, theta')
num_actions = env.action_space.n  # (left, right)
state_bounds = list(
    zip(env.observation_space.low,
        env.observation_space.high))  # [(l,u), (l,u), (l,u), (l,u)]
state_bounds[1] = [-0.5, 0.5]  # bound criteria
state_bounds[3] = [-np.radians(50), np.radians(50)]  # bound criteria

q_table = np.zeros(num_buckets + (num_actions,))  # (1,1,6,3,2)


def state_to_bucket(state):
    '''Convert continous states to discrete buckets'''
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= state_bounds[i][0]:
            bucket_index = 0
        elif state[i] >= state_bounds[i][1]:
            bucket_index = num_buckets[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = state_bounds[i][1] - state_bounds[i][0]
            offset = (num_buckets[i] - 1) * state_bounds[i][0] / bound_width
            scaling = (num_buckets[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


def select_action(state):
    '''Select action given state'''
    action = np.argmax(q_table[state])
    return action


for episode in range(max_episodes):
    obs = env.reset()  # (4,)
    prev_state = state_to_bucket(obs)

    for t in range(max_timestep):
        env.render()

        # selet action
        action = select_action(prev_state)
        # run
        obs, reward, done, _ = env.step(action)
        # observe state
        next_state = state_to_bucket(obs)
        # update Q
        q_table[prev_state + (action,)] += reward + \
            discount * np.amax(q_table[next_state])
        prev_state = next_state

        if done:
            # if failed
            print('Episode:{}/{} finished at timestep:{}'.format(
                episode, max_episodes, t))
            break
