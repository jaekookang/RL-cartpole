'''
2018-02-10

State: frame
Action: paddle up/down (binary)
Reward:
  +1 if ball bounces off,
  -2 if ball passes AI
   0 otherwise (e.g. waiting)
'''
import numpy as np
import pickle
import gym


# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99  # for rmsprop
resume = False

# init model
D = 80 * 80  # input dimension (80*80=6400 grid)
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    # xavier initialization
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # 200x6400
    model['W2'] = np.random.randn(H) / np.sqrt(H)  # (200,)
grad_buffer = {k: np.zeros_like(v)
               for k, v in model.items()}  # store gradient
rmsprop_cache = {k: np.zeros_like(v)
                 for k, v in model.items()}  # store rms values


def sigmoid(x):
    '''activation function'''
    return 1.0 / (1.0 + np.exp(-x))


def prepro(I):
    # receive game frame (I) and convert
    I = I[35:195]  # crop image
    I = I[::2, ::2, 0]  # down sample by a factor of 2
    I[I == 144] = 0  # erase background
    I[I == 109] = 0  # erase background
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()  # flattens to 1-dim


def discount_rewards(r):
    # Q update (From future to past)
    discounted_r = np.zeros_like(r)
    running_add = 0  # store sum of rewards
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = r[t] + running_add * gamma
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    '''Feedforward function'''
    h = np.dot(model['W1'], x)  # (200x6400)x(6400xN)=(200xN)
    h[h < 0] = 0  # 200xN
    logp = np.dot(model['W2'], h)  # (200,)x(200xN)=(N,)
    p = sigmoid(logp)  # (N,)
    return p, h


def policy_backward(eph, epdlogp):
    '''
    Backprop function
    eph: intermediate hidden state
    epdlogp: modulates gradient with advantage
    '''
    # compute: dC/dW2
    dW2 = np.dot(eph.T, epdlogp).ravel()
    # compute: derivative hidden
    dh = np.outer(epdlogp, model['W2'])
    # apply activation
    dh[eph <= 0] = 0
    # compute: dC/dW1
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


# Environment
env = gym.make('Pong-v0')
observation = env.reset()  # initial states
prev_x = None
# obs, hidden state, gradient, reward
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    cur_x = prepro(observation)
    # get image difference
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)  # 80*80=6400
    prev_x = cur_x  # (6400,)

    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3
    # Here action is coded as 2 and 3

    xs.append(x)  # append current image difference
    hs.append(h)  # append first hidden activation

    y = 1 if action == 2 else 0  # make action coding binary
    dlogps.append(y - aprob)

    env.render()
    observation, reward, done, _ = env.step(action)
    reward_sum += reward

    drs.append(reward)  # record reward after env.step(action)

    if done:
        episode_number += 1

        epx = np.vstack(xs)  # observation (states)
        eph = np.vstack(hs)  # hidden activation
        epdlogp = np.vstack(dlogps)  # gradient
        epr = np.vstack(drs)  # reward
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        discounted_epr = discount_rewards(epr)
        # normalize rewards (z-scoring)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr
        grad = policy_backward(eph, epdlogp)
        for k in model:
            grad_buffer[k] += grad[k]  # accumulate grad over batch

        # rmsprop parameter update every batch_size
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + \
                    (1 - decay_rate) * g**2
                model[k] += learning_rate * g / \
                    (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)

        running_reward = reward_sum if running_reward is None \
            else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' %
              (reward_sum, running_reward))
        if episode_number % 100 == 0:
            pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset()
        prev_x = None

    if reward != 0:
        print('ep {}: game finished, reward: {}{}'.format(
            episode_number, reward, ('' if reward == -1 else ' Bounced!!!!!')))
