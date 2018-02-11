import numpy as np
import tensorflow as tf
import gym
import pdb

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, directory="07-results/", force=True)

hidden_neurons = 24
learn_rate = 1e-2
gamma = .99

input_size = env.observation_space.shape[0]  # state=4
output_size = 1  # output of sigmoid

# X: Nx4
# Y: Nx1
# advantages: (N,)
X = tf.placeholder(tf.float32, [None, input_size], name='input_x')
Y = tf.placeholder(tf.float32, [None, output_size], name='input_y')
advantages = tf.placeholder(tf.float32, name='reward_signal')


def setup_network(x, y, advantages, hidden=[24, 24], learning_rate=1e-2, name='fc'):
    '''Define fully connected layer(s)'''
    with tf.variable_scope(name):
        net = x
        for nhid in hidden:
            net = tf.layers.dense(net, nhid,
                                  activation=tf.nn.relu,
                                  use_bias=False,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
        action_pred = tf.layers.dense(net, output_size,
                                      activation=tf.nn.sigmoid,
                                      use_bias=False,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        log_likes = -y * tf.log(action_pred) - (1 - y) * \
            tf.log(1 - action_pred)
        loss = tf.reduce_sum(log_likes * advantages)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learn_rate).minimize(loss)
    return loss, optimizer, action_pred


def discount_rewards(rlist, gamma=0.99):
    '''Compute discounted rewards given episode'''
    discounted_r = np.zeros_like(rlist, dtype=np.float32)
    curr_r = 0
    for t in reversed(range(len(rlist))):
        curr_r = curr_r * gamma + rlist[t]
        discounted_r[t] = curr_r
    return discounted_r


def test_rewards():
    '''Test discount_rewards()'''
    input = [1, 1, 1]
    output = discount_rewards(input)
    expect = [1 + 0.99 + 0.99**2, 1 + 0.99, 1]
    return np.testing.assert_almost_equal(output, expect)


loss, optimizer, action_pred = setup_network(X, Y, advantages)

# If average reward is more than 195 over 100 games,
# training will stop.
game_hist = []
sess = tf.Session()
sess.run(tf.global_variables_initializer())
max_episodes = 500

for episode in range(max_episodes):
    xs = np.empty(shape=[0, input_size])  # (0,4)
    ys = np.empty(shape=[0, 1])  # (0,1)
    rewards = np.empty(shape=[0, 1])  # (0,1)

    step_cnt = 0
    step_hist = []
    reward_sum = 0
    observation = env.reset()  # (4,); states

    while True:
        # Record all movements until the game ended
        x = np.reshape(observation, [1, input_size])  # reshape
        action_prob = sess.run(action_pred, {X: x})
        action = 0 if action_prob < np.random.uniform() else 1
        xs = np.vstack([xs, x])  # append states
        ys = np.vstack([ys, action])  # append y (y is stochastic)

        observation, reward, done, _ = env.step(action)
        rewards = np.vstack([rewards, reward])
        reward_sum += reward
        step_cnt += 1

        if done:
            discounted_r = discount_rewards(rewards)
            discounted_r = (  # normalize
                discounted_r - discounted_r.mean()) / (discounted_r.std() + 1e-7)
            l, _ = sess.run([loss, optimizer],
                            {X: xs, Y: ys, advantages: discounted_r})

            game_hist.append(reward_sum)
            step_hist.append(step_cnt)
            if len(game_hist) > 100:
                env.render()
                game_hist = game_hist[1:]
            break

    print('Episode:{:>5d} Reward:{:>4} Loss:{:>10.5f}'.format(
        episode, reward_sum, l))
    if np.mean(game_hist) >= 195:
        print('Game finished with episode: {} and avg reward: {}'.format(
            episode, np.mean(game_hist)))
        break

# See our trained bot in action
observation = env.reset()
reward_sum = 0

while True:
    env.render()
    x = np.reshape(observation, [1, input_size])
    action_prob = sess.run(action_pred, feed_dict={X: x})
    action = 0 if action_prob < 0.5 else 1  # No randomness
    observation, reward, done, _ = env.step(action)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break

sess.close()
