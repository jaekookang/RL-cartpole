import gym
import numpy as np
import tensorflow as tf

env = gym.make('CartPole-v0')

max_episodes = 100
max_timestep = 200

discount = 0.9
explore_rate = 0.01
learning_rate = 0.1

num_states = env.observation_space.shape[0]  # 4 (x, x', theta, theta')
num_actions = env.action_space.n  # 2 (left, right)

X = tf.placeholder(tf.float32, [1, num_states], name='input')  # (1,4)
W = tf.get_variable('W', shape=[num_states, num_actions],
                    initializer=tf.contrib.layers.xavier_initializer())  # (4,2)
Qpred = tf.matmul(X, W)  # (1,2)
Y = tf.placeholder(shape=[1, num_actions], dtype=tf.float32)

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


def select_action(Q, explore_rate):
    '''Select action given state'''
    if np.random.rand(1) < explore_rate:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q)
    return action


def update_Q(Q, action):
    '''Update Q'''
    if done:
        # if failed (fell)
        Q[0, action] = -100
    else:
        # if succeeded (stayed still)
        x_next = np.reshape(next_state, [1, num_states])
        Q_next = sess.run(Qpred, {X: x_next})
        Q[0, action] = reward + discount * np.max(Q_next)
    return Q


def update_explore_rate(t):
    '''Decay explore_rate over time'''
    return max(explore_rate, min(1, 1.0 - np.log10((t + 1) / 25)))


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for episode in range(max_episodes):
        r = 0  # reward
        ls = 0  # loss
        prev_state = env.reset()

        for t in range(max_timestep):
            env.render()
            x = np.reshape(prev_state, [1, num_states])
            Q = sess.run(Qpred, {X: x})  # prediction

            # selet action
            action = select_action(Q, explore_rate)
            # run
            next_state, reward, done, _ = env.step(action)
            # update Q
            Q = update_Q(Q, action)  # target

            # train Q-Net
            l, _ = sess.run([loss, train], {X: x, Y: Q})
            ls += l

            prev_state = next_state
            explore_rate = update_explore_rate(t)
            r += reward

            if done:
                # if failed
                print('Episode:{}/{} finished at timestep:{} reward:{} loss:{}'.format(
                    episode, max_episodes, t, r, ls / (t + 1)))
                break
