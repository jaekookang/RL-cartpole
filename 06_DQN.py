import tensorflow as tf
import numpy as np
import random
import DQN
import gym
from collections import deque
from typing import List
import pdb

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, 'gym-results/', force=True)
input_size = env.observation_space.shape[0]  # 4 (x, x', v, v')
output_size = env.action_space.n  # 2 (left, right)

disc_factor = 0.99
replay_memory = 50000
max_episodes = 50000
batch_size = 64
target_update_freq = 5


def replay_train(mainDQN: DQN.DeepQNet, targetDQN: DQN.DeepQNet, train_batch: list) -> float:
    '''
    Each element in train_batch of replay memory consists of:
    [(state, action, reward, next_state, done), ...]
    '''
    states = np.vstack([x[0] for x in train_batch])  # Nx4
    actions = np.array([x[1] for x in train_batch])  # (N,); 0 or 1 coded
    rewards = np.array([x[2] for x in train_batch])  # (N,); 0 or 1 coded
    next_states = np.vstack([x[3] for x in train_batch])  # Nx4
    done = np.array([x[4] for x in train_batch])  # (N,)

    X = states

    # consider cases when it is not done (not failed)
    Q_target = rewards + disc_factor * \
        np.max(targetDQN.predict(next_states), axis=1) * ~done  # (N,); target

    y = mainDQN.predict(states)  # Nx2; prediction
    y[np.arange(len(X)), actions] = Q_target
    return mainDQN.update(X, y)


def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    '''
    TF operations copying weights from 'src_scope' to 'dest_scope'
    '''
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name
    )
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name
    )

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(
            dest_var.assign(src_var.value()))  # assign src to dest
    return op_holder


def bot_play(mainDQN: DQN.DeepQNet, env: gym.Env) -> None:
    '''
    Test run
    '''
    state = env.reset()
    reward_sum = 0

    while True:
        env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward

        if done:
            print("Total score: {}".format(reward_sum))
            break


def main():
    replay_buffer = deque(maxlen=replay_memory)  # 50000
    last_100_game_reward = deque(maxlen=100)

    with tf.Session() as sess:
        mainDQN = DQN.DeepQNet(sess, input_size, output_size, name="main")
        targetDQN = DQN.DeepQNet(sess, input_size, output_size, name="target")
        sess.run(tf.global_variables_initializer())

        copy_ops = get_copy_var_ops(
            dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(max_episodes):
            e = 1. / (episode / 10 + 1)
            done = False
            step_cnt = 0
            state = env.reset()

            while not done:
                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                next_state, reward, done, _ = env.step(action)

                if done:
                    # if failed, add penalty
                    reward = -1

                # save replay buffer
                replay_buffer.append((state, action, reward, next_state, done))

                if len(replay_buffer) > batch_size:
                    minibatch = random.sample(replay_buffer, batch_size)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)

                if step_cnt % target_update_freq == 0:
                    sess.run(copy_ops)

                state = next_state
                step_cnt += 1

            print("Episode: {} steps: {}".format(episode, step_cnt))

            last_100_game_reward.append(step_cnt)

            if len(last_100_game_reward) == last_100_game_reward.maxlen:
                avg_reward = np.mean(last_100_game_reward)

                if avg_reward > 199:
                    print(
                        f"Game Cleared in {episode} episodes with avg reward {avg_reward}")
                    break


if __name__ == '__main__':
    main()
