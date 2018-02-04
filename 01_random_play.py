import gym

env = gym.make('CartPole-v0')

max_episodes = 10

for episode in range(max_episodes):
    step_cnt = 0
    r = 0
    state = env.reset()
    done = False

    while not done:
        env.render()
        step_cnt += 1
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        r += reward

        if done:
            print('Episode {} finised at time:{} with reward:{}'.format(
                episode, step_cnt, r))
            break
