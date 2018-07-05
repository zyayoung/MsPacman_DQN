import gym,cv2
import numpy as np
from RL_noise import DeepQNetwork
import matplotlib.pyplot as plt


def train():
    step = 0
    avg_score = 70
    history = []
    plt.ion()
    plt.figure()
    for episode in range(50000):
        observation = env.reset()
        # plt.imshow(cv2.resize(observation, (80, 105))[17:97,:,:])
        # plt.show()
        observation = observation.reshape(-1) / 255.0
        # plt.imshow(observation)
        # plt.show()
        real_obs = np.zeros((210*160*3, 4))
        real_obs[:, 0] = observation
        # print(observation)
        score = 0
        while True:

            # print(real_obs.shape)
            action = RL.choose_action(real_obs.reshape(-1))
            reward = 0
            for _ in range(2):
                observation, reward_, done, info = env.step(action)
                reward+=reward_

            observation = observation.reshape(-1) / 255.0
            score+=reward

            # if done or info['ale.lives'] < 3:
            #     reward = -1

            old_obs = real_obs.copy()
            real_obs[:, 3] = real_obs[:, 2]
            real_obs[:, 2] = real_obs[:, 1]
            real_obs[:, 1] = real_obs[:, 0]
            real_obs[:, 0] = observation
            RL.store_transition(old_obs.reshape(-1), action, reward/10, real_obs.reshape(-1))

            if step > 2000 and step % 16 == 0:
                RL.learn()
                env.render()

            step += 1
            if done or info['ale.lives'] < 3:
                avg_score = 0.01 * score + 0.99 * avg_score
                print(episode, int(score), avg_score)
                if step > 2000 and episode % 10 == 0:
                    history.append(avg_score)
                    plt.plot(history, c='blue')
                    plt.draw()
                    plt.pause(0.001)
                break

        if episode % 250 == 0:
            print('RL saved!')
            RL.save()


if __name__ == '__main__':
    env = gym.make('MsPacman-v0')
    env = env.unwrapped
    RL = DeepQNetwork(
        n_actions=4,
        n_features=210*160*12,
        e_greedy_start=0.0,
        e_greedy_increment=2e-4,
        e_greedy=0.90,
        replace_target_iter=60,
        reward_decay=0.9,
        memory_size=int(3000),
        batch_size=64,
        learning_rate=0.000025
    )
    try:
        RL.load()
    except:
        print('no model found')
    train()
    RL.save()
