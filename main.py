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
        observation_with_previous_four_frames = np.zeros((210 * 160 * 3, 4))

        observation_with_previous_four_frames[:, 0] = observation.reshape(-1) / 255.0
        tot_score_in_episode = 0
        while True:
            # Choose action
            action = RL.choose_action(observation_with_previous_four_frames.reshape(-1))

            # Step env
            reward = 0
            for _ in range(2):
                observation, reward_, terminated, extra_info = env.step(action)
                reward += reward_
            tot_score_in_episode += reward

            # Store observation
            old_obs = observation_with_previous_four_frames.copy()
            for i in range(3):
                observation_with_previous_four_frames[:, 3-i] = observation_with_previous_four_frames[:, 2-i]
            observation_with_previous_four_frames[:, 0] = observation.reshape(-1) / 255.0
            RL.store_transition(old_obs.reshape(-1), action, reward/10, observation_with_previous_four_frames.reshape(-1))

            # Learn
            if step > 2000 and step % 16 == 0:
                RL.learn()
                env.render()

            step += 1

            # Summarise when terminated
            if terminated or extra_info['ale.lives'] < 3:
                avg_score = 0.01 * tot_score_in_episode + 0.99 * avg_score
                print(episode, int(tot_score_in_episode), avg_score)
                if step > 2000 and episode % 10 == 0:
                    history.append(avg_score)
                    plt.plot(history, c='blue')
                    plt.draw()
                    plt.pause(0.001)
                break

        # Save model
        if episode % 125 == 0:
            print('RL saved!')
            RL.save()


if __name__ == '__main__':
    env = gym.make('MsPacman-v0')
    env = env.unwrapped
    RL = DeepQNetwork(
        n_actions=9,
        n_features=210*160*12,
        e_greedy_start=0.0,
        e_greedy_increment=1e-4,
        e_greedy=0.85,
        replace_target_iter=60,
        reward_decay=0.99,
        memory_size=int(3000),
        batch_size=64,
        learning_rate=0.00025
    )
    try:
        RL.load()
    except:
        print('no model found')
    train()
    RL.save()
