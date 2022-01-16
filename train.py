import gym
import numpy as np
import argparse
from DDPG import DDPG
from utils import create_directory, plot_learning_curve, scale_action

parser = argparse.ArgumentParser("DDPG parameters")
parser.add_argument('--max_episodes', type=int, default=1000)
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/DDPG/')
parser.add_argument('--figure_file', type=str, default='./output_images/reward.png')

args = parser.parse_args()


def main():
    env = gym.make('LunarLanderContinuous-v2')
    agent = DDPG(alpha=0.0003, beta=0.0003, state_dim=env.observation_space.shape[0],
                 action_dim=env.action_space.shape[0], actor_fc1_dim=400, actor_fc2_dim=300,
                 critic_fc1_dim=400, critic_fc2_dim=300, ckpt_dir=args.checkpoint_dir,
                 batch_size=256)
    create_directory(args.checkpoint_dir,
                     sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])

    reward_history = []
    avg_reward_history = []
    for episode in range(args.max_episodes):
        done = False
        total_reward = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation, train=True)
            action_ = scale_action(action.copy(), env.action_space.high, env.action_space.low)
            observation_, reward, done, info = env.step(action_)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            total_reward += reward
            observation = observation_

        reward_history.append(total_reward)
        avg_reward = np.mean(reward_history[-100:])
        avg_reward_history.append(avg_reward)
        print('Ep: {} Reward: {:.1f} AvgReward: {:.1f}'.format(episode+1, total_reward, avg_reward))

        if (episode + 1) % 200 == 0:
            agent.save_models(episode+1)

    episodes = [i+1 for i in range(args.max_episodes)]
    plot_learning_curve(episodes, avg_reward_history, title='AvgReward',
                        ylabel='reward', figure_file=args.figure_file)


if __name__ == '__main__':
    main()