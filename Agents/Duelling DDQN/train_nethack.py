import random
import numpy as np
import gym
import torch

from duelling_agent import *
from duelling_architecture import *
from wrappers import *

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    ACTIONS = (
        nethack.CompassDirection.N,
        nethack.CompassDirection.E,
        nethack.CompassDirection.S,
        nethack.CompassDirection.W,
        nethack.Command.PICKUP,
        nethack.Command.ZAP,
        nethack.Command.FIRE)

    env = gym.make('MiniHack-LavaCross-v0',
                    observation_keys = ['pixel_crop'],
                    penalty_time=-0.1,
                    penalty_step=-0.1,
                    reward_lose=-1,
                    reward_win=5,
                    seeds = [0],
                    actions = ACTIONS)

    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = FrameStack(env, 4)
    state = np.array(env.reset())

    hyper_params = {
            "replay-buffer-size": int(5000),  # replay buffer size 5e3
            "learning-rate": 1e-4,  # learning rate for Adam optimizer
            "discount-factor": 1,  # discount factor
            "num-steps": int(1e6),  # total number of steps to run the environment for
            "batch-size": 256,  # number of transitions to optimize at the same time 256
            "learning-starts": 5000,  # number of steps before learning starts
            "learning-freq": 5,  # number of iterations between every optimization step
            "use-double-dqn": True,  # use double deep Q-learning
            "target-update-freq": 1000,  # number of iterations between every target network update
            "eps-start": 1.0,  # e-greedy start threshold
            "eps-end": 0.01,  # e-greedy end threshold
            "eps-fraction": 0.6,  # fraction of num-steps
            "print-freq": 10,
    }

    fname = str(env) + '_env_' + str(hyper_params['learning-rate']) + '_lr_' + str(hyper_params['num-steps']) + '_num-steps_' + str(hyper_params['eps-fraction']) + '_eps-fraction_'\
            + '_repl_' + str(hyper_params['batch-size']) + '_batch_size_' +str(hyper_params['replay-buffer-size']) + '_replay-buffer_'

    tb = SummaryWriter(comment=fname)

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])
    agent = Duelling_DDQN(observation_space=env.observation_space,
                    action_space=env.action_space,
                    replay_buffer=replay_buffer,
                    use_double_dqn=hyper_params['use-double-dqn'],
                    lr=hyper_params['learning-rate'],
                    batch_size=hyper_params['batch-size'],
                    gamma=hyper_params['discount-factor']
    )

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]

    best_score = env.reward_range[0]

    lava_count = 0
    key_count = 0
    last_action = None
    prev_action = None

    state = env.reset()
    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (
            hyper_params["eps-end"] - hyper_params["eps-start"]
        )
        sample = random.random()
        if sample <= eps_threshold:
            action = np.random.choice(env.action_space.n)
        else:
            action = agent.act(state)

        next_state, reward, done, _ = env.step(action)

        '''Uncomment the following code to use as a custom reward manager for the LavaCross-v0 environment'''
        '''if env.key_in_inventory("wand") == 'f':
          key_count += 1
          if key_count == 1:
            reward = 0.5
        elif(env.key_in_inventory("wand") == 'f' and prev_action == 5 and last_action == 6 and action == 1):
          lava_count += 1
          if lava_count == 1:
            reward = 1'''

        prev_action = last_action
        last_action = action

        replay_buffer.add(state, action, reward, next_state, float(done))
        tb.add_scalar("reward_per_step", reward)
        ep_rew.append(reward)
        state = next_state

        episode_rewards[-1] += reward
        avg_score = np.mean(ep_rew[-100:])

        if done:
            state = env.reset()
            tb.add_scalar('reward', episode_rewards[-1], t)
            episode_rewards.append(0.0)
            lava_count = 0
            key_count = 0
            last_action = None
            prev_action = None

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["learning-freq"] == 0
        ):
            loss = agent.optimise_td_loss()
            ep_loss.append(loss)
            tb.add_scalar('loss', loss, t)

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["target-update-freq"] == 0
        ):
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if (
            avg_score > best_score
        ):
            best_score = avg_score
            torch.save(agent, fname)

        if (
            done
            and hyper_params["print-freq"] is not None
            and len(episode_rewards) % hyper_params["print-freq"] == 0
        ):
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("epsilon: {}".format(eps_threshold))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")

    torch.save(agent, fname)
