""" 
    Ian Anderson
    CSCI 4800 AI with Reinforcement Learning 
    OpenAI GYM Space Invaders (Image Based) Agent
    Final Project Submission
    12/8/2020
    
    Code is functional
"""


import gym
from model import DeepQNetwork, Agent
from utils import plotLearning
import numpy as np

#   ACTION SPACE FOR SPACE INVADERS GAME IS:
#   0-No Action, 1-fire weapon, 2-move right,
#   3-move left, 4-move right and fire, 5-move left and fire

if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')
    player = Agent(gamma = 0.95, epsilon = 1.0, alpha = 0.003, maxMemorySize = 5000, replace = 1000)


    # begin by letting the agent play a few games with totally random actions
    while player.memory_counter < player.memory_size:
        current_observation = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_observation, reward, done, info = env.step(action)

            # penalize the agent for losing
            if done and info['ale.lives'] == 0:
                reward = -100

            player.storeTransition(np.mean(current_observation[0:210, 0:160], axis = 2), action, reward, np.mean(next_observation[0:210, 0:160], axis = 2))
            current_observation = next_observation
    print('done with memory initialization')

    scores = []
    epsilon_history = []
    # @num_games determines the total number of games played by the AI.
    num_games = 30

    # batch size is important for test performance, although it effects the outcome
    # of the agent, only higher powered GPUs can handle a higher batch size
    batch_size = 32
   
    # uncomment the line below to record every episode.
    # env = wrappers.Monitor(env, "tmp/space-invaders-1", video_callable=lambda episode_id: True, force=True)

    for i in range(num_games):
        print('starting game ', i+1, 'epsilon: %.4f' % player.EPSILON)
        epsilon_history.append(player.EPSILON)
        done = False
        current_observation = env.reset()
        frames = [np.sum(current_observation[0:210, 0:160], axis=2)]

        score = 0
        last_action = 0

        # In order to ensure consistent data recording from our observations,
        # we will pass in 3 frames at a time and record our data in correlation to these
        # 'steps'.  This is neccesary because OpenAI gym has the agent perform actions
        # for a psuedorandom number of frames.

        while not done:
            if len(frames) == 3:
                action = player.chooseAction(frames)
                frames = []
            else:
                action = last_action

            next_observation, reward, done, info = env.step(action)
            score += reward
            frames.append(np.sum(next_observation[0:210, 0:160], axis=2))

            # penalty for losing
            if done and info ['ale.lives'] == 0:
                reward = -100

            player.storeTransition(np.mean(current_observation[0:210, 0:160], axis = 2), action, reward, np.mean(next_observation[0:210, 0:160], axis = 2))

            current_observation = next_observation
            player.learn(batch_size)
            last_action = action

            # uncomment next line to have game render live
            #env.render()

        scores.append(score)
        print('score: ', score)

# generate filename and and plot the scorechart into file using generated name.
x = [i + 1 for i in range(num_games)]
filename = 'ScoreChart' + str(num_games) + 'Games' + 'Gamma' + str(player.GAMMA) + \
               'Alpha' + str(player.ALPHA) + 'Memory' + str(player.memory_size)+ '.png'  

plotLearning(x, scores, epsilon_history, filename)
