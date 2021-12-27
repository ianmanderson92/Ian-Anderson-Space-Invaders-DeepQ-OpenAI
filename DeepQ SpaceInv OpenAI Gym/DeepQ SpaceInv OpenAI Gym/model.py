""" 
    Ian Anderson
    CSCI 4800 AI with Reinforcement Learning 
    OpenAI GYM Space Invaders (Image Based) Agent
    Final Project Submission
    12/8/2020
    
    Code is functional
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# Deep Q Network class that uses 3 convolutional layers and 2 fully connected layers.
class DeepQNetwork(nn.Module):
    def __init__(self, ALPHA):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, stride = 4, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128*23*16, 512)
        self.fc2 = nn.Linear(512, 6)

        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, current_observation):
        current_observation = T.Tensor(current_observation).to(self.device)
        current_observation = current_observation.view(-1, 1, 210, 160)
        current_observation = F.relu(self.conv1(current_observation))
        current_observation = F.relu(self.conv2(current_observation))
        current_observation = F.relu(self.conv3(current_observation))
        current_observation = current_observation.view(-1, 128*23*16)

        current_observation = F.relu(self.fc1(current_observation))

        actions = self.fc2(current_observation)

        return actions

class Agent(object):
    def __init__(self, gamma, epsilon, alpha, maxMemorySize, epsEnd = 0.05, replace = 10000, actionSpace = [0,1,2,3,4,5]):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.ALPHA = alpha
        self.actionSpace = actionSpace
        self.memory_size = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memory_counter = 0
        self.replace_target_cnt = replace
        self.Q_eval = DeepQNetwork(alpha)
        self.Q_next = DeepQNetwork(alpha)

    def storeTransition(self, current_state, action, reward, next_state):
        if self.memory_counter < self.memory_size:
            self.memory.append([current_state, action, reward, next_state])
        else:
            self.memory[self.memory_counter%self.memory_size] = [current_state, action, reward, next_state]
        self.memory_counter += 1

    # @chooseAction Used to choose the next action for the AI based on epsilon's current value.
    def chooseAction(self, current_observation):
        rand = np.random.random()
        actions = self.Q_eval.forward(current_observation)
        if rand < 1 - self.EPSILON:
            action = T.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.actionSpace)
        self.steps += 1
        return action

    # @learn used to determine the reward for each step, and update the Q evaluation arrays.
    def learn(self, batch_size):
        #zero out gradients for batch optimization
        self.Q_eval.optimizer.zero_grad()

        # Check to see if we need to replace the target network,
        # if it is time load the current_state dictionary from the Q evaluation
        # onto the Q next network
        if self.replace_target_cnt is not None and \
            self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        # calculate the start of the memory batch sample to get a small subset of
        # the array
        memStart = int(np.random.choice(range(self.memory_size - batch_size - 1)))


        # convert the memory batch to a numpy array
        miniBatch = self.memory[memStart:memStart + batch_size]
        memory = np.array(miniBatch)

        # forward the current and next current_state onto the device
        Q_prediction = self.Q_eval.forward(list(memory[:, 0][:])).to(self.Q_eval.device)
        Q_next = self.Q_next.forward(list(memory[:, 3][:])).to(self.Q_eval.device)

        # calculate the best action for the next current_state and the rewards accosiated
        maxA = T.argmax(Q_next, dim = 1).to(self.Q_eval.device)
        rewards = T.Tensor(list(memory[:, 2])).to(self.Q_eval.device)
        Q_target = Q_prediction

        # update the Q evaluation for the determined best action or @maxA, which equals
        # calculated reward + GAMMA * actual reward
        Q_target[:, maxA] = rewards + self.GAMMA*T.max(Q_next[1])

        # diminish epsilon over time
        if self.steps > 1500:
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END

        # Calculate loss function and back propagate it
        loss = self.Q_eval.loss(Q_target, Q_prediction).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

