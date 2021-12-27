*******************************************************
*  Name      :  Ian Anderson 
*  Student ID:  109166476      
*  Class     :  CSC 4800         
*  Lab#       :  Final Project              
*  Due Date  :  December 8th 2020
*******************************************************


                 Read Me


*******************************************************
*  Description of the program
*******************************************************

This program creates and trains an agent to play the "Space Invaders" game using the 'SpaceInvaders-v0' environment
available from the OpenAI gym in Python3.  Using a combination of Deep Q learning and a convolutional neural
network, it applies reinforcement learning priniciples while analyzing frame transitions from the game in order
to imrove the agent's gameplay patterns and achieve higher scores.


*******************************************************
*  Source files
*******************************************************

Name:  mainLoop.py

   Controls the main gameplay loop

Name:  model.py

    Contains DeepQNetwork and Agent classes

Name:  utils.py

    Contains subroutine used to plot test results


   
*******************************************************
*  Circumstances of programs
*******************************************************

   The program compiles and runs successfully using python 3.7
   

*******************************************************
*  How to build and run the program
*******************************************************

Make sure all 3 files listed above are in the same directory.
Execution should start using 'mainLoop.py'

on line 26 of mainLoop.py there is a line similar to 'player = Agent(gamma = 0.95, epsilon = 1.0, alpha = 0.003, maxMemorySize = 5000, replace = 1000)'

changing these values in the agent initialization will control the hyperparameters for the test as follows:
gamma (range [0,1]) - discount rate
epsilon (range [0,1]) - initial epsilon value. will decrease over time and is used during epsilon/greedy decision process.
alpha - learning rate (i did not mess with this during my testing at the time of writing this. This was a reccommended default value)
replace - dictates how many steps to wait before our Q_next state dictionary is updated.

on line 47 of mainLoop.py there is a line similar to 'num_games = 30'

This controls how many episodes will run before the test terminates.

on line 51 of mainLoop.py there is a line similar to 'batch_size = 32'

This can be increased to speed up testing however it will increase computational requirements as well.

on line 54 of mainLoop.py there is a line similar to ' env = wrappers.Monitor(env, "tmp/space-invaders-1", video_callable=lambda episode_id: True, force=True)'

This can be commented out in order to stop recording episodes automatically.
