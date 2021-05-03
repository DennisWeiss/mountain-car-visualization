import torch
import numpy as np
import gym
from time import sleep
import math
from QAgent import QAgent





# Define environment, agent and hyperparameters
env = gym.make('MountainCar-v0')
A=QAgent([15,15],3,env.observation_space.low,env.observation_space.high)#,Q="q-75k")
LR=0.1
eps=0.2
DISCOUNT=0.95
EPISODES=10000
EPSILON_DECAY_START=0
EPSILON_DECAY_END=EPISODES//2
deltaeps=eps/(EPSILON_DECAY_END-EPSILON_DECAY_START)
RENDER_EACH=2000
SAVE_EACH=1000
# Number of episodes to wait before writing on the console
SAY_HI_EACH=200
hi=0
for ep in range(EPISODES):
    if ep%SAVE_EACH==0:
        torch.save(A.Q, "q-" + str(int(ep / SAVE_EACH)) + "k")
    if ep%SAY_HI_EACH==0:
        # Print out code to show progress
        print("Hi from episode "+str(hi*SAY_HI_EACH))
        hi+=1
    X=env.reset() # X is the state var

    done=False
    if ep%RENDER_EACH==0:
        print("Episode " + str(ep))
    while not done:
        if ep % RENDER_EACH == 0:
            env.render()
            sleep(0.01)

        ac=A.getAction(X,eps)
        (newX, r, done, _) = env.step(ac)
        if not done:
            A.updateQ(X,ac,r,newX,DISCOUNT,LR)
        elif newX[0]>=env.goal_position:
            A.updateQFinal(X,ac)
        X=newX
    if ep>=EPSILON_DECAY_START and ep<=EPSILON_DECAY_END:
        eps-=deltaeps
env.close()
torch.save(A.Q,"q-"+str(int(EPISODES/1000))+"k")

"""
A=QAgent([18,14],[0,1,2],env.observation_space.low,env.observation_space.high)
for i in range(1000):
    X=env.reset()
    done=False
    while not done:
        env.render()
        sleep(0.01)
        pass
        ac=A.getAction(X)
        (X, r, done, _) = env.step(ac)
        #print("Action: "+str(ac)+"Reward: "+str(r))
        if done:
            print("Final X: "+str(X[0]))
env.close()"""