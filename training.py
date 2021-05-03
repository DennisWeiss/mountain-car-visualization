import torch
import numpy as np
import gym
from time import sleep
import math
from QAgent import QAgent






env = gym.make('MountainCar-v0')
A=QAgent([15,15],3,env.observation_space.low,env.observation_space.high,Q="aq75k")
LR=0.1
eps=0.0
DISCOUNT=0.95
EPISODES=25000
EPSILON_DECAY_START=0
EPSILON_DECAY_END=EPISODES//2
deltaeps=eps/(EPSILON_DECAY_END-EPSILON_DECAY_START)
RENDER_EACH=3000
SAY_HI_EACH=200
hi=0
for ep in range(EPISODES):
    if ep%SAY_HI_EACH==0:
        hi+=1
        print("Hi from episode "+str(hi*SAY_HI_EACH))
    X=env.reset() # X is the state var

    done=False
    rew=0
    t=0
    while not done:
        t+=1
        if ep % RENDER_EACH == 0:
            env.render()
            sleep(0.01)
        ac=A.getAction(X,eps)
        (newX, r, done, _) = env.step(ac)
        rew+=r
        A.updateQ(X,ac,r,newX,DISCOUNT,LR)
        X=newX
    if ep%RENDER_EACH==0:
        print("Episode "+str(ep))
        print("R = "+str(float(r)/float(t)))
    if ep>=EPSILON_DECAY_START and ep<=EPSILON_DECAY_END:
        eps-=deltaeps
env.close()
torch.save(A.Q,"aq100k")

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