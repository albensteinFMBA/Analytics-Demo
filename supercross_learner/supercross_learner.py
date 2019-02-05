# imports
import numpy as np
#import math as math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from supercross_track_maker import mk_trk1, mk_trk2
from supercross_env import supercross_env
from supercross_utilities import max_dict, random_action
import copy
import time

# define common hyperparameters
GAMMA = 0.9
ALPHA = 0.1

# AGENT 001: full throttle until end of track
  
# AGENT 002: semi-grad SARSA, with feature vector fv001
class AgentAprxSmGrdSarsa:
  def __init__(self, sLvl, xLvl):
    # sLvlDict keeps track of the number of data streams used to create the "state"
    # we use a dict since there may be 2 different formulations of the state with same number of data streams
    self.sLvl = sLvl
    self.sLvlDict = {}
    self.sLvlDict[1] = 6
    self.sLvlDict[2] = 4
    self.sLvlDict[3] = 36 # added trank elevation samples
    # initialize a vector of "feature extremes", values used for the min/max inputs to min-max normalization.
    # initialize to 0.01, and learn the exetreme for each as the game is played.
    self.sNorm = np.multiply(0.01,  np.ones(self.sLvlDict[self.sLvl]))
    # xLvl keeps track of the number of bins used to differentiate the action space
    if xLvl == -1:
      self.xLvlArr = np.concatenate((np.arange(0.2,0.5,0.1),np.arange(0.5,1.05,0.05)))
      self.xLvl = self.xLvlArr.size
    else:
      self.xLvl = xLvl
    # we can then use the dimensions of the sLvl and xLvl to create the dimension of theta and x
    self.theta_dim = self.sLvlDict[sLvl] * self.xLvl
    self.theta = np.random.randn(self.theta_dim) / np.sqrt(self.theta_dim)
  
  def getState(self, env):
    # the "state" is defined in the agent class since the environment has many data streams, 
    # and any combination of these streams could be used to create a "state", so called feature egineering
    # Agents are differentiated by:
    # 1] how they define the state using enviroment data, 
    # 2] what they do with that state information to create actions
    # create a vector of feature magnitudes used for min-max normalization. 
      # not sure if/how to use mean normalization since we dont have the features until we finish playing the game, at which point we dont need them anymore
      # must define feature exetremums by estimation intially, and tune them as exereince grows
    if self.sLvl == 1:
      s = np.array([
        env.bkaX[env.i],
        env.bkaY[env.i],
        env.bkvX[env.i],
        env.bkvY[env.i],
        env.bkY[env.i],
        env.trkYt[env.i],
      ])
    elif self.sLvl == 2:
      s = np.array([
        env.bkaX[env.i],
        env.bkaY[env.i],
        env.bkY[env.i],
        env.trkYt[env.i],
      ])
    elif self.sLvl == 3:
      trkLookAheadDistTgt = 30 # meters of distance the agent sees ahead
      trkLookAheadSampleDist = 1 # meters of track distance between each track sample point "seen" by agent
      if trkLookAheadDistTgt > env.trkX[-1]:
        # if the end of track is clser than the agent look ahead distance, only samepl track elevenation to end of track
        trkLookAheadSamples = np.arange(env.bkX[env.i], env.bkX[-1], 1)
        trkFeatures = np.interp(trkLookAheadSamples,env.trkX,env.trkY)
        # however, since agent feature vector size must be constant, and fed information represeenting the "same feature" for each learn pass, concat zeros to represent that track is flat beyond the end
        missingZeros = trkLookAheadDistTgt/trkLookAheadSampleDist - trkFeatures.size
        trkFeatures = np.concatenate((trkFeatures, np.zeros([missingZeros])))
      else:
        trkLookAheadSamples = np.arange(env.bkX[env.i], env.bkX[env.i] + trkLookAheadDistTgt, trkLookAheadSampleDist)
        trkFeatures = np.interp(trkLookAheadSamples,env.trkX,env.trkY)
      trkNorm = np.multiply(env.trkYmax, np.ones(trkFeatures.size))
      
      motionFeatures = np.array([
                                    1, # bias term
                                    env.bkaX[env.i],
                                    env.bkaY[env.i],
                                    env.bkvX[env.i],
                                    env.bkvY[env.i],
                                    env.bkY[env.i],
                                    ])
      
      # create non-normalized state vector
      s = np.concatenate((motionFeatures, trkFeatures))
      # update sNorm
      self.sNorm = np.maximum(self.sNorm, np.absolute(np.concatenate(motionFeatures, trkNorm)))
      # normalize elements of s. https://en.wikipedia.org/wiki/Feature_scaling
      s = (s - self.sNorm*-1)/(self.sNorm*2)
      # end self.sLvl == 3:
    return s
  
  def sa2x(self, s, a):
#    x = np.concatenate((a,s),axis=1)
    if self.xLvl == 1:
      x = s
    else:
      x = np.zeros((s.size * self.xLvl))
      for n in range(self.xLvlArr):
        binLim = self.xLvlArr[n]
        if a < binLim:
          startIdx = (n)*s.size
          endIdx = (n+1)*s.size
          x[startIdx:endIdx] = s      
    return x

  def predict(self, s, a):
    x = self.sa2x(s, a)
    prediction = self.theta.dot(x)
    return prediction

  def grad(self, s, a):
    grad = self.sa2x(s, a)
    return grad


def getQs(model, s):
  # we need Q(s,a) to choose an action
  # i.e. a = argmax[a]{ Q(s,a) }
  Qs = {}
  for a in np.arange(0,1,0.01):
    q_sa = model.predict(s, a)
    Qs[a] = q_sa
  return Qs 

def selectAction(offpolicyactions,it,Qs,t):
  if it < offpolicyactions.size:
      a = offpolicyactions[it]
  else:
    a = max_dict(Qs)[0]
    a = random_action(a, eps=0.5/t) # epsilon-greedy
  return a


if __name__ == '__main__':
  startTime=time.time()
  # AGENT 001: sweet of const thrttle over episode
  trk = mk_trk2(units='m')
  env  = supercross_env(trk)
  
  score = {}
  best_score = -1e9
  worst_score = 0
  for a in np.array([0.4, 0.98, 1]):
    env.__init__(trk)
    print(a)
    while not env.done:
      env.step(a)
    print('done a race')
    print(env.reward)
    score[a] = env.reward
    if env.reward > best_score:
      print('found better')
      best_score = env.reward
      env_best = copy.deepcopy(env)
      best_action = a
    if env.reward < worst_score:
      print('found worse')
      worst_score = env.reward
      env_worst = copy.deepcopy(env)
      worst_action = a
  
  print(worst_action)
  print(best_action)
#  fig2, ax2 = plt.subplots()
#  ax2.plot(env_best.trkX[0:env_best.i],env_best.trkY[0:env_best.i], label='trk')
#  ax2.plot(env_best.bkX[0:env_best.i],env_best.bkY[0:env_best.i], label='bk_best')
#  ax2.plot(env_worst.bkX[0:env_worst.i],env_worst.bkY[0:env_worst.i], label='bk_worst')
#  ax2.plot(env.bkX[0:env.i],env.bkY[0:env.i], label='bk_full')
#  ax2.legend()
  
  
  # AGENT 002: semi-grad SARSA, with feature vector fv001
  env2  = supercross_env(trk)
  agtSarsa = AgentAprxSmGrdSarsa(sLvl=3, xLvl=-1)
  offpolicyactions = agtSarsa.xLvlArr[agtSarsa.xLvlArr>0.4]
  offpolicyactions = np.repeat(offpolicyactions,4)
  
  # repeat until convergence
  totalIterations = 4000
  t = 1.0
  t2 = 1.0
  deltas = []
  bkY_mat = np.zeros([env2.bkX.size, totalIterations])
  a_mat = np.zeros([env2.bkX.size, totalIterations])
  theta_mat = np.zeros([env2.bkX.size, totalIterations])
  for it in range(totalIterations):
    if it % 100 == 0:
      t += 0.01
      t2 += 0.01
    if it % 1000 == 0:
      print("it:", it)
      checkPointTime=time.time()
      exeTime = (checkPointTime - startTime)
      print("cummulative execution time:", exeTime)
    alpha = ALPHA / t2
    
    # start episode
    env2.__init__(trk)
    s = agtSarsa.getState(env2)
    
    # get Q(s) so we can choose the first action
    Qs = getQs(agtSarsa, s)

    # the first (s, r) tuple is the state we start in and 0
    # (since we don't get a reward) for simply starting the game
    # the last (s, r) tuple is the terminal state and the final reward
    # the value for the terminal state is by definition 0, so we don't
    # care about updating it.
    a = selectAction(offpolicyactions,it,Qs,t)
    a_v = []
    a_v.append(a)
    biggest_change = 0
    while not env2.done:
      env2.step(a)
#      print("a:", a)
#      print("sim time:", env2.t[env2.i])
      r = env2.reward
      s2 = agtSarsa.getState(env2)

      # we need the next action as well since Q(s,a) depends on Q(s',a')
      # if s2 not in policy then it's a terminal state, all Q are 0
      old_theta = agtSarsa.theta.copy()
      if env2.done:
        agtSarsa.theta += alpha*(r - agtSarsa.predict(s, a))*agtSarsa.grad(s, a)
      else:
        # not terminal
        Qs2 = getQs(agtSarsa, s2)
        a2 = selectAction(offpolicyactions,it,Qs2,t)
        
        # we will update Q(s,a) AS we experience the episode
        agtSarsa.theta += alpha*(r + GAMMA*agtSarsa.predict(s2, a2) - agtSarsa.predict(s, a))*agtSarsa.grad(s, a)
        
        # next state becomes current state
        s = s2
        a = a2
        a_v.append(a)

      biggest_change = max(biggest_change, np.abs(agtSarsa.theta - old_theta).sum())
    deltas.append(biggest_change)
    bkY_mat[:,it] = np.interp(env2.trkX,env2.bkX, env2.bkY)
    a_mat[:,it] = np.interp(env2.trkX,env2.bkX, a_v)
    theta_mat[:,it] = agtSarsa.theta
  

  np.save('bkY_mat',bkY_mat)
  np.save('a_mat',a_mat)
  np.save('theta_mat',theta_mat)
  
  fig3, ax3 = plt.subplots()
  ax3.plot(env_best.trkX[0:env_best.i],env_best.trkY[0:env_best.i], label='trk')
  ax3.plot(env_best.bkX[0:env_best.i],env_best.bkY[0:env_best.i], label='bk_best')
  ax3.plot(env_worst.bkX[0:env_worst.i],env_worst.bkY[0:env_worst.i], label='bk_worst')
  ax3.plot(env2.bkX[0:env2.i],env2.bkY[0:env2.i], label='bk_agtSarsa')
  ax3.legend()
  
  fig4, ax4 = plt.subplots()
  ax4.plot(deltas, label='deltas')
  ax3.legend()
  
  
  # plot positioins vs time
  #fig1, ax1 = plt.subplots()
  #ax1.plot(t,bkY, label='bkY')
  #ax1.plot(t,whlY, label='whlY')
  #ax1.plot(t,sTY,'g--', label='sTY')
  #ax1.legend()
  # ax1.set_ylim(0,None)
  
  #  fig2, ax2 = plt.subplots()
  #  ax2.plot(env.trkX[0:env.i],env.trkY[0:env.i], label='trk')
  #  ax2.plot(env.bkX[0:env.i],env.bkY[0:env.i], label='bk')
  #  ax2.plot(env.bkX[0:env.i],env.whlY[0:env.i], label='whl')
  #  ax2.legend()
  
  # plot velocity vs time
  #fig5, ax5 = plt.subplots()
  #ax5.plot(bkX[0:i],bkvX[0:i], label='bkvX vs bkX')
  #ax5.plot(bkX[0:i],bkvY[0:i], label='bkvY vs bkX')
  #ax5.legend()
  
  #fig8, ax8 = plt.subplots()
  #ax8.plot(t,whlW, label='whlW')
  #ax8.legend()
  
  # plot tire beahvior
  #fig6, ax6 = plt.subplots()
  #ax6.plot(t,bkDragX, label='bkDragX')
  #ax6.legend()
  #fig7, ax7 = plt.subplots()
  #ax7.plot(t,whlfX, label='whlfX')
  #ax7.legend()
  
  # fig9, ax9 = plt.subplots()
  # ax9.plot(t,whlAlpha, label='whlAlpha')
  # ax9.legend()
  #fig10, ax10 = plt.subplots()
  #ax10.plot(t,whlCntctMthd, label='whlCntctMthd')
  #ax10.plot(t,inAir, label='inAir')
  #ax10.legend()
  # fig11, ax11 = plt.subplots()
  # ax11.plot(t,whlSrMthd, label='whlSrMthd')
  # ax11.legend()
  
  
  # plot suspension forces
  #fig2, ax2 = plt.subplots()
  #ax2.plot(t,sFk, label='spring')
  #ax2.plot(t,sFb, label='damper')
  #ax2.plot(t,whlfY, label='whlfY')
  #ax2.legend()
  
  # plot distance and height vs time
  #fig4 = plt.figure()
  #ax4 = fig4.add_subplot(111, projection='3d')
  #ax4.plot(bkX,t,bkY, label='bike')
  #ax4.plot(bkX,t,whlY, label='wheel')
  #ax4.legend()
  #ax4.set_ylim(0,None)
  # rotate the axes and update
  # for angle in range(0, 360):
  #     ax4.view_init(30, angle)
  #     plt.draw()
  #     plt.pause(.001)