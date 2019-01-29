# imports
import numpy as np
#import math as math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from supercross_track_maker import mk_trk1, mk_trk2
from supercross_env import supercross_env
from supercross_utilities import max_dict, random_action
import copy

# define common hyperparameters
GAMMA = 0.9
ALPHA = 0.1

# AGENT 001: full throttle until end of track
  
# AGENT 002: semi-grad SARSA, with feature vector fv001
class Agent002:
  def __init__(self):
    self.theta = np.random.randn(5) / np.sqrt(5)

  def sa2x(self, env, a):
    return np.array([
      a,
      env.bkvX[env.i],
      env.bkvY[env.i],
      env.bkY[env.i],
      env.trkYt,
    ])

  def predict(self, env, a):
    x = self.sa2x(env, a)
    return self.theta.dot(x)

  def grad(self, env, a):
    return self.sa2x(env, a)


def getQs(model, env):
  # we need Q(s,a) to choose an action
  # i.e. a = argmax[a]{ Q(s,a) }
  Qs = {}
  for a in np.arange(0,1,0.01):
    q_sa = model.predict(env, a)
    Qs[a] = q_sa
  return Qs 


if __name__ == '__main__':
  
  # AGENT 001: sweet of const thrttle over episode
  trk = mk_trk2(units='m')
  env  = supercross_env(trk)
  
  score = {}
  best_score = -1e9
  worst_score = 0
  for a in np.arange(0.4,1.01,0.01):
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
  fig2, ax2 = plt.subplots()
  ax2.plot(env_best.trkX[0:env_best.i],env_best.trkY[0:env_best.i], label='trk')
  ax2.plot(env_best.bkX[0:env_best.i],env_best.bkY[0:env_best.i], label='bk_best')
  ax2.plot(env_worst.bkX[0:env_worst.i],env_worst.bkY[0:env_worst.i], label='bk_worst')
  ax2.plot(env.bkX[0:env.i],env.bkY[0:env.i], label='bk_full')
  ax2.legend()
  
  
  # AGENT 002: semi-grad SARSA, with feature vector fv001
#  env2  = supercross_env(trk)
#  agent002 = Agent002()
#  
#  # repeat until convergence
#  t = 1.0
#  t2 = 1.0
#  deltas = []
#  for it in range(20000):
#    if it % 100 == 0:
#      t += 0.01
#      t2 += 0.01
#    if it % 1000 == 0:
#      print("it:", it)
#    alpha = ALPHA / t2
#
#    # get Q(s) so we can choose the first action
#    Qs = getQs(agent002, env2)
#
#    # the first (s, r) tuple is the state we start in and 0
#    # (since we don't get a reward) for simply starting the game
#    # the last (s, r) tuple is the terminal state and the final reward
#    # the value for the terminal state is by definition 0, so we don't
#    # care about updating it.
#    a = max_dict(Qs)[0]
#    a = random_action(a, eps=0.5/t) # epsilon-greedy
#    biggest_change = 0
#    while not env2.done:
#      r = grid.move(a)
#      s2 = grid.current_state()
#
#      # we need the next action as well since Q(s,a) depends on Q(s',a')
#      # if s2 not in policy then it's a terminal state, all Q are 0
#      old_theta = model.theta.copy()
#      if grid.is_terminal(s2):
#        model.theta += alpha*(r - model.predict(s, a))*model.grad(s, a)
#      else:
#        # not terminal
#        Qs2 = getQs(model, s2)
#        a2 = max_dict(Qs2)[0]
#        a2 = random_action(a2, eps=0.5/t) # epsilon-greedy
#
#        # we will update Q(s,a) AS we experience the episode
#        model.theta += alpha*(r + GAMMA*model.predict(s2, a2) - model.predict(s, a))*model.grad(s, a)
#        
#        # next state becomes current state
#        s = s2
#        a = a2
#
#      biggest_change = max(biggest_change, np.abs(model.theta - old_theta).sum())
#    deltas.append(biggest_change)
#
#  plt.plot(deltas)
#  plt.show()
  
  
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