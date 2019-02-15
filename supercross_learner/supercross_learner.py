# imports
import numpy as np
#import math as math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from supercross_track_maker import mk_trk1, mk_trk2
from supercross_env import supercross_env
from supercross_utilities import max_dict, random_action, AutoDict, find_nearest
import copy
import time
from math import isinf

# define common hyperparameters
GAMMA = 0.9
ALPHA = 0.1



def selectAction(offpolicyactions,it,Q,sx,sy,t,bcA_v):
  if it < offpolicyactions.size:
      a = offpolicyactions[it]
  else:
    a, _ = max_dict(Q[sx][sy])
    a = random_action(a, eps=0.5/t) # epsilon-greedy
    a = find_nearest(bcA_v, a)
  return a


if __name__ == '__main__':
  startTime=time.time()
  # AGENT 001: sweet of const thrttle over episode
  trk = mk_trk2(units='m')
  env  = supercross_env(trk)
  
  score = {}
  best_time = 1e9
  worst_time = 0
  breakAll = False
  for a in np.array([0.4, 0.98, 1]):
    env.__init__(trk)
    print(a)
    while not env.done:
      env.step(a,1)
    print('done a race')
    print(env.reward)
    score[a] = env.reward
    if env.time < best_time:
      print('found better')
      best_score = env.reward
      env_best = copy.deepcopy(env)
      best_action = a
    if env.time > worst_time:
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
  
  
  # AGENT 002: q-learing using X,Y position. 
  # this will not be generalizable to any track, but a learning experience for me only, 
  # just to see if this type of agent can learn to go fast.
  env2  = supercross_env(trk)
  bcA_v = np.arange(0.01,1,0.05) # vector of bin centers for action dimension of Q table (3d matrix)
  bcX_v = env2.trkXSampled
  maxY = np.max(env2.trkY)
  bcY_v = np.arange(0,maxY*2.5,0.5)
  
  # initialize Q(s,a)
  Q = {}
  # let's also keep track of how many times Q[s] has been updated
  update_counts_s = {}
  update_counts_sa = {}
  for x in bcX_v:
    Q[x] = {}
    update_counts_s[x] = {}
    update_counts_sa[x] = {}
    for y in bcY_v:
      Q[x][y] = {}
      update_counts_s[x][y] = 0.0
      update_counts_sa[x][y] = {}
      for a in bcA_v:
        Q[x][y][a] = 0.0
        update_counts_sa[x][y][a] = 1.0
        

  # we'll start by running some races at const throttle to at least get some learning
  offpolicyactions = np.extract(bcA_v>0.4,bcA_v)
  offpolicyactions = np.repeat(offpolicyactions,100)
  
  # repeat until convergence
  totalIterations = 50000
  t = 1.0
  t2 = 1.0
  deltas = []
  bkY_mat = np.zeros([env2.trkXSampled.size, totalIterations])
  throttle_mat = np.zeros([env2.trkXSampled.size, totalIterations])
  raceTimes = []
  bestTime = env2.t_end
  bestTimeIt = 0
  for it in range(totalIterations):
    if it % 100 == 0:
      t += 0.01
      t2 += 0.01
    if it % 1000 == 0:
      print("it:", it)
      checkPointTime=time.time()
      exeTime = (checkPointTime - startTime)
      print("cummulative execution time:", exeTime)
    
    # start episode
    env2.__init__(trk)
    

    # the first (s, r) tuple is the state we start in and 0
    # (since we don't get a reward) for simply starting the game
    # the last (s, r) tuple is the terminal state and the final reward
    # the value for the terminal state is by definition 0, so we don't
    # care about updating it.
    sx = find_nearest(bcX_v, env2.bkX[env2.i])
    sy = find_nearest(bcY_v, env2.bkY[env2.i])
    a, _ = max_dict(Q[sx][sy])
    biggest_change = 0
    while not env2.done:
      # get state
      sx = find_nearest(bcX_v, env2.bkX[env2.i])
      sy = find_nearest(bcY_v, env2.bkY[env2.i])
      # chose action
      a = selectAction(offpolicyactions,it,Q,sx,sy,t,bcA_v)
      # apply action and step env
      env2.step(a,100)
      # get new state
      sx2 = find_nearest(bcX_v, env2.bkX[env2.i])
      sy2 = find_nearest(bcY_v, env2.bkY[env2.i])
      # get reward
      r = env2.reward

      # adaptive learning rate
      alpha = ALPHA / update_counts_sa[sx][sy][a]
      update_counts_sa[sx][sy][a] += 0.005

      # we will update Q(s,a) AS we experience the episode
      old_qsa = Q[sx][sy][a]
      # the difference between SARSA and Q-Learning is with Q-Learning
      # we will use this max[a']{ Q(s',a')} in our update
      # even if we do not end up taking this action in the next step
      a2, max_q_s2a2 = max_dict(Q[sx2][sy2])
      Q[sx][sy][a] = Q[sx][sy][a] + alpha*(r + GAMMA*max_q_s2a2 - Q[sx][sy][a])
      biggest_change = max(biggest_change, np.abs(old_qsa - Q[sx][sy][a]))

      # we would like to know how often Q(s) has been updated too
      update_counts_s[sx][sy] = update_counts_s[sx][sy] + 1



    if env2.time < bestTime:
      print('new best time', env2.time)
      bestTime = env2.time
      bestTimeIt = it
    if breakAll:
      break
      
      
    raceTimes.append(env2.time)
    deltas.append(biggest_change)
    
    bkY_mat[:,it] = np.interp(env2.trkXSampled,env2.bkX[0:env2.i],env2.bkY[0:env2.i])
    throttle_mat[:,it] = np.interp(env2.trkXSampled,env2.bkX[0:env2.i], env2.throttle[0:env2.i])
  

  np.save('bkY_mat',bkY_mat)
  np.save('throttle_mat',throttle_mat)
  np.save('raceTimes',raceTimes)
  bestTime_mat = np.array([bestTime,bestTimeIt])
  np.save('bestTime_mat',bestTime_mat)
  
  fig3, ax3 = plt.subplots()
  ax3.plot(env_best.trkX[0:env_best.i],env_best.trkY[0:env_best.i], label='trk')
  ax3.plot(env_best.bkX[0:env_best.i],env_best.bkY[0:env_best.i], label='bk_best')
  ax3.plot(env_worst.bkX[0:env_worst.i],env_worst.bkY[0:env_worst.i], label='bk_worst')
  ax3.plot(env2.trkXSampled,bkY_mat[:,bestTimeIt], label='bk_agtSarsa_bestTime')
  ax3.legend()
  
  fig4, ax4 = plt.subplots()
  ax4.plot(deltas, label='deltas')
  ax4.legend()
  
  fig5, ax5 = plt.subplots()
  ax5.plot(raceTimes, label='raceTimes')
  ax5.legend()
  
  plt3dX_bkX, plt3dX_it = np.meshgrid(env2.trkXSampled, np.arange(0,it+1,1))
  fig6 = plt.figure()
  ax6 = fig6.add_subplot(111, projection='3d')
  ax6.plot_surface(plt3dX_bkX, plt3dX_it, bkY_mat.T, cmap=cm.jet, label='bkY')
  ax6.set_title('bkY')
  
  fig7 = plt.figure()
  ax7 = fig7.add_subplot(111, projection='3d')
  ax7.plot_surface(plt3dX_bkX, plt3dX_it, throttle_mat.T, cmap=cm.jet, label='throttle')
  ax7.set_title('throttle')
  
  fig9 = plt.figure()
  ax9 = fig9.add_subplot(111, projection='3d')
  throttle_mat_T = throttle_mat.T.copy()
  ax9.plot_surface(plt3dX_bkX[0:bestTimeIt+20,:], plt3dX_it[0:bestTimeIt+20,:], throttle_mat_T[0:bestTimeIt+20,:], cmap=cm.jet, label='throttle_trunk')
  ax9.set_title('throttl_trunk')
  
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