import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from supercross_track_maker import mk_trk1, mk_trk2
from supercross_env import supercross_env
from tensorforce.contrib.openai_gym import OpenAIGym
# Create an OpenAIgym environment
env = OpenAIGym('CartPole-v0', visualize=True)

a=env.actions
s=env.states


#trk = mk_trk2(units='m')
#env2  = supercross_env(trk)
#bcA_v = np.arange(0.01,1,0.05) # vector of bin centers for action dimension of Q table (3d matrix)
#bcX_v = env2.trkXSampled
#inAir_v = [False, True]
#Q2 = np.load('Q.npy').item()
#
#mat = np.zeros((bcX_v.size,2,bcA_v.size))
#i=0
#j=0
#k=0
#for k1, v1 in Q2.items():
#  j=0
#  for k2, v2 in v1.items():
#    k=0
#    for k3, v3 in v2.items():
#      mat[i,j,k] = v3
#      k+=1
#    j+=1
#  i+=1
#
#
#fig3, ax3 = plt.subplots()
#
#for i in range(bcA_v.size):
#  lblstr = str(bcA_v[i])
#  ax3.plot(bcX_v,mat[:,0,i], label=lblstr)
#
#ax3.legend()
#
#fig4, ax4 = plt.subplots()
#besta_v = np.zeros(bcX_v.size)
#bestaj_v = np.zeros(bcX_v.size)
#for i in range(bcX_v.size):
#  besta_v[i] = bcA_v[np.argmax(mat[i,0,:])]
#  bestaj_v[i] = bcA_v[np.argmax(mat[i,1,:])]
#  if i % 10 == 0:
#    lblstr = str(bcX_v[i])
#    ax4.plot(bcA_v,mat[i,0,:], label=lblstr)
#
#ax4.legend()
#
#bcYnorm_v = np.divide(env2.trkYSampled,np.max(env2.trkYSampled))
#
#fig5, ax5 = plt.subplots()
#ax5.plot(bcX_v,besta_v, label='best onGnd')
#ax5.plot(bcX_v,bestaj_v, label='best inAir')
#ax5.plot(bcX_v,bcYnorm_v, label='trk norm')

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#for i in range(bcA_v.size):
#  ys = mat[:,0,i]
#
#  ax.bar(bcX_v, ys, zs=bcA_v[i], zdir='y', alpha=0.8)
#  
#fig2 = plt.figure()
#bx = fig2.add_subplot(111, projection='3d')
#
#for i in range(bcA_v.size):
#  ys = mat[:,1,i]
#
#  ax.bar(bcX_v, ys, zs=bcA_v[i], zdir='y', alpha=0.8)

##https://matplotlib.org/gallery/mplot3d/bars3d.html
#np.random.seed(19680801)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#colors = ['r', 'g', 'b', 'y']
#yticks = [3, 2, 1, 0]
#for c, k in zip(colors, yticks):
#    # Generate the random data for the y=k 'layer'.
#    xs = np.arange(20)
#    ys = np.random.rand(20)
#
#    # You can provide either a single color or an array with the same length as
#    # xs and ys. To demonstrate this, we color the first bar of each set cyan.
#    cs = [c] * len(xs)
#    cs[0] = 'c'
#
#    # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
#    ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)
#
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#
## On the y axis let's only label the discrete values that we have data for.
#ax.set_yticks(yticks)