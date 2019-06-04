import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from supercross_track_maker import mk_trk1, mk_trk2
from supercross_env import supercross_env
from tensorforce.contrib.openai_gym import OpenAIGym
# Create an OpenAIgym environment
#env = OpenAIGym('CartPole-v0', visualize=True)
#a=env.actions
#s=env.states

trk = mk_trk2(units='m')
envS = supercross_env(trk)
aS=envS.actions
sS=envS.states
envS.reset

# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
#from tensorforce.agents import PPOAgent
#from tensorforce.execution import Runner
#
## Create an OpenAIgym environment.
#environment = OpenAIGym('CartPole-v0', visualize=False)
#
## Network as list of layers
## - Embedding layer:
##   - For Gym environments utilizing a discrete observation space, an
##     "embedding" layer should be inserted at the head of the network spec.
##     Such environments are usually identified by either:
##     - class ...Env(discrete.DiscreteEnv):
##     - self.observation_space = spaces.Discrete(...)
#
## Note that depending on the following layers used, the embedding layer *may* need a
## flattening layer
#
#network_spec = [
#    # dict(type='embedding', indices=100, size=32),
#    # dict(type'flatten'),
#    dict(type='dense', size=32),
#    dict(type='dense', size=32)
#]
#
#agent = PPOAgent(
#    states=environment.states,
#    actions=environment.actions,
#    network=network_spec,
#    # Agent
#    states_preprocessing=None,
#    actions_exploration=None,
#    reward_preprocessing=None,
#    # MemoryModel
#    update_mode=dict(
#        unit='episodes',
#        # 10 episodes per update
#        batch_size=10,
#        # Every 10 episodes
#        frequency=10
#    ),
#    memory=dict(
#        type='latest',
#        include_next_states=False,
#        capacity=5000
#    ),
#    # DistributionModel
#    distributions=None,
#    entropy_regularization=0.01,
#    # PGModel
#    baseline_mode='states',
#    baseline=dict(
#        type='mlp',
#        sizes=[32, 32]
#    ),
#    baseline_optimizer=dict(
#        type='multi_step',
#        optimizer=dict(
#            type='adam',
#            learning_rate=1e-3
#        ),
#        num_steps=5
#    ),
#    gae_lambda=0.97,
#    # PGLRModel
#    likelihood_ratio_clipping=0.2,
#    # PPOAgent
#    step_optimizer=dict(
#        type='adam',
#        learning_rate=1e-3
#    ),
#    subsampling_fraction=0.2,
#    optimization_steps=25,
#    execution=dict(
#        type='single',
#        session_config=None,
#        distributed_spec=None
#    )
#)
#
## Create the runner
#runner = Runner(agent=agent, environment=environment)
#
#
## Callback function printing episode statistics
#def episode_finished(r):
#    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
#                                                                                 reward=r.episode_rewards[-1]))
#    return True
#
#
## Start learning
#runner.run(episodes=3000, max_episode_timesteps=200, episode_finished=episode_finished)
#runner.close()
#
## Print statistics
#print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
#    ep=runner.episode,
#    ar=np.mean(runner.episode_rewards[-100:]))
#)
#


#import sys
#sys.path


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