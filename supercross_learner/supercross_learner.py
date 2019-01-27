# imports
import numpy as np
#import math as math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from supercross_track_maker import mk_trk1, mk_trk2
from supercross_env import supercross_env

trk = mk_trk2(units='m')
env  = supercross_env(trk)

throttle=1
while not env.done:
  env.step(throttle)
  
  



# plot positioins vs time
#fig1, ax1 = plt.subplots()
#ax1.plot(t,bkY, label='bkY')
#ax1.plot(t,whlY, label='whlY')
#ax1.plot(t,sTY,'g--', label='sTY')
#ax1.legend()
# ax1.set_ylim(0,None)

fig2, ax2 = plt.subplots()
ax2.plot(env.trkX[0:env.i],env.trkY[0:env.i], label='trk')
ax2.plot(env.bkX[0:env.i],env.bkY[0:env.i], label='bk')
ax2.plot(env.bkX[0:env.i],env.whlY[0:env.i], label='whl')
ax2.legend()

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