# imports
import numpy as np
#import math as math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from supercross_track_maker import mk_trk1, mk_trk2

# Define constants for environment model
g = -9.81

# wheel
whlR = (19/2+2)*0.0254 # ~0.3meters
whlM = 15
whlJ = 0.25*10

# bike
bkM = (220 - whlM + 160) / 2.2
bkPwr = 55 / 0.75
bkTrq = 2*-g*bkM*whlR
bkCrr = 0.015
bkAero = 0.5*1.2*1*0.7 # 1/2*rho*A*Cd

# suspension
sK = 2e4
sB = 1e4
sFbSat=np.abs(bkM*-g*10)
sC = 3
sT = 0.3
sSag = 0.2
sPL = bkM*-g - sK*sT*sSag

# tire grip model - nu*Fy
nu = 0.7

# define simulation
dt = 0.001
t=np.arange(0,20,dt)

# defined track
pts = mk_trk2(units='m')
trkStep = 0.05
trkX = np.arange(pts[0,0],pts[0,-1],0.05)
trkY = np.interp(trkX,pts[0,:],pts[1,:])
#fig12, ax12 = plt.subplots()
#ax12.plot(trkX,trkY, label='trk')

# defined initial conditions
bkX1 = trkX[0]
whlY1 = trkY[0]+whlR+0.5*0 # add initial positive offset to drop bike at begining
bkY1 = whlY1 + sT*(1-sSag)

# pre-allocate results data vectors
whlY = np.multiply(whlY1, np.ones(t.size))
whlvY= np.zeros(t.size)
whlaY= np.zeros(t.size)
bkX  = np.multiply(bkX1,  np.ones(t.size))
bkY =  np.multiply(bkY1,  np.ones(t.size))
bkvX = np.zeros(t.size)
bkvY = np.zeros(t.size)
bkaX = np.zeros(t.size)
bkaY = np.zeros(t.size)
whlAlpha = np.zeros(t.size)
whlW = np.zeros(t.size)
inAir = np.zeros(t.size)
whlCntctMthd = np.zeros(t.size)
whlfY = np.zeros(t.size)
whlfX = np.zeros(t.size)
sTY  = np.multiply(sT,  np.ones(t.size))
sFk = np.zeros(t.size)
sFb = np.zeros(t.size)
sFt = np.zeros(t.size)
pkTrq = np.multiply(bkTrq,  np.ones(t.size))
bkDragX = np.zeros(t.size)
bkDragY = np.zeros(t.size)
# for loop in time
for i in range(t.size):

# while loop in track length
# i=-1
# while True:
#   i+=1

  # break if time has reached end of preallocated results vectors or out of track length
  if (i >= t.size - 1) or (bkX[i] >= trkX[-1]):
    break
  
  # compute suspension forces. convention, extension=+ve, tension=-ve
  if i > 0:
    if inAir[i]:
      sFk[i]=0
      sFb[i]=0
      sFt[i]=whlM*g*0 
    else:
      sFk[i] = sPL + (sT - sTY[i])*sK
      sFb[i] = (sTY[i-1] - sTY[i])/dt*sB
      sFb[i] = min((sFbSat,max(sFb[i],-sFbSat)))
      sFt[i]=0
  else:
    sFk[i] = sPL + (sT - sTY[i])*sK
    sFb[i]=0
    sFt[i]=0
  # compute Y component drag forces
  bkDragY[i] = bkAero*bkvY[i]*bkvY[i]*np.sign(bkvY[i])*-1
  # compute bike free body in Y-direction and integrate
  bkaY[i] = (bkM*g + sFk[i] + sFb[i] + sFt[i] + bkDragY[i]) / bkM
  bkvY[i+1]  = bkvY[i]  + bkaY[i] *dt
  bkY[i+1]   = bkY[i]   + bkvY[i] *dt
  
  # find available peak torque
  if whlW[i] > 0:
    pkTrq = min(bkTrq, bkPwr/whlW[i])
  else:
    pkTrq = bkTrq

  # compute torque command
  cmdTrq = pkTrq*1 # no agent yet, apply fraction of peak torque
  
  # compute bike longitudinal free body
  if abs(bkvX[i]) > 0.01:
    bkDragX[i] = bkAero*bkvX[i]*bkvX[i]*np.sign(bkvX[i])*-1 - bkCrr*-g*(bkM+whlM)
  if inAir[i]:
    bkaX[i] = bkDragX[i]/(bkM+whlM)
  else:
    # compute peak longitidinal force as function of vertical force
    whlfY[i] = np.array([ sFk[i] + sFb[i] + whlM*-g])
    whlfXMax = whlfY[i]*nu
    # compute applied tractive force
    if cmdTrq >= 0:
      whlfX[i] = min(cmdTrq/whlR,  whlfXMax)
    else:
      whlfX[i] = max(cmdTrq/whlR, -whlfXMax)
    # compute bike acceleration that results from longitudinal force  
    bkaX[i] = (whlfX[i] + bkDragX[i])/(bkM+whlM)

  # integrate
  # bike longitudinal (bike X and whlX are the same)
  bkvX[i+1]  = bkvX[i]  + bkaX[i] *dt
  bkX[i+1]   = bkX[i]   + bkvX[i] *dt
  
  
  # find track elevation at this time step and next time steo
  trkYt = np.interp(bkX[i],trkX,trkY)
  trkYt2 = np.interp(bkX[i+1],trkX,trkY)
  
  # detect whether the bike and wheel will still be in contact with ground next time step
  if (bkY[i+1] - sT - whlR) > trkYt2:
    inAir[i+1] = True
  
  # detect if wheel will contact the ground this loop
  if inAir[i]:
    # compute wheel free body. a = F/m
    whlaY[i] = (whlM*g - sFk[i] - sFb[i] + sFt[i]) / whlM
    whlvY[i+1] = whlvY[i] + whlaY[i]*dt
    whlY[i+1]  = whlY[i]  + whlvY[i]*dt
    # detect and assert suspension travel extension limits
    if bkY[i+1] - whlY[i+1] > sT:
#      print('extension limit')
      whlY[i+1] = bkY[i+1] - sT
    # detect and assert wheel has cotacted the ground
    if (whlY[i+1]-whlR) < trkYt2:
      whlY[i+1] = trkYt2 + whlR
      inAir[i+1] = False
  else:
    whlY[i+1] = trkYt2 + whlR
      
  # compute suspension travel for next loop
  sTY[i+1] = bkY[i+1] - whlY[i+1]

# plot positioins vs time
#fig1, ax1 = plt.subplots()
#ax1.plot(t,bkY, label='bkY')
#ax1.plot(t,whlY, label='whlY')
#ax1.plot(t,sTY,'g--', label='sTY')
#ax1.legend()
# ax1.set_ylim(0,None)

fig2, ax2 = plt.subplots()
ax2.plot(trkX[0:i],trkY[0:i], label='trk')
ax2.plot(bkX[0:i],bkY[0:i], label='bk')
ax2.plot(bkX[0:i],whlY[0:i], label='whl')
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
fig6, ax6 = plt.subplots()
ax6.plot(t,bkDragX, label='bkDragX')
ax6.legend()
fig7, ax7 = plt.subplots()
ax7.plot(t,whlfX, label='whlfX')
ax7.legend()

# fig9, ax9 = plt.subplots()
# ax9.plot(t,whlAlpha, label='whlAlpha')
# ax9.legend()
fig10, ax10 = plt.subplots()
#ax10.plot(t,whlCntctMthd, label='whlCntctMthd')
ax10.plot(t,inAir, label='inAir')
ax10.legend()
# fig11, ax11 = plt.subplots()
# ax11.plot(t,whlSrMthd, label='whlSrMthd')
# ax11.legend()


# plot suspension forces
fig2, ax2 = plt.subplots()
ax2.plot(t,sFk, label='spring')
ax2.plot(t,sFb, label='damper')
ax2.plot(t,whlfY, label='whlfY')
ax2.legend()

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