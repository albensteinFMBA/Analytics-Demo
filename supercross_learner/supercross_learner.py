# imports
import numpy as np
#import math as math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from supercross_track_maker import mk_trk1

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
sC = 3
sT = 0.3
sSag = 0.2
sPL = bkM*-g - sK*sT*sSag

# tire grip model - nu*Fy
nu = 0.7

# tire contact model 
contactCloseEnough = 0.005 # 5mm tolerance is considered "close enough" for finding the true contact spacial position

# define simulation
dt = 0.01
t=np.arange(0,20,dt)

# defined track
pts = mk_trk1(units='m')
trkStep = 0.05
trkX = np.arange(pts[0,0],pts[0,-1],0.05)
trkY = np.interp(trkX,pts[0,:],pts[1,:])
fig12, ax12 = plt.subplots()
ax12.plot(trkX,trkY, label='trk')

#trkL = 30
#trkX = np.arange(0,trkL,trkStep)
#trkY = np.multiply(0, np.ones(trkX.size))
#trkY[150:160] = 0.05

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
  
  # find track elevation at time step
  trkYt = np.interp(bkX[i],trkX,trkY)

    

  if (bkX[i] < whlR) or (bkX[i] > (trkX[-1]-whlR)):
    # detect and assert wheel contact with track using single point method
    whlCntctMthd[i] = 1
    if (whlY[i] - whlR - trkYt) <= 0:  
      whlY[i] = trkYt + whlR
      inAir[i] = False
    else:
      inAir[i] = True  
  else:
    whlCntctMthd[i] = 2
    # detect and assert wheel contact with track using multi-point root finding method
    iterations=0  
     # find all track point which may be in contact with wheel, e.g. all point 1 wheel radius in front, and all points 1 wheel radius behind
      
    trkX_idxMin = (np.abs(trkX - (bkX[i]-whlR))).argmin()
    trkX_idxMax = (np.abs(trkX - (bkX[i]+whlR))).argmin()
    if trkX_idxMin >= trkX_idxMax:
      print(" trkX_idxMin=",trkX_idxMin,'trkX_idxMax=',trkX_idxMax,'bkX[i]=',bkX[i],'trkX=',trkX, end="")
    
    xDistancesSquared = np.square(np.subtract(bkX[i],trkX[np.arange(trkX_idxMin, trkX_idxMax)])) # we will reused these several times, so compute only once
    dMin = (np.amin(np.sqrt(xDistancesSquared
           + np.square(np.subtract(whlY[i],trkY[np.arange(trkX_idxMin, trkX_idxMax)])))) - whlR)
    if dMin < contactCloseEnough:
      # if the shortest distance between the wheel center and a nearby track point is smaller than (wheel radius)-(contactCloseEnough), 
      # then the wheel is in contact with the track, and we must find the new whlY[i] that satisfies our "contactCloseEnough" criteria
      inAir[i] = False
      
      # define wheel Y positions and compute their distance to initialze bisection method inputs
      whlY_tmpHi = bkY[i]+whlR/2
      whlY_tmpLo = whlY[i]
      whlY_tmpMed = np.mean(np.array([whlY_tmpHi,whlY_tmpLo]))
    
      dMinHi = (np.amin(np.sqrt(xDistancesSquared
           + np.square(np.subtract(whlY_tmpHi,trkY[np.arange(trkX_idxMin, trkX_idxMax)])))) - whlR)
      dMinMed = (np.amin(np.sqrt(xDistancesSquared
           + np.square(np.subtract(whlY_tmpMed,trkY[np.arange(trkX_idxMin, trkX_idxMax)])))) - whlR)
      dMinLo = dMin
#      print(" Hi=",whlY_tmpHi,'Med=',whlY_tmpMed,'Lo=',whlY_tmpLo,'dMinHi=',dMinHi,'dMinMed=',dMinMed,'dMinLo=',dMinLo, end="")
#      print("\n")
      if dMinHi < contactCloseEnough:
          # if the suspension is bottomed out, and then whlY[i] = bkY[i], not great, 
          # but we're hoping this resolves itself next time step. 
          # if this is a regular occurence over multiple consecutive time steps, we'll need a better solution
          whlY[i] = whlY_tmpHi
          bkY[i] = whlY_tmpHi
          print('ERROR: wheel too far penetrated into track at step')
          print(i)
      elif np.abs(dMinMed) <= contactCloseEnough:
          # the inital med position is ok, all done
          whlY[i] = whlY_tmpMed
      else:
        # apply bisection method to find acceptable whlY[i]
        iterations=0
        while True:
          iterations+=1
          if dMinMed < 0:
            # the med point results in too much penetration between wheel and track
            # repeat search between Hi and Med
            whlY_tmpLo = whlY_tmpMed
            whlY_tmpMed = np.mean(np.array([whlY_tmpHi,whlY_tmpLo]))
          else:
            # the med point results in no contact between wheel and track
            # repeat search between Med and Lo
            whlY_tmpHi = whlY_tmpMed
            whlY_tmpMed = np.mean(np.array([whlY_tmpHi,whlY_tmpLo]))
          
          dMinMed = (np.amin(np.sqrt(xDistancesSquared
           + np.square(np.subtract(whlY_tmpMed,trkY[np.arange(trkX_idxMin, trkX_idxMax)])))) - whlR)
          if np.abs(dMinMed) <= contactCloseEnough or iterations > 30:
            # whlY_tmpMed is acceptably in contact with track
            whlY[i] = whlY_tmpMed
            break
    else:
      inAir[i] = True

  # detect and assert suspension travel compression and extension limits
  sTY[i] = bkY[i] - whlY[i]

  # compute suspension forces. convention, extension=+ve, tension=-ve
  sFk[i] = sPL + (sT - sTY[i])*sK
  if i > 0:
    sFb[i] = (sTY[i-1] - sTY[i])/dt*sB
  else:
    sFb[i] = 0
  if inAir[i]:
    sFt[i] = whlM*g 
  # compute Y component drag forces
  bkDragY[i] = bkAero*bkvY[i]*bkvY[i]*np.sign(bkvY[i])*-1
  # compute bike free body in Y-direction and integrate
  bkaY[i] = (bkM*g + sFk[i] + sFb[i] - sFt[i] + bkDragY[i]) / bkM
  bkvY[i+1]  = bkvY[i]  + bkaY[i] *dt
  bkY[i+1]   = bkY[i]   + bkvY[i] *dt
  
  # integrate
  # wheel
  if inAir[i]:
    # compute wheel free body. a = F/m
    whlaY[i] = (whlM*g - sFk[i] - sFb[i] + sFt[i]) / whlM
    whlvY[i+1] = whlvY[i] + whlaY[i]*dt
    whlY[i+1]  = whlY[i]  + whlvY[i]*dt
    # detect and assert suspension travel extension limits
    if bkY[i+1] - whlY[i+1] > sT:
      print('extension limit')
      whlY[i+1] = bkY[i+1] - sT

  # find available peak torque
  if whlW[i] > 0:
    pkTrq = min(bkTrq, bkPwr/whlW[i])
  else:
    pkTrq = bkTrq

  # compute torque command
  cmdTrq = pkTrq*0.1 # no agent yet, apply fraction of peak torque
  
  # compute wheel rotational free body and bike longitudinal free body
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

# plot positioins vs time
fig1, ax1 = plt.subplots()
ax1.plot(t,bkY, label='bkY')
ax1.plot(t,whlY, label='whlY')
ax1.plot(t,sTY, label='sTY')
ax1.legend()
# ax1.set_ylim(0,None)

fig2, ax2 = plt.subplots()
ax2.plot(trkX[0:i],trkY[0:i], label='trk')
ax2.plot(bkX[0:i],bkY[0:i], label='bk')
ax2.plot(bkX[0:i],whlY[0:i], label='whl')
ax2.legend()

# plot velocity vs time
fig5, ax5 = plt.subplots()
ax5.plot(bkX[0:i],bkvX[0:i], label='bkvX vs bkX')
ax5.plot(bkX[0:i],bkvY[0:i], label='bkvY vs bkX')
ax5.legend()
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
ax10.plot(t,whlCntctMthd, label='whlCntctMthd')
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
fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.plot(bkX,t,bkY, label='bike')
ax4.plot(bkX,t,whlY, label='wheel')
ax4.legend()
ax4.set_ylim(0,None)
# rotate the axes and update
# for angle in range(0, 360):
#     ax4.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)