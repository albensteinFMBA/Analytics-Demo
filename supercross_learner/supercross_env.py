# imports
import numpy as np

class supercross_env: 
  # Define constants for environment model
  def __init__(self,trk):
    self.g = -9.81
  
    # wheel
    self.whlR = (19/2+2)*0.0254 # ~0.3meters
    self.whlM = 15
    self.whlJ = 0.25*10
    
    # bike
    self.bkM = (220 - self.whlM + 160) / 2.2
    self.bkPwr = 55 / 0.75
    self.bkTrq = 2*-self.g*self.bkM*self.whlR
    self.bkCrr = 0.015
    self.bkAero = 0.5*1.2*1*0.7 # 1/2*rho*A*Cd
    
    # suspension
    self.sK = 2e4
    self.sB = 1e4
    self.sFbSat=np.abs(self.bkM*-self.g*10)
    self.sC = 3
    self.sT = 0.3
    self.sSag = 0.2
    self.sPL = self.bkM*-self.g - self.sK*self.sT*self.sSag
    
    # tire grip model - nu*Fy
    self.nu = 0.7
    
    # define simulation
    self.t_end = 30
    self.dt = 0.001
    self.t=np.arange(0,self.t_end,self.dt)
  
    # defined track
    # pts = mk_trk2(units='m')
    self.trkStep = 0.05
    self.trkX = np.arange(trk[0,0],trk[0,-1],0.05)
    self.trkY = np.interp(self.trkX,trk[0,:],trk[1,:])
    # some track data for other computation outside the env
    self.trkYmax = np.max(self.trkY)
    self.trkXSampled = self.trkX[0::10]
    self.trkYSampled = self.trkY[0::10]
    #fig12, ax12 = plt.subplots()
    #ax12.plot(trkX,trkY, label='trk')
  
    # defined initial conditions
    self.bkX1 = self.trkX[0]
    self.whlY1 = self.trkY[0]+self.whlR+0.5*0 # add initial positive offset to drop bike at begining
    self.bkY1 = self.whlY1 + self.sT*(1-self.sSag)
    
    # pre-allocate results data vectors
    self.whlY = np.multiply(self.whlY1, np.ones(self.t.size))
    self.whlvY= np.zeros(self.t.size)
    self.whlaY= np.zeros(self.t.size)
    self.bkX  = np.multiply(self.bkX1,  np.ones(self.t.size))
    self.bkY =  np.multiply(self.bkY1,  np.ones(self.t.size))
    self.bkvX = np.zeros(self.t.size)
    self.bkvY = np.zeros(self.t.size)
    self.bkaX = np.zeros(self.t.size)
    self.bkaY = np.zeros(self.t.size)
    self.whlAlpha = np.zeros(self.t.size)
    self.whlW = np.zeros(self.t.size)
    self.inAir = np.zeros(self.t.size)
    self.whlCntctMthd = np.zeros(self.t.size)
    self.whlfY = np.zeros(self.t.size)
    self.whlfX = np.zeros(self.t.size)
    self.sTY  = np.multiply(self.sT,  np.ones(self.t.size))
    self.sFk = np.zeros(self.t.size)
    self.sFb = np.zeros(self.t.size)
    self.sFt = np.zeros(self.t.size)
    self.bkDragX = np.zeros(self.t.size)
    self.bkDragY = np.zeros(self.t.size)
    self.trkYt = np.zeros(self.t.size)
    self.throttle = np.zeros(self.t.size)
    
    # time step tracker
    self.i = 0
    
    # episode tracker
    self.done = False
    self.time = 0
    self.reward = 0
  
  # define physics (i.e. rules) of the environment, and how the agents actions impact the environment
  def step(self,throttle):
    # save action
    self.throttle[self.i] = throttle
    # step ahead in time
    
    # break if time has reached end of preallocated results vectors or out of track length
    if (self.t[self.i] >= self.t_end) or (self.bkX[self.i] >= self.trkX[-1]):
      self.done = True
      self.time = self.t[self.i]
      self.reward = 1
      return
    else:
      # set reward to small negative value to encourage finishing race faster
      self.reward = -self.dt
    
    # compute suspension forces. convention, extension=+ve, tension=-ve
    if self.i > 0:
      if self.inAir[self.i]:
        self.sFk[self.i]=0
        self.sFb[self.i]=0
        self.sFt[self.i]=self.whlM*self.g*0 
      else:
        self.sFk[self.i] = self.sPL + (self.sT - self.sTY[self.i])*self.sK
        self.sFb[self.i] = (self.sTY[self.i-1] - self.sTY[self.i])/self.dt*self.sB
        self.sFb[self.i] = min((self.sFbSat,max(self.sFb[self.i],-self.sFbSat)))
        self.sFt[self.i]=0
    else:
      self.sFk[self.i] = self.sPL + (self.sT - self.sTY[self.i])*self.sK
      self.sFb[self.i]=0
      self.sFt[self.i]=0
    # compute Y component drag forces
    self.bkDragY[self.i] = self.bkAero*self.bkvY[self.i]*self.bkvY[self.i]*np.sign(self.bkvY[self.i])*-1
    # compute bike free body in Y-direction and integrate
    self.bkaY[self.i] = (self.bkM*self.g + self.sFk[self.i] + self.sFb[self.i] + self.sFt[self.i] + self.bkDragY[self.i]) / self.bkM
    self.bkvY[self.i+1]  = self.bkvY[self.i]  + self.bkaY[self.i] * self.dt
    self.bkY[self.i+1]   = self.bkY[self.i]   + self.bkvY[self.i] * self.dt
    
    # find available peak torque
    if self.whlW[self.i] > 0:
      pkTrq = min(self.bkTrq, self.bkPwr/self.whlW[self.i])
    else:
      pkTrq = self.bkTrq
  
    # compute torque command
    cmdTrq = pkTrq*throttle # apply agent command
    
    # compute bike longitudinal free body
    if abs(self.bkvX[self.i]) > 0.01:
      self.bkDragX[self.i] = self.bkAero*self.bkvX[self.i]*self.bkvX[self.i]*np.sign(self.bkvX[self.i])*-1 - self.bkCrr*-self.g*(self.bkM+self.whlM)
    if self.inAir[self.i]:
      self.bkaX[self.i] = self.bkDragX[self.i]/(self.bkM+self.whlM)
    else:
      # compute peak longitidinal force as function of vertical force
      self.whlfY[self.i] = np.array([ self.sFk[self.i] + self.sFb[self.i] + self.whlM*-self.g])
      self.whlfXMax = self.whlfY[self.i]*self.nu
      # compute applied tractive force
      if cmdTrq >= 0:
        self.whlfX[self.i] = min(cmdTrq/self.whlR,  self.whlfXMax)
      else:
        self.whlfX[self.i] = max(cmdTrq/self.whlR, -self.whlfXMax)
      # compute bike acceleration that results from longitudinal force  
      self.bkaX[self.i] = (self.whlfX[self.i] + self.bkDragX[self.i])/(self.bkM+self.whlM)
  
    # integrate
    # bike longitudinal (bike X and whlX are the same)
    self.bkvX[self.i+1]  = self.bkvX[self.i]  + self.bkaX[self.i] * self.dt
    self.bkX[self.i+1]   = self.bkX[self.i]   + self.bkvX[self.i] * self.dt
    
    
    # find track elevation at this time step and next time steo
    self.trkYt[self.i] = np.interp(self.bkX[self.i],self.trkX,self.trkY)
    self.trkYt2 = np.interp(self.bkX[self.i+1],self.trkX,self.trkY)
    
    # detect whether the bike and wheel will still be in contact with ground next time step
    if (self.bkY[self.i+1] - self.sT - self.whlR) > self.trkYt2:
      self.inAir[self.i+1] = True
    
    # detect if wheel will contact the ground this loop
    if self.inAir[self.i]:
      # compute wheel free body. a = F/m
      self.whlaY[self.i] = (self.whlM*self.g - self.sFk[self.i] - self.sFb[self.i] + self.sFt[self.i]) / self.whlM
      self.whlvY[self.i+1] = self.whlvY[self.i] + self.whlaY[self.i] * self.dt
      self.whlY[self.i+1]  = self.whlY[self.i]  + self.whlvY[self.i] * self.dt
      # detect and assert suspension travel extension limits
      if self.bkY[self.i+1] - self.whlY[self.i+1] > self.sT:
        self.whlY[self.i+1] = self.bkY[self.i+1] - self.sT
      # detect and assert wheel has cotacted the ground
      if (self.whlY[self.i+1]-self.whlR) < self.trkYt2:
        self.whlY[self.i+1] = self.trkYt2 + self.whlR
        self.inAir[self.i+1] = False
    else:
      self.whlY[self.i+1] = self.trkYt2 + self.whlR
        
    # compute suspension travel for next loop
    self.sTY[self.i+1] = self.bkY[self.i+1] - self.whlY[self.i+1]
    
    # increment time index
    self.i+=1
    
    return

