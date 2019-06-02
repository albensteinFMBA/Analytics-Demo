# imports
import numpy as np

class supercross_env: 
  # Define constants for environment model
  def __init__(self,trk,height=0.0,endTime=30.0,sK=11e3,sB=2700.0,sSag=0.3,bkXstart=0.0,sKnonLin_flg=True,stateTrkResolution=1,stateTrkVisonDist=30):
    self.g = -9.81
    
    # track
    self.trk = trk
    
    # agent attributes
    self.stateTrkResolution = stateTrkResolution #distance increment between samples for agents view of track ahead
    self.stateTrkVisonDist = stateTrkVisonDist #how far in disatnce the agent sees ahead
    self.stateTrkNumel = np.floor(self.stateTrkVisonDist/self.stateTrkResolution)
    self.stateShape = self.stateTrkNumel + 3
    
    
    # wheel
    self.whlR = (19/2+2)*0.0254 # ~0.3meters
    self.whlM = 15
    self.whlJ = 0.25*10
    
    # bike
    self.bkM = (220 - self.whlM + 160) / 2.2
    self.bkPwr = 35 * 745.7 # 55HP->Watts
    self.bkTrq = 0.8*-self.g*self.bkM*self.whlR
    self.bkCrr = 0.015
    self.bkAero = 0.5*1.2*1*0.7 # 1/2*rho*A*Cd
    
    # suspension
    self.sK = sK # spring constant #https://www.dirtrider.com/features/understanding-your-suspension-sag#page-3, then using 105mm-35mm=70mm, and 170lb ridere => 10800N/m
    self.sB = sB # damping const. compression and rebound. critically damped for sK=11e3N/m, and m=166kg => 2700
    self.sFbSat=np.abs(self.bkM*-self.g*10) # damping force saturation
    self.sT = 0.3 # total travel
    self.sSag = sSag # intial sag as a fraction of sT. aSag=0 means means there is no sag at rest, sSag=1 means the suspension is fully compressed at rest. link used for sK says 33% is good target
    self.sPL = self.bkM*-self.g - self.sK*self.sT*self.sSag # spring preload=force gravity minus the force generated by the target amount of sag
    self.sKnonLin_flg = sKnonLin_flg
    
    # brakes
    self.brkSpdThr = 0.5; # mps, speed threshold below which, brake trq is ramped down to zero
    
    # tire grip model - nu*Fy
    self.nu = 2
    
    # define simulation
    self.t_end = endTime
    self.dt = 0.001
    self.t=np.arange(0,self.t_end,self.dt)
  
    # defined track
    # pts = mk_trk2(units='m')
    self.trkStep = 0.05
    self.trkX = np.arange(trk[0,0],trk[0,-1],0.05)
    self.trkY = np.interp(self.trkX,trk[0,:],trk[1,:])
    self.trkTheta = np.arctan(np.interp(self.trkX,trk[0,:],trk[2,:]))

    # some track data for other computation outside the env
    self.trkYmax = np.max(self.trkY)
    self.trkXSampled = self.trkX[0::10]
    self.trkYSampled = self.trkY[0::10]
  
    # defined initial conditions
    self.bkX1 = self.trkX[0] + bkXstart
    self.whlY1 = self.trkY[0]+self.whlR + height # "+ height" allows user to lift the bike off the ground for a drop test
    if height == 0:
      self.bkY1 = self.whlY1 + self.sT*(1-self.sSag) 
    else:
      # "+ height" allows user to lift the bike off the ground for a drop test
      # no inital sag for drop test
      self.bkY1 = self.whlY1 + self.sT + height       
    
    # pre-allocate results data vectors
    self.whlY = np.multiply(self.whlY1, np.ones(self.t.size))
    self.whlvY= np.zeros(self.t.size)
    self.whlaY= np.zeros(self.t.size)
    self.bkX  = np.multiply(self.bkX1,  np.ones(self.t.size))
    self.bkY =  np.multiply(self.bkY1,  np.ones(self.t.size))
    self.bkvX = np.zeros(self.t.size)
    self.bkvY = np.zeros(self.t.size)
    self.bkv = np.zeros(self.t.size)
    self.bkaX = np.zeros(self.t.size)
    self.bkaY = np.zeros(self.t.size)
    self.whlAlpha = np.zeros(self.t.size)
    self.whlW = np.zeros(self.t.size)
    self.inAir = np.zeros(self.t.size)
    self.whlCntctMthd = np.zeros(self.t.size)
    self.whlfn = np.zeros(self.t.size)
    self.whlft = np.zeros(self.t.size)
    self.whlfY = np.zeros(self.t.size)
    self.whlfX = np.zeros(self.t.size)
    self.bkfY = np.zeros(self.t.size)
    self.bkfX = np.zeros(self.t.size)
    self.sTY  = np.multiply(self.sT,  np.ones(self.t.size))
    self.sFk = np.zeros(self.t.size)
    self.sFb = np.zeros(self.t.size)
    self.sFt = np.zeros(self.t.size)
    self.bkDragX = np.zeros(self.t.size)
    self.bkDragY = np.zeros(self.t.size)
    self.trkYt = np.zeros(self.t.size)
    self.trkThetat = np.zeros(self.t.size)
    self.throttle = np.zeros(self.t.size)
    
    # time step tracker
    self.i = 0
    
    # episode tracker
    self.done = False
    self.time = 0
    self.reward = 0
  
  # Define tensorForce API functions
  def actions():
    # np.arange(-1,1,0.1)
    a = dict(type='float', shape=(), min_value=-1, max_value=1)
    return a
  
  def close(self):
    self.done = True
    return
  
  def execute(self,a):
    self.step(self,a,100) # 100 physics time steps between each agent time step.
    # 1] find trkX index that is stateTrkVisonDist meters ahead
    idx1 = (np.abs(self.trkX - self.bkX[self.i])).argmin()
    idx2 = (np.abs(self.trkX - (self.bkX[self.i]+self.stateTrkVisonDis*1.05))).argmin()
    # 2] get trkX and trkY data points that are stateTrkVisonDist ahead
    # 3] resample to X direction resolution of stateTrkResolution meters
    trkX_samples = np.arange(self.bkX[self.i],(self.bkX[self.i]+self.stateTrkVisonDist),self.stateTrkResolution)
    trkY_samples = np.interp(trkX_samples,self.trkX[idx1:idx2],self.trkY[idx1:idx2])
    # 4] ensure the number of points is equal to stateTrkNumel, crop additional points from furthest ahead
    trkY_samples = trkY_samples[0:self.stateTrkNumel]
    # 5] get all other state variables
    whlElev = (self.whlY[self.i] - self.whlR) - trkY_samples[0]
    bkSpd = self.bkV[self.i]
    sT = self.sTY[self.i]
    # 6] concatenate into a single state vector
    next_state = np.concatenate((trkY_samples, whlElev, bkSpd, sT), axis=0)
    return  next_state, self.done, self.reward
  
  def reset(self):
    self.__init__(self,self.trk)
    return
    
  def states(self):
    # state is vector concatenated of vectors and scalars
    #   [sampled track elevation points for some distance ahead, e.g. 30 meter ahead, sampled every 1 meter ...
    #   distance between wheel and track surface ...
    #   bike speed ...
    #   suspension travel]
    s = dict(shape=(self.stateShape), type='float')
    return s
  
  # define physics (i.e. rules) of the environment, and how the agents actions impact the environment
  def step(self,throttle,n):
    for r in range(n):
      # save action
      self.throttle[self.i] = throttle
      # step ahead in time
      
      # break if time has reached end of preallocated results vectors or out of track length
      if (self.t[self.i]+self.dt >= self.t_end) or (self.bkX[self.i] >= self.trkX[-1]):
        self.done = True
        self.time = self.t[self.i]
        self.reward = 10
        return
      
      # compute suspension forces. convention, extension=+ve, tension=-ve
      if self.sKnonLin_flg:
        if self.sTY[self.i] < self.sT*0.3:
          sK = self.sK + (self.sT - self.sTY[self.i])/self.sT*self.sK*5
        else:
          sK = self.sK
      else:
        sK = self.sK
        
      if self.i > 0:
        if self.inAir[self.i]:
          self.sFk[self.i]=0
          self.sFb[self.i]=0
          self.sFt[self.i]=self.whlM*self.g*0 
        else:
          self.sFk[self.i] = self.sPL + (self.sT - self.sTY[self.i])*sK
          self.sFb[self.i] = (self.sTY[self.i-1] - self.sTY[self.i])/self.dt*self.sB # compute
          self.sFb[self.i] = min((self.sFbSat,max(self.sFb[self.i],-self.sFbSat))) # saturate
          self.sFt[self.i]=0
      else:
        self.sFk[self.i] = self.sPL + (self.sT - self.sTY[self.i])*sK
        self.sFb[self.i]=0
        self.sFt[self.i]=0
      
      # find available peak torque
      if self.whlW[self.i] > 0:
        pkTrq = min(self.bkTrq, self.bkPwr/self.whlW[self.i])
      else:
        pkTrq = self.bkTrq
    
      # compute torque command
      if (throttle < 0) and (self.bkv[self.i] < self.brkSpdThr):
        cmdTrq = pkTrq*throttle*(self.brkSpdThr - self.bkv[self.i])/self.brkSpdThr # apply agent command for braking when speed is below ramp down threshold
      else:
        cmdTrq = pkTrq*throttle # apply agent command for accelerating in forward direction
      
      # compute wheel forces
      # compute track angle at this time instant
      self.trkThetat[self.i] = np.interp(self.bkX[self.i],self.trkX,self.trkTheta)
      if self.inAir[self.i]:
        self.whlfn[self.i]=0
        self.whlft[self.i]=0
        self.whlfX[self.i]=0
        self.whlfY[self.i]=0 # self.whlM*self.g-; this is irrelevant since it could only be used to extend susp when in air, but susp should be extended to leave the groud
      else:
        self.whlfn[self.i]=max(0, (self.whlM*self.g*-1+(self.sFk[self.i] + self.sFb[self.i]))*np.cos(self.trkThetat[self.i])) # normal force cannot have tension, therefore saturate to 0
        self.whlftMax = self.whlfn[self.i]*self.nu
        if cmdTrq >= 0:
          self.whlft[self.i] = min(cmdTrq/self.whlR,  self.whlftMax) + (self.whlM*self.g-(self.sFk[self.i] + self.sFb[self.i]))*np.sin(self.trkThetat[self.i])
        else:
          self.whlft[self.i] = max(cmdTrq/self.whlR,  -self.whlftMax) + (self.whlM*self.g-(self.sFk[self.i] + self.sFb[self.i]))*np.sin(self.trkThetat[self.i])
        self.whlfX[self.i] = -self.whlfn[self.i]*np.sin(self.trkThetat[self.i]) + self.whlft[self.i]*np.cos(self.trkThetat[self.i])
        self.whlfY[self.i] = self.whlfn[self.i]*np.cos(self.trkThetat[self.i]) + self.whlft[self.i]*np.sin(self.trkThetat[self.i])
      
      # compute bike free body in Y-direction and integrate
      # compute Y component drag forces
      self.bkDragY[self.i] = self.bkAero*self.bkvY[self.i]*self.bkvY[self.i]*np.sign(self.bkvY[self.i])*-1
      # HACK: include whlft*sin(theta) to account for vertial acceleration caused by traction on a slope, without going through wheel force balance and suspsension force balance to get the force throuogh from hweel to bike
      self.bkfY[self.i] = (self.bkM*self.g + self.sFk[self.i] + self.sFb[self.i] + self.sFt[self.i] + self.bkDragY[self.i] + self.whlft[self.i]*np.sin(self.trkThetat[self.i]))
      self.bkaY[self.i] = self.bkfY[self.i] / self.bkM 
      self.bkvY[self.i+1]  = self.bkvY[self.i]  + self.bkaY[self.i] * self.dt
      self.bkY[self.i+1]   = self.bkY[self.i]   + self.bkvY[self.i] * self.dt
        
      # compute bike free body in X-direction and integrate
      # compute X component drag forces
      if abs(self.bkvX[self.i]) > 0.01:
        self.bkDragX[self.i] = (self.bkAero*self.bkvX[self.i]*self.bkvX[self.i] - self.bkCrr*-self.g*(self.bkM+self.whlM))*np.sign(self.bkvX[self.i])*-1
      # compute bike acceleration that results from longitudinal force
      if self.inAir[self.i]:
        self.bkfX[self.i] = self.bkDragX[self.i]
        self.bkaX[self.i] = self.bkfX[self.i]/(self.bkM+self.whlM)
      else:
        self.bkfX[self.i] = self.whlfX[self.i] + self.bkDragX[self.i]
        self.bkaX[self.i] = self.bkfX[self.i]/(self.bkM+self.whlM)

      # integrate acceleration for veolcity and position
      # bike longitudinal (bike X and whlX are the same)
      self.bkvX[self.i+1]  = self.bkvX[self.i]  + self.bkaX[self.i] * self.dt
      self.bkX[self.i+1]   = self.bkX[self.i]   + self.bkvX[self.i] * self.dt
      
      # bike speed
      self.bkv[self.i+1] = np.sqrt(np.square(self.bkvX[self.i+1]) + np.square(self.bkvY[self.i+1]))
      # wheel rotational speed
      self.whlW[self.i+1] = self.bkv[self.i+1]/self.whlR
      
      # find track elevation at this time step and next time step
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
      
    # done for loop
    # if not terminated, set reward to small negative value to encourage finishing race faster
    self.reward = -self.dt*n
    
    return

