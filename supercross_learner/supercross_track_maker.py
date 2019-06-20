# generates 2D profiles of supercross track rythm sections
# References:
# 1] https://dirttwister.com/PracticeTracks/Building_A_Practice_Track.html
import numpy as np
import matplotlib.pyplot as plt
import math as math
import random

# all dimension in meters, but input might be in feet.
ft2m = 0.3048
deg2rad = np.pi/180
MINRADIUS = 2/ft2m # X meters converted to feet
DangDxMax = 2*deg2rad # X degrees converted to radians
secLen = 2*MINRADIUS*math.sin(DangDxMax) # feet
secLenNotCurve_ft = 0.2 

def secLen_f(minRad,DangDxMax):
  return 2*minRad*math.sin(DangDxMax) # feet

def convert_units_to_meters(pts):
  pts = np.multiply(ft2m, pts)
  return pts

def mk_jump(face_deg, land_deg, height_ft, flat_ft=0, startX_ft=-1, ctrX_ft=-1, endX_ft=-1, minRadFace_ft=20, minRadTop_ft=3,minRadLand_ft=10,secLenNotCurve_ft=secLenNotCurve_ft):
  # create jump, creates the 3 or 4 set of tuples for X,Y positions for a jump or table
  ptsTmp = np.zeros((2,1000))
  # face curve section is meant to cover from the first inclined section, to the inclined section which is 1*DangDxMax from face_deg
  nFaceCurveSec_raw = face_deg*deg2rad/DangDxMax
  nFaceCurveSec = math.ceil(nFaceCurveSec_raw)
  secLenFace = secLen_f(minRadFace_ft,DangDxMax)
  for i in range(nFaceCurveSec+1): # python doesn't include the last index when indexing arrays
    n=i+1
    ptsTmp[0,i+1] = ptsTmp[0,i] + secLenFace*math.cos(n*DangDxMax)
    ptsTmp[1,i+1] = ptsTmp[1,i] + secLenFace*math.sin(n*DangDxMax)
  faceCurveEnd=i+1 # python doesn't include the last index when indexing arrays, so add one. e.g. to see elements 0,1,2, we need to query for 0:3
  faceCurveHeight = ptsTmp[1,i+1]
  
  i+=1  
  nTopCurveFirstPartSec = math.ceil(face_deg*deg2rad/DangDxMax)
  secLenTop = secLen_f(minRadTop_ft,DangDxMax)
  topCurveHeight = 0
  for k in range(nTopCurveFirstPartSec+1):
    n=k+1
    topCurveHeight += secLenTop*math.sin(n*DangDxMax)

  remHeight = height_ft -  faceCurveHeight - topCurveHeight
  remLength = remHeight/math.sin(face_deg*deg2rad)
  nFaceSec = math.ceil(remLength/secLenNotCurve_ft)
  for j in range(nFaceSec+1):
    i = j + faceCurveEnd-1
    ptsTmp[0,i+1] = ptsTmp[0,i] + secLenNotCurve_ft*math.cos(face_deg*deg2rad)
    ptsTmp[1,i+1] = ptsTmp[1,i] + secLenNotCurve_ft*math.sin(face_deg*deg2rad)
 
  i+=1 
  
  iOffset = i
  nTopCurveSec = math.ceil((face_deg+land_deg)*deg2rad/DangDxMax)
  iOffsetFlat=0
  flatCreated=0
  if flat_ft > 0:
    nFlatSec = math.ceil(flat_ft/secLenNotCurve_ft)
  for j in range(nTopCurveSec+1): # python doesn't include the last index when indexing arrays,
    i = j + iOffset + iOffsetFlat
    ptsTmp[0,i+1] = ptsTmp[0,i] + secLenTop*math.cos(face_deg*deg2rad - j*DangDxMax)
    ptsTmp[1,i+1] = ptsTmp[1,i] + secLenTop*math.sin(face_deg*deg2rad - j*DangDxMax)
    if flat_ft > 0 and (face_deg*deg2rad - j*DangDxMax) == 0 and flatCreated == 0:
      flatCreated = 1
      for k in range(nFlatSec+1):
        i = j + iOffset + iOffsetFlat
        ptsTmp[0,i+1] = ptsTmp[0,i] + secLenNotCurve_ft
        ptsTmp[1,i+1] = ptsTmp[1,i]
        iOffsetFlat +=1
      iOffsetFlat -=1
  i+=1 # increment i to catch the last i+1 element written to ptsTmp
  topCurveEnd=i+1 # python doesn't include the last index when indexing arrays,
  nLandCurveSec = math.ceil(land_deg*deg2rad/DangDxMax)
  secLenLand = secLen_f(minRadLand_ft,DangDxMax)
  LandCurveHeight = 0
  for j in range(nLandCurveSec+1):
    n=j+1
    LandCurveHeight += secLenLand*math.sin(n*DangDxMax)
  remHeight = ptsTmp[1,topCurveEnd-1] - LandCurveHeight
  remLength = remHeight/math.sin(land_deg*deg2rad)
  nLandSec = math.ceil(remLength/secLenNotCurve_ft)
  for j in range(nLandSec+1):
    i = j + topCurveEnd-1
    ptsTmp[0,i+1] = ptsTmp[0,i] + secLenNotCurve_ft*math.cos(land_deg*deg2rad)
    ptsTmp[1,i+1] = ptsTmp[1,i] - secLenNotCurve_ft*math.sin(land_deg*deg2rad)
  
  i+=1
  landEnd=i+1
  
  for j in range(nLandCurveSec+1):
    i = j + landEnd-1
    n=j+1*0
    ptsTmp[0,i+1] = ptsTmp[0,i] + secLenLand*math.cos(land_deg*deg2rad - n*DangDxMax)
    ptsTmp[1,i+1] = ptsTmp[1,i] - secLenLand*math.sin(land_deg*deg2rad - n*DangDxMax)
  landCurveEnd=i+1
  
  # translatee the landCurve so that the last point is at y=0, and first point is one the line of landFace
  yErr = ptsTmp[1,landCurveEnd-1]
  xErr = yErr/math.tan(land_deg*deg2rad)
  errStartDelta = 1
  errEndDelta = 0
  ptsTmp[0,landEnd-errStartDelta:landCurveEnd-errEndDelta] = np.add(ptsTmp[0,landEnd-errStartDelta:landCurveEnd-errEndDelta],xErr)
  ptsTmp[1,landEnd-errStartDelta:landCurveEnd-errEndDelta] = np.subtract(ptsTmp[1,landEnd-errStartDelta:landCurveEnd-errEndDelta],yErr)
#  ptsTmp[1,landCurveEnd-1] = 0
  
  pts=ptsTmp[:,0:landCurveEnd]
  
  if startX_ft > 0:
    pts[0,:] = pts[0,:] + startX_ft
  elif ctrX_ft > 0:
    pts[0,:] = pts[0,:] + ctrX_ft - pts[0,-1]/2
  elif endX_ft > 0:
    pts[0,:] = pts[0,:] + endX_ft - pts[0,-1]
  
  return pts

def mk_flat(startX_ft=0,endX_ft=100,len_ft=-1):
  if len_ft > -1:
    ptsX = np.arange(startX_ft,(startX_ft+len_ft),secLenNotCurve_ft)
  else:
    ptsX = np.arange(startX_ft,endX_ft,secLenNotCurve_ft)
  pts = np.zeros((2,ptsX.size))
  pts[0,:] = ptsX
  return pts

def mk_trpl(startX_ft=0,gap_ft=60):
  #  create a triple jump, with start, and gap specified
  pts1 = mk_jump(30, 30, 6, minRadFace_ft=16, minRadTop_ft=2,minRadLand_ft=10)
  pts1[0,:] += startX_ft
  jump1PkX = np.mean(pts1[0,:])
  pts3 = mk_jump(30, 10, 3, ctrX_ft=(jump1PkX+gap_ft),minRadFace_ft=6, minRadTop_ft=3,minRadLand_ft=10)
  pts2 = mk_jump(30, 20, 4.5, ctrX_ft=(pts1[0,-1]+(pts3[0,0]-pts1[0,-1])/2),minRadFace_ft=6, minRadTop_ft=3,minRadLand_ft=10)
  ptsFlat1 = mk_flat(pts1[0,-1]+secLenNotCurve_ft,pts2[0,0])
  ptsFlat2 = mk_flat(pts2[0,-1]+secLenNotCurve_ft,pts3[0,0])
  pts = np.concatenate((pts1, ptsFlat1, pts2, ptsFlat2, pts3), axis=1)
  
  return pts

def mk_table(startX_ft=0,flat_ft=20):
  pts = mk_jump(20,20,5,flat_ft=flat_ft,startX_ft=startX_ft, minRadFace_ft=10, minRadTop_ft=3,minRadLand_ft=15)
  return pts

def mk_onoff(startX_ft=0,gap1_ft=6,gap2_ft=1):
#  startX_ft=0
#  gap_ft = 10
  pts1 = mk_jump(20,30,4.5, minRadFace_ft=20, minRadTop_ft=1.5,minRadLand_ft=10)
  pts1[0,:] += startX_ft
  pts2 = mk_jump(20,20,5,flat_ft=20, minRadFace_ft=10, minRadTop_ft=3,minRadLand_ft=15)
  pts2[0,:] = pts2[0,:] + pts1[0,-1] + gap1_ft
  pts3 = mk_jump(30,10,3, minRadFace_ft=5, minRadTop_ft=4,minRadLand_ft=15)
  pts3[0,:] = pts3[0,:] + pts2[0,-1] + gap2_ft
  ptsFlat1 = mk_flat(pts1[0,-1]+secLen,pts2[0,0])
  ptsFlat2 = mk_flat(pts2[0,-1]+secLen,pts3[0,0])
  pts = np.concatenate((pts1, ptsFlat1, pts2, ptsFlat2, pts3), axis=1)
  
  return pts

def addTrkGrad(pts):
  grd = np.gradient(pts[1,:],pts[0,:])
  pts = np.stack((pts[0,:],pts[1,:],grd),axis=0)
  return pts
  

def mk_trk1(units='ft'): #just a triple jump
  pts1 = mk_flat(endX_ft=120)
  pts2 = mk_trpl(startX_ft=(pts1[0,-1]+secLen))
  pts3 = mk_flat(startX_ft=(pts2[0,-1]+secLen),len_ft=30)
  
  pts = np.concatenate((pts1, pts2, pts3), axis=1)
  if units == 'm':
    pts = convert_units_to_meters(pts)
  
  pts = addTrkGrad(pts)
  
  return pts

def mk_trk2(units='ft'): #in run, triple, onoff, triple
  pts1 = mk_flat(endX_ft=60)
  pts2 = mk_trpl(startX_ft=(pts1[0,-1]+secLen))
  pts3 = mk_flat(startX_ft=(pts2[0,-1]+secLen),len_ft=20)
  pts4 = mk_onoff(startX_ft=(pts3[0,-1]+secLen))
  pts5 = mk_flat(startX_ft=(pts4[0,-1]+secLen),len_ft=20)
  pts6 = mk_trpl(startX_ft=(pts5[0,-1]+secLen))
  pts7 = mk_flat(startX_ft=(pts6[0,-1]+secLen),len_ft=30)
  
  pts = np.concatenate((pts1, pts2, pts3, pts4, pts5, pts6, pts7), axis=1)
  if units == 'm':
    pts = convert_units_to_meters(pts)
  
  pts = addTrkGrad(pts)
  
  return pts

def mk_trkAccel(units='ft',endX_ft=2000):
  pts = mk_flat(startX_ft=0,endX_ft=endX_ft,len_ft=-1)
  
  if units == 'm':
    pts = convert_units_to_meters(pts)
    
  pts = addTrkGrad(pts)
    
  return pts

def mk_trkRand(endX_m=800):
  # feature encoding: 
  feature_list = [1,2,3]
    # 0.X=flat of length X*2 meters, X limits [1,3]
    # 1.X=table top, with top length of X meters, X limits [3,6]
    # 2.X=on/off, with X changing nothing, X limits [0]
    # 3.X=triple jump, with X changing nothing, X limits [0]
    # 4.X=whoops, with X*3 meter length of whoops section, X limits [2,6]
  flatChar_list = [1,2,3]
  featureChar_dict = {0:[1,2,3],1:[3,4,5,6],2:[0],3:[0],4:[2,3,4,5,6]}
  # create a feature encoding that results in endX_m, by getting a sequence of random number that respect the limits. 0-4.[see limits for each]
  # use feature encoding to generate track profile
  featureEncoding = []
  pts = mk_flat(len_ft=50)
  chk = False
  while_cnt = 0
  while not chk:
    feature_tmp = random.choice(feature_list)
    characteristic_tmp = random.choice(featureChar_dict[feature_tmp])
    val = float(feature_tmp + characteristic_tmp/10)
    featureEncoding.append(val)
    startX_ft = pts[0,-1] + 1
#    print('startX_ft={val}'.format(val=startX_ft))
    if feature_tmp == 1:
      pts_tmp = mk_table(startX_ft=startX_ft,flat_ft=characteristic_tmp/ft2m)
    elif feature_tmp == 2:
      pts_tmp = mk_onoff(startX_ft=startX_ft)
    elif feature_tmp == 3:
      pts_tmp = mk_trpl(startX_ft=startX_ft)
    elif feature_tmp == 4:
      pass # whoop feature function does exist yet
    
    flatChar_tmp = random.choice(flatChar_list)
    val = float(0 + flatChar_tmp/10)
    featureEncoding.append(val)
    startX_ft = pts_tmp[0,-1] + 1
    pts_flat = mk_flat(startX_ft=startX_ft,len_ft=flatChar_tmp*2/ft2m)
    pts = np.concatenate((pts, pts_tmp, pts_flat), axis=1)
  
    chk = pts[0,-1] >= endX_m/ft2m
    if pts[0,-1] >= endX_m/ft2m:
      print('break due to track length')
      break
    while_cnt += 1
    if while_cnt >=8:
      print('break due to loop count')
      break
    
  # add gradient
  pts = convert_units_to_meters(pts) 
  pts = addTrkGrad(pts)
  # package track and encoding in np array to be passed to supercross_env
#  _,trkLen = pts.shape()
  print('featureEncoding={d}'.format(d=featureEncoding))
  return pts, featureEncoding
    

if __name__ == '__main__':
  ang=30
  face_deg = 30
  land_deg = 10
  height_ft = 10
  flat_ft = 0
  ctrX_ft = -1
  startX_ft = -1
  endX_ft = -1
  
#  pts = mk_jump(30, 30, 6, minRadFace_ft=15, minRadTop_ft=3,minRadLand_ft=10)
#  pts = mk_jump(face_deg, land_deg, height_ft)
  pts = mk_trpl()
#  pts = mk_onoff()
#  pts = mk_trk1()
#  pts = mk_trk2()
#  pts = mk_trk1(units='m')
#  pts, featureEncoding = mk_trkRand()
  
  
  
#  print(ptsTmp)
#  print(pts)
  fig5, ax5 = plt.subplots()
#  ax5.plot(pts[0,0:faceCurveEnd],pts[1,0:faceCurveEnd],'ro', label='faceCurve')
#  ax5.plot(pts[0,faceCurveEnd:faceEnd],pts[1,faceCurveEnd:faceEnd],'mo', label='face')
#  ax5.plot(pts[0,topCurveStart:topCurveEnd],pts[1,topCurveStart:topCurveEnd],'bo', label='topCurve')
#  ax5.plot(pts[0,topCurveEnd:landEnd],pts[1,topCurveEnd:landEnd],'co', label='land')
#  ax5.plot(pts[0,landEnd:landCurveEnd],pts[1,landEnd:landCurveEnd],'go', label='landCurve')
  ax5.plot(pts[0,:],pts[1,:],label='ft')
#  ax5.plot(pts[0,:],pts[2,:],label='grd')
#  ax5.plot(pts_m[0,:],pts_m[1,:],'k+',label='m')
  ax5.grid()
  ax5.legend()
  print(pts[1,-1])
    
    
#  pts1 = mk_jump(30, 10, 6)
#  fig6, ax6 = plt.subplots()
#  ax6.plot(pts1[0,:],pts1[1,:], label='jump')
#  ax6.legend()
#  
#  pts2 = mk_jump(30, 10, 6, ctrX_ft=10)
#  fig7, ax7 = plt.subplots()
#  ax7.plot(pts2[0,:],pts2[1,:], label='jumpCtr')
#  ax7.legend()
#  
#  pts3 = mk_jump(30, 10, 6, flat_ft=10)
#  fig8, ax8 = plt.subplots()
#  ax8.plot(pts3[0,:],pts3[1,:], label='table')
#  ax8.legend()
#  
#  pts4 = mk_jump(30, 10, 6, flat_ft=10, ctrX_ft=10)
#  fig9, ax9 = plt.subplots()
#  ax9.plot(pts4[0,:],pts4[1,:], label='tableCtr')
#  ax9.legend()
  
  #mk_trpl(startX=0,gap_ft=60)
#  ptst60 = mk_trpl()
#  fig10, ax10 = plt.subplots()
#  ax10.plot(ptst60[0,:],ptst60[1,:], label='triple60ft')
#  ax10.legend()
#  
#  ptst75 = mk_trpl(gap_ft=75)
#  fig11, ax11 = plt.subplots()
#  ax11.plot(ptst75[0,:],ptst75[1,:], label='triple75ft')
#  ax11.legend()

#  ptsOnOff = mk_onoff()
#  
#  fig12, ax12 = plt.subplots()
#  ax12.plot(ptsOnOff[0,:],ptsOnOff[1,:], label='onoff')
#  ax12.legend()
  
  
#  pts = mk_trk1()
  

  
#  pts2x = np.arange(pts[0,0],pts[0,-1],0.05)
#  pts2y = np.interp(pts2x,pts[0,:],pts[1,:])
#  
#  slp = np.gradient(pts2y,pts2x)
#  DSlpDx = np.gradient(slp,pts2x)
#  DSlpDxLim = np.clip(DSlpDx,-5,5)
#  slpLim = integrate.cumtrapz(DSlpDxLim,pts2x)
#  print(pts2x[0:-1].shape)
#  print(slpLim.shape)
#  pts2yLim = integrate.cumtrapz(slpLim,pts2x[0:-1])
#  
#  fig12, ax12 = plt.subplots()
#  ax12.plot(pts[0,:],pts[1,:], label='pts')
#  ax12.plot(pts2x[0:-2],pts2yLim, label='lim')
#  ax12.legend()
#  applyMinRad
  # https://stackoverflow.com/questions/22954078/formula-to-draw-arcs-ending-in-straight-lines-y-as-a-function-of-x-starting-sl/22982623#22982623
#  ang = np.zeros(pts.shape[1]-2)
#  print(range(pts.shape[1]-2))
#  for i in range(pts.shape[1]-2):
#  i=0
#  ax=pts[0,i]
#  ay=pts[1,i]
#  bx=pts[0,i+2]
#  by=pts[1,i+2]
#  p0x=pts[0,i+1]
#  p0y=pts[1,i+1]
#  
#  unitapomag = np.sqrt(np.square(p0x-ax) + np.square(p0y-ay))
#  normap0y = (p0x - ax)/unitapomag*MINRADIUS
#  normap0x = (p0y - ay)/unitapomag*MINRADIUS
#  
#  normap0Mag = np.sqrt(np.square(normap0x) + np.square(normap0y))
#  print(normap0Mag)
#    
#  ax2=ax+normap0x
#  ay2=ay+normap0y
#  p0x2a=p0x+normap0x
#  p0y2a=p0y+normap0y
#  
#  unitbpomag = np.sqrt(np.square(p0x-bx) + np.square(p0y-by))
#  print(unitbpomag)
#  normbp0y = (p0x - bx)/unitbpomag*MINRADIUS
#  normbp0x = (p0y - by)/unitbpomag*MINRADIUS
#  print(normbp0y)
#  print(normbp0x)
#  normbp0Mag = np.sqrt(np.square(normbp0x) + np.square(normbp0y))
#  print(normbp0Mag)
#    
#  bx2=bx+normbp0x
#  by2=by+normbp0y
#  p0x2b=p0x+normbp0x
#  p0y2b=p0y+normbp0y
#  
#  fig13, ax13 = plt.subplots()
#  ax13.scatter(ax,ay, label='a')
#  ax13.scatter(bx,by, label='b')
#  ax13.scatter(p0x,p0y, label='p0')
#  ax13.scatter(ax2,ay2, label='a2')
#  ax13.scatter(bx2,by2, label='b2')
#  ax13.scatter(p0x2a,p0y2a, label='p02a')
#  ax13.scatter(p0x2b,p0y2b, label='p02b')
#  ax13.legend()
  
  
#    ang[i] = (math.atan2(pts[1,i+2]-pts[1,i],pts[0,i+2]-pts[0,i])-math.atan2(pts[1,i+1]-pts[1,i],pts[0,i+1]-pts[0,i]))/deg2rad
  
#  print(ang)
#  print(pts)
# create track features, e.g. double, triple, table, reverse ski-jump, etc.
# each track feature is a set of 3-tuples for X,Y points, and min radius

# create functions for "standard" tracks, e.g. track 1, 2, 3, etc.
# create function for generating random tracks, input is max length, and min/max diantce between features