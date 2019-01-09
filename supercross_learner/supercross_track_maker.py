# generates 2D profiles of supercross track rythm sections
# References:
# 1] https://dirttwister.com/PracticeTracks/Building_A_Practice_Track.html
import numpy as np
import matplotlib.pyplot as plt
import math as math
from scipy import integrate

# all dimension in meters, but input might be in feet.
ft2m = 0.3048*0+1
deg2rad = np.pi/180
MINRADIUS = 1

def mk_jump(face_deg, land_deg, height_ft, flat_ft=0, ctrX_ft=-1):
  # create jump, creates the 3 or 4 set of tuples for X,Y positions for a jump or table
  # test inputs
#  face_deg = 30
#  land_deg = 30
#  height_ft = 6
#  flat_ft = 10
  ctrX_m = ctrX_ft*ft2m
  height_m = height_ft*ft2m + MINRADIUS
  if flat_ft == 0:
    pts = np.zeros((2,3))
  else:
    pts = np.zeros((2,4))
  pts[0,:] = 0
  pts[1,1] = height_m
  pts[0,1] = height_m/np.tan(face_deg*deg2rad)
  if flat_ft != 0:
    pts[0,2] = pts[0,1] + flat_ft*ft2m
    pts[1,2] = pts[1,1]
  pts[0,-1] = pts[0,-2] + pts[1,-2]/np.tan(land_deg*deg2rad)
  pts[1,-1] = 0
  if ctrX_ft != -1:
    if flat_ft == 0:
      pts[0,0] = ctrX_m - pts[0,1]
      pts[0,2] = ctrX_m + pts[0,2] - pts[0,1]
      pts[0,1] = ctrX_m
    else:
      pts[0,0] = ctrX_m - flat_ft/2*ft2m - pts[0,1]
      pts[0,3] = ctrX_m + flat_ft/2*ft2m + (pts[0,-1] - pts[0,-2])
      pts[0,1] = ctrX_m - flat_ft/2*ft2m
      pts[0,2] = ctrX_m + flat_ft/2*ft2m
  
  return pts

def mk_trpl(startX_ft=0,gap_ft=60):
  #  create a triple jump, with start, and gap specified
  pts1 = mk_jump(30, 30, 6)
  pts1[0,:] += startX_ft
  pts3 = mk_jump(30, 10, 3, ctrX_ft=(pts1[0,1]+gap_ft)) 
  pts2 = mk_jump(30, 20, 4.5, ctrX_ft=(pts1[0,2]+(pts3[0,0]-pts1[0,2])/2))
  pts = np.concatenate((pts1, pts2, pts3), axis=1)
  
  return pts

def mk_onoff(startX_ft=0,gap_ft=60):
  startX_ft=0
  gap_ft = 10
  pts1 = mk_jump(30,30,3)
  pts1[0,:] += startX_ft
  pts2 = mk_jump(20,20,4,flat_ft=15)
  pts2[0,:] = pts2[0,:] + pts1[0,2] + gap_ft
  pts3 = mk_jump(30,20,3)
  pts3[0,:] = pts3[0,:] + pts2[0,3] + gap_ft
  pts = np.concatenate((pts1, pts2, pts3), axis=1)
  
  return pts


def mk_trk1(): #just a triple jump
  pts1 = np.array([[0],[0]])
  pts2 = mk_trpl()
  pts2[0,:] = pts2[0,:] + 30
  pts3 = np.array([[30],[0]])
  pts3[0,:] = pts3[0,:] + pts2[0,-1]
  
  pts = np.concatenate((pts1, pts2, pts3), axis=1)
  
  return pts

if __name__ == '__main__':
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
  
  
  pts = mk_trk1()
  

  
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
  
  fig12, ax12 = plt.subplots()
  ax12.plot(pts[0,:],pts[1,:], label='pts')
#  ax12.plot(pts2x[0:-2],pts2yLim, label='lim')
  ax12.legend()
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