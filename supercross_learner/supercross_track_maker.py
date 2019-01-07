# generates 2D profiles of supercross track rythm sections
# References:
# 1] https://dirttwister.com/PracticeTracks/Building_A_Practice_Track.html
import numpy as np
import matplotlib.pyplot as plt


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




if __name__ == '__main__':
  pts1 = mk_jump(30, 10, 6)
  fig6, ax6 = plt.subplots()
  ax6.plot(pts1[0,:],pts1[1,:], label='jump')
  ax6.legend()
  
  pts2 = mk_jump(30, 10, 6, ctrX_ft=10)
  fig7, ax7 = plt.subplots()
  ax7.plot(pts2[0,:],pts2[1,:], label='jumpCtr')
  ax7.legend()
  
  pts3 = mk_jump(30, 10, 6, flat_ft=10)
  fig8, ax8 = plt.subplots()
  ax8.plot(pts3[0,:],pts3[1,:], label='table')
  ax8.legend()
  
  pts4 = mk_jump(30, 10, 6, flat_ft=10, ctrX_ft=10)
  fig9, ax9 = plt.subplots()
  ax9.plot(pts4[0,:],pts4[1,:], label='tableCtr')
  ax9.legend()
  
  #mk_trpl(startX=0,gap_ft=60)
  ptst60 = mk_trpl()
  fig10, ax10 = plt.subplots()
  ax10.plot(ptst60[0,:],ptst60[1,:], label='triple60ft')
  ax10.legend()
  
  ptst75 = mk_trpl(gap_ft=75)
  fig11, ax11 = plt.subplots()
  ax11.plot(ptst75[0,:],ptst75[1,:], label='triple75ft')
  ax11.legend()

  #mk_onoff
  flat_ft = 15
  gap_ft = 10
  

# create track features, e.g. double, triple, table, reverse ski-jump, etc.
# each track feature is a set of 3-tuples for X,Y points, and min radius

# double
#trpl = np.array([])
# create functions for "standard tracks, e.g. track 1, 2, 3, etc.
# create function for generating 