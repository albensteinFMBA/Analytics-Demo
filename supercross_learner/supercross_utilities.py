import numpy as np
import matplotlib.pyplot as plt


def max_dict(d):
  # returns the argmax (key) and max (value) from a dictionary
  # put this into a function since we are using it so often
  max_key = None
  max_val = float('-inf')
  for k, v in d.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val


def random_action(a, eps=0.1, a_step=0.01):
  # we'll use epsilon-soft to ensure all states are visited
  # what happens if you don't do this? i.e. eps=0
  ALL_POSSIBLE_ACTIONS = np.arange(0,1,a_step)
  p = np.random.random()
  if p < (1 - eps):
    return a
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)
  
  
class AutoDict(dict):
  def __missing__(self, key):
      x = AutoDict()
      self[key] = x
      return x


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
  
  
def draw_race(env):
  fig306, ax306 = plt.subplots()
  ax306.plot(env.bkX[0:env.i],env.throttle[0:env.i], label='throttle')
  ax306.plot(env.bkX[0:env.i],env.bkY[0:env.i], label='bk')
  ax306.plot(env.trkX[0:env.i],env.trkY[0:env.i], label='trk')
  ax306.legend()
  ax306.grid()
  
  
  fig306, ax306 = plt.subplots()
  ax306.plot(env.t[0:env.i],np.multiply(env.throttle[0:env.i],10), label='throttle')
  ax306.plot(env.t[0:env.i],env.bkX[0:env.i], label='bk')
  ax306.legend()
  ax306.grid()
    