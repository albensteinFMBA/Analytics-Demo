# imports
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import get_outliers_inliers
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.sod import SOD
from pyod.models.feature_bagging import FeatureBagging as FB

from skopt import gp_minimize 


objValCase = 3
glb_verbose = False

def Rprecision_f(Y_train, scores_pred):
  # compute R-precision
  idxTrue = np.where(Y_train == 1) # get indices of true anomalies
  t = np.count_nonzero(Y_train == 1) # get count of true anomalies
  idxPred = np.argpartition(scores_pred, -t)[-t:] # get indices of top t predicted anomalies
  intersect = np.intersect1d(idxTrue,idxPred) # get intersection between the 2 indices sets
  Rprecision = len(intersect)/t 
  return Rprecision

def objVal_f(Rprecision,y_pred,Y_train):
  # define different option of objVal
  if objValCase == 1  :
    objVal = Rprecision*-1 # *-1 ause larger Rprecision is better, but gp_minimize seeks to minimize the objVal 
  elif objValCase == 2:
    objVal = (y_pred != Y_train).sum() # Num of Errors in prediction
  elif objValCase == 3:
    objVal = 1 - Rprecision # Num of Errors in prediction
  else:
    objVal = Rprecision*-1
  return objVal

def obj_func_kNN(params):
  ## objective function used in baseian optimization
  outlier_fraction=params[0]
  n_neighbors=params[1]
  method=params[2]
  radius=params[3]
  
  # load data set to function work space
  Y_train = np.load('Y_train.npy')
  X_train = np.load('X_train.npy')
    
  # create model  
  clf = KNN(contamination=outlier_fraction, n_neighbors=n_neighbors,method=method,radius=radius);
  # fit the dataset to the model
  clf.fit(X_train)
  
  scores_pred = clf.decision_function(X_train)*-1 # predict raw anomaly score
  Rprecision = Rprecision_f(Y_train, scores_pred)
  if glb_verbose:
    print('R Precision : ',Rprecision)
  
  y_pred = clf.predict(X_train) # prediction of a datapoint category outlier or inlier
  objVal = objVal_f(Rprecision,y_pred,Y_train)
  
  return objVal

def obj_func_LOF(params):
  ## objective function used in baseian optimization
  outlier_fraction=params[0]
  n_neighbors=params[1]
  algorithm=params[2]
  leaf_size=params[3]
  
  # load data set to function work space
  Y_train = np.load('Y_train.npy')
  X_train = np.load('X_train.npy')
    
  # create model  
  clf = LOF(n_neighbors=n_neighbors,algorithm=algorithm,leaf_size=leaf_size,contamination=outlier_fraction);
  # fit the dataset to the model
  clf.fit(X_train)
  
  scores_pred = clf.decision_function(X_train)*-1 # predict raw anomaly score
  Rprecision = Rprecision_f(Y_train, scores_pred)
  if glb_verbose:
    print('R Precision : ',Rprecision)
    
  y_pred = clf.predict(X_train) # prediction of a datapoint category outlier or inlier
  objVal = objVal_f(Rprecision,y_pred,Y_train)
  
  return objVal

def save_print_output_f(mdlNam_s,res_gp,lastTime):
  exeTime = (time.time() - lastTime)
  lastTime = time.time()
  print("cummulative execution time :", mdlNam_s, exeTime)
  print("idx for best run :", mdlNam_s, np.argmin(res_gp.func_vals))
  print("params for best run :", mdlNam_s, res_gp.x_iters[np.argmin(res_gp.func_vals)])
  
  fnam_s = 'od_ionosphere_opt_results_' + mdlNam_s + '.txt'
  with open(fnam_s, 'w') as f:
    f.write("idx for best run: %s\n" % np.argmin(res_gp.func_vals))
    f.write("params for best run: %s\n" % res_gp.x_iters[np.argmin(res_gp.func_vals)])
    f.write("\n\nres_gp.x_iters:\n")
    for item in res_gp.x_iters:
        f.write("%s\n" % item)
    f.write("\n\nres_gp.func_vals:\n")
    for item in res_gp.func_vals:
        f.write("%s\n" % item)
        
  return lastTime

def opt_visualization_f(mdlNam_s, res_gp):
  iterations = np.arange(1,len(res_gp.func_vals)+1,1)
  iters_params = np.array(res_gp.x_iters)
  
  fig=plt.figure()
  ax5 = fig.add_subplot(211)
  ax5.plot(iterations,res_gp.func_vals)
  ax5.set_ylabel('Objective Value')
  
  ax6 = fig.add_subplot(212)
  ax6.plot(iterations,iters_params[:,1], label='k')
  xlable_s = 'opt loops' + mdlNam_s
  ax6.set_xlabel(xlable_s) # common x label
  ax6.set_ylabel('k')

if __name__ == '__main__':
  lastTime=time.time()
  loadData = False
  runMdlCheat = False
  runBayesOptKNN = True
  runBayesOptLOF = True
  runResultsPlots = False
  runAll = False
  breakAll = False
  
  # first time, load the web data, process, and save to numpy files. every other time, load numpy files
  if loadData:
    df = pd.read_csv('ionosphere.data', header = None)
    tmp=df.replace({34:{'g': 0.,'b': 1.}}) # this changes the strings to floats, which is what we want for PyOD
    Y_train = tmp.iloc[:,34].to_numpy()
    X_train = df.iloc[:,0:34].to_numpy() # this produces an array of floats, which is what we want for PyOD
    np.save('Y_train',Y_train)
    np.save('X_train',X_train)
  else:
    Y_train = np.load('Y_train.npy')
    X_train = np.load('X_train.npy')  
  
  if runMdlCheat or runAll:  
    # run a model fit for a model that cheats for knowing the contamination of the dataset a-priori
    # also use this model as the sand box for implementating and testing new algorithm features
    
    # find the actual contaminationof the dataset
    x_outliers, x_inliers = get_outliers_inliers(X_train,Y_train) 
    n_inliers = len(x_inliers)
    n_outliers = len(x_outliers)
    outlier_fraction = n_outliers/(n_inliers+n_outliers) # we are cheating here since we have used prio knowledge to select this hyper parameter
    
    clf = KNN(contamination=outlier_fraction);
    
    # fit the dataset to the model
    clf_name = 'kNN Cheat'
    clf.fit(X_train)

    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X_train)
  
    # Num of Errors in prediction
    n_errors = (y_pred != Y_train).sum()
    print('Num of Errors : ',clf_name, n_errors)
    
    # compute R-precision
    idxTrue = np.where(Y_train == 1) # get indices of true anomalies
    t = np.count_nonzero(Y_train == 1) # get count of true anomalies
    
    scores_pred = clf.decision_function(X_train)*-1 # predict raw anomaly score
    idxPred = np.argpartition(scores_pred, -t)[-t:]
    
    intersect = np.intersect1d(idxTrue,idxPred)
    Rprecision = len(intersect)/t
    if glb_verbose:
      print('R Precision : ',Rprecision)
    
  if runBayesOptKNN:
    res_gp_knn = gp_minimize(obj_func_kNN,
                       [(0.001, 0.5), (2, 10),['largest','mean','median'],(0.1, 10)], # outlier_fraction, n_neighbors, method, radius
                       n_calls=30,
                       n_random_starts=10,
                       verbose=glb_verbose)
    mdlNam_s = 'kNN'
    lastTime = save_print_output_f(mdlNam_s,res_gp_knn,lastTime)  
    opt_visualization_f(mdlNam_s, res_gp_knn)
          
  if runBayesOptLOF:
    res_gp_lof = gp_minimize(obj_func_LOF,
                       [(0.001, 0.5), (2, 10),['auto', 'ball_tree', 'kd_tree', 'brute'],(10,40)], # outlier_fraction, n_neighbors, algo, leaf size
                       n_calls=30,
                       n_random_starts=10,
                       verbose=glb_verbose)
    mdlNam_s = 'LOF'
    lastTime = save_print_output_f(mdlNam_s,res_gp_lof,lastTime)
    opt_visualization_f(mdlNam_s, res_gp_lof)

  if runResultsPlots or runAll:
    iterations = np.arange(1,len(res_gp_knn.func_vals)+1,1)
    iters_params = np.array(res_gp_knn.x_iters)
    
    fig=plt.figure()
    ax5 = fig.add_subplot(211)
    ax5.plot(iterations,res_gp_knn.func_vals, label='# errors')
    ax5.legend()
    ax5.set_ylabel('Objective Value')
    
    ax6 = fig.add_subplot(212)
    ax6.plot(iterations,iters_params[:,1], label='k')
    ax6.set_xlabel('opt loops') # common x label
    ax6.set_ylabel('k')