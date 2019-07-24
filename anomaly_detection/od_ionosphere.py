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


def Rprecision_f(Y_train, scores_pred):
  # compute R-precision
  idxTrue = np.where(Y_train == 1) # get indices of true anomalies
  t = np.count_nonzero(Y_train == 1) # get count of true anomalies
  idxPred = np.argpartition(scores_pred, -t)[-t:] # get indices of top t predicted anomalies
  intersect = np.intersect1d(idxTrue,idxPred) # get intersection between the 2 indices sets
  Rprecision = len(intersect)/t 
  return Rprecision

def objVal_f(Rprecision):
  objVal = Rprecision*-1 # *-1 ause larger Rprecision is better, but gp_minimize seeks to minimize the objVal 
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
  print('R Precision : ',Rprecision)
  
  return objVal_f(Rprecision)

if __name__ == '__main__':
  lastTime=time.time()
  loadData = False
  runMdlCheat = True
  runBayesOpt = False
  runResultsPlots = False
  runAll = True
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
    print('R Precision : ',Rprecision)
    
  if runBayesOpt or runAll:
    res_gp = gp_minimize(obj_func_kNN,
                       [(0.001, 0.5), (2, 10),['largest','mean','median'],(0.1, 10)], # outlier_fraction, n_neighbors, method, radius
                       n_calls=30,
                       n_random_starts=10,
                       verbose=False)
  
    exeTime = (time.time() - lastTime)
    print("cummulative execution time:", exeTime)
    print("idx for best run:", np.argmin(res_gp.func_vals))
    print("params for best run:", res_gp.x_iters[np.argmin(res_gp.func_vals)])
    
    with open('od_ionosphere_opt_results.txt', 'w') as f:
      f.write("idx for best run: %s\n" % np.argmin(res_gp.func_vals))
      f.write("params for best run: %s\n" % res_gp.x_iters[np.argmin(res_gp.func_vals)])
      f.write("\n\nres_gp.x_iters:\n")
      for item in res_gp.x_iters:
          f.write("%s\n" % item)
      f.write("\n\nres_gp.func_vals:\n")
      for item in res_gp.func_vals:
          f.write("%s\n" % item)

  if runResultsPlots or runAll:
    iterations = np.arange(1,len(res_gp.func_vals)+1,1)
    iters_params = np.array(res_gp.x_iters)
    
    fig=plt.figure()
    ax5 = fig.add_subplot(211)
    ax5.plot(iterations,res_gp.func_vals, label='# errors')
    ax5.legend()
    ax5.set_ylabel('Num of Errors')
    
    ax6 = fig.add_subplot(212)
    ax6.plot(iterations,iters_params[:,1], label='k')
    ax6.set_xlabel('opt loops') # common x label
    ax6.set_ylabel('k')






















#
#
##generate random data with two features
#X_train, Y_train = generate_data(n_train=200,train_only=True, n_features=2)
#
## by default the outlier fraction is 0.1 in generate data function 
#outlier_fraction = 0.1
#
## store outliers and inliers in different numpy arrays
#x_outliers, x_inliers = get_outliers_inliers(X_train,Y_train)
#
#n_inliers = len(x_inliers)
#n_outliers = len(x_outliers)
#
##separate the two features and use it to plot the data 
#F1 = X_train[:,[0]].reshape(-1,1)
#F2 = X_train[:,[1]].reshape(-1,1)
#
## create a meshgrid 
#xx , yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
#
## scatter plot 
#plt.scatter(F1,F2)
#plt.xlabel('F1')
#plt.ylabel('F2') 
#
#
#classifiers = {
#     'Angle-based Outlier Detector (ABOD)'   : ABOD(contamination=outlier_fraction),
#     'K Nearest Neighbors (KNN)' :  KNN(contamination=outlier_fraction)
#}
#
##set the figure size
#plt.figure(figsize=(10, 10))
#
#for i, (clf_name,clf) in enumerate(classifiers.items()) :
#    # fit the dataset to the model
#    clf.fit(X_train)
#
#    # predict raw anomaly score
#    scores_pred = clf.decision_function(X_train)*-1
#
#    # prediction of a datapoint category outlier or inlier
#    y_pred = clf.predict(X_train)
#
#    # Num of Errors in prediction
#    n_errors = (y_pred != Y_train).sum()
#    print('Num of Errors : ',clf_name, n_errors)
#
#    # rest of the code is to create the visualization
#
#    # threshold value to consider a datapoint inlier or outlier
#    threshold = stats.scoreatpercentile(scores_pred,100 *outlier_fraction)
#
#    # decision function calculates the raw anomaly score for every point
#    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
#    Z = Z.reshape(xx.shape)
#
#    subplot = plt.subplot(1, 2, i + 1)
#
#    # fill blue colormap from minimum anomaly score to threshold value
#    subplot.contourf(xx, yy, Z, levels = np.linspace(Z.min(), threshold, 10),cmap=plt.cm.Blues_r)
#
#    # draw red contour line where anomaly score is equal to threshold
#    a = subplot.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
#
#    # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
#    subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
#
#    # scatter plot of inliers with white dots
#    b = subplot.scatter(X_train[:-n_outliers, 0], X_train[:-n_outliers, 1], c='white',s=20, edgecolor='k') 
#    # scatter plot of outliers with black dots
#    c = subplot.scatter(X_train[-n_outliers:, 0], X_train[-n_outliers:, 1], c='black',s=20, edgecolor='k')
#    subplot.axis('tight')
#
#    subplot.legend(
#        [a.collections[0], b, c],
#        ['learned decision function', 'true inliers', 'true outliers'],
#        prop=matplotlib.font_manager.FontProperties(size=10),
#        loc='lower right')
#
#    subplot.set_title(clf_name)
#    subplot.set_xlim((-10, 10))
#    subplot.set_ylim((-10, 10))
#plt.show()