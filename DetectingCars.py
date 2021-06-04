# -*- coding: utf-8 -*-
#!/usr/bin/env python

from ToolsMachineLearning import sliding_window, PCA, plot_confusion_matrix,\
                                 BackgroundSubstractor, SecondCountour,\
                                 GenerateRep
import matplotlib.pyplot as plt

import cv2
import os
import re
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

if __name__ == '__main__':
  
  # Data For Training
  Dir_Train = "../DataADAS/TrainImages/"
  Dir_Test  = "../DataADAS/TestImages/"
  # Class Names
  class_names = ['NoCar','Car']
  fdescriptors = True
  # Image Size
  WinH = 40
  WinW = 100
  # PCA dimension reduction
  RDimension = 1100
  # Error Tolerance for discovering windows
  Tolerance = 0.99
  # Kernels for Morphological Operations
  kernel = np.ones((7,7),np.uint8)
  kernel1 = np.ones((3,3),np.uint8)
  kernel2 = np.ones((11,11),np.uint8)
  # Regular Expressions
  stneg = re.compile('neg')
  stpos = re.compile('pos')
  
  C1 = []
  C2 = []
  ########################### Reading Data #########################
  print('Reading Training Data')
  train_list = list()
  for root, dirs, files in os.walk(Dir_Train, topdown=False):
    for ii, file in enumerate(files):
      if stneg.match(file):
        img = cv2.imread(Dir_Train+file, 0)
        # Image Equalization
        img = cv2.equalizeHist(img)
        # Send img to float for flattening
        imgf = img.astype(float)
        if fdescriptors == True:
          flat = GenerateRep(img)
        else:
          flat = np.reshape(imgf,(1, WinH*WinW))
        C1.append(flat)
      else:
        img = cv2.imread(Dir_Train+file, 0)
        img = cv2.equalizeHist(img)
        rimg = BackgroundSubstractor(img)
        opening = cv2.morphologyEx(rimg, cv2.MORPH_OPEN, kernel)
        rimg[opening==0]=0
        #rimg = SecondCountour(rimg)  
        
        if ii<0:
          print(1)
          plt.figure()
          #plt.imshow(img, cmap='gray')
          #plt.figure()
          #plt.imshow(rimg, cmap='gray')
          #plt.show(block=False)
        imgf = rimg.astype(float)
        if fdescriptors == True:
          flat = GenerateRep(imgf)
        else:
          flat = np.reshape(imgf,(1, WinH*WinW))
        C2.append(flat)

  print('Reshape Data Matrix')      
  # Reshape and Reassamble
  n1 = len(C1)
  n2 = len(C2)
  if fdescriptors == False:
    C1 = (np.stack(C1, axis=1)).reshape(n1, WinH*WinW)
    C2 = (np.stack(C2, axis=1)).reshape(n2, WinH*WinW)
  else:
    C1 = np.stack(C1, axis=1)
    _, n,m = C1.shape
    C1 = C1.reshape(n,m)
    C2 = np.stack(C2, axis=1)
    _, n,m = C2.shape
    C2 = C2.reshape(n,m)
  X = np.concatenate((C1,C2), axis=0)
  # Get less information if necessary
  print('Apply PCA')   
  EigenValeus, EigenVect = PCA(X)
  U = EigenVect[:,0:RDimension ].T
  Xf = U.dot(X.T).T

  Y1 = np.zeros([n1,])
  Y2 = np.ones([n2,])
  Yf = np.concatenate((Y1,Y2), axis=0)
  print('Training Logistic Regression')
  LR = LogisticRegression()
  LR.fit(Xf, Yf)
  Yp = LR.predict(Xf)

  
  print('Confusion Matrix')
  # Compute confusion matrix
  cnf_matrix = confusion_matrix(Yf, Yp)
  np.set_printoptions(precision=2)
  
  # Plot non-normalized confusion matrix
  #plt.figure()
  #plot_confusion_matrix(cnf_matrix, classes=class_names,\
  #                      title='Confusion matrix LR, without normalization')  

  print('Training MLP')
  Y1 = np.ones([n1,])
  Y2 = 2*np.ones([n2,])
  Ym = np.concatenate((Y1,Y2), axis=0)
  MLP = MLPClassifier(hidden_layer_sizes=(2*RDimension, ))
  MLP.fit(Xf, Ym) 
    

  for root, dirs, files in os.walk(Dir_Test, topdown=False):
    for ii, file in enumerate(files):
      if ii%20==0:
        image = cv2.imread(Dir_Test+file, 0)
        ListRec = list() 
        for (x, y, window) in sliding_window(image, stepSize=8, windowSize=(WinW, WinH)):
          #if the window does not meet our desired window size, ignore it
          if window.shape[0] != WinH or window.shape[1] != WinW:
            continue
          # Here is your classification after training
          window = cv2.equalizeHist(window)
          rwindow = BackgroundSubstractor(window)
          opening = cv2.morphologyEx(rwindow, cv2.MORPH_OPEN, kernel)
          rwindow[opening==0]=0
          #rwindow = SecondCountour(rwindow)
          windowf = rwindow.astype(float)
          if fdescriptors == True:
            xwindow = GenerateRep(windowf)
            xwindow = U.dot(xwindow.T).T
          else:
            flat = np.reshape(windowf,(1,WinH*WinW))
            xwindow = U.dot(flat.T).T
          prLR = LR.predict_proba(xwindow)
          prMLP = MLP.predict_proba(xwindow)
          if prLR[0,1]>Tolerance and prMLP[0,1]>Tolerance :
            ListRec.append([x,y,x+WinW,y+WinH])
            
        rectList, weights = cv2.groupRectangles(ListRec, 10, eps=2.0)
        if list(rectList):
          coord = rectList[0]
          cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 255, 0), 1)
          #plt.figure()
          #plt.imshow(image, cmap='gray')
          #plt.show(block=False)
        else:
          print(1)
          #plt.figure()
          #plt.imshow(image, cmap='gray')
          #plt.show(block=False)
