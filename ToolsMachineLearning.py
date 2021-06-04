# -*- coding: utf-8 -*-
#!/usr/bin/env python


import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import chi2
import itertools
import cv2
from sklearn.cluster import KMeans


def sliding_window(image, stepSize, windowSize):
  # slide a window across the image
  for y in range(0, image.shape[0], stepSize):
    for x in range(0, image.shape[1], stepSize):
      # yield the current window
      yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def Normalization(Data):
    Mean1 = np.mean(Data, axis = 0)
    Std1  = np.std(Data, axis = 0)
    return (Data-Mean1)/Std1

def MahalonobisDetection(Data, alpha):
    Data = Data - np.mean(Data, axis = 0)
    n1,n2 = Data.shape
    Cov = (1/float(n1-1))*Data.T*Data
    M = np.zeros(n1)
    for i in range(0,n1):
        M[i] = Data[i,:]*np.linalg.inv(Cov)*Data.T[:,i]
    c = chi2.isf(alpha,n2) 
    return  M, c , Cov
    
def PCA(NData):
    NDataMean = NData - np.mean(NData,axis = 0)

    n1 , n2 = NDataMean.shape

    NCov = (NDataMean.T).dot(NDataMean)
    NCov = (1/float(n1-1))*NCov
    NEigenvaluesc, NEigenvectorsc = np.linalg.eigh(NCov) 
    idx = NEigenvaluesc.argsort()[::-1]  
    NEigenvaluesc = NEigenvaluesc[idx]
    NEigenvectorsc  =  NEigenvectorsc [:,idx]
    return NEigenvaluesc, NEigenvectorsc

def SelectingBestSubset2class(Data, nfeat, fmask,mmask):
    
    t1 , t2 = Data.shape
    
    C1 = np.asmatrix(Data[fmask,:])
    C2 = np.asmatrix(Data[mmask,:])
    n1, dummy = C1.shape
    n2, dummy = C2.shape    
    
    P1 = float(n1)/float(t1)
    P2 = float(n2)/float(t1)
    
    Flag = True 
    
    L1   = range(t2)
    
    t2 = t2 -1
    
    J = -100000.0
    
    while(Flag):
        p1 = list(itertools.combinations(L1,t2))
        print(len(p1))
        for j in p1:
            TData = Data[:,j]
            C1 = np.asmatrix(TData[fmask,:])
            C2 = np.asmatrix(TData[mmask,:])
            Cov1 = (1/float(n1-1))*C1.T*C1
            Cov2 = (1/float(n2-1))*C2.T*C2         
            Sw = P1*Cov1+P2*Cov2
            m1 = (1/float(n1))*np.sum(C1,axis = 0)
            m2 = (1/float(n2))*np.sum(C2,axis = 0)
            m0 = P1*m1+P2*m2
            Sm = (1/float(t1-1))*(TData - m0).T*(TData-m0)
            
            Jt = np.trace(Sm)/np.trace(Sw)
            
            if (Jt > J):
                print(L1)
                J = Jt
                L1 = j
                
        if (t2 == nfeat):
            Flag = False
            print('The selected features ')
            print(L1)
            print('J value for selection '+str(J))

        t2 = t2-1
         
    return L1, J

def kmeans(Data,centroids,error):
    lbelong = []
    x1,x2 = Data.shape
    y1,y2 = centroids.shape
    oldcentroids = np.matrix(np.random.random_sample((y1,y2)))
    # Loop for the epochs
    # This allows to control the error
    trace = [];
    while ( np.sqrt(np.sum(np.power(oldcentroids-centroids,2)))>error):
        # Loop for the Data
        for i in range(0,x2):
            dist = []
            point = Data[:,i]
            #loop for the centroids
            for j in range(0, y2):
                centroid = centroids[:,j]
                dist.append(np.sqrt(np.sum(np.power(point-centroid,2))))
            lbelong.append(dist.index(min(dist)))        
        oldcentroids = centroids
        trace.append(centroids)
        
        #Update centroids     
        for j in range(0, y2):
            indexc = [i for i,val in enumerate(lbelong) if val==(j)]
            Datac = Data[:,indexc]
            print(len(indexc))
            if (len(indexc)>0):
                centroids[:,j]= Datac.sum(axis=1)/len(indexc)
    return centroids, lbelong, trace


def gen_line(w,minr,maxr,nsamp):
    # Generate samples for x
    x = np.array(np.linspace(minr,maxr,nsamp))

    # Generate the samples for y
    y = -w[0,0]/w[2,0]-(w[1,0]/w[2,0])*x

    return x,y

def CM_own(Y1,Y2):
  """
  Print the Confusion Matrix for binary clases with labels +1 Y1 and -1 Y2
  """
  P,M1 = Y1.shape 
  N,M2 = Y2.shape
  TP = np.sum(1*(Y1>0))
  TN = np.sum(1*(Y2<0))
  FP = np.sum(1*(Y1<=0))
  FN = np.sum(1*(Y2>=0))
  print('{}'.format(15*'='))
  print('Confusion Matrix')
  print('{}'.format(20*'='))
  print(tabulate([['C1', TP , FP], ['C2', FN, TN]], headers=['', 'C1', 'C2']))
  print('{}'.format(20*'='))
  print(' ')
  print('{}'.format(20*'='))
  print('Confusion Matrix As Probabilities')
  print('{}'.format(20*'='))
  print(tabulate([['C1', '{0:0.2f}'.format(float(TP)/float(P)) , '{0:0.2f}'.format(float(FP)/float(P))  ],
                   ['C2', '{0:0.2f}'.format(float(FN)/float(N)) , '{0:0.2f}'.format(float(TN)/float(N)) ]], 
                    headers=['', 'C1', 'C2']))
  print('{}'.format(20*'='))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show(block=False)

def BackgroundSubstractor(img):
  """
  Removing Background
  """
  thr = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                              cv2.THRESH_BINARY,21,-4)
  _, cnts, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cnts = sorted(cnts, key = cv2.contourArea, reverse = True) 
  mask = img.copy()
  mask[mask > 0] = 0
  cv2.fillConvexPoly(mask, cnts[0], 255)
  mask = np.logical_not(mask)
  rimg = img.copy()
  rimg[mask] = 0
  
  return rimg

def SecondCountour(img):
  """
  Locking for the second contour
  """
  _, cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  if list(cnts):
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True) 
    mask = img.copy()
    mask[mask > 0] = 0
    cv2.fillConvexPoly(mask, cnts[0], 255)
    mask = np.logical_not(mask)
    rimg = img.copy()
    rimg[mask] = 0
  else:
    rimg = img.copy()
  
  return rimg

def GenerateRep(img, number_books=40):
  """
  Using sift descriptors
  """
  # Convert to uint8 for SIFT descriptors
  ImageU8 = np.array(img, np.uint8)
  # Object SIFT Descriptor
  sift = cv2.xfeatures2d.SIFT_create()
  # Calculate Descriptors
  kps, des = sift.detectAndCompute(ImageU8, None)
  # Branching for creating the descriptors 
  if len(kps)<number_books: # Case one not enough for clustering
    ADesc2 = np.zeros([number_books-len(kps),128])
    if len(kps)>0:
      ADesc1 = np.asarray(des)
      Rep = np.concatenate((ADesc1,ADesc2), axis=0)
    else:
      Rep = ADesc2
  else: # Enough for clustering
    # Get KMeans
    Code = KMeans(n_clusters=number_books)
    ADesc3 = np.asarray(des)
    Code.fit(ADesc3)
    # Get Labels of points and Center of Clusters
    Labels = Code.labels_
    Centers = Code.cluster_centers_
    LCode = []
    # Use the Sum(v-c) for representative code book
    for i in range(number_books):
      NN = ADesc3[Labels==i,:]
      LCode.append(np.sum(NN-Centers[i,:], axis=0))
    Rep = np.stack(LCode, axis=0)
  n,m = Rep.shape
  # Flatten and Return
  return np.reshape(Rep,(1,n*m))
