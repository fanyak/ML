# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:34:31 2020

@author: fanyak
"""
import numpy as np;
import math;
import scipy.integrate as integrate

import matplotlib.pyplot as plt


c1 = np.array([[-1,2], [-2,1], [-1,0] ]);
z1 = np.array([-1,1]);
c2 = np.array([[2,1], [3,2]]);
z2 = np.array([2,2])

def totalSquaredEuclideanDistance(c,z):
    #print('cluster', c, 'candidate', z)
    total = 0
    for i in c:
      total += np.sum(np.square( i - z ));
    return total;

def squaredEuclideanDistance(c,z):
    return np.sum(np.square( c - z ));

data = np.array([
        [0,-6], [4,4],[0,0],[-5,2]
        ])
    
    
def minkowskiDst(v1, v2, p):
    dist = 0;
    for i in range(len(v1)):
        dist += abs(v1[i] - v2[i])**p;
    return dist**(1/p);

def totalminkowskiDst(c, z, p):
    total = 0
    for i in c:
      total += minkowskiDst(i, z,p);
    return total;


### K-medoids
def kMedoids(k, data):
   
        
    ## STEP 1 - RANDOMLY CHOOSE CENTROIDS
    # incides for initialized centroids:
    centroids = [[-5,2], [0,-6]];                        
   
            
    #print(initialCentroidIndx)
    for t in range(1):
        #create k empty clusters
        clusters = [];
        for n in range(k):
            clusters.append([]);
            
        #ITERATE

        ## STEP 2 - CREATE CLUSTERS BASED ON CHOSEN CENTROIDS
        for el in data:
            distances =[];
            for c in centroids:
#                distances.append(
#                        squaredEuclideanDistance(np.array(el), np.array(c)) );
                distances.append(minkowskiDst(el, c, 2))
            #print(el, distances)
            #find minimum distance:
            minDistanceIndx = np.argmin(np.array(distances));
            #print(clusters, minDistanceIndx)
            #print(el, centroids)
            clusters[minDistanceIndx].append(el);
        
        #STEP3 - FIND THE BEST REPRESENTATIVE FOR EACH CLUSTER
        centroids = [];

        for c in clusters:
            minDistance = np.inf;
            newCentroid = None;
            for candidate in data:
                # Find the total distance of candidate centroids from elements in cluster
               # candidateDistance = totalSquaredEuclideanDistance (np.array(c), np.array(candidate));
               candidateDistance = totalminkowskiDst(c, candidate, 2)
               if(candidateDistance < minDistance):
                    minDistance = candidateDistance;
                    newCentroid = candidate;
            centroids.append(
                    newCentroid
            );            
        centroids = np.array(centroids)
    
    return (clusters, centroids);


### K-Means
def kMeans(k, data):
   
        
    ## STEP 1 - RANDOMLY CHOOSE CENTROIDS
    # incides for initialized centroids:
    centroids = [[-5,2], [0,-6]];                       
   
            
    #print(initialCentroidIndx)
    for t in range(2):
        #create k empty clusters
        clusters = [];
        for n in range(k):
            clusters.append([]);
            
        #ITERATE

        ## STEP 2 - CREATE CLUSTERS BASED ON CHOSEN CENTROIDS
        for el in data:
            distances =[];
            for c in centroids:#               
                distances.append(minkowskiDst(el, c, 1))
            #print(el, distances)
            #find minimum distance:
            minDistanceIndx = np.argmin(np.array(distances));
            #print(clusters, minDistanceIndx)
            #print(el, centroids)
            clusters[minDistanceIndx].append(el);
        
        #STEP3 - FIND THE BEST REPRESENTATIVE FOR EACH CLUSTER
        centroids = [];

        for c in clusters:
            #compute the mean
            sm = np.array([0,0])
            for el in c:
                sm+=el;
            newCentroid = np.array(sm)/len(c)
            centroids.append(
                    newCentroid.tolist()
            );  
        centroids = np.array(centroids)
    
    return (clusters, centroids);
            
        
        

####################
#####EM Algorithm
def guassianEvaluatedAtPointX(mu, v, x):
    N =  1/((2*np.pi*v)**(1/2)) * np.exp( (-(x-mu)**2)/(2*v));
    return N;

#thetas
#thetasForCluster = np.array([[-3, 4], [2,4]])
#priorForCluster = np.array([0.5, 0.5]);
#
##data
#x = np.array([0.2, -0.9, -1, 1.2, 1.8])

   
def CheckPointAgainstEachGuassian(thetasForCluster, priorForCluster, x):
    p = np.empty((len(thetasForCluster), len(x)));
    #mu =np.empty((len(thetasForCluster), len(x)));
    l = 0;
    for i in x:
        for j in range(len(thetasForCluster)):
            [m,v] = thetasForCluster[j];
            numerator = priorForCluster[j] * guassianEvaluatedAtPointX(m,v,i);
            totalProb = 0;
            for k in range(len(thetasForCluster)):
                [m,v] = thetasForCluster[k];
                totalProb += priorForCluster[k] * guassianEvaluatedAtPointX(m,v,i);
            #print(numerator/totalProb);
            p[j][l]=numerator/totalProb;
        l+=1;
    return np.array(p);
            
    
  
    
def calculateUpdatedProbabiliy(thetasForCluster, priorForCluster, x):
    probs = CheckPointAgainstEachGuassian(thetasForCluster, priorForCluster, x);
    summed = np.sum(probs, axis=1);
    return summed / len(x)
    
def checkUpdatedMean(thetasForCluster, priorForCluster, x):
    res = [];
    probs = CheckPointAgainstEachGuassian(thetasForCluster, priorForCluster, x);
    for j in range(len(thetasForCluster)):
        sm = probs[j]@x.T;
        res.append(sm / np.sum(probs[j]));
    return np.array(res);
        
def checkUpdatedVar(thetasForCluster, priorForCluster, x):
    probs = CheckPointAgainstEachGuassian(thetasForCluster, priorForCluster, x); 
    means = checkUpdatedMean(thetasForCluster, priorForCluster, x);
    res = [];
    for j in range(len(thetasForCluster)):
        sm = probs[j]@np.square((x.T-means[j])); 
        res.append(sm / np.sum(probs[j]));
    return np.array(res);




#thetas
meansAndVariances = np.array([[6, 1], [7,4]])
priors = np.array([0.5, 0.5]);

#data
x = np.array([-1, 0, 4, 5, 6])

def logLikelihood(x, theta, prior):    
    p = np.empty((len(theta), len(x)));
    #mu =np.empty((len(theta), len(x)));
    logLikelihood = 0;
    n =0;
    for i in x:
        for j in range(len(meansAndVariances)):
            [m,v] = meansAndVariances[j];
            numerator = priors[j] * guassianEvaluatedAtPointX(m,v,i);
            totalProb = 0;
            for k in range(len(meansAndVariances)):
                [m,v] = meansAndVariances[k];
                totalProb += priors[k] * guassianEvaluatedAtPointX(m,v,i);

            logLikelihood+=(numerator/totalProb)*np.log(numerator/(numerator/totalProb));
            p[j][n] = (numerator/totalProb);
        n+=1;
    return logLikelihood, np.argmax(p,axis=0);


for i in range(20):
      p = calculateUpdatedProbabiliy(meansAndVariances, priors, x); 
      m = checkUpdatedMean(meansAndVariances, priors, x);
      v = checkUpdatedVar(meansAndVariances, priors, x);
      meansAndVariances = np.array([[m[0], v[0]], [m[1], v[1]]]);
      priors = p;
      
print(meansAndVariances)
    