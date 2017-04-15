# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import scipy as sp
from  pylab import *
from numpy import ndarray
import numpy as np
import random
import matplotlib.pyplot as plt
from io import BytesIO
from captcha.image import ImageCaptcha
from PIL import Image
import string
from scipy import optimize

################ CLASS Based code ##############
"""
class nn()
    def __init__(self,Input,tol=None):
        self.inA = np.array(Input)
        self.inAsize = np.size(Input,0)

    
    
def triggerfunc(self,z):
    y = 1/(1+e**(-z))
    return(y)
    
def triggerprime(self,z):
    return(e**z/(1 + e**z)**2)

def forward(self, tol):
    pass
def backward(self,inA):
    pass

def train(self,tol):
    pass
    
"""
############### Function based code ##################

def sigmoid(z):
    y = 1/(1+np.exp(-z))
    return(y)

def sigmoidprime(z):
    return(np.exp(-z)/((1+np.exp(-z))**2))
    
def wmstarter(size):
    b = []
    for i in range(len(size)-1):
        a = np.random.randn(size[i+1],size[i])
        b.append(a)
    return(b)
    

def costfunction(x,y,weights,size): 
    yhat = forward(x,size,weights)
    return(0.5*sum((y-yhat[0])**2))

    
def costfuncprime(imput,size,y,weights):
    yhat,Z,A = forward(imput,size,weights)
    delta = []
    djdw = []
    delta.append(np.multiply(-(y-yhat),sigmoidprime(Z[-1])))
    djdw.append(np.dot(delta[0],A[-2].T))
#    print(shape(A[-1]))
    
    for i in range(len(size)-2):
        delt = np.dot(weights[-i-1].T,delta[i])
        delta.append(np.multiply(delt,sigmoidprime(Z[-i-2])))
        djdw.append(np.dot(delta[i+1],A[-i-3].T))

    return(djdw,yhat)
    
    
def forward(inputs,size,weights):
    Z = []
    A=[inputs]
#    print(weights)
    for i in range(len(size)-1):
        Z.append(np.dot(weights[i],A[i]))
        A.append(sigmoid(Z[i]))
    return(A[-1],Z,A)    
    
    


def plotneural(A,weights):
    #Weights är inte än in programmerade 
    #Gör dem färgkordinerade när nevrala nätverket är klart
    w = len(A)
    for i in range(w):
        for j in range(A[i]):
            plt.scatter(i,j/A[i])
          
    for i in range(w-1):
        for w in range(A[i+1]):
            for j in range(A[i]):
                if weights[i][w][j]<0:
                    colur = (sigmoid(weights[i][w][j]),sigmoid(weights[i][w][j]),0)
                    plt.plot((i,i+1),(j/A[i],w/A[i+1]),color = colur)
                else:
                    colur = (0,sigmoid(weights[i][w][j]),sigmoid(weights[i][w][j]))
                    plt.plot((i,i+1),(j/A[i],w/A[i+1]),color = colur)
    plt.show()
    
def changeweights(weights,djdw,scalar,size):
    for j in range(len(size)-1):
        weights[j] = weights[j]-scalar*djdw[-j-1]
    return weights

    
def getParams(A):
    w = np.array([])
    for i in range(len(A)):
        w = np.concatenate((w, A[i].ravel()))
    return(w)
def setParams(A,size):
    j=0
    B = []
    for i in range(len(size)-1):
        s = size[i]*size[i+1]
        a = A[j:s+j]
        j += size[i]*size[i+1]
        c = np.reshape(a, (size[i+1] ,size[i]))
        B.append(c)
    return(B)

##################### MAIN #################
def main(imp,out,scalar,size,loops,weights,wstarter):
    size[0]=shape(imp)[0]
    size[-1]=shape(out)[0]
#    print(size)
    if wstarter==True:
        weights = wmstarter(size)
    listanx=[]
    listan = []
    ylist = []
    wlist = []
    glist = []
    for i in range(loops):
        djdw,yhat = costfuncprime(imp,size,out,weights)
        listan.append(costfunction(imp,out,weights,size))
        listanx.append(i)
        weights = changeweights(weights,djdw,scalar,size)
        wlist.append(weights)
        glist.append(djdw)
#        if len(wlist)>1 and scalar <7:
            #5/(-np.linalg.norm(yhat-out)+7)
#            print(scalar)
#            scalar = bbmethod(glist,wlist)
        if np.linalg.norm(yhat-out) < 10**-9:
            print(i,"hej")
            print(np.linalg.norm(yhat-out),"NNNOOOORRRMM")
            print(yhat)
            print(out,"hello")
            plotneural(size,weights)
            return(yhat,listanx,listan,weights,djdw)
          
    print(np.linalg.norm(yhat-out),listan[-1],"NNNOOOORRRMM")
    print(yhat)
    print(out,"hello")
    plotneural(size,weights)
    
    return(yhat,listanx,listan,weights,djdw)


def costFunctionWrapper(params, X, y,size):
    params = setParams(params,size)
    cost = costfunction(X, y,params,size)
    print(params,shape(params))
    grad = getParams(costfuncprime(X,size,y,params)[0])
    return cost, grad
def callbackF(X,y, params):
    setParams(params,size)
    
      
def train(X,y,size):
    #Make empty list to store costs:
    J = []
    
    params0 = getParams(wmstarter(size))
#    print(params0)
    options = {'maxiter': 10, 'disp' : True}
    _res = optimize.minimize(costFunctionWrapper, params0, jac=True, method='BFGS', \
                             args=(X, y,size), options=options)#,callback=callbackF(X,y,params0))

    setParams(_res.x,size)
    optimizationResults = _res
    return(_res)
#print(optimize(weightsdata,inputdata,outputdata,sizedata))
########## Varibles #######################3


sinx=[]
siny=[]
number = 10
sizesin=[1,10,1]

for i in range(1,number):
    j = 1/i
    sinx.append([j])
    siny.append([np.sin(j)])
sinxlist =sinx
sinx = array(sinx).T

#print(sinx)
siny = array(siny).T
#print(siny)
sizesin = [1,10,4,1]
weightssin = wmstarter(sizesin)
#hej = main(sinx,siny,scalar,sizesin,loops,weightssin,False)
#weightssin = hej[3]
#print(hej[0])


inputdata = array([[3,2,3,4],[5,1,2,4],[10,2,4,2],[2,3,4,5]]).T
outputdata = array([[0.75,0.10],[0.82,0.3],[0.93,0.3],[0.2,0.5]]).T
scalar=0.01
loops=30000
sizedata= [1,10,10,1]
weightsdata = wmstarter(sizedata)
#hej = main(inputdata,outputdata,scalar,sizedata,loops,weightsdata,True)
#om man vill återanvända vikterna sätt weight sätt true till false
#weights = hej[3] 

#plotneural(size,randomweights)
#plt.plot(hej[1],hej[2])
#plt.show()
#plt.plot(sinxlist,hej[0])
#plt.show()


X = np.array(([3,5], [5,1], [10,2]), dtype=float).T
y = np.array(([75], [82], [93]), dtype=float).T
X = X/np.amax(X, axis=0)
y = y/100 #Max test score is 100
size = [2,3,1]
train(X,y,size)
#weights = 0
#results = main(X,y,3,size,loops,weights,False)
#plt.plot(results[1],results[2])
#weights = results[3]
################ CAPATCH #########################

def randomstring(size=6, chars=string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
#print(randomstring())

def imagecaptcha(name,font='/Users/niklasinde/Library/Fonts/billy.ttf'):
    
    image = ImageCaptcha(fonts=[font])
    data = image.generate(name)
    assert isinstance(data, BytesIO)
    image.write(string, "capache/"+name)
    print("hej")

#imagecaptcha(randomstring())
    




    
    





