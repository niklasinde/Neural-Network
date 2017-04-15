#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:21:46 2017

@author: niklasinde
"""
import scipy as sp
from  pylab import *
from numpy import ndarray
import numpy as np
import random
import matplotlib.pyplot as plt
import string
from scipy import optimize
from io import BytesIO
import csv
from itertools import product
def wmstarter(s):
    """returns a random weights matrix given the size s"""
    b = []
    for i in range(len(s)-1):
        a = np.random.randn(s[i],s[i+1])
        b.append(a)
    return(b)


def numbermatrix(size):
    A = np.zeros(size)
    counter = 0
    for row in range(size[0]):
        for col in range(size[1]):
            A[row][col]= counter
            counter += 1
            
    return(A)
            
            
            
def load(Howmanynumbers):
    if Howmanynumbers== 100:
        f = open("/Users/niklasinde/Dropbox/Universitetet/Kandidatupsats/mnist/mnist_train_100.csv", 'r')
    elif Howmanynumbers == 10:
        f = open("/Users/niklasinde/Dropbox/Universitetet/Kandidatupsats/mnist/mnist_test_10.csv", 'r')
    elif Howmanynumbers == 60000:
        f = open("/Users/niklasinde/Dropbox/Universitetet/Kandidatupsats/mnist/mnist_train (1).csv", 'r')
    a = f.readlines()
#    f.close()
#    f = figure(figsize=(15,15))
    count=0
    pictureMatrix = []
    X=[]
    y = np.zeros([len(a),10])
    for line in a:
        linebits = line.split(',')
        imarray = np.asfarray(linebits[1:]).reshape((28,28))

        pictureMatrix.append(imarray)

        vector = imarray.ravel()
        
        y[count][int(linebits[0])] = 1
      
        X.append(vector)
        count = count + 1

    X =1/256*np.array(X)   
    return(X,y,pictureMatrix)
def showanumber(a,loops):
    """only plots number a from vector x2 givven that y2 is the right number in x2"""
    x,y,z = load(60000)
    for i in range(loops):
        if y[i][a]==1:
            showimg(x[i])
        
def blackwhite(pictureMatrix):
    """Turns a picture with grey colors in to only black"""
    a = pictureMatrix
    s = shape(a)
    for i in range(s[0]):
        for k in range(s[1]):
            if a[i][k]>0:
                a[i][k]=1
            else:
                a[i][k]=0
    return(a)
def showimg(A,numberInA=None):
    """turns a matrix in to a picure"""
    if type(A) == type([]):
        a = numberInA
        imshow(A[a],cmap="Greys",interpolation="None")
    elif type(A) == type(np.array([])):
        if shape(A)[0]==784:
            A = np.asfarray(A.reshape((28,28)))
            imshow(A,cmap="Greys",interpolation="None")
        elif shape(A)[0]==shape(A)[1]:
            imshow(A,cmap="Greys",interpolation="None")
        elif shape(A) != shape(A) and shape(A[numberinA])[0]==shape(A[numberinA])[1]:
            imshow(A[a],cmap="Greys",interpolation="None")
        elif shape(A)[0] != shape(A)[1] and shape(A[numberInA])[0]==784:
            A = np.asfarray(A[numberInA].reshape((28,28)))
            imshow(A,cmap="Greys",interpolation="None")
        else:
            print("Unknown matrix")
    else:
        print("Unknown type")
    plt.show()

    
def dlpng(filename,number):
    """Imports a file from the folder kandidatuppsats/ritadebilder/ with filename=filename and the real number
       Exports a picture matrix, a vector and the y vector (right answear)"""
    inportedImg = sp.misc.imread("/Users/niklasinde/Dropbox/Universitetet/Kandidatupsats/ritadebilder/"+filename+".png")
    picture = inportedImg[:,:,3]
    x = 1/255*picture.ravel().T
    y = np.zeros([1,10])
    y[0][number] = 1
    return(picture,x,y)

    

class NN:
    """
    Size is a list with the number of neurons in respective layer
    Weights is the initial value of the weights.
        """
    def __init__(self,size,weights):
        self.size = size
        self.error = 10
        self.weights =  weights
#        self.weights = [np.random.rand(y,x) for x,y in zip(size[:-1],size[1:])]
        self.bias = [np.random.randn(1,y) for y in size[1:]]

        
    def checker(self,x,y):
        """checks that the size of the network is correct with aspect to the size
            of the input and output matrix"""
        self.size[0]=shape(x)[1]
        self.size[-1]=shape(y)[1]

    def sigmoid(self,z):
        """outputs the sigmoidfunction of z"""
        y = 1.0/(1.0+np.exp(-z))
#        print(y)
        return (y)
        

    def sigmoidprime(self,z):
        """outputs the sigmoidprimefunction of z"""
        y = np.exp(z)/((np.exp(z)+1)**2)
        return(y)
    
    def costfunction(self,x,y): 
        """ returns the error of forward(x) and y (truevalue)"""
        self.yHat = self.forward(x)
        self.error=(0.5*sum((y-self.yHat)**2))
        return(self.error)

    def results(self,x):
        y = self.forward(x)
        sort =sorted(range(len(y.T)), key=lambda k: y.T[k])
        sort = sort[::-1]
        results1= []
        results2 = []
        yt = y.T
        for i in range(len(sort)):
            results1.append(sort[i])
            results2.append(yt[sort[i]])
        print("First guess",results1[0],"with",results2[0][0],"%")
        
        print("Second guess",results1[1],"with",results2[1][0],"%")
            
    def forward(self,x):
        """forward propogation of x"""
        print(shape(x),shape(self.weights[0]))
        print(self.weights[0])
        Z = []
        A = [x]
        for i in range(len(self.size)-1):
            Z.append(np.dot(A[i],self.weights[i])+self.bias[i])
            A.append(self.sigmoid(Z[i]))
        self.Z = Z
        self.A = A
#        print("forward")
        return(A[-1])    
    
        
    def showimg(self,A,numberInA=None):
        if type(A) == type([]):
            a = numberInA
            imshow(A[a],cmap="Greys",interpolation="None")
        elif type(A) == type(np.array([])):
            if shape(A)[0]==784:
                A = np.asfarray(A.reshape((28,28)))
                imshow(A,cmap="Greys",interpolation="None")
            elif shape(A)[0]==shape(A)[1]:
                imshow(A,cmap="Greys",interpolation="None")
            elif shape(A) != shape(A) and shape(A[numberinA])[0]==shape(A[numberinA])[1]:
                imshow(A[a],cmap="Greys",interpolation="None")
            elif shape(A)[0] != shape(A)[1] and shape(A[numberInA])[0]==784:
                A = np.asfarray(A[numberInA].reshape((28,28)))
                imshow(A,cmap="Greys",interpolation="None")
            else:
                print("Unknown matrix")
        else:
            print("Unknown type")
        plt.show()
    def costfuncprime(self,x,y):
#        print("costfuncprim")
        self.yhat = self.forward(x)
        delta = []
        djdw = []
        delt = np.multiply(-(y-self.yhat),self.sigmoidprime(self.Z[-1]))
        delta.append(delt)
        djdw.append(np.dot(self.A[-2].T,delta[0]))
        for i in range(2,len(self.size)):
            delt = np.multiply(np.dot(delt, self.weights[-i+1].T),self.sigmoidprime(self.Z[-i]))
            delta.append(delt)
            djdw.append(np.dot(self.A[-i-1].T,delt))
        djdw2 = []
        for i in range(len(djdw)):
            djdw2.append(djdw[-1-i])
        
        return(djdw2)
    def dropout(self):
        listnr = np.random.uniform(low=0, high=len(self.weights))
        
    
    
    def plotneural(self):
        #Weights är inte än in programmerade 
        #Gör dem färgkordinerade när nevrala nätverket är klart
        A = self.size
        w = len(A)
        for i in range(w):
            for j in range(A[i]):
                plt.scatter(i,j/A[i])
              
        for i in range(w-1):
            for w in range(A[i+1]):
                for j in range(A[i]):

                    if abs(self.weights[i][j][w])<10**-4:
                        colur = "k"
                        plt.plot((i,i+1),(j/A[i],w/A[i+1]),color = colur)
                    elif self.weights[i][j][w]>0:
                        colur = "g"
                        plt.plot((i,i+1),(j/A[i],w/A[i+1]),color = colur)
                    else:
                        colur = "r"
                        plt.plot((i,i+1),(j/A[i],w/A[i+1]),color = colur)
        plt.show()
    
#### Theis functions are helper functions. The make the list or matrises 
#### to long vectors instead. We need this when optimizing. 
#### We also need the to make vectors in to matrices when doing the forward
#### and backprob ei. computing the facual gradients the will go in to the
#### optimizing algoritms. Because they need vectorbased gradients to be able
#### to pair a certain gradient with a certhen weight that need to be optimized.
#### All this to addapt to the input type of the built in functions in python.

    def getparams(self):
        w = np.array([])
        for i in range(len(self.weights)):
            w = np.concatenate((w, self.weights[i].T.ravel()))
#        print("getparams")
        return(w)   
   
    def computegradients(self, X, y):
        djdw = self.costfuncprime(X, y)
        w = np.array([])
        for i in range (len(djdw)):
            w = np.concatenate((w,djdw[i].ravel()))
#        print("computegradients")
        return w
    def setparams(self,A):
        j=0
        B = []
#        print("setparamsfunctioN")
        for i in range(len(self.size)-1):
            s = self.size[i+1]*self.size[i]
            a = A[j:s+j]
            c = np.reshape(a, (self.size[i] ,self.size[i+1]))
            j += self.size[i+1]*self.size[i]
            B.append(c)
        self.weights=B
        return (self.weights)
    
    
    
    
    
class BFGS(object):
    def __init__(self, NN):
        #Make Local reference to network:
        self.NN = NN
        self.i = 1
        self.list = []
        self.cost =1
        self.grad = 0
    def callbackF(self, params):
#        print("hejsan")
        self.NN.setparams(params)
#        print(shape(self.NN.setparams(params)),"setparams",self.i,"i")
        self.J.append(self.NN.costfunction(self.X, self.y))  
    def costFunctionWrapper(self, params, X, y):
        self.NN.setparams(params)
        self.i = self.i +1 
        cost = self.NN.costfunction(X, y)
        grad = self.NN.computegradients(X, y)
        self.J.append(cost)
        print(cost,shape(grad),"costfunctionwrapper",self.i,"i")
        return cost, grad
        
    def trains(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y
        #Make empty list to store costs:
        self.J = []
        params0 = self.NN.getparams()
#        print(self.i,"params0")
        option = {'maxiter': 300, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='L-BFGS-B', \
                                 args=(self.X, self.y),tol = 10**-10,options=option, callback=self.callbackF)
        
        self.NN.setparams(_res.x)
        self.optimizationResults = _res
#        self.NN.plotneural()

class clayer:
    def __init__(self,pic,filtersize, nrfilters, isPic=True, shape = 28):
        ###### We skriv in så vi kan göra en step size(den längden som matrixen
        ###### hoppar när den downsamplar)
        self.isPic = isPic
        self.filter = self.wmstarter(filtersize,nrfilters)
        self.filtersize = filtersize
        self.pic,self.shape= self.transformation(pic,shape)
        self.weights = self.wmstarter(nrfilters,filtersize)
        self.zeropad = 0
        self.stride =1

        
        
    def wmstarter(self,filtersize,nrfilters):
        print(nrfilters)
        assert len(filtersize) == len(nrfilters),"length of filtersize is not equal to nrfilter"
        a = []
        count=0
        for i in nrfilters:
            A = [np.random.randn(filtersize[count],filtersize[count]) for j in range(i)]
            count+=1
            a.append(A)
        return(a)
    def outputzeromatrix(self,picwidt,filt,stride = 1,zeropadding = 0):
        s = (self.shape-self.filtersize+2*self.zeropad)/self.stride
        return(np.z((s,s)))
        
        
        
        
    def transformation(self,pic,shape):
        #### To make life easier with the iterations we want a matrixbased image.
        #### So this function is just to make sure we have just that.
        if self.isPic == True:
            return(pic,np.shape(pic[0])[0])
        elif self.isPic == False:
            assert len(pic)==shape**2, """the vector cannot be reshaped to the specified size"""
            A = [np.asfarray(pic[i].reshape((shape,shape))) for i in range(len(pic))]
            return(A,shape)

    def consmallerpic(self):
        """format of x == list(pictures)"""
        filt = self.filter
        listofoutputs = []
        for i in range(len(self.pic)):
            entilllista = []
            for row in range(self.shape-filt+1):
                for col in range(self.shape-filt+1):
                    rowfilt = row+filt
                    colfilt = col+filt
                    A = self.pic[i][row:rowfilt,col:colfilt]
                    entilllista.append(A)   
            listofoutputs.append(A)
        self.listofoutputs = listofoutputs
        return (listofoutputs)
    def convforward(self,pics):
        """ Format of pics == list(list(smallpictures of big picture)"""
        x = consmallerpic(pics)
        outputlist=[]
        listofoutputs = []
        nrofbigpic = len(x)
        nrofsmallpic = len(x[0])
        for i in range(nrofbigpic):
            for j,k in product(range(nrofsmallpic),range(self.nroffilters)):
                  lista.append(np.multiply(x[i][j],weights[k]))
                  x.append(listofout)
            
        for i in range(nrofbigpic):
            outputmatrix = np.zeros(())
            for j in range(nrofsmallpic):
                for k in range(self.nroffilters):
                    listofoutputs.append(np.multiply(x[i][j],weights[k]))
                    x.append(listofout)
            
                        
        
            
        pass
    def relu(self,x):
        return(max(0,x))
       
    
#numberarray = numbermatrix((9,9))
#a = clayer([numberarray],[2],[3])

#b = a.consmallerpic()
#print(b)

        

######################### Handwritten letters import########################


#x2,y2,picturMatrix = load(60000)
#testx,testy,picture = load(10)
#picture = blackwhite(picturMatrix[0])
#size2 = [784,400,100,81,10]
#wm2 = wmstarter(size2)
#neural = NN(size2,wm2)
#t = BFGS(neural)
#t.trains(x2,y2)
#plt.plot(t.J)
#plt.grid(1)
#plt.xlabel('Iterations')
#plt.ylabel('Cost')
#plt.show()
#testy1 = neural.forward(testx)-testy
#testy2 = neural.forward(testx)
#print("hello")

########################## Test function ###################
x1 = np.array(([3,5,10,11,12], [5,1,1,11,12], [10,2,4,11,12]), dtype=float)
y1 = np.array(([75], [82], [93]), dtype=float)
#x1 = x1/np.amax(x1, axis=0)
#y1 = y1/100 #Max test score is 100
size = [5,4,1]
wm = wmstarter(size)
Nn =NN(size,wm)
Nn.plotneural()
ti = BFGS(Nn)
ti.trains(x1,y1)
#plt.plot(BFGS.j)
#plt.grid(1)
#plt.xlabel('Iterations')
#plt.ylabel('Cost')
#plt.show()
#print(NN.forward(X))

















#Pokerdataset load
#def ipd():
#    results = []
#    with open('/Users/niklasinde/Dropbox/Universitetet/Kandidatupsats/poker-hand-training-true.txt') as inputfile:
#        for line in inputfile:
#            results.append(line.strip().split(','))
#            for i in range(len(results[-1])):
#                results[-1][i]= int(results[-1][i])
#                
#    
##    print("TANK YOU")
#                
#    return (results)
##ipd()


####################### PONG ########################

try:
    weightsfromclass = neural.weights

except NameError:
    print("")