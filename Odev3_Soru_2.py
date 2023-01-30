import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd


"Pandas kullanılarak veriler sütünlara göre okunuyor"

reading = pd.read_csv('D:/Schools PDF/2021-2022 6. Dönemim Güz/EHB 420 Yapay Sinir Ağları/Ödevler/Ödev 3/iris.txt', sep=",", header=None)
reading.columns = ["a", "b", "c", "d","e"]


"Veriler tek bir array içinde toplanıyor."
Data_iris = np.zeros((150,4))
Data_norm = np.zeros((150,4))
for i in range(0,150):
    Data_iris[i,0:4] = [reading.a[i],reading.b[i],reading.c[i],reading.d[i]]

for i in range(4):
    Data_norm[:,i] = 2.*(Data_iris[:,i] - np.min(Data_iris[:,i]))/np.ptp(Data_iris[:,i])-1


Y_train = np.zeros(150)
Y_train[0:50] = 0
Y_train[50:100] = 1
Y_train[100:150] = 2

Data = np.array([Data_norm[:,0],Data_norm[:,1],Data_norm[:,2],Data_norm[:,3],Y_train])
Data = np.transpose(Data)


Mean_of_classes = np.zeros((3,5))

for i in range(4):
    Mean_of_classes[0,1+i] = np.mean(Data[0:50,i])
    Mean_of_classes[1,1+i] = np.mean(Data[50:100,i])
    Mean_of_classes[2,1+i] = np.mean(Data[100:150,i])
    
Mean_of_classes[1,0] = 1
Mean_of_classes[2,0] = 2
    
number_neurons_x = 5
number_neurons_y = 5
total_neurons = number_neurons_x*number_neurons_y


prng1 = np.random.RandomState(2143560879)
W = prng1.random_sample(total_neurons*4)                    
W = np.reshape(W,(total_neurons,4))*6

D = np.ones((total_neurons,total_neurons))

for i in range(0,number_neurons_x):
    for j in range(0,number_neurons_y):
        for k in range(0,number_neurons_x):
            for l in range(0,number_neurons_y):
                D[number_neurons_y*i+j,k*number_neurons_y+l] = math.sqrt((i-k)**2+(j-l)**2)
                              
np.random.shuffle(Data) 

X_train = Data[0:120,0:4]
X_test = Data[120:150,0:4]

learn_rate = 0.8
sigma = 0.7

Class = Data[0:120,4]



V = np.zeros((120,total_neurons))
result = np.zeros(120)

for l in range(0,1000):
    for i in range(0,120):
        for j in range(0,total_neurons):
            V[i,j] = np.matmul(X_train[i,:],W[j,:])
            
        Max_V = np.max(V[i,:])
        result_x = np.argmax(V[i,:])
        result[i] = result_x
        for k in range(0,total_neurons):
            W[k,:] = W[k,:] + learn_rate*np.exp(-D[result_x,k]/(2*sigma**2))*(X_train[i,:]-W[k,:])
            
            
Y_test = Data[120:150,4]
V_test = np.zeros((30,total_neurons))
result_test = np.zeros(30)

for i in range(0,30):
    for j in range(0,total_neurons):
        V_test[i,j] = np.matmul(X_test[i,:],W[j,:])    
        
    Max_V = np.max(V_test[i,:])
    result_x = np.where(V_test[i,:] == np.max(V_test[i,:]))
    result_test[i] = result_x[0] 