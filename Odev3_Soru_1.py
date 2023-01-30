import numpy as np
import matplotlib.pyplot as plt
import math

'''1. KISIM
Veri üretimi ve çizdirilmesi
'''

#Merkez noktaları ve sigmaları belirlenen 4 adet küme
A = np.random.multivariate_normal([3, 3, 2], 0.4*np.eye(3), 150)    #Merkez noktası 3 3 2 olan ve 0.4 sigma ile gaussian dağılan noktalar
B = np.random.multivariate_normal([3, 3, -2], 0.4*np.eye(3), 150)
C = np.random.multivariate_normal([3, -3, 2], 0.4*np.eye(3), 150)
D = np.random.multivariate_normal([3,-3, -2], 0.4*np.eye(3), 150)

#Her bir sınıf için farklı renkler atandı.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(A[:, 0], A[:, 1], A[:, 2], color='blue')
ax.scatter(B[:, 0], B[:, 1], B[:, 2], color='red')
ax.scatter(C[:, 0], C[:, 1], C[:, 2], color='green')
ax.scatter(D[:, 0], D[:, 1], D[:, 2], color='yellow')

Data_ABCD = np.concatenate((A, B, C, D), axis=0)           #Tüm veriler tek matris içerisinde toplandı

Y_train = np.zeros(600)         #Daha sonrasında eğitim kümesine koymamak ancak kıyaslayabilmek adına her bir sınıfa bir değer atandı.
Y_train[0:150] = 0              #En son çıkan ağırlıklar ve kazanan nöronlarla bu sınıflar arasında yorum yapılmaya çalışıldı.
Y_train[150:300] = 1
Y_train[300:450] = 2
Y_train[450:600] = 3

Data = np.array([Data_ABCD[:,0],Data_ABCD[:,1],Data_ABCD[:,2],Y_train])     #Sınıf bilgileri veri kümesine son eleman olarak eklendi
Data = np.transpose(Data)       #Verileri ağa uygun bir hale getirmek için transpose alındı.

"""2.KISIM
Verilerin ağa sunulup eğitimin gerçekleştirilmesi ve ağ parametrelerinin belirlenmesi
"""

number_neurons_x = 5        #X eksenindeki nöron sayısının belirlenmesi
number_neurons_y = 5        #Y eksenindeki nöron sayısının belirlenmesi
total_neurons = number_neurons_x*number_neurons_y

#Veriler -3, +3 arasında olduğu için ilk ağırlıklar da rastgele olarak -3, +3 arasında seçildi.
prng1 = np.random.RandomState(1234560897)
W = prng1.random_sample(total_neurons*3)*6-3
W = np.reshape(W,(total_neurons,3))

#Uzaklık matrisi yazdırıldı. X ve Y ekseninde 5 er nöron var ise ortaya çıkacak olan matris 25,25 boyutunda olacaktır.
D = np.ones((total_neurons,total_neurons))

for i in range(0,number_neurons_x):
    for j in range(0,number_neurons_y):
        for k in range(0,number_neurons_x):
            for l in range(0,number_neurons_y):
                D[number_neurons_y*i+j,k*number_neurons_y+l] = math.sqrt((i-k)**2+(j-l)**2)

                
np.random.shuffle(Data)     #Data karıştırıldı
X_train = Data[0:480,0:3]   #Eğitim verisi ayrıldı. (Son sütun yani sınıf bilgisi alınmadı)
X_test = Data[480:600,0:3]  #Test verisi alındı. (Son sütun yani sınıf bilgisi alınmadı)
learn_rate = 0.5            #Öğrenme hızı
sigma = 0.5                 #Sigma değeri

Class = Data[0:480,3]       #Daha rahat kontrol edebilmek adına sınıflar buraya kaydedildi.
V = np.zeros((480,total_neurons))   #Herbir epok başında kazanan nöronları bulabilmek adına tüm nöron ağırlıklarıyla yapılan iç çarpımlar bu arrayde tutuldu.
result = np.zeros(480)              #Herbir epoktaki kazanan nöronun index' ini tutan array

for l in range(0,400):
    for i in range(0,480):
        for j in range(0,total_neurons):
            V[i,j] = np.matmul(X_train[i,:],W[j,:])     #Herbir datanın tüm nöronlarla olan iç çarpımlarının sonuçları burada tutuluyor.
            
        Max_V = np.max(V[i,:])                          #Maksimum gelen iç çarpım Max_V içerisinde kaydediliyor
        result_x = np.where(V[i,:] == np.amax(V[i,:]))  #Kazanan nöronun index'i belirleniyor.
        result[i] = result_x[0]
        res = result_x[0]
        for k in range(0,total_neurons):
            W[k,:] = W[k,:] + learn_rate*np.exp(-D[res,k]/(2*sigma**2))*(X_train[i,:]-W[k,:])   #Her bir nöronun ağırlıkları kazanan nörona yakınlıklarına göre güncelleniyor.

'''

'''

Y_class = Data[480:600,3]               #Test verilerinin sınıfları daha rahat yorum yapabilmek adına bu array içinde tutuldu.
V_test = np.zeros((120,total_neurons))  #Test verisinde iç çarpım sonuçlarını tutan array
result_test = np.zeros(120)             #Kazanan nöronların index'ini depolayan array

for i in range(0,120):
    for j in range(0,total_neurons):
        V_test[i,j] = np.matmul(X_test[i,:],W[j,:])    #İç çarpımlar yapılıyor.
        
    Max_V_Test = np.max(V_test[i,:])
    result_x = np.where(V_test[i,:] == np.max(V_test[i,:]))     #Kazanan nöronun index'i belirleniyor
    result_test[i] = result_x[0]                                #Kazanan nöron indexleri kaydediliyor





