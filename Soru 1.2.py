import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

""" 1.KISIM
Verilerin .txt dosyasından okunması, Hamming katsayılarının eklenmesi
test kümesi ve data kümesi olarak ayrılmaları
"""


"Pandas kullanılarak veriler sütünlara göre okunuyor"

reading = pd.read_csv('D:/Schools PDF/2021-2022 6. Dönemim Güz/EHB 420 Yapay Sinir Ağları/Ödevler/Ödev 2/iris.txt', sep=",", header=None)
reading.columns = ["a", "b", "c", "d","e"]


"Veriler tek bir array içinde toplanıyor."
data = np.zeros((150,8))
for i in range(0,150):
    data[i,0:4] = [reading.a[i],reading.b[i],reading.c[i],reading.d[i]]

maximum_value = np.max(data)
data = data/maximum_value               #Veri normalize edilerek tüm veriler 0-1 arasına getiriliyor

"Bias ve Hamming katsayıları ekleniyor"
for i in range(150):
    if i < 50:
        data[i,4:8] = [1,1,0,0]
    if 50 <= i < 100:
        data[i,4:8] = [1,0,1,0]
    if 100 <=i < 150:
        data[i,4:8] = [1,0,0,1]
        
np.random.shuffle(data)         #Veriler eğitim ve test kümesine farklı sırada verilmek için karıştırılıyor.

#Toplamları 150 olmalıdır!!!
train_data_set_sayısı = 120     #Kaç veriyle eğitim yapılmak istendiği burada seçilebilir
test_data_set_sayısı = 30       #Kaç veriyle test yapılacağını belirleyen değişken

train_data = data[0:train_data_set_sayısı]      #Train datası oluşturuluyor
test_data = data[150-test_data_set_sayısı:150]  #Test datası oluşturuluyor



""" 2.KISIM
Train verilerinin networke konularak eğitilmesi
"""

katman1_norön = 50      #1.katmandaki nöron sayıları
katman2_norön = 30      #2.katmandaki nöron sayıları


momentum1 = 0.5         #1.katmandaki momentum değeri
momentum2 = 0.7
momentum0 = 0.9
learning_rate1 = 0.1    #1.katmandaki learning değeri
learning_rate2 = 0.2
learning_rate0 = 0.3
sigmoid_a = 0.5         #Sigmoid alfa değeri
exp = math.exp(1)

w1 = np.ones((katman1_norön,5))*0.1                 #İlk Ağırlıklar belirlendi
w2 = np.ones((katman2_norön,katman1_norön+1))*0.1
w0 = np.ones((3,katman2_norön+1))*0.1

"Momentum terimi için oluşturulmuş w(k-1) terimleri"
w1_older = np.random.random_sample((katman1_norön,5))*0.1+0.01                  #İlk momentum değeri için 0.01-0.11 arasında değerler oluşturuldu
w2_older = np.random.random_sample((katman2_norön,katman1_norön+1))*0.1+0.01
w0_older = np.random.random_sample((3,katman2_norön+1))*0.1+0.01



E = np.zeros(train_data_set_sayısı)         #Hata değerleri tutulmak için array oluşturuluyor
e = np.zeros((3,train_data_set_sayısı))


for epoch in range(0,500):
    E_ort = 0
    for i in range(0,train_data_set_sayısı):
        "FORWARD PATH"     
        v1 = np.matmul(w1,train_data[i,0:5])                                        #inputlarla ilk ağırlıklar çarpıldı.
        y1 = 1/(1+exp**(-sigmoid_a*v1))                                             #v2 aktivasyon fonksiyonuna girdi.
        y1 = np.insert(y1,len(y1),1)                                                #y2 ye bias eklendi.
        y1_prime = sigmoid_a*exp**(-sigmoid_a*v1)/(1 + exp**(-sigmoid_a*v1))**2     #teta'(v)
        
        v2 = np.matmul(w2,y1)
        y2 = 1/(1+exp**(-sigmoid_a*v2))
        y2 = np.insert(y2,len(y2),1)
        y2_prime = sigmoid_a*exp**(-sigmoid_a*v2)/(1 + exp**(-sigmoid_a*v2))**2     
        
        v0 = np.matmul(w0,y2)
        y0 = 1/(1+exp**(-sigmoid_a*v0))
        y0_prime = sigmoid_a*exp**(-sigmoid_a*v0)/(1 + exp**(-sigmoid_a*v0))**2     
        e[:,i] = train_data[i,5:8] - y0
        E[i] = 1/2*np.matmul(np.transpose(e[:,i]),e[:,i])                            #Hata fonksiyonu
        
        E_ort = E_ort + E[i]                                                         #Ortalama hata hesaplamak için E değerleri toplanıyor
       
        w1_old = w1                                                                  #Bir sonraki iterasyonda w(k-1) terimi olarak kullanılmak üzere ağırlıklar kaydediliyor. 
        w2_old = w2
        w0_old = w0
        
        "BACKWARD PATH"
        "Yerel gradyenlerin hesaplanması"
        sigma0 = e[:,i]*y0_prime                                                     #Yerel gradyen hesapları
        sigma2 = np.matmul(np.transpose(w0[:,0:katman2_norön]),sigma0) * y2_prime
        sigma1 = np.matmul(np.transpose(w2[:,0:katman1_norön]),sigma2) * y1_prime
             
        "Ağırlıkların güncellenmesi"
        w0 = w0 + learning_rate0 * np.outer(sigma0,y2) + momentum0*(w0-w0_older)     #Ağırlıkların güncellenmesi
        w1 = w1 + learning_rate1 * np.outer(sigma1,train_data[i,0:5]) + momentum1*(w1-w1_older)   
        w2 = w2 + learning_rate2 * np.outer(sigma2,y1) + momentum2*(w2-w2_older)        

        w1_older = w1_old                                                            #Bir sonraki w(k-1) değerleri burada güncelleniyor
        w2_older = w2_old
        w0_older = w0_old

    if E_ort < 0.001*train_data_set_sayısı:        #Ortalama hata 0.001 den küçükse durdur komutu
       print(epoch) 
       break
        
""" 4. KISIM
Eğitilmiş ağırlık değerleriyle test kümesinin denenme süreci
"""


E_test = np.zeros(test_data_set_sayısı)
e_test = np.zeros((3,test_data_set_sayısı))
y0_test = np.zeros((3,test_data_set_sayısı))                                    #***Hamming katsayıları ile kıyaslanmak üzere oluşturulmuş tahminleri gösteren vektör***

for i in range(0,test_data_set_sayısı):
    "FORWARD PATH"     
    v1 = np.matmul(w1,test_data[i,0:5])                                         #inputlarla ilk ağırlıklar çarpıldı.
    y1 = 1/(1+exp**(-sigmoid_a*v1))                                             #v2 aktivasyon fonksiyonuna girdi.
    y1 = np.insert(y1,len(y1),1)                                                #y2 ye bias eklendi.
    y1_prime = sigmoid_a*exp**(-sigmoid_a*v1)/(1 + exp**(-sigmoid_a*v1))**2     #teta'(v)
        
    v2 = np.matmul(w2,y1)
    y2 = 1/(1+exp**(-sigmoid_a*v2))
    y2 = np.insert(y2,len(y2),1)
    y2_prime = sigmoid_a*exp**(-sigmoid_a*v2)/(1 + exp**(-sigmoid_a*v2))**2     
        
    v0 = np.matmul(w0,y2)
    y0_test[:,i] = 1/(1+exp**(-sigmoid_a*v0))                                   #Ağ çıkışındaki tahminleri gösteren y0 değeridir.
    y0_prime = sigmoid_a*exp**(-sigmoid_a*v0)/(1 + exp**(-sigmoid_a*v0))**2     
    e_test[:,i] = test_data[i,5:8] - y0_test[:,i]
    E_test[i] = 1/2*np.matmul(np.transpose(e_test[:,i]),e_test[:,i])            #Hata fonksiyonu