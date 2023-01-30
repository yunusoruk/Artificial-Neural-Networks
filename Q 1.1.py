import matplotlib.pyplot as plt
import numpy as np
import math


"Gürültülü işaretleri üretmek için fonksiyon"
"image oluşturulan örüntüyü, n ve r ise ne kadar noisy ve pixel hatalı şeklin üretilmesi gerektiğini söyler"
def my_function(image,n,r):                              
    #plt.imshow(image, cmap='gray')                      ### Hatasız şekil
    #plt.show()
    
    noisy_saved = np.ones(((5,10,n)))
    for i in range(0,n):
        noisy = image + 0.2 * np.random.rand(5, 10)     ### Gri gürültü eklenmiş şekil
        noisy = noisy/noisy.max()
        noisy_saved[:,:,i] = noisy                      ### farklı şekiller 3. boyutta kaydediliyor ve her bir noise'e sahip şekil görülmek istenirse görülüyor
        #plt.imshow(noisy, cmap='gray')          
        #plt.show()   

    rand_image_saved = np.ones(((5,10,r)))
    for i in range(0,r):        
        rand_index = np.random.randint(0, 50)           ### Bir pixelin değiştiği şekil
        rand_image = np.array(image)
        rand_image[int(rand_index % 5),int(rand_index / 5)] = np.where(rand_image[int(rand_index % 5), int(rand_index / 5)] == 1,0.0, 1.0)
        rand_image_saved[:,:,i] = rand_image
        #plt.imshow(rand_image, cmap='gray')     
        #plt.show()
    return noisy_saved,rand_image_saved

"0 ları 0.05, 1 leri 0.95 yapan fonksiyon"
def zero_one_converter(rand_image):
    rand_image[rand_image == 0] = 0.05
    rand_image[rand_image == 1] = 0.95
    return rand_image



""" 1. KISIM
Sınıf sınıf olan örüntülerin üretilme aşaması
"""



"1. Sınıf Örüntüler"
image1 = np.ones((5, 10))
image1[:, 0] = 0
image1[:, -1] = 0
image1[0, :] = 0
image1[-1, :] = 0
noisy1,rand_image1 = my_function(image1,4,4) #Her bir sınıf için 4 adet noisy 4 adet pixel hatalı örüntü üretildi.

rand_image1 = zero_one_converter(rand_image1) #0 lar 0.05'e, 1 ler 0.95 e çevrildi 

"2. Sınıf Örüntüler"
image2 = np.ones((5, 10))
image2[:, 0:5] = 0
noisy2,rand_image2 = my_function(image2,4,4)

rand_image2 = zero_one_converter(rand_image2)

"3. Sınıf Örüntüler"
image3 = np.ones((5, 10))
image3[[0,2,4],:] = 0
noisy3,rand_image3 = my_function(image3,4,4)

rand_image3 = zero_one_converter(rand_image3)

"4. Sınıf Örüntüler"
image4 = np.ones((5, 10))
image4[:,[0,2,4,6,8]] = 0
noisy4,rand_image4 = my_function(image4,4,4)
 
rand_image4 = zero_one_converter(rand_image4)



""" 2.KISIM
Verileri vektör haline, tek boyutlu hale, getirmek ve bias eklemek.
 Her bir örüntü için 4 adet noisy 4 adet pixel hatalı şekil bulunuyor
"""

"Dataların 51. elemanı bias olarak son 4 elemanı ise Hamming katsayıları olarak ayarlandı"
train_data = np.ones((55,32))                           #Yukarıda üretilen şekiller 4'lü 4'lü olarak tek array içinde saklanıyor
train_data[0:50,0:4] = noisy1.reshape(50,4) 
train_data[0:50,4:8] = rand_image1.reshape(50,4)
train_data[0:50,8:12] = noisy2.reshape(50,4)
train_data[0:50,12:16] = rand_image2.reshape(50,4)
train_data[0:50,16:20] = noisy3.reshape(50,4)
train_data[0:50,20:24] = rand_image3.reshape(50,4)
train_data[0:50,24:28] = noisy4.reshape(50,4)
train_data[0:50,28:32] = rand_image4.reshape(50,4)

train_data = np.swapaxes(train_data, 0, 1)              #Row ve column yeri değiştirildi

"Hamming katsayılarını data'ların en sonuna eklemek image1 için 1000, image2 için 0100 şeklinde Hamming katsayıları belirlendi"
for i in range(32):
    if i < 8:
        train_data[i,51:55] = [1,0,0,0]
    if 8 <= i < 16:
        train_data[i,51:55] = [0,1,0,0]
    if 16 <=i < 24:
        train_data[i,51:55] = [0,0,1,0]
    if 24 <=i < 32:
        train_data[i,51:55] = [0,0,0,1]
        
np.random.shuffle(train_data)                           #Verileri farklı sırada eğitim ağına sunmak için datalar karıştırıldı. 
        
        
        
""" 3.KISIM
Veri Üretimi ve düzenlenmesi kısmı bitti. Artık verileri eğitmesi için neural network'e verebiliriz
"""     
        
    
"3 katmanlı bir neural network kurulacak.Bu katmanlar sırasıyla 8 6 4 olarak belirlendi. İstenilirse değiştirilebilir şekilde ayarlandı."
"Bu kısımda başlangıç ağırlık değerleri,momentum,learning rate gibi parametreler belirleniyor"
katman1_norön = 8     #Eğitim ağının ilk katmanındaki norön sayısı
katman2_norön = 6     #Eğitim ağının ikinci katmanındaki norön sayısı


momentum1 = 0.5         #Eğitim ağının ilk katmanındaki momentum değeri
momentum2 = 0.7        #Eğitim ağının 2. katmanındaki momentum değeri
momentum0 = 0.9         #Eğitim ağının son katmanındaki momentum değeri
learning_rate1 = 0.2   #Eğitim ağının ilk katmanındaki learning değeri
learning_rate2 = 0.4  #Eğitim ağının 2. katmanındaki learning değeri
learning_rate0 = 0.6   #Eğitim ağının son katmanındaki learning değeri
sigmoid_a = 0.5         #Sigmoid alfa değeri
exp = math.exp(1)


w1 = np.ones((katman1_norön,51))*0.1                #1. katman ilk ağırlık değerleri
w2 = np.ones((katman2_norön,katman1_norön+1))*0.1   #2. katman ilk ağırlık değerleri
w0 = np.ones((4,katman2_norön+1))*0.1               #3. katman ilk ağırlık değerleri

"Momentum terimi için oluşturulmuş w(k-1) terimleri"
w1_older = np.random.random_sample((katman1_norön,51))*0.1+0.01                 #Momentum terimindeki ilk w(k-1) değerleri 0.01-0.11 arasında belirleniyor
w2_older = np.random.random_sample((katman2_norön,katman1_norön+1))*0.1+0.01
w0_older = np.random.random_sample((4,katman2_norön+1))*0.1+0.01



E = np.zeros(32)            #Her bir epokdaki her bir veri için hataları gözlemlemek için E array'i oluşturuldu.
e = np.zeros((4,32))


for epoch in range(0,500):
    E_ort = 0
    for i in range(0,32):
        "FORWARD PATH"     
        v1 = np.matmul(w1,train_data[i,0:51])                                       #inputlarla ilk ağırlıklar çarpıldı.
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
        e[:,i] = train_data[i,51:55] - y0
        E[i] = 1/2*np.matmul(np.transpose(e[:,i]),e[:,i])                            #Hata fonksiyonu
        
        E_ort = E_ort + E[i]                                                         #Ortalama hata hesabı için E[i] değerleri toplanıyor
       
        w1_old = w1                                                                  #Bu iterasyondaki ağırlık değerlerini kaybetmemek için ağırlık değerlerini tutuyoruz
        w2_old = w2
        w0_old = w0
        
        "BACKWARD PATH"
        "Yerel gradyenlerin hesaplanması"
        sigma0 = e[:,i]*y0_prime                                                    #Yerel gradyenler algoritmaya göre hesaplanıyor
        sigma2 = np.matmul(np.transpose(w0[:,0:katman2_norön]),sigma0) * y2_prime
        sigma1 = np.matmul(np.transpose(w2[:,0:katman1_norön]),sigma2) * y1_prime
             
        "Ağırlıkların güncellenmesi"
        w0 = w0 + learning_rate0 * np.outer(sigma0,y2) + momentum0*(w0-w0_older)    #Ağırlıklar güncelleniyor
        w1 = w1 + learning_rate1 * np.outer(sigma1,train_data[i,0:51]) + momentum1*(w1-w1_older)   
        w2 = w2 + learning_rate2 * np.outer(sigma2,y1) + momentum2*(w2-w2_older)        

        w1_older = w1_old                                                           #Bir sonraki iterasyonda kullanılacak momentum terimindeki w(k-1) terimleri belirleniyor
        w2_older = w2_old
        w0_older = w0_old

    if E_ort < 0.001*32:        #Ortalama hata 0.001 den küçükse durdur komutu
       E_ort = E_ort/32
       print(epoch,E_ort) 
       break
        
""" 4. KISIM
Test kümesinin oluşturulması ve network'ün denenmesi
"""

"Her bir class için 2 adet noisy ve 2 adet pixel hatalı image oluşturuluyor"
noisy1_test,rand_image1_test = my_function(image1,2,2)
noisy2_test,rand_image2_test = my_function(image2,2,2)
noisy3_test,rand_image3_test = my_function(image3,2,2)
noisy4_test,rand_image4_test = my_function(image4,2,2)

rand_image1_test = zero_one_converter(rand_image1_test)
rand_image2_test = zero_one_converter(rand_image2_test)
rand_image3_test = zero_one_converter(rand_image3_test)
rand_image4_test = zero_one_converter(rand_image4_test)

"Tüm test datası tek bir array altında toplanıyor"
test_data = np.ones((55,16))
test_data[0:50,0:2] = noisy1_test.reshape(50,2) 
test_data[0:50,2:4] = rand_image1_test.reshape(50,2)
test_data[0:50,4:6] = noisy2_test.reshape(50,2)
test_data[0:50,6:8] = rand_image2_test.reshape(50,2)
test_data[0:50,8:10] = noisy3_test.reshape(50,2)
test_data[0:50,10:12] = rand_image3_test.reshape(50,2)
test_data[0:50,12:14] = noisy4_test.reshape(50,2)
test_data[0:50,14:16] = rand_image4_test.reshape(50,2)

test_data = np.swapaxes(test_data, 0, 1)              #Row ve column yeri değiştirildi

"Test datalarının sonuna eşlenmesi gereken Hamming katsayıları ekleniyor."
for i in range(16):
    if i < 4:
        test_data[i,51:55] = [1,0,0,0]
    if 4 <= i < 8:
        test_data[i,51:55] = [0,1,0,0]
    if 8 <=i < 12:
        test_data[i,51:55] = [0,0,1,0]
    if 12 <=i < 16:
        test_data[i,51:55] = [0,0,0,1]

np.random.shuffle(test_data)                    #Test dataları karışık sırada denenmek üzere karıştırılıyor

E_test = np.zeros(16)                           #Hata fonksiyonu incelenmek üzere tutuluyor
e_test = np.zeros((4,16))

" ***ÖNEMLİ***: En sonda verilen tahminleri görmek üzere y0 verileri tutuluyor. Hamming katsayıları ile buradaki tahminler karşılaştırılacak"
y0_test = np.zeros((4,16))                      

for i in range(0,16):
    "FORWARD PATH"     
    v1 = np.matmul(w1,test_data[i,0:51])                                       #inputlarla ilk ağırlıklar çarpıldı.
    y1 = 1/(1+exp**(-sigmoid_a*v1))                                             #v2 aktivasyon fonksiyonuna girdi.
    y1 = np.insert(y1,len(y1),1)                                                #y2 ye bias eklendi.
    y1_prime = sigmoid_a*exp**(-sigmoid_a*v1)/(1 + exp**(-sigmoid_a*v1))**2     #teta'(v)
        
    v2 = np.matmul(w2,y1)
    y2 = 1/(1+exp**(-sigmoid_a*v2))
    y2 = np.insert(y2,len(y2),1)
    y2_prime = sigmoid_a*exp**(-sigmoid_a*v2)/(1 + exp**(-sigmoid_a*v2))**2     
        
    v0 = np.matmul(w0,y2)
    y0_test[:,i] = 1/(1+exp**(-sigmoid_a*v0))
    y0_prime = sigmoid_a*exp**(-sigmoid_a*v0)/(1 + exp**(-sigmoid_a*v0))**2     
    e_test[:,i] = test_data[i,51:55] - y0_test[:,i]
    E_test[i] = 1/2*np.matmul(np.transpose(e_test[:,i]),e_test[:,i])            #Hata fonksiyonu
    
"Test için BACKWARD PATH'e gerek olmadığı için kodun bu kısmı burası için silindi"
        








