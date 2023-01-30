from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#Merkez noktaları ve sigmaları belirlenen 4 adet küme
A = np.random.multivariate_normal([3, 3, 2], 0.4*np.eye(3), 150)                 #Merkez noktası 3 3 2 olan ve 0.4 sigma ile gaussian dağılan noktalar
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

data = np.concatenate((A, B, C, D), axis=0)                                     #Tüm veriler tek matris içerisinde toplandı

Y_class = np.zeros(600)                                                         #Daha sonrasında eğitim kümesine koymamak ancak kıyaslayabilmek adına her bir sınıfa bir değer atandı.
Y_class[0:150] = 0                                                              #En son çıkan ağırlıklar ve kazanan nöronlarla bu sınıflar arasında yorum yapılmaya çalışıldı.
Y_class[150:300] = 1
Y_class[300:450] = 2
Y_class[450:600] = 3
labels = Y_class


#data = np.genfromtxt('iris.txt', delimiter=',', usecols=(0, 1, 2, 3))
#labels = np.genfromtxt('iris.txt', delimiter=',', usecols=(4), dtype=str)

#Veriler konumlarına yarıştırıldı.
def classify(som, data):
    winmap = som.labels_map(X_train, y_train)                                   #Kazanan haritası oluşturuldu.
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:                                                              
        win_position = som.winner(d)                                            
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

#Train Test Split ile veri eğitildi.
X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels)

#MiniSom ile veri eğitildi.
som = MiniSom(6, 6, 3, sigma=3.5, learning_rate=0.5, 
              neighborhood_function='triangle', random_seed=10)
som.pca_weights_init(X_train)
som.train_random(X_train, 1000, verbose=False)                                      


print(classification_report(y_test, classify(som, X_test)))


