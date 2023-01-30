import tensorflow as tf
import pandas as pd
import numpy as np
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

# iris.data verilerini aktarma.
df = pd.read_csv("D:\Ders\YSA HW\Ödev 2\irisSoru3.data",sep=',', header=0)    

# Data shuffle                      
df = df.sample(frac = 1) 
  
# Değişken verilerin 0 ile 1 arasında olması için normalizasyon. 
columns = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]                
maximum_value = np.max (columns)
X_Data = columns / maximum_value                                                                  

# String olarak belirtilen tür değerlerini array içerisinde integer değerleri olarak toparlamak.                
Y_Data = df['Species']                                                                          
Y_Data = df['Species'].map({"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2})          


# train_test_split komutu kullanılarak %75 lik bir bölüm eğitim kümesi %25 lik kalan bölüm test kümesi olarak ayrıldı.
# random_state datanın random seçimini sağlamaktadır.
def TrainTestData():                           
    
    x_train, x_test = train_test_split( X_Data, test_size = 0.2, random_state = 80)
    
    return x_train, x_test 

x_train, x_test  = TrainTestData()

# Feauture column oluşturmak 
a = tf.feature_column.numeric_column('SepalLengthCm')
b = tf.feature_column.numeric_column('SepalWidthCm')
c = tf.feature_column.numeric_column('PetalLengthCm')
d = tf.feature_column.numeric_column('PetalWidthCm')

feature_column = [a, b, c, d] 

#Estimator Model
train_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x = x_train,  batch_size = 40, num_epochs = 1000, shuffle = True)
test_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x = x_test,  batch_size = 40, num_epochs = 1, shuffle = False)


#Linear Classifier Model
model = tf.compat.v1.estimator.LinearClassifier(feature_columns = [a, b, c, d] , n_classes=3)

#Model Training
model_train = model.train(input_fn = train_func, steps = 1000)

#Result
result = model.evaluate(test_func)
print('Accuracy: ', result["accuracy"])

#Prediction
predict_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x = x_test, num_epochs = 1, shuffle = False)
prediction_data = model.predict(input_fn = predict_func)
prediction_list = list(prediction_data)
prediction = [i["class_ids"][0] for i in prediction_list]
#prediction_report = classification_report(y_test, prediction)

#Confusion Matrix
#confusion_matrix = confusion_matrix(y_test, prediction)
#skplt.metrics.plot_confusion_matrix(y_test, prediction, figsize = (6,6), title = "Confusion Matrix")




