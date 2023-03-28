'''Python içine gerekli kütüphaneler import edilir.Diğer kütüphaneler ilgili işlem yapılmadan önce aşağıda import edilecektir. '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''Kullanacağımız veri seti Sosyal meyda veri setidir.Python'ın pandas kütüphanesinin read_csv methodu ile veri setini import ediyoruz.Veri seti yaş,cinsiyet,gelir düzeyi gibi bağımsız 
değişkenlerden oluşup bağımlı değişken olan satın alıp almama durumu incelenmektedir.'''

data=pd.read_csv('Social_Network_Ads.csv')
print(data)

'''Makine öğrenmesinde tahminlerde bulunabilmek için değişkenlerin sürekli olması gerekmektedir.Label encoder ile veriler kategorik bir veriden sayısal bir veriye dönüştürülmesini sağlar.
Veri ön hazırlık işlemi olarak bu işlemin yapılabilmesi için Scikit-Learn kütüphanesinden Label encoder import edilir.Veri setinde 'Gender' kolonu kategorik verilerden oluştuğu için 
label encoder işlemi uygulanır ve kolon son haliyle '0' ve '1' verilerinden oluşmuş olur.'''

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Gender']=le.fit_transform(data['Gender'])

'''Veri seti toplamda 5 kolondan oluşmaktadır.Kolonlardan biri bağımlı değişken diğerleri ise bağımsız değişkenlerdir.Bağımsız değişken kolonlarda verilen 
ölçüm özelliklerine purchased kolonu için sınıflandırma yapacağız.Öncesinde bağımsız değişkenlerdeki nitelikler için bir x matrisi,bağımlı değişken için ise bir y vektörü 
oluşturacağız.'''

x=data.iloc[: , :-1].values
y=data.iloc[: ,4:].values

'''Bağımlı ve bağımsız değişkenlerimizi belirledikten sonra Iris veri seti 4 bölüme ayrılır.Bu bölümlerden %67'lik kısım olan x_train ve y_train eğitim için kullanılırken
%33'lük kısım olan  X_test ve Y_test ise makineye tahmin ettirilmeye çalışılacaktır.'''

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.33,random_state=0)

'''Veriler önceklendirmeden önce yukarıda test ve tarin olmak üzere 4 kısma ayrılmıştır.Aynı işlemi Standar scaler adı verilen ölçeklendirme işlemi yapıldıktan sonrada uygulayacağız.
Standard scaler,veri setinde bulunan verileri aynı düzlemde birbiri ile karşılaştırılabilecek değerlere dönüştürür.
Bu sayede veri setinde bulunan çok yüksek bir değerle çok düşük bir değerin karşılaştırılması mümkün hale gelir.'''

from sklearn.preprocessing import StandardScaler
ss = StandardScaler() 
x_scaled=ss.fit_transform(x)
x_scaled_train, x_scaled_test, y_scaled_train, y_scaled_test = train_test_split(x_scaled, y,train_size=0.33,random_state=0)

'''Makine öğrenmesinde kullanılan sınıflandırma yöntemlerinden biri de lojistik Regresyondur.Lojistik regresyon,birçok bağımsız değişkeni kullanarak bir sonucu tahmin etmeye çalışır.
Lojistik regresyon, tahmin etmek istediğimiz sonucun sınıfını (0 veya 1 gibi) tahmin etmek için kullanılır.Örneğimizde de bağımsız değişkenlere göre kişilerin sosyal medya satın alıp 
almaması gibi 0 ve 1 ile ifade edebileceğimiz ikili bir durum tahmin edilmeye çalışılmıştır.Lojistik regresyon, kolayca anlaşılabilir olması, iyi çalıştığı pek çok durum ve hızlı 
hesaplama özellikleri nedeniyle makine öğrenmesi alanında yaygın bir algoritmadır.'''

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_train,y_train.ravel())
tahmin=log.predict(x_test)

log.fit(x_scaled_train,y_scaled_train.ravel())
tahmin2=log.predict(x_scaled_test)

'''Confusion matrix,sınıflandırma problemlerinde kullanılan bir performans ölçümüdür. Karışıklık matrisi, gerçek sınıfı ve tahmin edilen sınıfı içeren bir tablodur. 
Bu tablo, dört farklı değere sahip olabilir: true positive (TP), false positive (FP), true negative (TN) ve false negative (FN).TP, modelin doğru bir şekilde bir sınıfı
 belirlediği durumlarda oluşurken, FP modelin yanlış bir şekilde bir sınıfı belirlediği durumlarda oluşur.TN, modelin bir sınıfı doğru bir şekilde olmadığını belirlediği 
 durumlarda, FN ise modelin bir sınıfı yanlış bir şekilde olmadığını belirlediği durumlarda oluşur.Karmaşıklık matrisi, bu dört sonucu bir matris içinde gösterir.''' 
 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_scaled_test, tahmin2)
print(cm)
#Confusion matrix:
#[[156  10]    
#[ 43  59]]                                                                               
    
'''Accuracy değeri gibi f1 skor da sınıflandırma algoritmalarının başarısını ölçmek için kullanılan bir metriktir.Özellikle dengeli olmayan veri setleri için sınıflandırma problemlerinde
kullanılı bir metriktir.F1 skor 0 ve 1 arasında bir değer alıp 1 en iyi 0 ise en kötü performansa sahip olunduğu anlamına gelmektedir.'''

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_scaled_test, tahmin2)
print(accuracy)    
#Başarı oranı:0.8022388059701493

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, tahmin)
print(accuracy)    
#Başarı oranı:0.7574626865671642