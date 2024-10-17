import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

data_dict = pickle.load(open('./langgue_Detect\data.pickle','rb'))

data = np.asanyarray( data_dict['data'])
labels =np.asanyarray( data_dict['labels'])


# ແບ່ງໂມເດວ ໂດຍໃຊ້ train_test_split ກຳນົດອັດຕາສວ່ນ test ຂໍ້ມູນ test_size=0.2  20 %
x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.2,shuffle = True,stratify=labels)

model = RandomForestClassifier()

model.fit(x_train,y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict,y_test)

print('{}%  of samples  was classifier correctly  ! '.format(score*100)) 

#ເຮົາມາບັນທືກໂມເດວໄວ້ໂມເດວ

f = open ('./langgue_Detect\model2.p','wb')
pickle.dump({'model':model},f)
f.close()