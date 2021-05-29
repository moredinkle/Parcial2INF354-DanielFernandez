

import pandas as pd
import numpy as np
df= pd.read_csv(r'C:\Universidad\Septimo\354\audit_data\audit_risk.csv', skiprows=0, low_memory=False)
pd.set_option('max_columns', None)
df=df.replace(np.nan,"0")

X=df[['Sector_score','Risk_A','Risk_B','Risk_C','Risk_D','RiSk_E','Risk_F','Inherent_Risk']]
y=df['Risk']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,train_size=0.8)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(6,6,6,6),solver='lbfgs',max_iter=6000)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
