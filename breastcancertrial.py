import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from joblib import dump, load

cols = ['id', 'diagnosis','radius-mean', 'texture-mean', 'perimeter-mean', 'area-mean', 'smoothness-mean', 'compactness-mean', 'concavity-mean', 'concave-points-mean', 'symmetry-mean', 'fractal-dimension-mean', 'radius-se', 'texture-se', 'perimeter-se', 'area-se', 'smoothness-se', 'compactness-se', 'concavity-se', 'concave-points-se', 'symmetry-se', 'fractal-dimension-se', 'radius-ex', 'texture-ex', 'perimeter-ex', 'area-ex', 'smoothness-ex', 'compactness-ex', 'concavity-ex', 'concave-points-ex', 'symmetry-ex', 'fractal-dimension-ex']
df = pd.read_csv('https://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat', names=cols)
df.head()

x = df.iloc[:, 2:32]
y = df.iloc[:, 1]

x.head()
y.head() 

df.isnull().sum()
df.isna().sum()

from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values =  np.nan, strategy = 'mean') 
imputer.fit(x)
x = imputer.transform(x)

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
y = le.fit_transform(y) #dont need to cast as num py array bc dependent variabel does not have to be numpy array 
print (y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)