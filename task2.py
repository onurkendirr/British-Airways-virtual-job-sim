import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("data/customer_booking.csv", encoding="latin-1")

df.head()
df.dtypes
df.shape
df.isnull().sum()
df["booking_complete"].value_counts()

############# MUTUAL INFOS #####################

X = df.drop("booking_complete", axis=1)
y = df.booking_complete

for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

X.dtypes

from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)
mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Info Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
plt.show()

from sklearn.model_selection import train_test_split


def dataset(X, y):
    train_full_X, val_X, train_full_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

    train_X, test_X, train_y, test_y = train_test_split(train_full_X, train_full_y, test_size=0.25, random_state=0)
    return (train_X, test_X, train_y, test_y)


from sklearn.preprocessing import MinMaxScaler


def scale(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    return X


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

X = df.drop('booking_complete', axis=1)

#one hot encoding

X = pd.get_dummies(X)
X = scale(X)
y = df.booking_complete

X_train, X_val, y_train, y_val = dataset(X, y)

forest_model = RandomForestClassifier(random_state=1)
forest_model.fit(X_train, y_train)
preds = forest_model.predict(X_val)

print('ACCURACY: ', accuracy_score(y_val, preds) * 100)
print('AUC score: ', roc_auc_score(y_val, preds))
