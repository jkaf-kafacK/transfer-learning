from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
rambdonF = None

iris = datasets.load_iris()
X = iris.data
Y = iris.target
#print(X[:-10])
#print(Y[:10])
X_train , X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,stratify=Y,random_state=42)


def model_train():
    print("training model ok")
    rambdonF = RandomForestClassifier(n_estimators=10, max_depth=2,random_state=42)
    rambdonF.fit(X_train, y_train)
    return rambdonF
def predicte(x_):

    predicted = rambdonF.predict(x_)
    
    return predicted

rambdonF = model_train()
def val_accu():
    val_acc = accuracy_score(predicte(X_test),y_test)
    return {f"Mean accuracy score: {val_acc:.3}"}