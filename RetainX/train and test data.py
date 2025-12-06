import pandas as pd 
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# read data
data=pd.read_csv(r"N:\RetainX\clean_data.csv")
# split data feature and target
x=data.drop("Churn_Yes",axis=1)
y=data["Churn_Yes"]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)

# Balance classes using SMOTE
sm = SMOTE(random_state=42)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train)

# Define classification models with specific hyperparameters
LG  = LogisticRegression(class_weight="balanced")
DTC = DecisionTreeClassifier(max_depth=7,class_weight="balanced")
RFC = RandomForestClassifier(n_estimators=300,max_depth=12,class_weight="balanced")
BNB = BernoulliNB()
SVC = SVC(kernel="rbf", C=2,class_weight="balanced")
KNC = KNeighborsClassifier(n_neighbors=9)
# train models classification
LG.fit(x_train_res,y_train_res)
DTC.fit(x_train_res,y_train_res)
RFC.fit(x_train_res,y_train_res)
BNB.fit(x_train_res,y_train_res)
SVC.fit(x_train_res,y_train_res)
KNC.fit(x_train_res,y_train_res)
# predict in all model
pred_LG=LG.predict(x_test)
pred_DTC=DTC.predict(x_test)
pred_RFC=RFC.predict(x_test)
pred_BNB=BNB.predict(x_test)
pred_SVC=SVC.predict(x_test)
pred_KNC=KNC.predict(x_test)
# metrics : accuracy_score , classification_report , confusion_matrix in all model
acc_LG=accuracy_score(y_test,pred_LG)
acc_DTC=accuracy_score(y_test,pred_DTC)
acc_RFC=accuracy_score(y_test,pred_RFC)
acc_BNB=accuracy_score(y_test,pred_BNB)
acc_SVC=accuracy_score(y_test,pred_SVC)
acc_KNC=accuracy_score(y_test,pred_KNC)

cr_LG=classification_report(y_test,pred_LG)
cr_DTC=classification_report(y_test,pred_DTC)
cr_RFC=classification_report(y_test,pred_RFC)
cr_BNB=classification_report(y_test,pred_BNB)
cr_SVC=classification_report(y_test,pred_SVC)
cr_KNC=classification_report(y_test,pred_KNC)

cm_LG=confusion_matrix(y_test,pred_LG)
cm_DTC=confusion_matrix(y_test,pred_DTC)
cm_RFC=confusion_matrix(y_test,pred_RFC)
cm_BNB=confusion_matrix(y_test,pred_BNB)
cm_SVC=confusion_matrix(y_test,pred_SVC)
cm_KNC=confusion_matrix(y_test,pred_KNC)


Accuracy = pd.DataFrame({
                        "Model": ["Logistic Regression",
                                  "Decision Tree Classifier",
                                  "Random Forset Classifier ",
                                  "Naive Bayes BernoulliNB",
                                  "Support Vector Classifier",
                                  "K Neighbors Classifier"],
    
                       "Accuracy": [round(acc_LG*100,2),
                                    round(acc_DTC*100,2),
                                    round(acc_RFC*100,2),
                                    round(acc_BNB*100,2),
                                    round(acc_SVC*100,2),
                                    round(acc_KNC*100,2)]})
classification_report=pd.DataFrame({
                        "Model": ["Logistic Regression",
                                  "Decision Tree Classifier",
                                  "Random Forset Classifier ",
                                  "Naive Bayes BernoulliNB",
                                  "Support Vector Classifier",
                                  "K Neighbors Classifier"],
                        "Classification Report":[
                                                cr_LG,
                                                cr_DTC,
                                                cr_RFC,
                                                 cr_BNB,
                                                 cr_SVC,
                                                 cr_KNC]})                 

confusion_matrix=pd.DataFrame({   
                                "Model": ["Logistic Regression",
                                          "Decision Tree Classifier",
                                          "Random Forset Classifier ",
                                          "Naive Bayes BernoulliNB",
                                          "Support Vector Classifier",
                                          "K Neighbors Classifier"],
                               
                               "Confusion Matrix":[cm_LG,
                                                   cm_DTC,
                                                   cm_RFC,
                                                   cm_BNB,
                                                   cm_SVC,
                                                   cm_KNC]})
# upload model and metrics
Accuracy.to_csv("Accuracy.csv",index=False)
classification_report.to_csv("classification_report.csv",index=False)
confusion_matrix.to_csv("confusion_matrix.csv",index=False)
joblib.dump(LG,"Logistic Regression.pkl")
joblib.dump(DTC,"Decision Tree Classifier.pkl")
joblib.dump(RFC,"Random Forset Classifier.pkl",compress=3)
joblib.dump(BNB,"Naive Bayes BernoulliNB.pkl")
joblib.dump(SVC,"Support Vector Classifier.pkl")
joblib.dump(KNC,"K Neighbors Classifier.pkl")
