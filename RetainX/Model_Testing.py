import streamlit  as st
import joblib
import pandas as pd 

# load a model 
LG=joblib.load("RetainX/Logistic Regression.pkl")
DTC=joblib.load("RetainX/Decision Tree Classifier.pkl")
RFC=joblib.load("RetainX/Random Forset Classifier.pkl")
BNB=joblib.load("RetainX/Naive Bayes BernoulliNB.pkl")
SVC=joblib.load("RetainX/Support Vector Classifier.pkl") 
KNC=joblib.load("RetainX/K Neighbors Classifier.pkl")
# Create radio button choose the model
st.sidebar.image("RetainX/RetainX_image.png")
model_option=st.sidebar.radio("Choase the Model",["Metrics","Logistic Regression","Decision Tree Classifier",
                                                  "Random Forest Classifier","Naive Bayes BernoulliNB",
                                                  "Support Vector Classifier","K Neighbors Classifier"])

if model_option=="Metrics":
    st.table(pd.read_csv("RetainX/Accuracy.csv"))
    st.table(pd.read_csv("RetainX/classification_report.csv"))
    st.table(pd.read_csv("RetainX/confusion_matrix.csv"))
# input the user    
age = st.number_input("العمر", min_value=1, max_value=120, step=1)
if age > 50:
    senior_citizen = 1
else:
    senior_citizen = 0
tenure = st.number_input("عدد أشهر الاشتراك ", min_value=0)
monthly_charges = st.number_input("المبلغ الشهري ", min_value=0.0)
total_charges = st.number_input("إجمالي المبالغ المدفوعة ", min_value=0.0)
gender = st.selectbox("الجنس", ["ذكر", "أنثى"])
gender_Male = 1 if gender == "ذكر" else 0
partner = st.selectbox("هل لديه شريك؟", ["لا", "نعم"])
Partner_Yes = 1 if partner == "نعم" else 0
dependents = st.selectbox("هل لديه معالين؟", ["لا", "نعم"])
Dependents_Yes = 1 if dependents == "نعم" else 0
phone_service = st.selectbox("خدمة هاتف؟", ["لا", "نعم"])
PhoneService_Yes = 1 if phone_service == "نعم" else 0
paperless = st.selectbox("فاتورة إلكترونية؟", ["لا", "نعم"])
PaperlessBilling_Yes = 1 if paperless == "نعم" else 0
multiple_lines = st.selectbox("أكثر من خط اتصال؟", ["لا", "نعم"])
MultipleLines_Yes = 1 if multiple_lines == "نعم" else 0
online_security = st.selectbox("حماية الإنترنت؟", ["لا", "نعم"])
OnlineSecurity_Yes = 1 if online_security == "نعم" else 0
online_backup = st.selectbox("نسخ احتياطي؟", ["لا", "نعم"])
OnlineBackup_Yes = 1 if online_backup == "نعم" else 0
device_protection = st.selectbox("حماية الجهاز؟", ["لا", "نعم"])
DeviceProtection_Yes = 1 if device_protection == "نعم" else 0
tech_support = st.selectbox("دعم فني؟", ["لا", "نعم"])
TechSupport_Yes = 1 if tech_support == "نعم" else 0
stream_tv = st.selectbox("بث تلفزيون؟", ["لا", "نعم"])
StreamingTV_Yes = 1 if stream_tv == "نعم" else 0
stream_movies = st.selectbox("بث أفلام؟", ["لا", "نعم"])
StreamingMovies_Yes = 1 if stream_movies == "نعم" else 0
input_user=pd.DataFrame({"SeniorCitizen" :[senior_citizen],
                         "tenure":[tenure],
                         "MonthlyCharges":[monthly_charges],
                         "TotalCharges":[total_charges],
                         "gender_Male":[gender_Male],
                         "Partner_Yes":[Partner_Yes],
                         "Dependents_Yes":[Dependents_Yes],
                         "PhoneService_Yes":[PhoneService_Yes],
                         "PaperlessBilling_Yes":[PaperlessBilling_Yes],
                         "MultipleLines_Yes":[MultipleLines_Yes],
                         "OnlineSecurity_Yes":[OnlineSecurity_Yes],
                         "OnlineBackup_Yes":[OnlineBackup_Yes],
                         "DeviceProtection_Yes":[DeviceProtection_Yes],
                         "TechSupport_Yes":[TechSupport_Yes],
                         "StreamingTV_Yes":[StreamingTV_Yes],
                         "StreamingMovies_Yes":[StreamingMovies_Yes]})
# selected a model 
if model_option == "Logistic Regression":
    if st.button("Predict Logistic Regression"):
        Logistic=LG.predict(input_user)

        if Logistic[0]==0:
            st.success("المستخدم رح يبقى في الخدمه")
        else :
            st.error("المستخدم رح يترك في الخدمه")

elif model_option == "Decision Tree Classifier":
    if st.button("Predict Decision Tree Classifier"):
        Decision=DTC.predict(input_user)
        if Decision[0]==0:
            st.success("المستخدم رح يبقى في الخدمه")
        else :
            st.error("المستخدم رح يترك في الخدمه")

elif model_option == "Random Forest Classifier":
    if st.button("Predict Random Forest Classifier"):
        Random=RFC.predict(input_user)
        if Random[0]==0:
            st.success("المستخدم رح يبقى في الخدمه")
        else :
            st.error("المستخدم رح يترك في الخدمه")

elif model_option == "Naive Bayes BernoulliNB":
    if st.button("Predict Naive Bayes BernoulliNB"):
        BernoulliNB=BNB.predict(input_user)
        if BernoulliNB[0]==0:
            st.success("المستخدم رح يبقى في الخدمه")
        else :
            st.error("المستخدم  رح يترك في الخدمه")

elif model_option == "Support Vector Classifier":
    if st.button("Predict Support Vector Classifier"):
        Support=SVC.predict(input_user)
        if Support[0]==0:
            st.success("المستخدم رح يبقى في الخدمه")
        else :
            st.error("المستخدم رح يترك في الخدمه")

elif model_option == "K Neighbors Classifier":
    if st.button("Predict K Neighbors Classifier"):
        K=KNC.predict(input_user)
        if K[0]==0:
            st.success("المستخدم رح يبقى في الخدمه")
        else :
            st.error("المستخدم رح يترك في الخدمه")



