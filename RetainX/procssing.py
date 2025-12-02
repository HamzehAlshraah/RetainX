import pandas as pd
#Read the data
data =pd.read_csv(r"N:\RetainX\Telco_Cusomer_Churn.csv")
# transform type of data straing to numeric
data["TotalCharges"]=pd.to_numeric(data["TotalCharges"],errors="coerce")
# Fill in the missing values ​​in the parameter so that it is not affected by outliers.
data["TotalCharges"].fillna(data["TotalCharges"].median(),inplace=True)
# Replace "No internet service" with "No" while keeping other values unchanged
raplce={"No internet service": "No",
        "No":"No",
        'Yes':'Yes'}
column= ["MultipleLines" , "InternetService", "OnlineSecurity" ,"OnlineBackup" , 
          "DeviceProtection" ,"TechSupport" ,"StreamingTV" , "StreamingMovies" 
          ,"Contract" , "PaymentMethod"]
for col in column :
    data[col]=data[col].map(raplce)  
#one hot encoding
columnss=["gender","Partner","Dependents","PhoneService","PaperlessBilling","Churn","MultipleLines" , "InternetService", "OnlineSecurity" ,"OnlineBackup" , 
          "DeviceProtection" ,"TechSupport" ,"StreamingTV" , "StreamingMovies" 
          ,"Contract" , "PaymentMethod"]
data=pd.get_dummies(data,columns=columnss,drop_first=True,dtype="int")
# Deleting the column is pointless.
data=data.drop(["customerID"],axis=1)
#download data  after clean and procssing 
data.to_csv(r"N:\RetainX\clean_data.csv",index=False)
