import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import streamlit as st

ohe = joblib.load('onehot_encoder_geography.pkl')
lb = joblib.load('label_encoder_gender.pkl')
scalar = joblib.load('scalar.pkl')
model=tf.keras.models.load_model('churn_model.h5')


st.title("Customer Churn Prediction")

geography=st.selectbox("geography",ohe.categories_[0])
gender=st.selectbox("gender",lb.classes_)
age=st.slider('age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated_Salary')
tenure=st.slider('tenure',0,10)
num_of_products=st.slider('num_of_products',1,4)
has_cr_card=st.selectbox('has credit card',[0,1])
is_active_member=st.selectbox('is active member',[0,1])


input_data={
  'CreditScore':credit_score,
  'Geography':geography,
  'Gender':gender,
  'Age':age,
  'Tenure':tenure,
  'Balance':balance,
  'NumOfProducts':num_of_products,
  'HasCrCard':has_cr_card,
  'IsActiveMember':is_active_member,
  'EstimatedSalary':estimated_salary
}


input_df=pd.DataFrame([input_data])
geo_ohe=ohe.transform(input_df[['Geography']])
geo_df=pd.DataFrame(geo_ohe.toarray(),columns=ohe.get_feature_names_out(['Geography']))

input_df=pd.concat([input_df.drop('Geography',axis=1),geo_df],axis=1)

input_df['Gender']=lb.transform(input_df[['Gender']])

scaled_data=scalar.transform(input_df)
pred=model.predict(scaled_data)

pred_probab=pred[0][0]

if pred_probab>0.5:
    st.write(f'The customer is likely to churn with a probability of {pred_probab:.2f}')

else:
    st.write(f'The customer is unlikely to churn with a probability of {pred_probab:.2f}')