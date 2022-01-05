import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import os 

st.title('Heart Failure Predictor')

model = tf.keras.models.load_model("saved_model")
data = pd.read_csv('heart.csv')
cols = list(data.columns)[:-1]

st.header('Data')
st.dataframe(data)

fig, ax = plt.subplots(2,2)
fig.set_size_inches(10,10)
colormap = matplotlib.colors.ListedColormap(['Green','Red'])
ax[0][0].scatter(data['Age'],data['RestingBP'], cmap = colormap, c =list(data['HeartDisease']))
ax[0][0].set_xlabel('Age')
ax[0][0].set_ylabel('RestingBP')
ax[0][0].legend(['Healthy','Heart Disease'])

ax[0][1].hist(data[data['HeartDisease']==1]['Age'], histtype = 'stepfilled')
ax[0][1].set_xlabel('Age of People with Heart Disease')
ax[0][1].set_ylabel('Count')

ax[1][0].scatter(data['MaxHR'],data['RestingBP'], cmap = colormap, c =list(data['HeartDisease']))
ax[1][0].set_xlabel('MaxHR')
ax[1][0].set_ylabel('RestingBP')
ax[1][0].legend(['Healthy','1'])

ax[1][1].hist(data[data['HeartDisease']==1]['MaxHR'], histtype = 'stepfilled')
ax[1][1].set_xlabel('Max Heart Rate of People with Heart Disease')
ax[1][1].set_ylabel('Count')


st.header('Analysing Data')
if st.checkbox('View Plots'):
    st.pyplot(fig)
    image_lst = []
    for filename in os.listdir('./plot'):
        image_lst.append(Image.open('./plot/'+filename))
    st.image(image_lst)


st.header('Enter Details of the Patient')
with st.form("my_form"):
    age = st.slider('How old is the Patient',0,100)
    st.write('Age of Patient is {}'.format(age))

    #Sex
    sex = st.selectbox('Sex of Patient',data['Sex'].unique())
    st.write('Sex of the Patient is',sex)

    #ChestPainType
    chestpain = st.selectbox('Chest Pain Type',data['ChestPainType'].unique())

    #restingBP
    restingbp = st.slider('Resting Blood Pressure of Patient',0,200)

    #cholestrol
    chol = st.slider('Cholesterol Level of Patient',0,700)

    #fastingbs
    fastingbs = int(st.selectbox('Fasting blood sugar',data['FastingBS'].unique()))
    st.caption('1: if FastingBS > 120 mg/dl, 0: otherwise')

    #restingecg
    restingecg = st.selectbox("Resting ECG of Patient",data['RestingECG'].unique())
    str = 'Resting Electrocardiogram Results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes criteria]'
    st.caption(str)

    #maxhr
    maxhr = st.slider('Maximum Heart Rate of Patient Recorded', 60,200)

    #exercise
    exercise = st.selectbox('Exercise-Induced Angina',data['ExerciseAngina'].unique())
    st.caption('[Y: Yes, N: No]')

    #oldpeak
    oldpeak = st.number_input('Old Peak of Patient',min_value=-3.0,max_value = 7.0, value = 1.000,step = 0.001)
    st.caption('oldpeak = ST [Numeric value measured in depression], enter a number between -3 and 7')

    #st_slope
    stslope = st.selectbox('Slope of Peak Exercise',data['ST_Slope'].unique())
    st.caption('The slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]')
    
    st.form_submit_button()

test_val = [age,sex,chestpain,restingbp,chol,fastingbs,restingecg,maxhr,exercise,oldpeak,stslope]

if st.button('Predict the Patients chance of Heart Disease'):
    df = pd.DataFrame(columns=cols)
    df.loc[len(df.index)] = test_val
    test_dict = df.to_dict()
    for colname in test_dict.keys():
        test_dict[colname]  = tf.convert_to_tensor(list(test_dict[colname].values()))
    prediction = model.predict(test_dict)
    if prediction>=0.5:
        st.subheader('The Patient has a greater chance of Heart Failure with a probability of {:0.02f}%'.format(float(prediction*100)))
    if prediction<0.5:
        st.subheader('The Patient is predicted to be healthy with a probability of {:0.02f}% of having heart failure'.format(float(prediction*100)))
