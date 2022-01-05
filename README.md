# Heart Failure Predictor

## About
A Web UI deployed Dense Neural Network Model Made using Tensorflow that predicts whether the patient is healthy or has chances of heart disease with probability.

### Dataset
The Dataset used is the [Heart Failure Prediction](https://www.kaggle.com/fedesoriano/heart-failure-prediction) Dataset from kaggle.
-Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.
-People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.
-This dataset was created by combining different datasets already available independently but not combined before. In this dataset, 5 heart datasets are combined over 11 common features which makes it the largest heart disease dataset available so far for research purposes.
- For more inforamtion visit : https://www.kaggle.com/fedesoriano/heart-failure-prediction

### UI Demonstration
This is an interactive website made using a python library called streamlit that implements the neural network model. You can view dataset (scrollable and explandable), several plots that have good insights on data. For prediction, user has to input various details about the patient being tested into the form. User has to provide details like age,blood pressure, maximum heart rate which can be filled using numerical inputs, sliders for numerical values and a selectbox for categorical options. Click the submit button and then click the Predict button to infer whether the patient has chances of heart disease and the probablity of having a heart disease.

https://user-images.githubusercontent.com/69198671/148185697-dff226fd-c365-4f53-aa56-406c2ff7af17.mp4

>To run this ui open the directory in command terminal and use the command `streamlit run interface.py`

##### Attribute Information
- Age: age of the patient (years)
- Sex: sex of the patient (M: Male, F: Female)
- ChestPainType: chest pain type (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)
- RestingBP: resting blood pressure (mm Hg)
- Cholesterol: serum cholesterol (mm/dl)
- FastingBS: fasting blood sugar (1: if FastingBS > 120 mg/dl, 0: otherwise)
- RestingECG: resting electrocardiogram results (Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria)
- MaxHR: maximum heart rate achieved (Numeric value between 60 and 202)
- ExerciseAngina: exercise-induced angina (Y: Yes, N: No)
- Oldpeak: oldpeak = ST (Numeric value measured in depression)
- ST_Slope: the slope of the peak exercise ST segment (Up: upsloping, Flat: flat, Down: downsloping)
- HeartDisease: output class (1: heart disease, 0: Normal)

### DNN Model (Keras)
The model is used is shown in the codeblock below:

```
model = tf.keras.Sequential([
    layers.DenseFeatures(feature_cols.values()),
    layers.BatchNormalization(input_dim = (len(feature_cols.keys()),)),
    layers.Dense(256, activation='relu',kernel_regularizer='l2'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu',kernel_regularizer='l2'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),loss ='binary_crossentropy',metrics=['accuracy',tf.keras.metrics.AUC()])
```

The model is very dense and the dataset is small, so as to avoid overfitting various regularization methods are used like:
-Batch Normalization
-Dropout Layers
-L2 Regularization
-Early Stopping Callback

Feature Columns are used and datasets are of converted into tf.data.Dataset type for faster processing.
Age Feature is bucketized. Whereas all other numerical features are passed as numerical feature columns. Categorical as categorical feature columns.

The model has an accuracy of approximately 98% on Test Dataset and AUC(area under roc curve) of 1.00.
The model training is visualized in Tensorboard.

### About files in repo
- `pred_model.ipynb`: Jupyter Notebook of the code used to build the DNN and exploratory data analysis using pandas,matplotlib and seaborn
- `interface.py`: Used to run the website for interactive UI
- `model_py.py`: DNN Model code available in .py format
- `saved_model folder`: Contains the DNN Model saved in .pb format that can be imported into any python file.
