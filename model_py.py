# import base64
# from facets_overview.gecmdneric_feature_statistics_generator import GenericFeatureStatisticsGenerator
import os
import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow_addons as tfa
# import tensorflow_data_validation as tfdv
import seaborn as sns
import matplotlib.pyplot as plt
# from platform import python_version

print('Tensorflow Version: {}'.format(tf.__version__))
data = pd.read_csv('heart.csv')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 20
label_name = 'HeartDisease'

def df_to_dataset(df,shuffle=True,batch=True):
    df_copy = df.copy()
    label = df_copy.pop(label_name)
    ds = tf.data.Dataset.from_tensor_slices((dict(df_copy),label))
    if batch:
        ds = ds.batch(batch_size)
    if shuffle:
        ds = ds.shuffle(buffer_size = len(df_copy),seed=30)
    ds = ds.prefetch(1)
    return ds

def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        ds = ds.shuffle(shuffle_size,seed=30)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    train_ds = train_ds.batch(batch_size).prefetch(1)
    val_ds = val_ds.batch(batch_size).prefetch(1)
    test_ds = test_ds.batch(batch_size).prefetch(1)
    
    return train_ds, val_ds, test_ds

ds = df_to_dataset(data,batch=False)
train_ds,val_ds,test_ds = get_dataset_partitions_tf(ds,ds_size = len(ds))
print('Batches in Training, Validation and Test Datasets:',len(train_ds),len(val_ds),len(test_ds))

num_cols = ['RestingBP','Cholesterol','MaxHR','Oldpeak',]

cat_cols = ['Sex','ChestPainType','RestingECG','MaxHR','FastingBS','ExerciseAngina','ST_Slope']

bucketized_col = 'Age'

feature_cols = {colname: tf.feature_column.numeric_column(colname) for colname in num_cols}

for colname in cat_cols:
    fc = tf.feature_column.categorical_column_with_vocabulary_list(colname, data[colname].unique())
    feature_cols[colname] = tf.feature_column.indicator_column(fc)

nbuckets = np.arange(data.describe().Age['min'],data.describe().Age['max'])
fc = tf.feature_column.numeric_column('Age')
feature_cols['Age'] = tf.feature_column.bucketized_column(fc, boundaries=list(nbuckets))
assert (len(list(data.columns))-1) == len(list(feature_cols.keys()))
from tensorflow.keras import layers
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
callbacks = tf.keras.callbacks.EarlyStopping(
  patience=30,
  min_delta=0.001,
  restore_best_weights=True
)

history =  model.fit(
    train_ds,
    validation_data=val_ds,
    epochs = 500,
    verbose = 1,
    callbacks = [callbacks]
)

plt.figure(figsize = (16,10))

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
auc = history.history['auc']
val_auc = history.history['val_auc']
epochs = range(len(loss))

# plt.subplot(2,2,1)
# plt.plot(epochs,acc,val_acc)
# plt.legend(['Accuracy','Validation Accuracy'])
# plt.title('Accuracy Curve')

# plt.subplot(2,2,2)
# plt.plot(epochs,loss,val_loss)
# plt.legend(['Loss','Validation Loss'])
# plt.title('Loss Curve')

# plt.subplot(2,2,3)
# plt.plot(epochs,auc,val_auc)
# plt.legend(['AUC','Validation AUC'])
# plt.title('AUC Curve')
# plt.show()

print('Models Accuracy on Test Dataset is : {0:.3f}'.format(model.evaluate(test_ds)[1]))

model.save('saved_model')
