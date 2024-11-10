
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

directory="D:\Freelance_İşler\Berkay_UYSAL\Odev_2\car_data"
print(os.listdir(directory))

cnn = Sequential()
#Adding 1st Convolution and Pooling Layer
cnn.add(Conv2D(32,kernel_size=(3,3),input_shape=(128,128,3),activation='relu'))
cnn.add(MaxPool2D(pool_size=(2,2)))
cnn.add(Dropout(0.2))
#Adding 2nd Convolution and Pooling Layer
cnn.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
cnn.add(MaxPool2D(pool_size=(2,2)))
cnn.add(Dropout(0.2))
#Adding 3rd Convolution and Pooling Layer
cnn.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
cnn.add(MaxPool2D(pool_size=(2,2)))
cnn.add(Dropout(0.2))
#Adding 4th Convolution and Pooling Layer
cnn.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
cnn.add(MaxPool2D(pool_size=(2,2)))
cnn.add(Dropout(0.2))
#Adding 5th Convolution and Pooling Layer
cnn.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
cnn.add(MaxPool2D(pool_size=(2,2)))
cnn.add(Dropout(0.2))

#Flatten
cnn.add(Flatten())

#Adding Input and Output Layer
cnn.add(Dense(units=256,activation='relu'))
cnn.add(Dense(units=256,activation='relu'))
cnn.add(Dense(units=256,activation='relu'))
cnn.add(Dense(units=196,activation='sigmoid'))

cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Data agumentation
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory('D:\Freelance_İşler\Berkay_UYSAL\Odev_2\car_data\egitim',
                                              target_size=(128,128),
                                              batch_size=32,
                                              class_mode='categorical')
test_data = test_datagen.flow_from_directory('D:\Freelance_İşler\Berkay_UYSAL\Odev_2\car_data\den',
                                              target_size=(128,128),
                                              batch_size=32,
                                              class_mode='categorical')
history = cnn.fit_generator(train_data,
                            steps_per_epoch=100,
                            epochs=30,
                            validation_data=test_data,
                            validation_steps=50)
#Model Saving
cnn.save('cnn.h5')

#Eğitim ve test verilerinin doğruluk grafiği
vals = pd.DataFrame.from_dict(history.history)
vals = pd.concat([pd.Series(range(0,30),name='epochs'),vals],axis=1)
fig,(ax,ax1) = plt.subplots(nrows=2,ncols=1,figsize=(16,16))
sns.lineplot(x='epochs',y='accuracy',data=vals,ax=ax,color='r')
sns.lineplot(x='epochs',y='val_accuracy',data=vals,ax=ax,color='g')
sns.lineplot(x='epochs',y='loss',data=vals,ax=ax1,color='r')
sns.lineplot(x='epochs',y='val_loss',data=vals,ax=ax1,color='g')
ax.legend(labels=['Test Accuracy','Training Accuracy'])
ax1.legend(labels=['Test Loss','Training Loss'])

#Predicition
y_pred=cnn.predict(test_data)


