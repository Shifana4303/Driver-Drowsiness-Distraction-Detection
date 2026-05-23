import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os



datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,      
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    'dataset yawn',            
    target_size=(64,64),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'dataset yawn',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)



model = models.Sequential([

    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')   
])


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)



model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)


if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/yawn_model.h5')

print("✅ Yawn model trained and saved successfully!")