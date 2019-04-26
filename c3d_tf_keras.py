import tensorflow as tf
import numpy as np

actions = [
    "Diving", "Golf-Swing", "Kicking", "Lifting", "Riding-Horse", "Running",
    "SkateBoarding", "Swing-Bench", "Swing-SideAngle", "Walking"
]


def train(x_train, y_train, x_test, y_test):
    # print(x_test.shape)
    num_classes = 10


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv3D(32,kernel_size=(3,3,3), strides=(1,1,1), activation='relu',input_shape=(20,70,70,3)),
        tf.keras.layers.MaxPool3D(pool_size=(2,2,2)),
        tf.keras.layers.Conv3D(64,kernel_size=(3,3,3), strides=(1,1,1), activation='relu'),
        tf.keras.layers.MaxPool3D(pool_size=(2,2,2)),
        tf.keras.layers.Conv3D(128,kernel_size=(3,3,3), strides=(1,1,1), activation='relu'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    # print(y_test)
    model.fit(x_train, y_train,steps_per_epoch=10, epochs=5, verbose=1)
    model.save("preTrained.model")

    predictions = model.predict_classes(x_test)
    print("Starting evaluation...\n")
    for i in range(len(predictions)):
        print("Correct answer: %s" % actions[y_train[i]],"\nPrediction: %s\n" % actions[predictions[i]])

    print("Done!")
    # model.evaluate(x_test, y_test)
