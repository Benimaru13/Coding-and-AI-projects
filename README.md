# Coding-and-AI-projects
Building AI Course Project

# Handwriting Classifier

Final project for the Building AI course

## Summary
This model is an application of AI towards image processing. It uses machine learning and convolutional neural networks to analyze hand writing samples between different people and classify them appropriately.
For this project, I am training it on data from my family members, and I aim to implement computer vision for a life feed of the writing instead.
Describe briefly in 2-3 sentences what your project is about. About 250 characters is a nice length! 


## Background

Which problems does your idea solve? How common or frequent is this problem? What is your personal motivation? Why is this topic important or interesting?
This project was born from my desire to apply AI towards a problem I faced around me. I remember reading the writing of my siblings, and pondering of how similar they seemed and how much of a hard time I had distinguishing it. That was when the idea came to me; build an AI model that can be used to appropriately distinguish their writing samples. As I refined my idea, I realized the application of this model beyond this problem. In the realm of forensics, innovative technology that can make handwriting identification much easier is much appreciated and this would improve the quality of detective work. Additionally, various other advancements could be made in this realm, like using distinct handwriting gestures to access something or perform a specific task. The potential of this is limitless and I am excited by the various ways it can be implemented

## How is it used?

Describe the process of using the solution. In what kind situations is the solution needed (environment, time, etc.)? Who are the users, what kinds of needs should be taken into account?
If you were a forensic scientist, a detective, an innovator with an idea around this, or just an average person, you would use this model to make predictions based on the trainded data samples. It might need to be finetuned to exact cause but it is mostly case-dependent. 


Images will make your README look nice!
Once you upload an image to your repository, you can link link to it like this (replace the URL with file path, if you've uploaded an image to Github.)
![Writing](https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg](https://penvibe.com/wp-content/uploads/2009/10/Handwriting-Sample-Text.jpg)


<img src="https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg](https://penvibe.com/wp-content/uploads/2009/10/Handwriting-Sample-Text.jpg" width="300">
<img src="https://www.researchgate.net/profile/Sung-Hyuk-Cha/publication/228846030/figure/fig3/AS:667709830361089@1536205837124/Some-copybook-handwriting-styles-for-the-word-beheaded.png" width="300">

# Code block samples
```
import numpy as np
from tensorflow.keras.datasets import mnist

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

import matplotlib.pyplot as plt

index = 12
sample = x_test[index].reshape(1, 28, 28, 1)
prediction = model.predict(sample)
predicted_label = np.argmax(prediction)

plt.imshow(x_test[index].reshape(28, 28), cmap="gray")
plt.title(f"Predicted: {predicted_label}")
plt.show()
```


## Data sources and AI methods
I  manually create the data samples for my data.

## Challenges

What does your project _not_ solve? Which limitations and ethical considerations should be taken into account when deploying a solution like this?
it attempts to classify the data and does not offer a prediction instead. This could lead to false positives or negatives, and, depending on the situation, could cause logistic problems.

## What next?

How could your project grow and become something even more? What kind of skills, what kind of assistance would you  need to move on? 
I need to expand my data set and experiment on other forms of deep learning to achieve better results

## Acknowledgments

*  sources of inspiration - personal challenges
