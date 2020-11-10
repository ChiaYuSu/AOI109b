# Week 6 to Week 8 write-up
## ANN MNIST
## Code Part
#### Import library
```py
import tensorflow as tf
print(tf.__version__)
```
- Result:
```
2.3.0
```

#### Import dataset
```py
# Load in the data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```
- Result:
```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step
```
- Dataset:
    - **training set**: 60,000 28x28 pixel grayscale images
    - **test set**: 10,000 images of the same specification, with a total of 10 types of digital labels
- 2 tuples:
    - `x_train`, `x_test`: The **grayscale image** represented by the uint8 array, the size is **(num_samples, 28, 28)**
    - `y_train`, `y_test`: The **numeric label** represented by the uint8 array **(integer ranging from 0-9)**, the size is **(num_samples,)**
 - `/ 255.0`: For normalized (歸一化)

#### Print dataset shape
```py
print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)
print("y_train.shape", y_train.shape)
print("y_test.shape", y_test.shape)
```
- Result:
```
x_train.shape: (60000, 28, 28)
x_test.shape: (10000, 28, 28)
y_train.shape (60000,)
y_test.shape (10000,)
```

#### Build ANN
1. input shape = 28 x 28
2. units of first layer = 128 with relu
3. dropout = 20%
4. units of output layer = 10 with softmax
5. optimizer = adam'
6. loss = sparse_categorical_crossentropy
7. metrics = accuracy

```py
# Build the model - 1
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
])
```
- `Dropout`: A method to **prevent** neural network from **overfitting**
    - **Dropout can only be used in training**, you can't dropout during testing, you need to use a complete network test
- `softmax`: Softmax regression is to use the Softmax operation to make the probability distribution of the last layer output equal to 1 (Softmax 回歸是使用 Softmax 運算使得最後一層輸出的機率分佈總和為 1)
    - logistic regression vs. softmax regression
    <br><img src="Week 6\lr_vs_sr.PNG" width="300px" />
- `categorical_crossentropy` vs `sparse_categorical_crossentropy`
    - If labels are **one-hot encoding**, use **`categorical_crossentropy`**
        - e.g.: [0, 1, 0, 1, 0]
    - If your tagets are **digitally encoded**, use **`sparse_categorical_crossentropy`**
        - e.g.: [2, 0, 1, 5, 19]

#### Training Model
1. epochs = 10
2. use validation data

```py
# Train the model
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
```
- Result:
```
Epoch 1/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.2904 - accuracy: 0.9161 - val_loss: 0.1359 - val_accuracy: 0.9613
Epoch 2/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1414 - accuracy: 0.9590 - val_loss: 0.1029 - val_accuracy: 0.9696
Epoch 3/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1072 - accuracy: 0.9675 - val_loss: 0.0864 - val_accuracy: 0.9742
Epoch 4/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0869 - accuracy: 0.9730 - val_loss: 0.0785 - val_accuracy: 0.9768
Epoch 5/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0753 - accuracy: 0.9764 - val_loss: 0.0728 - val_accuracy: 0.9779
Epoch 6/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0647 - accuracy: 0.9791 - val_loss: 0.0759 - val_accuracy: 0.9766
Epoch 7/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0568 - accuracy: 0.9819 - val_loss: 0.0698 - val_accuracy: 0.9790
Epoch 8/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0517 - accuracy: 0.9827 - val_loss: 0.0699 - val_accuracy: 0.9809
Epoch 9/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0483 - accuracy: 0.9841 - val_loss: 0.0717 - val_accuracy: 0.9805
Epoch 10/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0449 - accuracy: 0.9848 - val_loss: 0.0744 - val_accuracy: 0.9790
```
- `validation_data`: A tuple of the form (X, y) or (X, y, sample_weights) is the specified validation set

#### Plot curves
```py
# Plot loss per iteration (每次迭代損失)
import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

# Plot accuracy per iteration (每次迭代的準確度)
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
```
- Result:
    - Loss per iteration:
    <br><img src="Week 6\plot-1.png" width="300px" />
    - Accuracy per iteration:
    <br><img src="Week 6\plot-2.png" width="300px" />
- `loss`: Loss value of **training set**
- `val_loss`: Loss value of **validation set**
- `loss` vs. `val_loss`:
    - `loss` ↓, `val_loss` ↓: The training network is normal, the **best case**
    - `loss` ↓, `val_loss` stable: Network over-fitting, dropout and max pooling can be added
    - `loss` stable, `val_loss` ↓: The data set has serious problems, it is recommended to reselect the dataset
    - `loss` stable, `val_loss` stable: Learning process encounters a bottleneck, it is necessary to reduce the learning rate or the number of batches
    - `loss` ↑, `val_loss` ↑: Network structure design problems, improper setting of training hyperparameters, data sets need to be cleaned, the **worst case**

#### Evaluate model (評估模型)
```py
# Evaluate the model
print(model.evaluate(x_test, y_test))
```
- Result:
```
313/313 [==============================] - 0s 1ms/step - loss: 0.0687 - accuracy: 0.9802
[0.06869552284479141, 0.9801999926567078]
```
- `evaluate`: Returned is the **test loss value** and the **test accuracy value**
```py
# Print the confusion matrix
from sklearn.metrics import confusion_matrix
print("-------------------- predict --------------------")
p_test = model.predict(x_test)
print(p_test)
print("-------------------- argmax --------------------")
p_test = p_test.argmax(axis=1)
print(p_test)
print("-------------------- confusion_matrix --------------------")
print(confusion_matrix(y_test, p_test))
```
- Result:
```
-------------------- predict --------------------
[[1.07882840e-08 4.57160088e-10 1.82849840e-06 ... 9.99948144e-01
  1.88240570e-08 2.12869645e-06]
 [6.20885898e-10 2.05201068e-05 9.99978185e-01 ... 2.55343300e-14
  2.87087758e-08 1.07379650e-19]
 [2.76707937e-08 9.99255359e-01 1.59466035e-05 ... 3.20466293e-04
  3.97870142e-04 9.75069625e-08]
 ...
 [1.02920234e-16 1.36563981e-12 3.54045203e-15 ... 2.72104700e-07
  4.51266940e-10 9.63035745e-07]
 [6.94468268e-12 1.33233550e-12 7.91420067e-15 ... 4.27693089e-11
  2.10271128e-06 2.10204727e-12]
 [2.11599449e-08 1.49794590e-15 2.40332163e-08 ... 2.98902252e-18
  3.26038363e-11 4.63211446e-13]]
-------------------- argmax --------------------
[7 2 1 ... 4 5 6]
-------------------- confusion_matrix --------------------
[[ 973    0    0    0    1    1    1    1    3    0]
 [   0 1127    2    1    0    0    1    0    4    0]
 [   2    2 1004    5    1    0    3    4   11    0]
 [   1    0    2  995    0    2    0    6    2    2]
 [   0    0    6    0  965    0    1    2    1    7]
 [   2    1    0   11    2  868    2    1    4    1]
 [   8    3    0    1    4    2  939    0    1    0]
 [   2    4    9    3    0    0    0 1002    2    6]
 [   4    1    1    7    4    6    0    3  944    4]
 [   2    2    0    3    6    4    0    6    1  985]]
```
- `predict`: Predict the probability of which category (number 0 - 9) the sample belongs to (預測照片是哪個數字)
- `argmax`: Find the index of the maximum value of each row (將最大的值的 index 找出)
- `confusion_matrix(true label, predict label)`

#### Show some misclassified examples
```py
import numpy as np

misclassified_idx = np.where(p_test != y_test)[0]
print(misclassified_idx.shape)
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i], cmap='gray')
plt.title("True label: %s Predicted: %s" % (y_test[i], p_test[i]))
```
- Result:
```
(198,)
Text(0.5, 1.0, 'True label: 4 Predicted: 7')
```
<br><img src="Week 6\plot-3.png" width="300px" />
- `(198,)`: 198 prediction errors out of 10,000 data 