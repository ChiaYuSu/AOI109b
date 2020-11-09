# Week 2 to Week 4 write-up
## 自動化光學檢測與機器視覺實務
### Machine Learning vs. Deep Learning
<img src="Week 2\MLvsDL.PNG" width="550px" />

### Structure of Course
- Week 1: Explain the syllabus
- **Week 2 - Week 5: ANN**
- Week 6 - Week 8: CNN-1
- Week 9: Midterm
- Week 10 - Week 12: CNN-2
- Week 13 - Week 15: RNN
- Week 16: GAN
- Week 17: Final
- Week 18: Results presentation

### Training the Artificial Neural Network (ANN)
1. 隨機初始化權重，使得權重接近 0 但不等於 0
2. 將第一個觀察數據輸入「輸入層」，每個自變量特徵輸入一個神經元
3. **正向傳播**：神經網路從左至右進行計算，每個神經元得到來自上一層神經元的輸入與權重的運算
4. 誤差函數：計算預測值與實際值的差異 (loss function)
5. **反向傳播**：神經網路從右至左進行計算，依據誤差函數相對於權重的梯度，對每個權重進行更新，學習速率與梯度將決定更新的速度 (optimizer)
6. 對每一組新的觀察數據，重複 Step 1 到 Step 5 (batch size)
7. 當所有數據都輸入神經網路後，稱之為一期 (epoch) 的訓練

## Code Part
#### Dataset download
- [Churn_modelling.csv](https://drive.google.com/file/d/1R5QouDghLLaJzFl_sxcP4bcnBQWFUAND/view?usp=sharing)

#### Importing the libraries
```py
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
```

#### View tensorflow version
```py
tf.__version__
```

### Part 1 - Data Preprocessing
#### Importing the dataset
```py
dataset = pd.read_csv("Churn_modelling.csv")
X = dataset.iloc[:, 3:-1].values # -1 means takes the last column, but does not include the last column
y = dataset.iloc[:, -1].values # -1 means takes the last column
print(X)
print(y)
```
- Result:
```
[[619 'France' 'Female' ... 1 1 101348.88]
 [608 'Spain' 'Female' ... 0 1 112542.58]
 [502 'France' 'Female' ... 1 0 113931.57]
 ...
 [709 'France' 'Female' ... 0 1 42085.58]
 [772 'Germany' 'Male' ... 1 0 92888.52]
 [792 'France' 'Female' ... 1 0 38190.78]]
 [1 0 1 ... 1 1 0]
```
- `y` means column "Exited"

#### Encoding categorical data
##### **Label** Encoding the "Gender" column
```py
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)
```
- Result:
```
[[619 'France' 0 ... 1 1 101348.88]
 [608 'Spain' 0 ... 0 1 112542.58]
 [502 'France' 0 ... 1 0 113931.57]
 ...
 [709 'France' 0 ... 0 1 42085.58]
 [772 'Germany' 1 ... 1 0 92888.52]
 [792 'France' 0 ... 1 0 38190.78]]
```

##### One Hot Encoding the "Geography" column
```py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X[:, 1:]
print(X)
```
- Result:
```
[[0.0 0.0 619 ... 1 1 101348.88]
 [0.0 1.0 608 ... 0 1 112542.58]
 [0.0 0.0 502 ... 1 0 113931.57]
 ...
 [0.0 0.0 709 ... 0 1 42085.58]
 [1.0 0.0 772 ... 1 0 92888.52]
 [0.0 0.0 792 ... 1 0 38190.78]]
```
- `transformers`: The argument is a list of tuples, and the structure of each tuple is: **(name, transformer, columns)**
    - name: The name of the transformer, just start a string
    - transformer: `drop`, `passthrough` or estimator
    - columns: Specify which columns to convert
- `remainder`: `drop`, `passthrough` or estimator, default = `drop`

#### Splitting the dataset into the Training set and Test set
```py
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```
- `test_size`: represents 20% of the data in `X` and `y` as the test set and 80% of the data as the training set
- `random_state`: int or RandomState instance, default = None

#### Feature Scaling (特徵縮放)
```py
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train.shape)
```
- Result:
```
(8000, 11)
```
- `StandardScaler`: Normally distribute the data, the average value will become 0, and the standard deviation will become 1, reducing the influence of outliers (將資料常態分布化，平均值會變為 0、標準差變為 1，使離群值影響降低)
- `fit_trasform`: **Fit to data**, then transform it
- `transform`: Perform **standardization** by **centering** and **scaling**
    - We call `fit_transform()` on the training set. In fact, we have found the mean μ and variance σ^2, that is, we have found the transformation rule. We use this rule on the training set. Similarly, we can directly apply it to the test set. **Therefore, in the processing on the test set, we only need to standardize the data and do not need to fit the data again** (我們在訓練集上調用 `fit_transform()`，其實找到了均值 μ 和方差 σ^2，即我們已經找到了轉換規則，我們把這個規則利用在訓練集上，同樣，我們可以直接將其運用到測試集上，所以在測試集上的處理，我們只需要標準化數據而不需要再次擬合數據。)

### Part 2 - Building the ANN
#### Initializing the ANN
```py
ann = keras.Sequential()
```

#### Adding the input layer and the first hidding layer
```py
ann.add(layers.Dense(units=6, activation='relu', input_dim = 11))
```
- `units`: Positive integer, dimensionality of the output space (大於 0 的整數，代表該層的輸出維度)
- `activation`: Activation function to use. If you don't specify anything, no activation is applied ([激活函數](https://www.tensorflow.org/api_docs/python/tf/keras/activations))
    - `relu`: Applies the **Rectified Linear Unit** (線性整流函數) activation function. If the value is positive, the value is output, if the value is negative, the output is 0 
    <br><img src="Week 2\ReLU.png" width="550px" />
- `input_dim`: Represents the dimensions of the tensor (We have 11 features in `X`)

#### Adding the second hidden layer
```py
ann.add(layers.Dense(units=6, activation='relu'))
```

#### Adding the output layer
```py
ann.add(layers.Dense(units=1, activation='sigmoid'))
```
- `sigmoid`: Sigmoid activation function (Sigmoid 函數), <img src="Week 2\sigmoid.png" width="70px" />. The sigmoid function is the **most frequently used** activation function at the beginning of the deep learning field. The `sigmoid()` function is simply a mapping function, which maps any variable to between [0, 1] (Sigmoid 函數簡單來講就是個映射函數，將任何變量映射到 [0, 1] 之間). The most commonly seen application scenario is when we are training the model to do binary classification (二分類), we only need to set the threshold, and judge the value less than 0.5 as 0 and the value greater than 0.5 as 1. Then we can make a two-class prediction.
<br><img src="Week 2\sigmoid_pic.png" width="550px" />

### Part 3 - Training the ANN
#### Compiling the ANN
```py
ann.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics= ['accuracy'])
```
- `optimizer`: Optimization function (優化函數)
    - `adam`: Adaptive Moment Estimation, more other optimizers can refer to [here](https://www.cnblogs.com/guoyaohua/p/8542554.html)
- `loss`: Loss function (損失函數)
    - `binary_crossentropy`: `binary_crossentropy` loss function, generally used for binary classification (二分類). For other loss functions, please refer to [here](https://www.itread01.com/content/1543994346.html)
- `metrics`: Performance measure (成效衡量指標)
    - `accuracy`: `accuracy` is the easiest indicator in machine learning to evaluate the quality of a model. For other metrics, please refer to [here](https://zhuanlan.zhihu.com/p/95293440)
        - The `accuracy` is the simplest accuracy that everyone knows. For example, we have 6 samples whose true label `y_true` is [0, 1, 3, 3, 4, 2], but is predicted by a model to be [0, 1, 3, 4, 4, 4], that is, `y_pred` = [ 0, 1, 3, 4, 4, 4], then the accuracy of the model = 4/6 = 66.67%

#### Training the ANN on the Training set
```py
r = ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
```
- Result:
```
Epoch 1/100
250/250 [==============================] - 0s 983us/step - loss: 0.7409 - accuracy: 0.5235
Epoch 2/100
250/250 [==============================] - 0s 984us/step - loss: 0.4987 - accuracy: 0.7950
...
Epoch 99/100
250/250 [==============================] - 0s 964us/step - loss: 0.3300 - accuracy: 0.8668
Epoch 100/100
250/250 [==============================] - 0s 935us/step - loss: 0.3304 - accuracy: 0.8656
```

#### Draw historical data of **loss** and **accuracy**
```py
import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['accuracy'], label='accuracy')
plt.legend(loc='best', shadow=True)
```
- Result:
<img src="Week 2\loss_accuracy_plot.png" width="550px" />

- `history`: Keras supports the callback API, in which the History function is called by default, and **the loss and accuracy of each round of training are collected**. If there is a test set, the data of the **test set will also be collected**. The historical data will collect the return value of the `fit()` function in the history object.
- `loc`: Legend position

### Part 4 - Making the predictions and evaluating the model
#### Predicting the result of a single observation (預測單個觀察的結果)
##### Homework
- Use our ANN model to predict if the customer with the following informations will leave the bank:
    - Geography: France
    - Credit Score: 600
    - Gender: Male
    - Age: 40 years old
    - Tenure: 3 years
    - Balance: $60000
    - Number of Products: 2
    - Does this customer have a credit card: Yes
    - Is this customer an Active Member: Yes
    - Estimated Salary: $50000
- Question: So, should we say goodbye to that customer?

##### Homework Solution
```py
print(ann.predict(sc.transform([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
```
- Result:
```
[[False]]
```
- We don't need to say goodbye to the customer

#### Predicting the Test set results
```py
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print("---------- y_pred.reshape ----------")
print(y_pred.reshape(len(y_pred), 1))
print("---------- y_test.reshape ----------")
print(y_test.reshape(len(y_test), 1))
print("------------ concatenate ------------")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))
```
- Result:
```
---------- y_pred.reshape ----------
[[False]
 [False]
 [False]
 ...
 [False]
 [False]
 [False]]
---------- y_test.reshape ----------
[[0]
 [1]
 [0]
 ...
 [0]
 [0]
 [0]]
------------ concatenate ------------
[[0 0]
 [0 1]
 [0 0]
 ...
 [0 0]
 [0 0]
 [0 0]]
```
- `concatenate`: Concate two matrices
    - `axis`: The axis along which the arrays will be joined. If axis is None, arrays are flattened before use. Default is 0.
        - `axis = 0`
        ```py
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6]])
        print(np.concatenate((a, b), axis=0)) # [[1, 2], [3, 4], [5, 6]]
        ```
        - `axis = 1`
        ```py
        print(np.concatenate((a, b), axis=1)) # [[1, 2, 5], [3, 4, 6]]
        ```
- `reshape`: Gives a new shape to an array without changing its data

#### Making the Confusion Matrix
```py
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("---------- confusion_matrix ----------")
print(cm)
print("----------- accuracy_score -----------")
a = accuracy_score(y_test, y_pred)
print(a)
```
- Result:
```
---------- confusion_matrix ----------
[[1510   85]
 [ 201  204]]
----------- accuracy_score -----------
0.857
```
- Confusion Matrix:
<table>
    <tbody>
    <tr>
        <th colspan="2" rowspan="2"></th>
        <th colspan="2">Actual class</th>
    </tr>
    <tr>
        <td>P</td>
        <td>N</td>
    </tr>
    <tr>
        <td rowspan="2">Predicted class</td>
        <td>P</td>
        <td>TP</td>
        <td>FP</td>
    </tr>
    <tr>
        <td>N</td>
        <td>FN</td>
        <td>TN</td>
    </tr>
    </tbody>
</table>

- `accuracy_score`:  *(**T**rue **P**ositive + **T**rue **N**egative) / Total*

### Part 5 - Evaluating, improving and tuing the ANN
#### Evaluating the ANN (評估)
```py
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
```

```py
ann_cv = KerasClassifier(build_fn = build_classifier, batch_size = 32, epochs = 100)
accuracies = cross_val_score(estimator = ann_cv, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std() # Standard deviation
print("----------- accuracies -----------")
print(accuracies)
print("----------- Mean + Variance -----------")
print('Mean = ', mean, '; Variance = ', variance)
```

- Result:
```
----------- accuracies -----------
[0.85750002 0.85874999 0.86250001 0.83999997 0.81625003 0.83125001
 0.83125001 0.83375001 0.81625003 0.85124999]
----------- Mean + Variance -----------
Mean =  0.8398750066757202 ; Variance =  0.016157904139983605
```

#### Tuning the ANN (調整)
```py
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
```
```py
ann_gs = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32], 'epochs': [100, 500], 'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = ann_gs, param_grid = parameters, scoring = 'accuracy', cv = 5)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
```