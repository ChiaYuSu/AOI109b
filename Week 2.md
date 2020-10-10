# Week 2 write-up
## 自動化光學檢測與機器視覺實務
### Machine Learning vs. Deep Learning
<img src="Week 2\MLvsDL.PNG" width="550px" />

### Structure of Course
| Week | Contents |
|------|----------|
| 1    | 說明     |
| 2    | ANN      |
| 3    | ANN      |
| 4    | ANN      |
| 5    | ANN      |
| 6    | CNN-1    |
| 7    | CNN-1    |
| 8    | CNN-1    |
| 9    | 期中     |
| 10   | CNN-2    |
| 11   | CNN-2    |
| 12   | CNN-2    |
| 13   | RNN      |
| 14   | RNN      |
| 15   | RNN      |
| 16   | GAN      |
| 17   | 期末     |
| 18   | 成果     |

### Training the Artificial Neural Network (ANN)
1. 隨機初始化權重，使得權重接近 0 但不等於 0
2. 將第一個觀察數據輸入「輸入層」，每個自變量特徵輸入一個神經元
3. **正向傳播**：神經網路從左至右進行計算，每個神經元得到來自上一層神經元的輸入與權重的運算
4. 誤差函數：計算預測值與實際值的差異 (loss function)
5. **反向傳播**：神經網路從右至左進行計算，依據誤差函數相對於權重的梯度，對每個權重進行更新，學習速率與梯度將決定更新的速度 (optimizer)
6. 對每一組新的觀察數據，重複 Step 1 到 Step 5 (batch size)
7. 當所有數據都輸入神經網路後，稱之為一期 (epoch) 的訓練

## Code Part
#### Importing the libraries
```py
import tensorflow as tf
import pandas as pd
import numpy as np
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
ann = tf.keras.models.Sequential()
```

#### Adding the input layer and the first hidding layer
```py
ann.add(tf.keras.layers.Dense(units=6, activation='relu', input_dim = 11))
```
- `units`: Positive integer, dimensionality of the output space (大於 0 的整數，代表該層的輸出維度)
- `activation`: Activation function to use. If you don't specify anything, no activation is applied ([激活函數](https://www.tensorflow.org/api_docs/python/tf/keras/activations))
    - `relu`: Applies the **Rectified Linear Unit** (線性整流函數) activation function. If the value is positive, the value is output, if the value is negative, the output is 0 
    <br><img src="Week 2\ReLU.png" width="550px" />
- `input_dim`: Represents the dimensions of the tensor (We have 11 features in `X`)

#### Adding the second hidden layer
```py
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
```

#### Adding the output layer
```py
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```
- `sigmoid`: Sigmoid activation function (Sigmoid 函數), <img src="Week 2\sigmoid.png" width="70px" />. The sigmoid function is the **most frequently used** activation function at the beginning of the deep learning field. The `sigmoid()` function is simply a mapping function, which maps any variable to between [0, 1] (Sigmoid 函數簡單來講就是個映射函數，將任何變量映射到 [0, 1] 之間). The most commonly seen application scenario is when we are training the model to do binary classification (二分類), we only need to set the threshold, and judge the value less than 0.5 as 0 and the value greater than 0.5 as 1. Then we can make a two-class prediction.
<br><img src="Week 2\sigmoid_pic.png" width="550px" />
