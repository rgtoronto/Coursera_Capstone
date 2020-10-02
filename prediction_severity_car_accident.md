# Predicting the Severity of Car Accident

## 1. Introduction

### 1.1 Background
With the development of high technology, there are many software companies started focusing on the AI field. We already have some mature navigation software either for GPS devices, or mobile application and others that can connect to vehicles. Imagine if we are going to drive to another city to visit our friend. before we start driving, checking on the navigation tool from which it can provide overall driving conditions based on the weather, road condition, light condition etc. It can predict the severity of an incident in a certain similiar circumstance and dynamically adjust the data transactions on a location basis during the driving. This will somehow help everyone avoid the injure tragic happening by avoiding one or more of the factors.

### 1.2 Problem
Driving a car can speed up the time to arrive at a place, it can help people moving faster, moreover, it also provides a lot of job opportunities. At the same time, driving safe also becomes an important topic for the community.
This project aims to predict the severity of the car accident based on the surrounding circumstance, and find out the factors that most determine the severity.

### 1.3 Interest
When a car is close to a place that satisfies the factors that ever had a historical accident, it can push a message to warn the driver or passage who is using the navigation tool. Also for some organizations or communities, the government, can provide a warning sign or notice to warn drivers to pay attention to avoid the accident.

## 2. Data visualization and pre-processing

### 2.1 Data sources

For this project, we are using the car accident report from Seattle city from `2004–01–01` to `2019–05–20`, it listed out `194673` records, and `37` columns in total. In this dataset, it described the accident severity, address type, road condition, light condition, coordinates etc. I noticed there are missing values for certain columns, this will be covered in the data cleaning section.

### 2.2 Data Cleaning

The data was downloaded from the provided [link](https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv). As I mentioned in the last section, there are missing values in some volumes, if we do the following command, we can see 13 columns match this criterion. they are `X`, `Y`, `INTKEY`, `LOCATION`, `EXCEPTRSNCODE`, `EXCEPTRSNDESC`, `COLLISIONTYPE`, `JUNCTIONTYPE`, `INATTENTIONIND`, `SDOTCOLNUM`, `SPEEDING`, `ST_COLCODE`, `ST_COLDESC`.

```
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 194673 entries, 0 to 194672
Data columns (total 38 columns):
SEVERITYCODE      194673 non-null int64
X                 189339 non-null float64
Y                 189339 non-null float64
OBJECTID          194673 non-null int64
INCKEY            194673 non-null int64
COLDETKEY         194673 non-null int64
REPORTNO          194673 non-null object
STATUS            194673 non-null object
ADDRTYPE          192747 non-null object
INTKEY            65070 non-null float64
LOCATION          191996 non-null object
EXCEPTRSNCODE     84811 non-null object
EXCEPTRSNDESC     5638 non-null object
SEVERITYCODE.1    194673 non-null int64
SEVERITYDESC      194673 non-null object
COLLISIONTYPE     189769 non-null object
PERSONCOUNT       194673 non-null int64
PEDCOUNT          194673 non-null int64
PEDCYLCOUNT       194673 non-null int64
VEHCOUNT          194673 non-null int64
INCDATE           194673 non-null object
INCDTTM           194673 non-null object
JUNCTIONTYPE      188344 non-null object
SDOT_COLCODE      194673 non-null int64
SDOT_COLDESC      194673 non-null object
INATTENTIONIND    29805 non-null object
UNDERINFL         189789 non-null object
WEATHER           189592 non-null object
ROADCOND          189661 non-null object
LIGHTCOND         189503 non-null object
PEDROWNOTGRNT     4667 non-null object
SDOTCOLNUM        114936 non-null float64
SPEEDING          9333 non-null object
ST_COLCODE        194655 non-null object
ST_COLDESC        189769 non-null object
SEGLANEKEY        194673 non-null int64
CROSSWALKKEY      194673 non-null int64
HITPARKEDCAR      194673 non-null object
dtypes: float64(4), int64(12), object(22)
memory usage: 56.4+ MB
```
I quickly check on the `SERVERITYCODE`, found the data also is unevenly distributed.
```
1    136485
2     58188
Name: SEVERITYCODE, dtype: int64
```

### 2.3 Feature Selection
Based on the observation of the raw data, I decided to select a few valuable columns for severity prediction. At first, I chose the following columns:
`df = df[['SEVERITYCODE','ADDRTYPE','ROADCOND','LIGHTCOND','INCDATE']].copy()`
and then cleaned the rows that have empty values.
`df.dropna()`
As a result, the dataset becomes `187630` rows × `5` columns format so far.

## 3. Exploratory Data Analysis

### 3.1. Calculation of target variable
The severity only have 2 values, 
`1- Property Damage Only Collision` 
`2- Injury Collision`. 
However, after in-depth data analysis, I found out that `ADDRTYPE` has 3 enum values, `ROADCOND` has 9 enum values, `LIGHTCOND` has 9 enum values, I decided to drop the `INCDATE` column since from now on, I realized that it does not contribute much to the calculation of target variable.

### 3.2 Relationship between SEVERITYCODE and `ADDRTYPE`,`ROADCOND` and `LIGHTCOND`
By doing the following commands, I found out the values for each column. 
Image1|Image2|Image3
-|-|-
[<img src="images/1.png">|<img src="images/2.png">|<img src="images/3.png">
Most of the accidents happened on block|Dry and Wet road condition has more accidents |Daylight is the top 1 compare to others.

`Block` has the most accident, however, `Intersection` has the most `Serverity=2` accident (42%), `Alley` has the most `Sererity=1` accident (89%).

```
ADDRTYPE      SEVERITYCODE
Alley         1               0.890812
              2               0.109188
Block         1               0.762885
              2               0.237115
Intersection  1               0.572476
              2               0.427524
Name: SEVERITYCODE, dtype: float64
```
In the known reason list, `Snow/Slush` has the most accident `Severity = 1` (69%), Oil has the most `Severiy=2` (37%)

```
ROADCOND        SEVERITYCODE
Dry             1               0.678227
                2               0.321773
Ice             1               0.774194
                2               0.225806
Oil             1               0.625000
                2               0.375000
Other           1               0.674242
                2               0.325758
Sand/Mud/Dirt   1               0.693333
                2               0.306667
Snow/Slush      1               0.833665
                2               0.166335
Standing Water  1               0.739130
                2               0.260870
Unknown         1               0.950325
                2               0.049675
Wet             1               0.668134
                2               0.331866
Name: SEVERITYCODE, dtype: float64
```
For the `Light condition` column, the expected result was that `Dark - No Street Lights` has the most `Severity=1` cases, `Dark - Unknown Lighting` has the most `Severity=2` cases.
```
LIGHTCOND                 SEVERITYCODE
Dark - No Street Lights   1               0.782694
                          2               0.217306
Dark - Street Lights Off  1               0.736447
                          2               0.263553
Dark - Street Lights On   1               0.701589
                          2               0.298411
Dark - Unknown Lighting   1               0.636364
                          2               0.363636
Dawn                      1               0.670663
                          2               0.329337
Daylight                  1               0.668116
                          2               0.331884
Dusk                      1               0.670620
                          2               0.329380
Other                     1               0.778723
                          2               0.221277
Unknown                   1               0.955095
                          2               0.044905
Name: SEVERITYCODE, dtype: float64
```

### 3.3 Convert Categorical features to numerical values
Based on previous analysis, I decided to convert the categorical features to numerical values, by doing this, I basically normalize all the data.
The final dataset becomes `187630` rows x `21` columns, we can see the sample output below:
I chose to drop the `Unknow` and `Other` columns since they are not providing any useful information.
Till now, it concluded the feature selections as output below:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 194673 entries, 0 to 194672
Data columns (total 21 columns):
SEVERITYCODE                194673 non-null int64
ADDRTYPE                    192747 non-null object
ROADCOND                    189661 non-null object
LIGHTCOND                   189503 non-null object
Alley                       194673 non-null uint8
Block                       194673 non-null uint8
Intersection                194673 non-null uint8
Dry                         194673 non-null uint8
Ice                         194673 non-null uint8
Oil                         194673 non-null uint8
Sand/Mud/Dirt               194673 non-null uint8
Snow/Slush                  194673 non-null uint8
Standing Water              194673 non-null uint8
Wet                         194673 non-null uint8
Dark - No Street Lights     194673 non-null uint8
Dark - Street Lights Off    194673 non-null uint8
Dark - Street Lights On     194673 non-null uint8
Dark - Unknown Lighting     194673 non-null uint8
Dawn                        194673 non-null uint8
Daylight                    194673 non-null uint8
Dusk                        194673 non-null uint8
dtypes: int64(1), object(3), uint8(17)
memory usage: 9.1+ MB
```

## 4. Predictive Modeling
There are two types of models, regression and classification. Regression models are to predict the continuous target data, I think classification is more fit for this project.
Classification models focus on the probabilities of the Severity level will be. The underlying algorithms are similar between regression and classification models, but various people may have a different preference. Therefore, in this study, I choose to use all four kinds of classification methods, compared the results and find out the best model.

### 4.1 Classification models
A supervised machine learning algorithm is one that relies on labelled input data to learn a function that produces an appropriate output when given unlabeled data. Classification is a supervised learning approach that can be thought of as a means of categorizing or classifying some unknown items into a discrete set of classes. It includes `decision trees`, `naive bayes`, `linear discriminant analysis`, `k-nearest neighbour`, `logistic regression`, `neural networks`, and `support vector machines`. There are many types of classification algorithms. I will only use `k-nearest neighbour`, `decision trees`, `logistic regression` and `support vector machines` in this project.

#### 4.1.1 K Nearest Neighbor(KNN)
K Nearest Neighbor(KNN) can be used in both regression and classification predictive problems. However, to my understanding, it's mostly used in classification since it fairs across all parameters evaluated when determining the usability of a technique.
I first split the filtered data into train and test to find the best k.
```
Train set: (136271, 18) (136271,)
Test set: (58402, 18) (58402,)
```
By dividing the dataset into the training set and test set, it actually didn't get the output result for this method.
It somehow leads me to conclude the following:
- KNN is not good for heavy data, the prediction stage might be slow or even can running out of memory
- KNN may require more memory compared to others since it needs to save all training data during run time
- KNN is an expensive method at least it needs more memory

#### 4.1.2 Decision Tree
The decision tree is the simplest method, it can be used for both regression and classification problems. It's very useful in machine learning and popular.
Also did the same, I divided data into 2 sets, a training set and test set. I got a really impressive result for this:
```
DecisionTrees's Accuracy:  1.0
DecisionTrees's Jaccard score is:  1.0
DecisionTrees's f1-score is:  1.0
```
Based on the result, it concludes that:
- Decision tree is easy to understand and interpret, it's very close to the human decision-making process.
- It can work with numerical and categorical features.
- It just needs a little data preprocessing whereas KNN needs one-hot encoding
- Unimportant features will not influence much to the result and it also doesn't affect the quality much.

#### 4.1.3 Supported Vector Tree
A supported vector tree is another tree-based technique and a popular tool to build prediction models.
In this project result, we have 2 categories in an x-y plane with points overlapping each other, and it's not easy for us to find a straight line that can perfectly separate them. When using this way, I noticed that the model actually can get a very good prediction result which was close to 1.

```
SVM's f1-score is:  0.9999486310391947
SVM's Jaccard score is:  0.9999486323359446
```

#### 4.1.4 Logistic Regression
Logistic regression is a statistical and machine learning technique for classifying records of a dataset based on the values of the input fields. In logistic regression, I use multiple independent variables such as `ADDRTYPE`, `ROADCOND` and `LIGHTCOND` to predict the car accident severity.
I also used another way of looking at the accuracy of the classifier - confusion matrix.
The Jaccard score is 1.0. From the confusion matrix, we can see that from the first row. The first row is for accidents whose actual churn value in the test set is 1. As we can calculate, out of 27425 customers, the churn value of 27425 of them is 1. They all meet the predicted value.
From the result, 
- logistic regression is pretty efficient in terms of time and memory requirements, unlike KNN.
- It's Convenient to get probability scores
- It doesn't handle the scale number of categorical features well

#### Result Analysis
In this study, From chart 5.1 below, we can see that `Decision Tree` and `Logistic Regression` got the same accuracy on `Jaccard score` and `f1-score score`. For some reason, the KNN method has not displayed any result for this project, `Supported Vector Tree` also can get decent results.
accuracy classification score| K Nearest Neighbor(KNN)|Decision Tree|Supported Vector Tree|Logistic Regression
-|-|-|-|-
Jaccard score|x|1.0|0.9999486323359446|1.0|
log-loss score|x|x|x|0.0035982614481627918|
f1-score score|x|1.0|0.9999486310391947|1.0|

chart 5.1
## 5. Conclusions
From this study, firstly, I observed the severity values, and found out they are unevenly distributed; moreover, I respectively analyzed the relationship between `Severity` and `road condition`, `address type` and `light condition`, leant that how the different independent features affect the severity targe value. At the same time, it also brings me to a stage that how we should clean some dummy data that was thinking useful than actually didn't contribute much. Finally, the models can be very helpful and valuable to help people stay safe and enjoy driving.

## 6 Future Direction
This report is almost ending, but this is just a good start, the machine learning field is wide and more areas we can explore on. For example, in this project, it predicts the severity based on the historical data where the accident already happened. This also can be used to a place there was no accident happened, however, can predict the severity if it happened. It warns the driver that if the accident happens, most likely the severity would be predicted.

Also, for the KNN method, I could improve the model to at least get the better K, to get the result.
