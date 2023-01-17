---
title: |
  | STATS 780
  | Written Report for Final Project
author: "Amandeep Sidhu (400076920)"
date: "`r format(Sys.time(), '%d %B, %Y')`"
output:
  html_document:
    toc: yes
    df_print: paged
  pdf_document:
    includes:
      in_header: header.tex
    toc: yes
bibliography: STATS780.bib
fontsize: 11pt
geometry: margin = 1in
linestretch: 1.5
---

\newpage

## Introduction

In this final project, the data set selected for our analysis was titled healthcare-dataset-stroke-data.csv and can be obtained from the kaggle website [@kaggle]. The data set includes healthcare information for stroke detection using the predictor variables recorded (e.g. gender, bmi, smoking_status). For literature review, this data set has not been used in any formal papers (according to the information on the kaggle website), but other kaggle contributors have attempted to produce effective models using this data. Models previously produced incorporated the use of supervised learning (i.e. classification techniques) involving boosting and SVM. According to the World Health Organization (WHO), stroke rank among the global leaders in causes of death [@WHO]. As a result, models that can efficiently predict the likeliness of a patient to have a stroke is crucial for medical practitioners. With this in mind, the problem at hand is to develop a model that can classify whether a patient is likely to get a stroke or not based on predictor variables provided by healthcare facilities. For example, a previous approach implemented by @ustand describes the use of a boosting algorithm to classify which patients were likely to get a stroke based on the predictors available in the data set. Conversely, another approach that @adam implemented, incorporates support vector machine (SVM) to classify patients into the appropriate categories. In this project, we will implement our two model approaches and compare the results accordingly.

Observing the data set entries, we see that the response variable is a binary categorical variable. The variable is called "stroke" and its values are only 0 or 1. As a result, it makes sense to implement classification methods to determine which patients are likely to get a stroke (i.e. this is a classification statistical problem) using theory discussed in lecture. Recall in literature that classification is a form of supervised machine learning that can classify observations into groups based on multiple features (or attributes). In particular, classification models essentially read some inputs and generates an output that classifies the input into a category that seems reasonable. For the purpose of this project, we will be classifying whether patients are likely to get a stroke or not, depending on the various predictors recorded in the data set by medical practitioners.

## Methods

For both approaches, the feature engineering and variable selection for this data set includes removing the id (patient id) variable away from the data set as it was found to have little to no effect on classifying the response variable. In addition to this approach, dummy variables were created for the categorical variables as the original values for these variables were not readable by the algorithm. For example, if we observe the gender variable that has values of Male, Female and Other, we code a dummy variable for each value as 1, 0 and 2 respectively. See the coding section for this process.

### Method 1: Random Forests Algorithm

The first method that we will implement on the data set is called the Random Forests algorithm. In essence, this algorithm is an ensemble classification approach whose aim is to construct a collection of decision trees with controlled variation [@RForests]. It is also important to note that because the random forest approach is made up of multiple decision trees, it may be considered computationally expensive. Because of the computation complexity of decision trees, the random forests algorithm was originally developed to address this limitation by entertaining a small number of predictors available at each node (or split). This will be called "m" during the coding procedure. Thus, two important hyper-parameters will need to be tuned during the creation of the model: the number of predictors available at each split (i.e. m) and the number of trees (i.e. ntrees). The result of this procedure is to potentially improve model performance of the data set. In order to tune these parameters, we can use the Random Forests package in R as we can simulate a random forest model across various values of "m" and "ntrees" and see which values yield the best results [@rf]. Then, the values that have the best results will be used in the final random forest model for analysis. As previously mentioned, the computational complexity is vital to evaluating model performance as a less computationally expensive model may be preferred over one that is relatively expensive. For the random forests approach, the computational complexity is O(n*log(n)*d*k) [@KumarP]. This will be compared with out second approach to evaluate efficiency. Finally, in order to guarantee reproducibility of results, we will initiate a seed to ensure that values are consistent for each trial.

### Method 2: Weighted K-NN Classifier

The second approach that we will implement (and was not discussed in lecture) is the weighted K-NN machine learning algorithm. This algorithm incorporates a similar approach to the K-NN algorithm but attempts to address potential issues with respect to the choosing of the hyper-parameter k. Recall that K is the number of nearest neighbours the model will consider for analysis. IN literature, values of K may influence the results of the model depending on whether K is too small or too big. If the value of K is too small, we may yield results with higher variance and lower bias. This would be a problem as the model may be more sensitive to data points that are considered outliers. If the value of K is too large, we would yield results with lower variance and a higher bias. This would also be a problem as a neighbourhood may pull in points that belong to another neighbourhood and mis-classify points. The solution that the Weighted K-NN approach provides is by creating and assigning weights to points. In a practical sense, a heavier weight is assigned to points that are nearby one another, and a smaller weight to points further away [@wKNN]. Using a mathematical explanation, the inverse distance function is used, which implies that as the distance increases, the weight decreases (and vice versa). Furthermore, as @wKNN explains, the weights are learned using both of gradient descent and cross-validation. The only important tuning parameter for this approach is the parameter K, which denotes the number of neighbourhoods. This can be done with the help of evaluating various error rates for different K values. The K value that yields the lowest error rate will be selected and implemented in the final version of the Weighted K-NN model. The computational cost for this method is O(knd), which is similar to the K-NN algorithm [@KumarP]. Thus, it is suitable to use the Weighted K-NN approach instead of the K-NN as it addresses potential limitations of the K-NN model and maintains its computational cost. Once, again, we implement a seed to guarantee the reproducibility of results.

### Description of Comparison Criterions 

In order to compare and contrast the two model approaches that we will be implementing, we will use the confusion matrix and mis-classification rate on the test set to compare and contrast the Random Forests model versus the Weighted K-NN model. Recall that the confusion matrix is a N x N matrix that can be used for evaluating the performance of a classification model, where N is the number of target classes in the response variable [@conf]. Because we are attempting the compare the actual target values with the predicted ones by our classification models, it makes sense to use the confusion matrix as a possible comparison criterion to compare our models. The second comparison criterion that we will use is the mis-classification rate. Since, the mis-classification rate is a metric that tells us the fraction (as a percent) of the predictions that were wrong, it is also a logical choice to incorporate this metric to compare the performance of our classification models. 

## Results

### Exploratory Data Analysis (EDA)

Upon initial inspection, we see that the data set selected has 1 response variable (titled "stroke") and 11 attributes that can serve as predictors in model creation. There are 5110 observations recorded prior to any processes applied to the data set. The response variable recorded in this data set is called "stroke" and is a binary categorical variable with values of 0 and 1. If a patient has a value of 0, the patient is not at risk of a stroke. If there is a value of 1, the patient is at risk of a stroke. As Figure **1** of the supplementary section displays, we see that there are 11 predictor variables recorded and 1 response variable denoted as "stroke". Moreover, to observe the data types of each variable, refer to Figure **1** once again. The variable names are also self-explanatory as the variable named "avg_glucose_lvl" denotes the average glucose level in the blood. Thus, no variable name changes need to be conducted. To observe the summaries of each variable, a five numerical summary calculation is computed in Figure **2** of the supplementary section. Please note, only the first 3 variables are outputted for the sake of space. If we run this summary for all variables in the data set, we notice that there are six character variables (that act as categorical variables, which we code as dummy variables). It is important to code these character variables as dummy variables to ensure the model can read their values appropriately. One key observation from the summary exploration is that the bmi variable is acting as a character type. This does not make sense as the bmi (or body mass index) should be a numerical type. Upon further investigation, we notice that there are missing entries in this column. As a result, we remove the rows where the body mass index are missing to avoid any skewness of our results during the model fitting (see code). Once the missing values are removed, the number of observations drop from 5110 to 4909. Next, we create a pairs plot to observe the relationship between the numerical values present in the data set.

```{r echo=FALSE, out.width="75%", out.height="75%", fig.align='center'}
knitr::include_graphics("pairsplot.png")
```
\begin{center}

  \textbf{Figure 3:} Pairs plot of the 3 non-categorical variables

\end{center}

As we can see in Figure **3**, there appears to be no discernible patterns between the three numerical variables. A minor pattern which exists is that younger people tend to have higher average glucose levels in their blood. Finally, a boxplot will be constructed to help identify the spread of data in the variables.

```{r echo=FALSE, out.width="60%", out.height="60%", fig.align='center'}
knitr::include_graphics("boxplot.png")
```
\begin{center}

  \textbf{Figure 4:} Boxplot Analysis of the three non-categorical variables present

\end{center}

Observing Figure **4**, we see that there might be some concern with the average glucose level variable, but because the values are tightly compact at the ends we can leave this variable in for analysis. No statistical or data transformations were conducted on any of these variables present as the values are relatively close to one another.


### Results of Random Forests Algorithm

A 50%-50% training-test split was conducted to evaluate the performance of the random forest classification method. The random forest model here was obtained with the help of cross-validation by tuning the hyper-parameters, m and ntrees. In R, we investigated various values of m (that range from 1 to 11 as this is the number of predictors) and for ntrees (that range from 100 to 500). Observing Figure **5**, we see a heat map of the two tuning parameters. Literature suggests using the two values that correspond to the darkest region on the map. As a result, we will use the values of m = 3 and ntrees = 100. Please see the code to see how this was implemented. 

```{r echo=FALSE, out.width="70%", out.height="70%", fig.align='center'}
knitr::include_graphics("heatmap.png")
```
\begin{center}

  \textbf{Figure 5:} Heat map of mtry and number of trees

\end{center}

Now that the optimal tuning parameters have been found, we can evaluate how successful the model performs given our test data set that we originally split from the training data. We will investigate the performance using our first comparison criterion: confusion matrix.

\begin{center}
  Table 1: Confusion Matrix of Random Forest Algorithm
\end{center}
  
|             | 0 | 1 |
| :---------- | -------: | :---:               |
| 0 | 2347     | 2                 |
| 1   | 106    | 0            | 

As seen in Table **1**, the algorithm does a good job classifying patients with a stroke, but struggles to do so for individuals whom are likely to get a stroke (i.e. observation 1). From this table as well, we can calculate the mis-classification rate to be 0.04399185 or 4.4 %. Conversely, this would mean the accuracy of this method would be 0.95600815 or 95.6%.


### Results of Weighted K-NN Classifier

Once again, 50%-50% training-test split was conducted to evaluate the performance of the weighted K-NN classification method. The weighted K-NN model was obtained here by tuning the parameter, K, using the same technique to tune the parameter in a normal K-NN approach. In other words, the K with the lowest error rate was selected and implemented in the final model. Please see the code section to see how this was implemented.

```{r echo=FALSE, out.width="70%", out.height="70%", fig.align='center'}
knitr::include_graphics("error_plot.png")
```
\begin{center}

  \textbf{Figure 6:} Error rates at various K values

\end{center}

Observing Figure **6**, we can see that the error rate is lowest at 10 but begins to tail off at around K = 7 neighbourhoods. We select K = 7 as our tuning parameter value because the changes in error rate past this value are minimal enough to not notice a big difference. Now let us observe the confusion matrix of the Weighted K-NN algorithm and calculate the mis-classification rate from this table. Recall that the mis-classification rate is a metric that tells us the fraction (as a percent) of the predictions that were wrong (and accuracy is the proportion of right predictions in contrast to this).

\begin{center}
  Table 2: Confusion Matrix of Weighted K-NN Algorithm
\end{center}
  
|             | 0 | 1 |
| :---------- | -------: | :---:               |
| 0 | 2334     | 17                 |
| 1   | 101    | 2            | 

Table **2** appears to do a good job classifying patients with entries of 0 but a poor job with patients with entries of 1 (i.e. likely to get a stroke). Moreover, from this Table we can compute the mis-classification rate and accuracy which are 0.04808476 (or 4.8%) and 0.95191524 (or 95.2%) respectively. In the conclusion section, we will compare the results and discuss why the confusion matrix may be poor in terms of classifying patients with '1' entries in the response variable column.

If we interpret the results of applying both methods in the context of the original data set, both methods do a great job in classifying patients whom are not at risk of getting a stroke, but fail to do a good job at classifying patients who are at risk (i.e. '1' observations in the data). Though the accuracy is high for both algorithms, it is heavily skewed as there are more patients with '0' observations.

## Conclusion

To observe the differences in results of the mis-classification rates of different techniques implemented, observe Table **3**. 

\newpage

\begin{center}
  Table 3: Model Performance Metrics
\end{center}
  
|             | Mis-classification Rate | Accuracy |
| :---------- | -------: | :---:               |
| Random Forests | 0.0436     | 0.9564                 |
| Weighted K-NN   | 0.0481    | 0.9519            | 
| Boosting          | 0.036      | 0.964             |
| SVM         | 0.13      | 0.87              |

As you can see, the boosting algorithm implemented by @ustand performs best of all the models investigated. The SVM approach implemented by @adam performs the worst with a mis-classification rate of approximately a 13% mis-classification rate. Between the two approaches implemented by our study, the Random Forest algorithm performs best with a mis-classification rate of approximately 0.0436 and an accuracy of 0.9564. 

### Findings in the Context of the Data and Problem

One interesting finding for the context of the data and the problem was that if we observe Figure **7** that shows the variable importance plot, we see that the most important variables are average glucose level, bmi and age, which are the 3 numerical variables (i.e. the non-categorical variables) in the data set. This suggests that the numerical types are more important in this model and if we were to incorporate more continuous variables in our study then our results for our models may improve.

```{r echo=FALSE, out.width="75%", out.height="75%", fig.align='center'}
knitr::include_graphics("varimpplot.png")
```
\begin{center}

  \textbf{Figure 7:} Variable Importance Plot of the Data

\end{center}

A second finding in the context of the problem is that the results show that the classification of the entries with '1' in the stroke column is not good. Upon inspection of the data, we observe that there are actually 4700 observations with 0 and 209 observations with 1 as its value in the stroke column. This suggests that there is severe unbalanced data present and this may be the reason for poor confusion matrices for the two methods investigated.

### Comparison of the Two Techniques Investigated

As previously mentioned in Table **3**, we see that the Random Forest classification algorithm outperforms the Weighted K-NN classifier by observing the mis-classification rate and the accuracy calculated. The mis-classification rate is 4.36 % (with 95.6% accuract) for the random forest and 4.81% mis-classification (with 95.2% accuracy) for weighted K-NN classifier. Again, both of these metrics could be better if we obtain more continuous (or numerical data types) instead of more categorical variables in future studies. In addition, because the data set is unbalanced in terms of observations in the stroke column (i.e. a lot more '0' entries than '1' entries), a more balanced data set may improve this data set even more.

### Analytical Challenges

Recall that the computational cost for the Random Forest algorithm is O(n*log(n)*d*k) and O(knd) [@KumarP]. If we evaluate the models strictly in terms of the computational complexity, we observe that the random forest method is better in terms of cost efficiency. Since this model also performed best out of the models we implemented, we can say that this model is more optimal to use. In order to guarantee reproducibility of the results was ensured, seeds were initiated in the training-test split (see the code). A challenge involving the random forest algorithm is that it tends to be biased when categorical variables are used and because we are dealing with a lot of categorical variables here (8 out of the 11 variables are categorical), this may skew our results and not allow us to get the most optimal results. Conversely, a challenge for the Weighted K-NN classifier is that is a significant improvement over the normal K-NN algorithm, the weighting process may be insufficient when outliers exist (such as ones in this data set). Revisting the boxplot in Figure 4, we see that there are outliers outside of the whiskers. A possible extension is to scale the data further prior to implementing the methods discussed, such as a log-transformation on these values to make them more compact.


\newpage

## References

<div id="refs"></div>

\newpage

## Supplementary Material

```{r message=FALSE, warning=FALSE}
library(dplyr)
library(e1071)
library(caret)
library(ggplot2)
library(randomForest)
library(kknn)

setwd("C:/Users/User/Documents/Stats_780_Final_Report")

df <- read.csv("healthcare-dataset-stroke-data.csv")

```

```{r}
#drop unnecessary columns

df <- df[c(2:12)]
head(df)
```
**Figure 1:** First Six rows of the variables in the data set

```{r}
summary(df[c(1:3)])
```
**Figure 2:** Summary of First three variables in the data set


```{r}
#assign dummy variables to the appropriate columns here

#fix gender variable (1 represents male and 0 represents female and 2 is other)
df[df == "Male"] <- 1
df[df == "Female"] <- 0
df[df == "Other"] <- 2

#fix ever_married to a numerical value
# (1 = yes, 0 = no)

df[df == "Yes"] <- 1
df[df == "No"] <- 0

#fix work_type variable
#(Private = 0, Self-employed = 1, children = 3, Govt_job = 2, Never_worked = 4)
df[df == "Private"] <- 0
df[df == "Self-employed"] <- 1
df[df == "Govt_job"] <- 2
df[df == "children"] <- 3
df[df == "Never_worked"] <- 4

#fix Residence_type variable 

df[df == "Urban"] <- 0 
df[df == "Rural"] <- 1

#fix smoking_status variable

df[df == "formerly smoked"] <- 0
df[df == "never smoked"] <- 1
df[df == "smokes"] <- 2
df[df == "Unknown"] <- 3


#drop bmi empty values (i.e. NaN)
df <- df[!(df$bmi == "NaN"|df$bmi == "N/A"),]

#convert to numerical values (useful for pairsplot)
df$bmi <- as.numeric(df$bmi)
df$gender <- as.numeric(df$gender)
df$ever_married <- as.numeric(df$ever_married)
df$work_type <- as.numeric(df$work_type)
df$Residence_type <- as.numeric(df$Residence_type)
df$smoking_status <- as.numeric(df$smoking_status)

```

```{r eval = FALSE}
#Code for Pairsplot:

pairs(df[,c(2,8,9)])
```

```{r eval = FALSE}
# Code for Boxplot:
par(mar=c(4,12,4,1)) 
boxplot(df[,c(2,8,9)], horizontal= TRUE, las= 1)

```

```{r eval=FALSE}
# Implement the Random Forest Algorithm Here

#split the data
set.seed(1)
df$stroke <- factor(df$stroke)
train <- sample(1:nrow(df), nrow(df) / 2)
df.test <- df[-train, "stroke"]

#Algorithm 1: Random Forests
#let us first tune the parameters for this classifier
#begin by iterating throughout all predictor variables available

df_rf = tune.randomForest(stroke~., data = df[train,], mtry = 1:10,
                          ntree = 100*1:5,
                          tunecontrol = tune.control(sampling = "cross", 
                                                     cross = 5))
#summary(df_rf)
plot(df_rf, main = "", xlab = "mtry", ylab="Number of Trees", cex.lab = 0.75,
     cex.axis= 1)

#the optimal m and ntrees were found to be 3 and 100 respectively

set.seed(1)
rf.df = randomForest(stroke~., data = df, subset = train, mtry = 3, ntree = 100,
                     importance = TRUE)

#avg_glucose, age, id, bmi most important

rf_pred <- predict(rf.df, df[-train,], type ="class")

#classification table

tab1 <- table(df.test, rf_pred)
tab1

#mis-classification rate
1-classAgreement(tab)$diag

#value of about 0.0436 or 4.36%!

```


```{r eval = FALSE}
#Algorithm 2: Weighted K-NN ML

set.seed(1)
df$stroke <- factor(df$stroke)
train <- sample(1:nrow(df), nrow(df) / 2)
df.learn <- df[-train,]
df.valid <- df[train,]

#find optimal k neighbours

obj2 <- tune.knn(df.learn, df.learn$stroke, k = 1:10,
                 tunecontrol = tune.control(sampling = "boot"))
#summary(obj2)
#plot(obj2) - error rate plot

#we notice the error rate is lowest at k = 7

#apply weighted kknn

df.kknn <- kknn(stroke~., df.learn, df.valid, k = 7, distance = 1,
                kernel = "triangular")
fit <- fitted(df.kknn)
tab2 <- table(df.valid$stroke, fit)

#mis-classification rate
1-classAgreement(tab2)$diag

#error rate is 0.04808476 or 4.80% - nice
```

```{r eval=FALSE}
# variable importance plot code
varImpPlot(rf.df, main ="", cex.lab = 0.75)

```
