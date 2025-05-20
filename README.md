# Amazon_reviews
This report is based on The Amazon Reviews for Sentiment Analysis dataset. 
## A REPORT USING MACHINE LEARNING ON AMAZON REVIEWS FOR SENTIMENT ANALYSIS
## INTRODUCTION
Machine Learning (ML) is a way for computers to learn and make predictions or decisions without being explicitly programmed. It involves analyzing data and finding patterns or relationships within it. With ML, computers can learn from examples and experiences to improve their performance over time.
In this report, we will discuss the basics of ML, including how to load and explore datasets. Datasets are collections of data that we use to train our ML models. We will also talk about handling missing values in the dataset through a process called data cleansing. Missing values are data points that are not available or have not been recorded. By properly handling missing values, we can ensure that our ML models are accurate and reliable. Data cleansing being another essential step in ML, is where we remove duplicate records from the dataset and address outliers. By understanding these concepts and applying appropriate techniques for loading, exploring, and cleansing datasets, we can build robust and reliable ML models.[13] These steps are crucial in ensuring the success of any ML project and laying the foundation for accurate predictions and decisions.

## Loading and Exploring the Dataset:
This report is based on The Amazon Reviews for Sentiment Analysis dataset. The first step is to have a good insight of the dataset. The dataset is loaded into a data structure, such as a libraries and packages, which allows for convenient manipulation and analysis.
 ![image](https://github.com/user-attachments/assets/543d74f0-cdf9-4ec7-b058-472118462067)
 Import the required packages and libraries

This involves loading the dataset and comprehending its characteristics. Loading a dataset requires the use of programming libraries like Pandas in Python. The dataset used is the amazon_reviews, as shown below this shows the csv file loaded and a few rows displayed.
 ![image](https://github.com/user-attachments/assets/f2532d06-6802-44c5-8022-b7d5c8cf5297)
 Loading the dataset.
 
This screenshot below shows how the shape pf the dataset was analyzed, showing the number of rows and columns. Also showcased the data type of each column in the dataset.
![image](https://github.com/user-attachments/assets/8c986d7f-d7f0-4a12-aabc-0d8a5833538f)
Shape of dataset and datatypes.

The next step is to analyze the dataset statistically showing the mean, standard. The overview of the dataset was captured by displaying the first 5 and last 5 rows of the dataset as shown below.
 ![image](https://github.com/user-attachments/assets/c46c7194-920f-4d3c-809e-1b464c86f5f1)
Detecting Missing Values:

Before addressing missing values, it is important to identify if any are present in the dataset. This can be achieved by utilizing functions like isnull() in Pandas to check for missing values. Shown below is detecting null valves and removing rows or columns with missing values. One of the columns has a null value, and this was dropped, then the dataset was reviewed and confirmed the null value was dropped.

![image](https://github.com/user-attachments/assets/bf22609c-4d0a-4b44-9bc6-b1dfeec4126f)
Checking for missing values and deleting them.

Data Cleansing: As shown below, I searched for duplicates in the dataset and none were found, then the dataset was reviewed again to check the shapes and null and duplicate values were checked. Using the Wilson lower bound method, the dataset was sorted in an ascending order.
 ![image](https://github.com/user-attachments/assets/448250cd-c52d-4607-8e0e-88217ad74466)

I truncated the columns to the needed columns only an unwanted columns,  below shows the command line that drop(delete) the Unnamed column. 
 ![image](https://github.com/user-attachments/assets/4f4ac9ef-0671-4ea9-8c23-5f66b65ecd38)

The code data.info() in pandas is used to get a quick summary of a dataset. It tells us things like how many rows and columns there are, how many values are missing, what types of data are in each column, and how much memory the dataset is using.
 
![image](https://github.com/user-attachments/assets/db19fafb-9a7b-442e-a11c-0fc0385f15e7)

To analyze the distribution of missing values and understand the summary statistics of numerical columns in a dataset, you can use the following methods:
•	data.isnull().sum(): This code helps you find and count the missing values in each column of the Dataset. It gives you a list of column names with the respective counts of missing values in each column.
•	data.describe(): This method provides a summary of the numerical columns in the dataset. It gives you important information like the count (number of non-null values), average, variation, minimum value, median and maximum value for each numerical column. This gives a comprehensive understanding of both the missing data distribution and the statistical properties of the numerical data in the dataset.
 
![image](https://github.com/user-attachments/assets/3fb9ccb7-1242-4a6c-b23a-9b730e2bbc67)

I proceeded to display the dataset to have an overview of after the preprocessing method as shown below:
 ![image](https://github.com/user-attachments/assets/7d6e16ef-5366-4e23-a2ce-1571743058e4)

The next step was to Analyze the shape(rows and columns) of the new preprocessed dataset. S required, we are to review the dataset and classify the reviews as positive, negative and neutral in the dataset. This shows the number of positive, negative and neutral reviews.
 ![image](https://github.com/user-attachments/assets/dcbefb17-da06-4f34-89a2-d1a72d74b19a)

Exploring the Data Analysis and Visualization
This shows exploring and visualizing of the dataset. I have classified the reviews as positive, negative and neutral in the dataset  using the overall ratings. This was visualized in a pie chart, showing in percentage of each start rating. 

 ![image](https://github.com/user-attachments/assets/7415ee9f-1763-4931-b828-36d06a4ed7a4)

 ![image](https://github.com/user-attachments/assets/ab28e534-a281-4be2-a883-e8061589ce78)


The train_test_split function is used to divide the dataset into two sets: training and testing sets. By passing the features (X) and the target variable (y) to the function, you can split the data accordingly. After using train_test_split, this practice helps evaluate the model's performance and prevent overfitting the resulting variables are:
X_train: This variable contains the features (textual reviews) that your machine learning model will learn from during the training process.
X_test: This variable contains the features that will be used to evaluate the performance of your model.
y_train: This variable contains the corresponding labels for the training set.
y_test: This variable contains the corresponding labels for the testing set.
 ![image](https://github.com/user-attachments/assets/a73f1ab4-f343-49fa-9ba1-74fde3f510b2)

TF-IDF, known as Term Frequency-Inverse Document Frequency, is a widely used method in natural language processing (NLP) and machine learning. It is employed to transform a group of text documents into numerical vectors. These numerical vectors can be utilized as input for machine learning algorithm.
 
Predicting and evaluating the model
As shown above, to predict the labels for the test set using a logistic regression model (logreg_model), the following line of code is used: y_pred_logreg = logreg_model.predict(X_test_tfidf). The variable `y_pred_logreg` will store the predicted labels obtained from the logistic regression model.

To calculate the Mean Squared Error (MSE) between the actual labels (y_test) and the predicted labels (y_pred_logreg) from the logistic regression model, the following line of code is used:
mse_logreg = mean_squared_error(y_test, y_pred_logreg), The variable `mse_logreg` will hold the calculated Mean Squared Error.

To determine the accuracy of the logistic regression model on the test set, the following line of code is used: accuracy_logreg = accuracy_score(y_test, y_pred_logreg).

The variable `accuracy_logreg` will store the accuracy score, which is the ratio of correctly predicted labels to the total number of labels, obtained by comparing the predicted labels (y_pred_logreg) with the actual labels (y_test).

The code calculates the confusion matrix, which is a valuable tool for evaluating the performance of a classification model. It provides a clear representation of the number of true positives, true negatives, false positives, and false negatives.
To enhance the interpretability of the confusion matrix, a heatmap visualization is generated. This visualization offers a more intuitive way to analyze the matrix, facilitating the identification of patterns and errors in the model's predictions.
By examining the confusion matrix and its visualization, you can gain insights into the model's performance, particularly in terms of correctly and incorrectly classified instances. This information helps us to understand how well the model is performing.
 
![image](https://github.com/user-attachments/assets/4d715667-6302-4f3b-ba16-85f7ba43e0d1)

### THE CONFUSION MATRIX
The code computes the confusion matrix, which serves as a robust tool for evaluating the performance of a classification model.

To facilitate the interpretation of the confusion matrix and enable the identification of patterns and prediction errors, a heatmap visualization is generated.

The printed confusion matrix presents a concise summary of the counts for true positives, true negatives, false positives, and false negatives.

By analyzing the confusion matrix and its associated visualization, you can gain a comprehensive understanding of your model's performance, particularly in terms of accurately and inaccurately classified instances.
![image](https://github.com/user-attachments/assets/d609de57-6738-4162-b73a-f526a7cb3f36)

![image](https://github.com/user-attachments/assets/7552409e-ee63-4f16-ac08-2d2f321094b9)

CONCLUSION
The sentiment analysis model developed for analyzing Amazon product reviews performed effectively in accurately categorizing user-generated content. *By effectively capturing the unique characteristics of each review, the TF-IDF vectorization approach enabled the logistic regression model to make well-informed predictions.* The model's performance was further enhanced and its full potential was realized through hyperparameter adjustment.
**Utilizing the logistic regression model with stored TF-IDF vectorizer offers a practical way to implement the sentiment analysis system in real-world scenarios.** These stored models provide a consistent and efficient method for analyzing the sentiment of new textual data, whether they are used in production environments or integrated into other applications.


In conclusion, **this case study demonstrates the effectiveness of natural language processing techniques in combination with machine learning algorithms for sentiment analysis tasks. The evaluation results suggest that the model can be applied in various scenarios where understanding user sentiment is crucial, as it has the ability to generalize to unseen data.** Thorough analysis and comprehensive assessment metrics provide transparency and insights into the model's strengths and areas for potential improvement.



*References:
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems.[15] O'Reilly Media.
- VanderPlas, J. (2016). Python Data Science Handbook: Essential Tools for Working with Data.[16] O'Reilly Media.*


