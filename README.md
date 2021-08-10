# Abalone_Gender_Age_Price_Predictor

**Table of Content-:**

* Problem Statement
* Demo Overview
* Goal of these project
* Methods and models used
* Why i used clustering,Regression and Classification
* Workdone
* Conclusion
* Result



**1)Problem Statement-:**

* I'm Vamshi let's say i'm hired as a data scientist in XYZ company based in India,i'm   hired as a fresher and hopefully i got my first task,it's about a abalone shell.Price     of the abalone linearly increases with it's age,age is calculated on the basis of no   of rings but here the problem is identifying no of rings is not possible by human.it       is calculated by microscope and it is very boring and time consuming procedure.I had   a task to solve this problem.
* Company provided the data,data is about the dimension of the abalone shell but it is   also a living being there are three different genders like male,female and                 infant.With respect to the dimensions i have to predict gender,age and current price   of the abalone shell.

**2)Demo Overview -:** Link()

**3)Goal of these project -:**
* To make a abalone prediction task simpler for anyone which is not possible possible for human eye just by seeing only,by using some machine learning techniques

**4)Methods and models used -:**

* a)Supervised Learning -:*
   1)Regression -: Linear regression,Support Vector Regressor,Xtreme Gradient Boost Regressor,Random Forest Regressor
   
   2)Classification -: KNeighborsClassifier,RandomForestClassifier,XGBClassifier,LogisticRegression,Support Vector Classifier
   
* b)Unsupervised Learning -:* Heirarical Clustering


**5)Why I used Clustering,Regression and Classification-:**

* For prediction of Gender by using different dimension is only possible by clustering.

* For prediction of age is only possible by regression only because it is a continuous data that's why i used regression

* By using regression we are not able to predict the age properly to prevent from these i converted it into 3 different classes which is only predicted by classification only.
   
   
**6)Workdone-:**

* By using heirarical Clustering i have created a three different classes by using that classes and by using that classes i have classified the genders by some research i.e male   abalone are find more,female abalone are found very less and infant abalones are found little less than male through these reasearch i classified gender

* By using regression i tryed to predicted the age label which is manually created through some research by models are not much accurately predicting the result,that might be a reason of continuous data to prevent from these i created a class column.

* By using Classification i tryed to predict the age class and it worked models are close to the optimal point

**Conclusion-:**
*By the above analysis done, we can conclude that,*
* Classification after clusteing

  Out of three classification models knn is giving more accuracy as compare to other models i.e 97%
  Regression after creating continuous age column

*Regression models are not accurately predicting Continuous data accuracy = 53%
 
*Classification after creating classes of age,fter creating classes of age models are giving maximum accuracy of 87% by almost all three models.


**Result**-:
In the light of evidence,we came to a result that by using dimensions of abalone now anyone can easily predict the gender,age and current price of the abalone shell
