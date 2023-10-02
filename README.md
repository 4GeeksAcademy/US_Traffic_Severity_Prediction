# US Accidents 2019 - Predicting Accident Severity Model 🚘 🚙

Using Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, 2019, 
the goal is to create a Model where we predict the Severity of such accidents. Such Model could help reduce accidents and create better driver safety by predicting what key factors weight in that provoke such accidents. The model would obviously help do better financial investments and allocate human resources in a proper way to create a safer conditions for drivers. 

## 📒 Dataset Overview
US-Accident dataset is a countrywide car accident dataset. It contains more than 7 million cases of traffic accidents that took place from February 2016 to December 2020. For this project it was only took in account all 2019 Data reported by Bing / MapQuest after our EDA and for Modeling, so that irrelevant can be eliminated for the greatest extent of it. 

This choice was made due to resources, as the full data was too much to work on my current machines, and as a challenge, to not use other tools than the current accessible computer at home. The first task was even to convert the .csv to a .parquet format which enables better compression without losing any data. All for the sake of speed and memory consumption. 

The Data in hands required a lot of investigation, and for this one as I went through Numerical and Categorical data doing Univariate and Multivariate analysis of the data, I've also proceeded to clean the data set to reduce more and more it's memory consumption, but also take out unecessary data what would mess up the model. 

This project was great to work with because many "mundane" conceptions about traffic were dismistified. The most important feature that affects our target "Severity" is the Coordinates, and by consequence the type of road. I've found out that Interstates are by far the more dangerous roads that lead do bigger severity of accidents. 
At the same time, weather conditions do play a very little role in the severity of an accident, which is a conception that drivers have (at least I did), snow or heavy rain lead to much more accidents, when in fact the difference is very little when compared to clear days. Same was disconvered for mechanisms like Traffic Signals which seem do not work properly to avoid accidents, but roundabouts or speed bumps do work well. 

For the Models I've tried to work with a Logistic Regression, since this is a categorical problem, but the results weren't the best. 

I then went immediately to a Random Forest with boosting, which lead to be the best result. With these features as the most important:

<img src="https://github.com/4GeeksAcademy/cesargustavo-Final_Project/blob/main/assets/rndforest.png" width="500">

XGBoost also didn't do better than Random Forest. 

An easy ensemble was also tried, but didn't beat the Random Forest. 

So in the End Random Forest, was the best model to use. 


## Streamlit App to Test the Model 

As final task I decided to create a Streamlit app which can be web embedded to try out the Model. 
You can input an address in the US that it will convert to Latitude and Longitude Coordinates, City and State on the background so it can work throughout the model. 
Afterwards user can select a series of features to predict on a scale of 1 to 4, 4 being the more severy level of a future accident happening in such conditions, locations and time. 
Here are a few screenshots of the app: 

<img src="https://github.com/4GeeksAcademy/cesargustavo-Final_Project/blob/main/assets/app01.png" width="600">
<img src="https://github.com/4GeeksAcademy/cesargustavo-Final_Project/blob/main/assets/app02.png" width="600">

Due to the size of the model, it wasn't possible to upload and deploy the app, but running all the code in the repo, will get the app working. 

For future reference, if I had more data, like speed of the vehicules, alcohol rate in the driver blood, state of the car, the model could be even more accurate. 
The Model can also be used on the inverse, for applications where someone wants to do new planification for roads and understand how to make them safer. 

