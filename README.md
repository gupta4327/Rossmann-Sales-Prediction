# Project Title - Sales Prediction : Predicting sales of a major store chain Rossmann

![104977231-ca344600-59dd-11eb-94c4-26f431bc2802](https://user-images.githubusercontent.com/88886118/213765233-a9265fb0-14de-4d5c-83ff-2c55e7c6ec4c.png)

---
## Contents <p id="contents"></p>
- <a href="#bquestions">Business Questions</a>
  - <a href="#business_impact">Business Impact</a>
- <a href="#methodology">Methodology</a>
  - <a href="#data_collect">Data Collect</a>
  - <a href="#data_cleaning">Data Cleaning</a>
    - <a href="#data_description">Data Description</a>
- <a href="#data_exploration">Data Exploration</a>
- <a href="#machine_learning">Machine Learning Algorithms</a>
- <a href="#final_model_selection">Final Modeling Selection</a>

## 📝 Problem Statement <p id="bquestions"></p>

Rossmann operates over 3,000 drug stores in 7 European countries.  Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, 
You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column . Note that some stores in the dataset were temporarily closed for refurbishment.

## 🤔 How it might be helpful for business? <p id="business_impact"></p>

Many of these managers contacted me for consult the data to make the prediction for their stores. So, I went looking for the real stakeholder to understand the main reason for this sales forecast. Talking to managers, they told me that was a request from the CFO at a meeting with all managers.

When discussing with the CFO, he informs me that wants to know the store sales forecast because wants to renovate them, and he would like to know how much money will have in cash to carry out the construction. I understod the main problem and start looking for the data that will help me make this prediction.

As this is a problem originating from a Kaggle competition, the data are available through their platform, and it is not necessary to use extraction from databases and other sources.

## 💻 Methodology <p id="methodology"></p>
![sales_forecast_32f42ee760](https://user-images.githubusercontent.com/88886118/213775274-3112ea1f-e468-4a77-9b02-8c5d34a47ec3.png)

## DATA COLLECT <p id="data_collect"></p>

The data used in this project are available through the Kaggle platform, and can be find [here](https://www.kaggle.com/c/rossmann-store-sales/data). But, if it were a real company environment, this data would be collected through database queries, and other sources of information.

## 🧹 DATA CLEANING <p id="data_cleaning"></p>

Cleaning the dataset downloaded from the previous step, performing operations such as:

### Data Description <p id="data_description"></p>

  - Check the size and type of the data.
  - Check for the existence of missing values, and if so, use an approach to fill in these values.
    - *In this step i found some columns with missing values, and filled them thinking about the business.*
  - Standardize the types of variables.
  

## 🔎  EXPLORATORY DATA ANALYSIS <p id="data_exploration"></p>

The exploratory data analysis proposes to analyze how the variables map the phenomenon we want to model, and what is the strength of this impact. Proposed basically for 3 objectives:

- Gain business experience.
- Validate business hypotheses and generate insights.
- Identify variables that are important to the model.


<p align="center">
<img src="https://user-images.githubusercontent.com/88886118/216805194-0a1c0b37-2038-4f50-ab68-4936478b706f.png">
</p> 


- Relation seems to be linear between customers and sales but seems to have high variance
- Also Graph helps us conclude that although there are some stores that have a same customer visit still some have higher sales and some have lower sales
It can be dependable on many factors like competitors distance, promo,etc


<p align="center">
<img src="https://user-images.githubusercontent.com/88886118/216804684-595e96f2-c78b-445f-954e-577894a123c4.png">
</p> 

- In Regards to month sales and customers graph we can depict that number of customers visiting store per day is almost similar every month and for sales per day there is a quite surge in daily sales in 12th month which is also a festive month(christmas and New Year) and week of the year graphs gives a prove of that as well (you can see a surge in 51st week)

- In Regards to Date of month sales and customers graph there is no any such great depiction but still customers visiting store and sales is higher in starting and ending of months

- In Regards to day of week sales and customers graph we can see surge in sales on Monday and Sunday and also number of customers visiting store is higher on sundays and almost similar for other weekdays


<p align="center">
<img src="https://user-images.githubusercontent.com/88886118/216805259-91c32a9b-549d-4d4d-b96f-7ffb8f2e2263.png" width="700" height="400"/>
</p> 

- We can simply see that where on other weekdays near to 1000 or above stores are opened and operating whereas on sundays on an average less than 50 stores are opened and thus resulting in increasing sale value for store on that day

- Most of the stores are closed on sundays thats why the rare ones that are operating on sunday are getting surge in sales on sundays
And the same reason as most of stores are closed on sundays then on next day of opening that is on monday they are getting bit higher sales

<p align="center">
<img src="https://user-images.githubusercontent.com/88886118/216805530-e16de7d7-85e6-434f-8f87-8c6c51a9d73f.png">
</p> 



- Promos are surely working in favour of increasing sales of stores. On a day Stores showing promos on an average makes a sale of approx 2000 dollars greater than the days they don't shows a promo
We can see for sales with promo, peak of distribution lies somewhere around 8500$ and sales without promo peak of distribution lies somewhere around 7500 which means maximum sales with promo is around 8500 and maximum sales without promo is around 7500 for stores
Lets randomly get 5-10 stores and see the difference in there sales on the days when they are showing promos and when they are not



<p align="center">
<img src="https://user-images.githubusercontent.com/88886118/216805803-b3ab7d5c-1859-4b2b-aed1-36e6c68096fa.png">
</p> 


- Looking over all the stores it seems like for most of the stores promo2 is actually working as a average sales of these stores after participating in promo2 is higher than before participating.

<p align="center">
<img src="https://user-images.githubusercontent.com/88886118/216806384-dea0fa2e-7570-47d3-97ad-75e32da5c3bb.png">
</p>

- One observing thing is that only storetype B is using Extar assortment startegies and that could be the potential reason for storetype B having greater sales as from the below analysis we can see that Extra assortment has higher sales.

<p align="center">
<img src="https://user-images.githubusercontent.com/88886118/216806282-f587fc1f-033e-4fe2-b4dd-d34c177f777e.png">
</p> 

- There are few exceptions but in general we can see that most of the time sales is decreased a bit after competition is established. This proves that competition and its distance is affecting the sales of rossman stores


## MACHINE LEARNING ALGORITHMS <p id="machine_learning"></p>

It is this project stage that involves the part that most data scientists like to work on, the use of machine learning algorithms. All the previous steps were designed to maximize the algorithms efficiency

## ⌛ Machine Learning Modelling <p id="machine_learning_modelling"></p>

**Linear Regression**

| Model Name	| RMSE | MAE | RMSPE | R2_score | Adjusted R2_score |
| ----------- | ----------- |  ----------- |  ----------- |-----------|-----------|
| Linear Regression |1768.0 | 1197.03 | 0.274 | 0.69 | 0.69 |

**Decision Tree**

| Model Name	| RMSE | MAE | RMSPE | R2_score | Adjusted R2_score |
| ----------- | ----------- |  ----------- |  ----------- |-----------|-----------|
| Decision Tree |1321.32 | 915.21 | 0.206| 0.83 | 0.83  |

**Random Forest**

| Model Name	| RMSE | MAE | RMSPE | R2_score | Adjusted R2_score |
| ----------- | ----------- |  ----------- |  ----------- |-----------|-----------|
| Random Forest | 1097.37 |761.01 | 0.168| 0.88| 0.88  |

**Gradient Boost Regression**

| Model Name	| RMSE | MAE | RMSPE | R2_score | Adjusted R2_score |
| ----------- | ----------- |  ----------- |  ----------- |-----------|-----------|
| Gradient Boost Regression | 1163.47 | 829.25| 0.193| 0.87| 0.87  |

**XGBoost**

| Model Name	| RMSE | MAE | RMSPE | R2_score | Adjusted R2_score |
| ----------- | ----------- |  ----------- |  ----------- |-----------|-----------|
| XGBoost Regressor | 1059.17 | 752.12|  0.174|  0.89| 0.89  |

## 📊 FINAL MODEL SELECTION <p id="final_model_selection"></p>

![Screenshot 2023-02-05 at 1 10 01 PM](https://user-images.githubusercontent.com/88886118/216807437-04170d03-1737-4a1d-b368-39530b95aa5c.png)
![Screenshot 2023-02-05 at 1 10 15 PM](https://user-images.githubusercontent.com/88886118/216807450-ded2f340-d5dc-45d8-b73c-5ad00e6932f6.png)

From the visual analysis of various performance metrics of different models, we can see although gradient boost and xtreme gradient boost are giving us a great performance but at the cost of execution time.

So if there is no kind of incremental modelling and we can just train the model once and store it then we can go with Xtreme gradient boost. 
