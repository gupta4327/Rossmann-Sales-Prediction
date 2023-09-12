from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd 
import datetime 
from xgboost import XGBRegressor

stores_data = pd.read_csv('stores_data.csv')

encoder_dict = dict({'Assortment': {'Basic': 0, 'Extended': 1, 'Extra': 2},
                 'StoreType': {'a': 0, 'b': 1, 'c': 2, 'd': 3},
                    'StateHoliday':{'0':0, 'Easter':2, 'PublicHoliday':3, 'Christmas':1}})

app = Flask(__name__)

model = pickle.load(open('rossman_sales_predictor.pkl', 'rb'))

def promo2_indicator(promo2_since_date, date):
  #check if promo2 since date is not equal to zero
  print(promo2_since_date)
  print(type(promo2_since_date))
  if promo2_since_date != '0':
    print(promo2_since_date)
    print(type(promo2_since_date))	
    promo2_since_date = datetime.datetime.strptime(promo2_since_date[0:10], "%Y-%m-%d")		
    #if days difference between sales date and date between promotion2 started is greater than 0 then set indicator as 1
    if date-promo2_since_date>= datetime.timedelta(days=0):
      return 1
    #if promo2 since date is 0 then store is not participating in promo2 and set indicator as 0
    else:
      return 0
  return 0

'''We are designing a function to check in a dataset if at specific day of sales of a specific store is competition is there 
   if so update the competition distance with the actual distance of competitor else put 100000 in place of 
   competition distance '''

def competition_distance(comp_dist, comp_since_date, date):
  #if competition since date has date only and not zero then only enters if value
  if comp_since_date != '0':
    comp_since_date = datetime.datetime.strptime(comp_since_date[0:10], "%Y-%m-%d")	
    #if difference between sales date and competitionopensincedate is greater than or equal to zero days means competiton is there before or on sales date
    #then return comp_dist
    if date-comp_since_date >= datetime.timedelta(days=0):
      return comp_dist
    #else imputr 100000
    else:
      return 100000
  #if comp since date is 0 then return 100000
  return 100000

#class for final predictions
class Rossman:

  #function to clean out the data i.e. null or missing value treatment
  def data_cleaning(self,data):   
    #merging data with stores data 
    data = pd.DataFrame(data, index = [0])
    data['Store'] = data['Store'].astype('int64')
    final_data = pd.merge(data,stores_data, on = 'Store', how = 'inner')

    #moving a current date to date column if date is not present in a row
    final_data['Date']  = final_data['Date'].apply(lambda x : date.today().strftime('%Y-%m-%d') if pd.isna(x) or x=='' else x)

    #getting a day of week from date to fill if day of week is not present 
    final_data['DayOfWeek'] = final_data['DayOfWeek'].apply(lambda x : x.weekday if pd.isna(x) else x)

    final_data['Date'] = pd.to_datetime(final_data['Date'])

    final_data['Promo'] = final_data['Promo'].astype('int64')

    final_data['DayOfWeek'] = final_data['DayOfWeek'].astype('int64')	

    final_data['SchoolHoliday'] = final_data['SchoolHoliday'].astype('int64')

    final_data['HolidayIndicator'] = final_data['HolidayIndicator'].astype('int64')


    #for all other features null value tratment has already done during training and will replicate same here 
    
    return final_data

  
  def feature_engineering(self,final_data):
    #extracting details from date column
    final_data['month'] = pd.DatetimeIndex(final_data['Date']).month
    final_data['year'] = pd.DatetimeIndex(final_data['Date']).year
    final_data['DateOfMonth'] = pd.DatetimeIndex(final_data['Date']).day
    final_data['WeekofYear'] = pd.DatetimeIndex(final_data['Date']).week

    # getting a promo2indicator feature denoting if promo2 is active on that sales data
    final_data['Promo2Indicator'] = final_data.apply(lambda x: promo2_indicator(x['Promo2SinceDate'], x['Date']),
                                                     axis=1)

    # updating a competitiondistance feature if competition is active on that sales data else impute 100000
    final_data['CompetitionDistance'] = final_data.apply(
        lambda x: competition_distance(x['CompetitionDistance'], x['CompetitionOpenSinceDate'], x['Date']), axis=1)
    
    #selecting the features only needed in our model prediction
    final_data = final_data[['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
                            'StoreType', 'Assortment', 'CompetitionDistance',
                            'avg_cust_visit_per_day', 'avg_sales_per_customer', 'month', 'year',
                            'DateOfMonth', 'WeekofYear', 'Promo2Indicator', 'HolidayIndicator']]
         

    #label encoding using previously defined label encoder to ensure same encoding has been applied
    for feature in encoder_dict.keys():
        mapping_dict = encoder_dict[feature]
        value = final_data[feature].values[0]
        value = str(value)
        final_data[feature] = mapping_dict[value]
        
    
    #returning prediction ready data 
    return final_data
    
  def prediction(self,data,final_data):

    #predicting output 
    output = model.predict(final_data)
    
    #creating a dataframe that will store predictor and predicted variable 
    predicted_df = data
    predicted_df['Predicted Sales'] = output

    #for feature in encode_features:
      #predicted_df[feature] = encoder_dict[feature].inverse_transform(predicted_df[feature]) 

    #returning the created dataframe
    return predicted_df 

def rossman_prediction(unit):

    #pipeline for predicting rossman sales
    pipeline = Rossman()

    #checking if whole data becomes null after treating and if not go further  
    if len(unit) >0:
      data = pipeline.data_cleaning(unit)
      final_data = pipeline.feature_engineering(data)
      prediction  = pipeline.prediction(data,final_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.items()]
    features = dict(features)
    if features['Open']!= '0':
        prediction = rossman_prediction(features)
        op= prediction['Predicted Sales'].values[0]
        date = prediction['Date'].values[0]
        output = round(op, 2)
    else:
     output = 0
    output_dict = features
    output_dict['Sales'] = output


    return render_template('predict.html', output_dict=output_dict)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)
