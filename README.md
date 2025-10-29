# Energy-consumption-prediction-model
this programm generates a fake data set describing the relationship between the temperature in C and the usage of electricity in kW/h, influenced by temperature, weekday and previous usage. Then it trains a small linear regression model which then predicts and evaluates the data. Then in the end it essentially visualises the data by plotting actual VS predicted numbers.

Dependencies:
import pandas
import numpy 
import matplotlib.pyplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

Output:
<img width="1000" height="566" alt="Screenshot 2025-10-29 at 12 07 25" src="https://github.com/user-attachments/assets/29c96c53-0b54-44b0-8132-544db1efb84a" />
