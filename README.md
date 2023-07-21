# Car-Price-Prediction-using-RANDOM-FOREST-REGRESSION
ML Python Project


---------------------------------------------------------------------------------------

Note :- All the code in this project has been tested and run successfully in Google Colab. I encourage you to try running it in Colab for the best experience and to ensure smooth execution. Happy coding!

---------------------------------------------------------------------------------------


This code is an example of using a Random Forest Regressor to predict car prices based on a given dataset. Here's an explanation of each step:

1. Importing libraries and reading data:
   - `import pandas as pd`: Imports the pandas library for data manipulation.
   - `from google.colab import files`: Imports the `files` module from the `google.colab` package to handle file uploads.
   - `uploaded = files.upload()`: Prompts the user to upload a CSV file, which is stored in the `uploaded` variable.
   - `dataset = pd.read_csv('CarPrice_data.csv')`: Reads the CSV file and creates a DataFrame called `dataset`.

2. Data preprocessing:
   - `dataset = dataset.drop(['car_ID'], axis=1)`: Drops the 'car_ID' column from the DataFrame, as it is not needed for prediction.
   - `Xdata = dataset.drop('price', axis='columns')`: Creates a DataFrame `Xdata` containing all columns except the 'price' column.
   - `numericalCols = Xdata.select_dtypes(exclude=['object']).columns`: Identifies the column names with numerical data and stores them in `numericalCols`.
   - `X = Xdata[numericalCols]`: Creates a DataFrame `X` containing only the numerical data.
   - `Y = dataset['price']`: Creates a Series `Y` containing the target variable 'price'.

3. Feature scaling:
   - `from sklearn.preprocessing import scale`: Imports the `scale` function from the `sklearn.preprocessing` module.
   - `X = pd.DataFrame(scale(X))`: Scales the numerical features in `X` to have zero mean and unit variance.
   - `X.columns = cols`: Assigns the original column names back to the scaled DataFrame `X`.

4. Data splitting:
   - `from sklearn.model_selection import train_test_split`: Imports the `train_test_split` function from the `sklearn.model_selection` module.
   - `x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)`: Splits the data into training and testing sets.

5. Random Forest Regressor model:
   - `from sklearn.ensemble import RandomForestRegressor`: Imports the `RandomForestRegressor` class from the `sklearn.ensemble` module.
   - `model = RandomForestRegressor()`: Initializes the Random Forest Regressor model.
   - `model.fit(x_train, y_train)`: Trains the model using the training data.

6. Prediction and evaluation:
   - `ypred = model.predict(x_test)`: Predicts car prices using the test data.
   - `from sklearn.metrics import r2_score`: Imports the `r2_score` function from the `sklearn.metrics` module.
   - `r2score = r2_score(y_test, ypred)`: Computes the R-squared score to evaluate the model's performance.
   - `print("R2Score", r2score*100)`: Prints the R-squared score as a percentage.
