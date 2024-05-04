# Multivariate Time-Series using VAR
# Predict the future season classification of serie A Tim

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller

# =============================================================================
#                                     Main
# =============================================================================
# Prepare the data
data = {
    'Season': [
        '2010/2011', '2011/2012', '2012/2013', '2013/2014', '2014/2015',
        '2015/2016', '2016/2017', '2017/2018', '2018/2019', '2019/2020',
        '2020/2021', '2021/2022', '2022/2023'],
    'Napoli':     [3, 5, 2, 3, 5, 2, 3, 2, 2, 7, 5, 3, 1],
    'Roma':       [6, 7, 6, 2, 2, 3, 2, 3, 6, 5, 7, 6, 6],
    'Atalanta':   [21, 12, 15, 11, 17, 13, 4, 7, 3, 4, 3, 8, 5],
    'Milan':      [1, 2, 3, 8, 10, 7, 6, 6, 5, 6, 2, 1, 4],
    'Fiorentina': [9, 13, 4, 4, 4, 5, 8, 8, 16, 9, 13, 7, 8],
    'Lazio':      [5, 4, 7, 9, 3, 8, 5, 5, 8, 3, 6, 5, 2],
    'Inter':      [2, 6, 9, 5, 8, 4, 7, 4, 4, 2, 1, 2, 3],
    'Juventus':   [7, 0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 7],  # 0 porque não pode ter constantes
    'Torino':     [28, 22, 16, 7, 9, 12, 9, 9, 7, 16, 17, 10, 10],
    'Bologna':    [16, 9, 13, 19, 23, 14, 15, 15, 10, 12, 12, 13, 9],
    'Sassuolo':   [36, 24, 21, 17, 12, 6, 12, 11, 11, 8, 8, 11, 13],
    'Udinese':    [4, 3, 5, 13, 16, 17, 13, 14, 12, 14, 14, 12, 12]
}

data_premier = {
    'Season': [
        '2010/2011', '2011/2012', '2012/2013', '2013/2014', '2014/2015',
        '2015/2016', '2016/2017', '2017/2018', '2018/2019', '2019/2020',
        '2020/2021', '2021/2022', '2022/2023'],
    'Arsenal':     [4, 3, 4, 4, 3,   2, 6, 6, 5, 8,   8, 5, 2],
    'Aston Villa': [9, 16, 15, 15, 15,    17, 15, 17, 13, 17, 11, 14, 7],
    'Brighton':    [41, 30, 24, 26, 40,    23, 22, 15, 17, 15, 16, 9, 5],
    'Chelsea':     [2, 6, 3, 3, 1,   10, 1, 5, 3, 4,   4, 3, 12],
    # 'Crystal Palace':    [40, 37, 25, 11, 10,   15, 15, 11, 12, 14,  14, 12, 11],
    'Everton':     [7, 7, 6, 5, 11,  11, 8, 8, 8, 12,   9, 16, 17],
    'Leicester':   [30, 29, 26, 21, 14,    1, 13, 9, 9, 5,     5, 8, 18],
    'Liverpool':   [6, 8, 7, 2, 6,   8, 4, 4, 2, 1,   3, 2, 6],
    'Man City':    [3, 1, 2, 1, 2,   1, 4, 3, 1, 1,   2, 1, 1],
    'Man Utd':     [1, 2, 1, 7, 4,   5, 7, 2, 7, 3,   2, 6, 3],
    'Newcastle':   [12, 5, 16, 10, 15,   18, 21, 10, 13, 13,   12, 11, 4],
    'Southampton': [41, 22, 14, 8, 7,    6, 9, 17, 16, 11,     15, 15, 20],
    'Stoke City':  [13, 14, 13, 9, 9,    9, 14, 19, 36, 35,    34, 34, 36],
    'Tottenham':   [7, 4, 5, 4, 6,       5, 3, 3, 4, 6,        7, 4, 8],
    'West Bromwich': [11, 10, 8, 17, 13,   14, 11, 20, 24, 22, 19, 30, 29],
    'West Ham':    [20, 23, 10, 13, 12,    7, 12, 13, 10, 16,  6, 7, 14],
    'Wolves':      [17, 20, 42, 41, 27,    34, 35, 21, 7, 7,   13, 10, 13],
}


def plot(data):
    # Extract the clubs' names for the legend
    clubs = list(data.keys())[1:]

# Plotting
    plt.figure(figsize=(10, 6))
    for club in clubs:
        plt.plot(data['Season'], data[club], marker='o', label=club)

    plt.xlabel('Season')
    plt.ylabel('Position')
    plt.title('Club Positions Over Seasons')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def reorder_df_predictions(predicted_positions, initial_year):
    year = initial_year
    df = pd.DataFrame()
    for _, row in predicted_positions.iterrows():
        sorted_series = row.rename("value")
        sorted_series = sorted_series.sort_values().reset_index().reset_index()
        sorted_series['position'] = sorted_series['level_0'].add(1)
        sorted_series = sorted_series.drop("level_0", axis=1)
        sorted_series = sorted_series.drop("value", axis=1)
        transposed_df = sorted_series.set_index('index').T
        df_classification = transposed_df.rename({'position': str(year - 1) + "/" + str(year)})

        df_sorted_by_name = df_classification.sort_index(axis=1)
        df = pd.concat([df, df_sorted_by_name], axis=0)
        year += 1

    return df


data = data_premier

plot(data)

df = pd.DataFrame(data)

# Check for data stationarity.
# P-value must be < 0.05
# Perform ADF test on the 'Napoli' column
result = adfuller(df['Arsenal'])
# Print the ADF test statistic
print("ADF Statistic:", result[0])
# Print the p-value
print("p-value:", result[1])

# Drop last row to have 2 more season to predict
years_remove = 2
df = df.iloc[:-years_remove]

# Prepare the time-series data
df_ts = df.set_index('Season')

# Split data
train_size = int(len(df_ts) * 0.8)
train_data, test_data = df_ts[:train_size], df_ts[train_size:]

print("\nTRAIN DATA")
print(train_data)

# Fit the VAR model
model = VAR(train_data)
model_fit = model.fit()

# Make predictions on the test data
lag_order = model_fit.k_ar
predictions = model_fit.forecast(train_data.values[-lag_order:], steps=len(test_data))
predicted_df = pd.DataFrame(predictions, index=test_data.index, columns=test_data.columns)
initial_year = 2019
predicted_df = reorder_df_predictions(predicted_df, initial_year)

print("\nTEST DATA")
print(predicted_df)

# Model Performance
mse = mean_squared_error(test_data, predicted_df)
print("Mean Squared Error:", mse)

# Make predictions
years_to_predict = 3
lag_order = model_fit.k_ar
predictions = model_fit.forecast(df_ts.values[-lag_order:], steps=years_to_predict)

# Print the predictions
predicted_positions = pd.DataFrame(predictions, columns=df_ts.columns)

initial_year = 2023 + years_remove - years_to_predict
df = reorder_df_predictions(predicted_positions, initial_year)
print("\nFINAL PREDICTIONS")
print(df)
