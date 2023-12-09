# -*- coding: utf-8 -*-

import shutil
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# NEED THIS API EXPOSED

def getStockTickers(data):

  return data["Name"].unique()


# NEED THIS API EXPOSED

def getStockData(data, stock_ticker):

  stock_data = data[data['Name'] == stock_ticker]

  return stock_data


def getTargetData(data, target):

  target_data = data[target]
  target_data = np.array(target_data).reshape(-1,1)

  return target_data


def standardizeData(data):

  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(data)

  return scaled_data, scaler


def createDataPartitions(scaled_data, original_data, split_ratio):

  trainset_length = int(len(scaled_data) * split_ratio)
  train_data = scaled_data[: trainset_length]

  x_train = []
  y_train = []

  # Creating dataset for predicting ahead for a maximum of 60 days.
  for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60 : i])
    y_train.append(train_data[i : i + 1])

  # Convert the x_train and y_train to numpy arrays
  x_train, y_train = np.array(x_train), np.array(y_train)

  # Reshape the data
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

  test_data = scaled_data[trainset_length - 60 :]

  # Create the data sets x_test and y_test
  x_test = []
  y_test = original_data[trainset_length :]

  for i in range(60, len(test_data)):
      x_test.append(test_data[i - 60 : i])

  # Convert the data to a numpy array
  x_test = np.array(x_test)

  # Reshape the data
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

  return x_train, y_train, x_test, y_test


def buildModel(x_train):

  # Build the LSTM model
  model = Sequential()
  model.add(LSTM(128, return_sequences = True, input_shape= (x_train.shape[1], 1)))
  model.add(LSTM(64, return_sequences = False))
  model.add(Dense(25))
  model.add(Dense(1))

  return model


def getPredictions(model, scaler, x_test, y_test):

  # Get the models predicted price values
  predictions = model.predict(x_test)
  predictions = scaler.inverse_transform(predictions)

  # Get the root mean squared error (RMSE)
  rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

  return predictions, rmse


def zipModels(source_folder, destination_folder):

  # Ensure the destination folder exists, create it if not
  os.makedirs(destination_folder, exist_ok=True)

  # List all folders in the source folder that start with 'lstm'
  lstm_folders = [folder for folder in os.listdir(source_folder) if
                  folder.startswith('lstm') and
                  os.path.isdir(os.path.join(source_folder, folder))]

  # Zip each folder individually and save it to the destination folder
  for folder in lstm_folders:

      source_path = os.path.join(source_folder, folder)
      destination_path = os.path.join(destination_folder, f'{folder}')

      shutil.make_archive(destination_path, 'zip', source_path)

  shutil.make_archive("/content/zipped_models", 'zip', destination_folder)

  print("Zipping completed!")


# Early stopping callback based on RMSE

class RMSECallback(tf.keras.callbacks.Callback):

    def __init__(self, model_filename, scaler, x_test, y_test):
        super(RMSECallback, self).__init__()
    
        self.model_filename = model_filename
        self.best_rmse = float('inf')
        self.scaler = scaler
        self.x_test = x_test
        self.y_test = y_test

    
def on_epoch_end(self, epoch, logs=None):

        predictions, rmse = getPredictions(self.model, self.scaler, self.x_test, self.y_test)

        print(f"RMSE on test data: {rmse:.2f}\n")

        if rmse < 5:

            print("Reached RMSE < 5. Stopping training.\n")
            self.model.stop_training = True

        if rmse < self.best_rmse:

            print(f"Saving model with RMSE: {rmse:.2f}\n")

            self.best_rmse = rmse
            self.model.save(f"/content/prediction_models/{self.model_filename}")


def main():
    
    data = pd.read_csv("/content/all_stocks_5yr.csv")
    data.set_index('date', inplace = True)
    data.dropna(inplace = True)
    
    # Training Pipeline

    stock_tickers = getStockTickers(data)

    for stock_ticker in stock_tickers:

      # Get Stock Data particular to the corresponding stock ticker.
      stock_data = getStockData(data, stock_ticker)

      if len(stock_data) < 60:
        continue

      # Get Target Data. In our case, we are predicting the Close Price of the Stock.
      target_data = getTargetData(stock_data, 'close')

      # Standardize the data for quicker convergence.
      scaled_data, scaler = standardizeData(target_data)

      # Create data sequences for training and testing. I am doing a 0.95 train/test split.
      x_train, y_train, x_test, y_test = createDataPartitions(scaled_data,
                                                              target_data, 0.95)

      # Create the model
      model = buildModel(x_train)

      # Compile the model
      model.compile(optimizer = 'adam', loss = 'mean_squared_error')

      print(f'\nTraining on {stock_ticker} data...\n')

      # Train the model
      model.fit(x_train, y_train, epochs=20, batch_size = 1,
                  callbacks=[RMSECallback(model_filename = f"lstm_{stock_ticker}",
                                          scaler = scaler, x_test = x_test,
                                          y_test = y_test)])

    zipModels(source_folder = '/content/prediction_models',
              destination_folder = '/content/zipped_models')
    
    
if __name__ == "__main__":
    main()