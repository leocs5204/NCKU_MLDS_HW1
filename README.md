# NCKU_ML_Homework
Homeworks for MLDS Stock trading tool.

## Inputs Data
Before training the model, firstly mark 4 colums data as ['open', 'highest', 'lowest', 'close'].
Labeled "open" as target data that I want to predict, and the rest features is training features I use to train model.
Add another column feature for marking the price difference of today's open and today's close. mark '1' as price climbing, '0' as unchanged and '-1' as decreasing.

## Normalization
Normalize stage is introduced here for normalize data sets.

## Training Model
Here use a DNN model for two layers training the model. Use Activation function 'relu' for adding non-linearity of model training.

## Predict
Predict the 'open' value, and store the open value to a yesterday price variable to compare the today's price with yesterday's opening price.
if today's opening price is higher than yesterday's price then buy the stock, if price is lower than yesterday then sell the stock.

