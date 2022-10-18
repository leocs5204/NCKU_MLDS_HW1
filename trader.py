#Trader HW

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras import layers
from keras import optimizers
from keras.layers import Normalization

column_names = ['open', 'highest', 'lowest', 'close']

def load_data(filename):
    print("Load: {}".format(filename))
    raw_data = pd.read_csv(filename, names=column_names)
    return raw_data

def prepare_today_climb(data):
    new_train_df = data.copy()
    today_diff = []
    for i in range(len(new_train_df)):
        today_diff.append(new_train_df.iloc[i, 3] - new_train_df.iloc[i, 0])
    new_train_df["today_diff"] = today_diff
    new_train_df["today_climb"] = new_train_df["today_diff"].apply(lambda x: 1 if x>0 else 0)
    return new_train_df

def normalization(data):
    normalized_data = Normalization(axis=-1)
    normalized_data.adapt(np.array(data))
    return normalized_data

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0,10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Open]')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_results(self, test_features, test_labels):
    results = []
    results['dnn_model'] = self.dnn_model.evaluate(test_features, test_labels, verbose=0)
    return results   


class Trader():
    def __init__(self):
        self.hold = 0
        self.yesterday_open = 0
    
    def buy_stock(self):
        if self.hold == 0:
            self.hold = 1
            return "1"
        elif self.hold == -1:
            self.hold = 0
            return "1"
        else:
            self.hold = 1
            return "0"
    
    def sell_stock(self):
        if self.hold == 1:
            self.hold = 0
            return "-1"
        elif self.hold == 0:
            self.hold = -1
            return "-1"
        else:
            self.hold = -1
            return "0"

    def hold_stock(self):
        return "0"

    def build_and_compile_model(self, norm):
        model = Sequential([
            norm,
            layers.Dense(100, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(100, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])
        model.compile(loss='mean_absolute_error',
                           optimizer=optimizers.Adam(0.01))
        return model


    def train(self, data):
        self.train_dataset = data
        self.train_dataset = self.train_dataset.dropna()
        new_train_dataset = prepare_today_climb(self.train_dataset)
        train_features = new_train_dataset.copy()
        # we would like to learn the open price of stock.
        train_lables = train_features.pop("open")
        normalizer = normalization(train_features)
        self.dnn_model = self.build_and_compile_model(normalizer)
        history = self.dnn_model.fit(
                           train_features,
                           train_lables,
                           validation_split=0.2,
                           verbose=0, epochs=100)
        plot_loss(history)

    def predict_action(self, row_data):
        print(row_data)
        predict = self.dnn_model.predict(row_data).flatten()
        print(predict)
        today_open = predict[0]
        print('predict open price: ', today_open)
        
        if (today_open - self.yesterday_open) > 0:
            self.yesterday_open = today_open
            return self.buy_stock()
        else:
            self.yesterday_open = today_open
            return self.sell_stock()
        
    


if __name__ == '__main__':     
    # You should not modify this part.     
    import argparse       
    parser = argparse.ArgumentParser()     
    parser.add_argument('--training',                        
                        default='training_data.csv',
                        help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv', 
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')     
    args = parser.parse_args()

    # The following part is an example.     
    # You can modify it at will.     
    
    training_data = load_data(args.training) 
    print(training_data)
    trader = Trader() 
    trader.train(training_data) 
    testing_data = load_data(args.testing)

    with open(args.output, 'w') as output_file: 
        for row in range(len(testing_data)): 
            row_df = testing_data.iloc[row]
            # We will perform your action as the open price in the next day. 
            if(row < len(testing_data)-1):
                action = trader.predict_action(row_df)
                output_file.write("{}\n".format(action)) 
            
            #this is your option, you can leave it empty. 
            #trader.re_training(i)