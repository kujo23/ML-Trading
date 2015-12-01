import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from collections import OrderedDict

# Utility Functions
def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates, addSPY=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


#Rolling Mean and Rolling Standard Deviation
def get_rolling_mean(values,window):
    #Return rolling mean of given values, using window size
    return pd.rolling_mean(values,window)

def get_rolling_std(values,window):
    #Return rolling std dev of given values, using window size
    return pd.rolling_std(values,window)

#KNN Learner class
class KNNLearner(object):

    def __init__(self,k):
    	self.k = k
    	self.dataX = None
    	self.dataY = None

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.dataX = dataX
        self.dataY = dataY

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """

        predictY = []
        for point in points:
        	euclidean_distances = np.array([np.linalg.norm(point-mydata) for mydata in self.dataX])
        	indices = np.argsort(euclidean_distances)
        	bestY = np.mean(self.dataY[indices[0:self.k]])
        	predictY.append(bestY)

        return predictY

#Bag Learner class
class BagLearner(object):

    def __init__(self,learner,kwargs,bags,boost):
    	self.dataX = None
    	self.dataY = None
    	learners = []
    	for i in range(0,bags):
    		learners.append(learner(**kwargs))
    	self.learners = learners
    	self.boost = boost

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.dataX = dataX
        self.dataY = dataY

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """

        predictYs = np.zeros([len(points),len(self.learners)])
        for i in range(len(self.learners)):
        	random_indices = np.random.choice(len(self.dataX),len(self.dataX),replace = True)
        	dataX = np.array([self.dataX[random_index] for random_index in random_indices])
        	dataY = np.array([self.dataY[random_index] for random_index in random_indices])
        	self.learners[i].addEvidence(dataX,dataY)
        	predictYs[:,i] = self.learners[i].query(points)
        return np.mean(predictYs,axis = 1)

def test_run():

    ########################## TRAINING the Learner ##############################

    # Define input parameters
    start_date = '2008-01-01'
    end_date = '2009-12-31'
    dates = pd.date_range(start_date,end_date)

    symbols = ['IBM']
    df = get_data(symbols,dates)

    #Bollinger Bands
    rm = get_rolling_mean(df['IBM'],window = 5)
    rstd = get_rolling_std(df['IBM'],window = 5)
    bb_value = (df['IBM'] - rm)/(2*rstd)

    #Volatility
    rstd = get_rolling_std((df['IBM']/df['IBM'].shift(1) - 1),window = 5)

    #Momentum
    momentum = df['IBM']/df['IBM'].shift(5) - 1

    #Value of Y. 5 Day return
    Y = df['IBM'].shift(-5)/df['IBM'] - 1

    data = np.zeros([bb_value.size,4])

    data[:,0] = bb_value
    data[:,1] = momentum
    data[:,2] = rstd
    data[:,3] = Y
    data[np.isnan(data)] = 0.0

    # compute how much of the data is training
    train_rows = math.floor(1* data.shape[0])
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]

    # create a learner and train it
    learner = BagLearner(learner = KNNLearner, kwargs = {"k":3}, bags = 10, boost = False)
    learner.addEvidence(trainX, trainY)

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]


    newTrainY = df['IBM'].copy()
    newPredY = df['IBM'].copy()
    newTrainY_scaled = df['IBM'].copy()
    newPredY_scaled = df['IBM'].copy()
    cnt = 0
    for i in range(len(newTrainY.index)):
        newTrainY[i] = trainY[cnt]
        newPredY[i] = predY[cnt]
        newTrainY_scaled[i] = (trainY[cnt] + 1) * df['IBM'][i]
        newPredY_scaled[i] = (predY[cnt] + 1) * df['IBM'][i]
        cnt += 1

    #Generating Orders
    short_entries = []
    short_exits = []
    long_entries = []
    long_exits = []
    signals = []
    i = 0
    while i <len(newPredY.index):
        if newPredY[i] > 0.05 and i+5 < len(newPredY.index):
            long_entries.append(str(newPredY.index[i]))
            signals.append([str(newPredY.index[i]).split()[0],'BUY','Long Entry'])
            long_exits.append(str(newPredY.index[i+5]))
            signals.append([str(newPredY.index[i+5]).split()[0],'SELL','Long Exit'])
            i += 6
        elif newPredY[i] < -0.05 and i+5 < len(newPredY.index):
            short_entries.append(str(newPredY.index[i]))
            signals.append([str(newPredY.index[i]).split()[0],'SELL','Short Entry'])
            short_exits.append(str(newPredY.index[i+5]))
            signals.append([str(newPredY.index[i+5]).split()[0],'BUY','Short Exit'])
            i += 6
        else :
            i += 1

    # Create a orders File for In Sample Data orders.csv
    ordersfile = open('orders.csv','w')
    ordersfile.write("Date,Symbol,Order,Shares\n")
    for signal in signals:
        ordersfile.write("%s,IBM,%s,100\n"%(signal[0],signal[1]))
    ordersfile.close()

    #Plot to show Training Y/Price/Predicted Y
    ax = df['IBM'].plot(title='Training Y/Price/Predicted Y',label='Price')
    newTrainY_scaled.plot(label='Training Y', color = 'c', ax = ax)
    newPredY_scaled.plot(label='Predicted Y', color = 'r', ax = ax)
    ymin,ymax = ax.get_ylim()
    ax.legend(loc='upper left')
    plt.show()

    #Plot to show Data In Sample Entries/Exits
    ax = df['IBM'].plot(title='IBM Data In Sample Entries/Exits',label='Price')
    newTrainY_scaled.plot(label='Training Y', color = 'c', ax = ax)
    newPredY_scaled.plot(label='Predicted Y', color = 'r', ax = ax)
    ymin,ymax = ax.get_ylim()
    plt.vlines(long_entries,ymin,ymax,color='g', label='Long Entries')
    plt.vlines(long_exits,ymin,ymax,label='Long Exits')
    plt.vlines(short_entries,ymin,ymax,color='r', label='Short Entries')
    plt.vlines(short_exits,ymin,ymax, label='Short Exits')
    ax.legend(loc='upper left')
    plt.show()


    ########################## TESTING the Learner ##############################

    # Define input parameters
    start_date = '2010-01-01'
    end_date = '2010-12-31'
    dates = pd.date_range(start_date,end_date)

    symbols = ['IBM']
    df = get_data(symbols,dates)

    #Bollinger Band
    rm = get_rolling_mean(df['IBM'],window = 5)
    rstd = get_rolling_std(df['IBM'],window = 5)
    bb_value = (df['IBM'] - rm)/(2*rstd)

    #Volatility
    rstd = get_rolling_std((df['IBM']/df['IBM'].shift(1) - 1),window = 5)

    #Momentum
    momentum = df['IBM']/df['IBM'].shift(5) - 1

    #Value of Y. 5 Day return
    Y = df['IBM'].shift(-5)/df['IBM'] - 1

    data = np.zeros([bb_value.size,4])
    data[:,0] = bb_value
    data[:,1] = momentum
    data[:,2] = rstd
    data[:,3] = Y
    data[np.isnan(data)] = 0.0

    # Compute how much of the data is testing
    test_rows = math.floor(1* data.shape[0])

    # Separate out training and testing data
    testX = data[:test_rows,0:-1]
    testY = data[:test_rows,-1]

    # Evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]

    newTestY = df['IBM'].copy()
    newPredY = df['IBM'].copy()
    newTestY_scaled = df['IBM'].copy()
    newPredY_scaled = df['IBM'].copy()
    cnt = 0
    for i in range(len(newTestY.index)):
        newTestY[i] = testY[cnt]
        newPredY[i] = predY[cnt]
        newTestY_scaled[i] = (testY[cnt] + 1) * df['IBM'][i]
        newPredY_scaled[i] = (predY[cnt] + 1) * df['IBM'][i]
        cnt += 1

    #Generating Orders
    short_entries = []
    short_exits = []
    long_entries = []
    long_exits = []
    signals = []
    i = 0
    while i <len(newPredY.index):
        if newPredY[i] > 0.05 and i+5 < len(newPredY.index):
            long_entries.append(str(newPredY.index[i]))
            signals.append([str(newPredY.index[i]).split()[0],'BUY','Long Entry'])
            long_exits.append(str(newPredY.index[i+5]))
            signals.append([str(newPredY.index[i+5]).split()[0],'SELL','Long Exit'])
            i += 6

        elif newPredY[i] < -0.05 and i+5 < len(newPredY.index):
            short_entries.append(str(newPredY.index[i]))
            signals.append([str(newPredY.index[i]).split()[0],'SELL','Short Entry'])
            short_exits.append(str(newPredY.index[i+5]))
            signals.append([str(newPredY.index[i+5]).split()[0],'BUY','Short Exit'])
            i += 6
        else :
            i += 1

    # Create a orders File for Out Sample Data orders_outsample.csv
    ordersfile = open('orders_outsample.csv','w')
    ordersfile.write("Date,Symbol,Order,Shares\n")
    for signal in signals:
        ordersfile.write("%s,IBM,%s,100\n"%(signal[0],signal[1]))
    ordersfile.close()

    # Plot Data Out of Sample Entries/Exits
    ax = df['IBM'].plot(title='IBM Data Out of Sample Entries/Exits',label='Price')
    newPredY_scaled.plot(label='Predicted Y', color = 'r', ax = ax)
    ymin,ymax = ax.get_ylim()
    plt.vlines(long_entries,ymin,ymax, color='g', label='Long Entries')
    plt.vlines(long_exits,ymin,ymax, label='Long Exits')
    plt.vlines(short_entries,ymin,ymax,color='r', label='Short Entries')
    plt.vlines(short_exits,ymin,ymax, label='Short Exits')
    ax.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    test_run()
