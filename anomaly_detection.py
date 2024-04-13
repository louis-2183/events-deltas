import pandas as pd
import numpy as np
from keras.layers import Input, Dense 
from keras.models import Model 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from scipy import stats

class CurveConstructor:
    def __init__(self,max_events,min_events):
        self.encoding_dim = 10
        self.max_events = max_events
        self.min_events = min_events
        
    def fit(self,df):
        df['y'] = df['y'].astype('float32')
        df = df.dropna()        

        df_t = df.drop('ds',axis=1).values
        # Define the autoencoder model 
        input_dim = df.shape[1] - 1
        encoding_dim = 10
        input_layer = Input(shape=(input_dim,)) 
        encoder = Dense(encoding_dim, activation='relu')(input_layer) 
        decoder = Dense(input_dim, activation='relu')(encoder) 
        autoencoder = Model(inputs=input_layer, outputs=decoder) 
          
        # Compile and fit the model 
        autoencoder.compile(optimizer='adam', loss='mse') 
        autoencoder.fit(df_t, df_t, epochs=50, 
                        batch_size=32, shuffle=True) 
          
        # Calculate the reconstruction error for each data point 
        reconstructions = autoencoder.predict(df_t) 
        mse = ((df_t-reconstructions)**2).mean(axis=1)
        
        anomaly_scores = pd.Series(mse, name='anomaly_scores') 
        anomaly_scores.index = df.index
        self.df = df
        
        threshold_max = anomaly_scores.quantile(0.99) 
        self.anomalous_max = anomaly_scores > threshold_max 
             
        threshold_min = anomaly_scores.quantile(0.01) 
        self.anomalous_min = anomaly_scores < threshold_min 
        
    """
        plot_thresholds(thresh_type)
        
        thresh_type: string of either 'max' or 'min'
        
            Decides which anomaly type to highlight, either a spike or dip.
            
        Plots the time series and identified anomaly points
    """
    def plot_thresholds(self,thresh_type='max'):
        df = self.df
        if thresh_type == 'max':
            anom = self.anomalous_max
            events = self.max_events
        elif thresh_type == 'min':
            anom = self.anomalous_min
            events = self.min_events
            
        # Plot the data with anomalies marked in red 
        plt.figure(figsize=(16, 8)) 
        plt.plot(df['ds'],df['y']) 
        plt.plot(df['ds'][anom], 
                 df['y'][anom], 'ro')         
        # Draw lines showing anomaly timeframes
        for event in events:
            plt.axvline(x = df[(df['ds'] == event[0])]['ds'], color = 'b')
            plt.axvline(x = df[(df['ds'] == event[1])]['ds'], color = 'b')
        
        if thresh_type == 'max':
            plt.title('Anomaly Detection Max threshold') 
        elif thresh_type == "min":
            plt.title('Anomaly Detection Min threshold') 
        plt.xlabel('ds') 
        plt.ylabel('y') 
        plt.show() 
        
    """
        construct_curve(mx,n_periods,thresh_type)
        
        mx: Float ranging from 0 to 1
        
            Position of curve peak
            
        n_periods: Int
        
            Number of periods to scale the output curve to
            
        thresh_type: string of either 'max' or 'min'
        
            Decides which anomaly type to construct the curve for, either a spike or dip.
            
        Returns a 1D array containing positions and amplitude along the produced curve
    """
    def construct_curve(self,mx,n_periods,thresh_type):
        # Create triangular distribution to fit skewed distribution on
        data = np.random.triangular(0, mx, 1, 1000)
        
        a, loc, scale = stats.skewnorm.fit(data)
        
        x = np.linspace(0, 1, n_periods)
        p = stats.skewnorm.pdf(x,a, loc, scale)
        # Scale curve to 0,1 to multiply by average triangle amplitude
        p = [x/np.max(p) for x in p]
        
        # If curve is based off a dip in the series, flip it back upside down 
        if thresh_type == "min":
            p = [-x for x in p]
        
        return p
    
    """
        predict(pred,thresh_type)
        
        pred: 1D array of length 2 containing strings of date format YYYY-MM-DD HH:MM:SS.
        
            Prediction period
        
        thresh_type: string of either 'max' or 'min'
        
            Decides which anomaly type to predict, either a spike or dip.
            
        Returns a pandas DataFrame containing date range and prediction
    """
    def predict(self,pred,thresh_type='max'):
        dt_range = pd.date_range(start=pred[0],end=pred[1],freq='H')
        n_periods = len(dt_range)
        df = self.df
        
        # Calculate anomalies on the right side of the mean to get amplitude
        if thresh_type == 'max':
            anomalous = self.anomalous_max
            events = self.max_events
            df['anomalous'] = anomalous
            df['anomalous'] = df.apply(lambda x: True if x['y'] > np.mean(df['y']) and x['anomalous'] == True else False,axis=1)
            thresh_anom = np.min(df[(df['anomalous'] == True)]['y'].values)
            
        elif thresh_type == 'min':
            anomalous = self.anomalous_min
            events = self.min_events    
            df['anomalous'] = anomalous
            df['anomalous'] = df.apply(lambda x: True if x['y'] < np.mean(df['y']) and x['anomalous'] == True else False,axis=1)
            thresh_anom = np.max(df[(df['anomalous'] == True)]['y'].values)
            
        df = df.reset_index()

        # Calculate average triangle for each anomaly
        ratios = []
        distances = []
        heights = []
        
        for event in events:
            temp = df[(df['ds'] > event[0]) & (df['ds'] < event[1])]
            temp = temp[(temp['anomalous'] == True)]
            
            scaler = StandardScaler()
            temp = temp[['index','y']]
            scaler.fit(temp)
            scaled = scaler.transform(temp)

            # If the data produces an upside down triangle as it is a dip, flip it            
            if thresh_type == 'min':
                scaled[:,1] = [-x for x in scaled[:,1]]
            
            # Get 2D position of highest point in triangle - where midpoint will be assumed to be
            for item in scaled:
                if item[1] == np.max(scaled[:,1]):
                    max_pt = item
            
            # Collate the points of the triangle
            tri_points = np.array([
                [scaled[0][0],np.min(scaled[:,1])],
                max_pt,
                [scaled[len(scaled)-1][0],np.min(scaled[:,1])]]
            )
            
            # Get ratio of x range * y range to calculate the weight for averaging
            ratios.append( (np.max(scaled[:,0]) - np.min(scaled[:,0]))*
                           (np.max(scaled[:,1]) - np.min(scaled[:,1])))
        
            # Get height of the triangles in order to calculate the weighted average height
            if thresh_type == 'max':
                heights.append((temp['y'].max()-thresh_anom)/len(temp))
            elif thresh_type == 'min':
                heights.append((thresh_anom-temp['y'].min())/len(temp))
        
            # As above with distance
            distances.append((tri_points[1][0]-tri_points[0][0])/(tri_points[2][0]-tri_points[0][0]))
        
        distances_norm = []
        heights_norm = []
        
        # Normalise ratios    
        ratios = [x / np.sum(ratios) for x in ratios]
        for idx,i in enumerate(ratios):
            distances_norm.append(i*distances[idx])
            heights_norm.append(i*heights[idx])
        
        # Combined midpoint
        mid_x = np.sum(distances_norm)
        mid_height = np.sum(heights_norm)
        
        # Retrieve average curve for output
        crv = self.construct_curve(mid_x,n_periods,thresh_type)
        crv = [x*mid_height for x in crv]
        
        return pd.DataFrame(data={'ds':dt_range,'yhat_delta':crv})