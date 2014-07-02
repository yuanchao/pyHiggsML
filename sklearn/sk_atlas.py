'''
Example code for processing Atlas Machine-Learning Challenge
Based on Darin Baumgartel's reference code.

2014/06/23 Yuan CHAO
'''   

print(__doc__)

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.cross_validation import train_test_split
import math
 
# Load training data
print 'Loading CSV training data.'
data_train = np.loadtxt( 'training.csv', delimiter=',', skiprows=1,
                         converters={32: lambda x:int(x=='s'.encode('utf-8')) } )
 
print 'Splitting training and testing samples.'

# Reading csv into NumPy arrays
# random_state used for training/validation splitting seed
Z_train, Z_valid, Y_train, Y_valid = train_test_split(
                                       data_train[:,1:32], data_train[:,32],
                                       test_size=0.5, random_state=42)

X_train = Z_train[:,0:30]
X_valid = Z_valid[:,0:30]
W_train = Z_train[:,30]
W_valid = Z_valid[:,30]

# Train the GradientBoostingClassifier using our good features
print 'Training classifier (this may take some time!)'
gbc = GBC(n_estimators=50, max_depth=5, min_samples_leaf=200,
          max_features=10, verbose=1)

gbc.fit(X_train,Y_train) 
 
# Get the probaility output from the trained method, using the testing sample
prob_predict_train = gbc.predict_proba(X_train)[:,1]
prob_predict_valid = gbc.predict_proba(X_valid)[:,1]
 
# Experience shows me that choosing the top 15% as signal gives a good AMS score.
# This can be optimized though!
pcut = np.percentile(prob_predict_train,85)
 
# This are the final signal and background predictions
Yhat_train = prob_predict_train > pcut 
Yhat_valid = prob_predict_valid > pcut
 
# To calculate the AMS data, first get the true positives and true negatives
# Scale the weights according to the r cutoff.
TruePositive_train = W_train*(Y_train==1.0)*(1.0/0.5)
TrueNegative_train = W_train*(Y_train==0.0)*(1.0/0.5)
TruePositive_valid = W_valid*(Y_valid==1.0)*(1.0/0.5)
TrueNegative_valid = W_valid*(Y_valid==0.0)*(1.0/0.5)
#print sum(TruePositive_train), sum(TrueNegative_train),
#print sum(TruePositive_valid), sum(TrueNegative_valid)
 
# s and b for the training 
s_train = sum ( TruePositive_train*(Yhat_train==1.0) )
b_train = sum ( TrueNegative_train*(Yhat_train==1.0) )
s_valid = sum ( TruePositive_valid*(Yhat_valid==1.0) )
b_valid = sum ( TrueNegative_valid*(Yhat_valid==1.0) )
print s_train, b_train,
print s_valid, b_valid

# Now calculate the AMS scores
print 'Calculating AMS score for a probability cutoff pcut=',pcut
def AMSScore(s,b): return  math.sqrt (2.*( (s + b + 10.)*math.log(1.+s/(b+10.))-s))
print '   - AMS based on 50% training   sample:',AMSScore(s_train,b_train)
print '   - AMS based on 50% validation sample:',AMSScore(s_valid,b_valid)

# making performance plots

from sklearn.externals import joblib
joblib.dump(gbc, 'atlas.pkl') 

