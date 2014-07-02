'''
Example code for processing Atlas Machine-Learning Challenge
Based on Darin Baumgartel's reference code.

2014/06/23 Yuan CHAO
'''   

print(__doc__)

import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.cross_validation import train_test_split
import math

from sklearn.externals import joblib

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
print 'Loading classifier'
gbc = joblib.load("atlas.pkl")
 
# Get the probaility output from the trained method, using the testing sample
# Likelihood background==0, signal==1
prob_predict_train = gbc.predict_proba(X_train)[:,1]
prob_predict_valid = gbc.predict_proba(X_valid)[:,1]

print 'Computing the GBC outputs'
gbc_train = gbc.decision_function(X_train)[:,0]
#gbc_valid = gbc.decision_function(X_valid)[:,0]

# Experience shows me that choosing the top 15% as signal gives a good AMS score.
# This can be optimized though!
pcut = np.percentile(prob_predict_train,85)
 
# This are the final signal and background predictions (likelihood)
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
#print s_train, b_train,
#print s_valid, b_valid

# Now calculate the AMS scores
print 'Calculating AMS score for a probability cutoff pcut=',pcut
def AMSScore(s,b): return  math.sqrt (2.*( (s + b + 10.)*math.log(1.+s/(b+10.))-s))
print '   - AMS based on 50% training   sample:',AMSScore(s_train,b_train)
print '   - AMS based on 50% validation sample:',AMSScore(s_valid,b_valid)

# making performance plots


from matplotlib import pyplot as plt

x_axis = np.linspace(min(gbc_train), max(gbc_train), 30)
#y_axis = W_train

sig_train = sum(Y_train==1.0)
bkg_train = sum(Y_train==0.0)

sig_eff = np.linspace(0., 0., len(x_axis))
bkg_eff = np.linspace(0., 0., len(x_axis))
sig_pur = np.linspace(0., 0., len(x_axis))
eff_pur = np.linspace(0., 0., len(x_axis))

idx = 0
print
for xx in x_axis :
  Yh_train = gbc_train > xx
  sig_eff[idx]=sum((Y_train==1)*(Yh_train==1))*1.0/sig_train
  bkg_eff[idx]=sum((Y_train==0)*(Yh_train==1))*1.0/bkg_train
  #sig_eff[idx]=sum(gbc.decision_function(X_train[Y_train>0.5])[:,0].ravel() > xx)*1.0/sig_train
  #bkg_eff[idx]=sum(gbc.decision_function(X_train[Y_train<0.5])[:,0].ravel() > xx)*1.0/bkg_train
  sig_pur[idx]=sig_eff[idx]*sig_train/sum(Yh_train==1)
  eff_pur[idx]=sig_eff[idx]*sig_pur[idx]

  print ".",
  sys.stdout.flush()
  idx = idx + 1

# Draw objects
ax1 = plt.subplot(111)
plt.plot(x_axis, bkg_eff,color="red",linewidth=2,label='Bkg. (Train)')
plt.plot(x_axis, sig_eff,color="blue",linewidth=2,label='Sig. (Train)')
plt.plot(x_axis, sig_pur,'--',color="green",linewidth=2,
         label='Purity (Train)')
plt.plot(x_axis, eff_pur,':',color="green",linewidth=2,
         label='Eff. Pur. (Train)')

# Make labels and title
plt.title("Higgs ML Signal-Background Efficiency and Purity")
plt.xlabel("Gradient Boosting Classifier")
plt.ylabel("Efficiency (Purity)")
 
# Adjust legends
legend = ax1.legend(loc='upper center', shadow=True,ncol=2)

# Adjust the axis boundaries (just cosmetic)
ax1.axis([min(x_axis), max(x_axis), 0., 1.2])

# Save the result to png
plt.savefig("Sklearn_eff.png")
 
Classifier_training_S = gbc.decision_function(X_train[Y_train>0.5])[:,0].ravel()
Classifier_training_B = gbc.decision_function(X_train[Y_train<0.5])[:,0].ravel()
Classifier_testing_A = gbc.decision_function(X_valid)[:,0].ravel()
  
c_max = max([Classifier_training_S.max(),Classifier_training_B.max(),
             Classifier_testing_A.max()])
c_min = min([Classifier_training_S.min(),Classifier_training_B.min(),
             Classifier_testing_A.min()])


# Get histograms of the classifiers
Histo_training_S = np.histogram(Classifier_training_S,bins=50,
                                range=(c_min,c_max))
Histo_training_B = np.histogram(Classifier_training_B,bins=50,
                                range=(c_min,c_max))
Histo_testing_A = np.histogram(Classifier_testing_A,bins=50,
                               range=(c_min,c_max))
  
# Lets get the min/max of the Histograms
AllHistos= [Histo_training_S,Histo_training_B]
h_max = max([histo[0].max() for histo in AllHistos])*1.25
h_min = max([histo[0].min() for histo in AllHistos])
  
# Get the histogram properties (binning, widths, centers)
bin_edges = Histo_training_S[1]
bin_centers = ( bin_edges[:-1] + bin_edges[1:]  ) /2.
bin_widths = (bin_edges[1:] - bin_edges[:-1])
  
# To make error bar plots for the data, take the Poisson uncertainty sqrt(N)
ErrorBar_testing_A = np.sqrt(Histo_testing_A[0])
  
# Draw objects
ax2 = plt.subplot(211)
ax2 = plt.subplot(111)
  
# Draw solid histograms for the training data
ax2.bar(bin_centers-bin_widths/2.,Histo_training_B[0],facecolor='red',
        linewidth=0,width=bin_widths,label='B (Train)',alpha=0.5)
ax2.bar(bin_centers-bin_widths/2.,Histo_training_S[0],
        bottom=Histo_training_B[0],facecolor='blue',linewidth=0,
        width=bin_widths,label='S (Train)',alpha=0.5)

ff = (1.0*(sum(Histo_training_S[0])+sum(Histo_training_B[0]))) / \
     (1.0*sum(Histo_testing_A[0]))
 
# # Draw error-bar histograms for the testing data
ax2.errorbar(bin_centers, ff*Histo_testing_A[0], yerr=ff*ErrorBar_testing_A,
             xerr=None, ecolor='black',c='black',fmt='.',
             label='Test (reweighted)')
  
# Make a colorful backdrop to show the clasification regions in red and blue
ax2.axvspan(pcut, c_max, color='blue',alpha=0.08)
ax2.axvspan(c_min,pcut, color='red',alpha=0.08)
  
# Adjust the axis boundaries (just cosmetic)
ax2.axis([c_min, c_max, h_min, h_max])
  
# Make labels and title
plt.title("Higgs ML Signal-Background Separation")
plt.xlabel("Gradient Boosting Classifier")
plt.ylabel("Counts/Bin")
 
# Make legend with smalll font
legend = ax2.legend(loc='upper center', shadow=True,ncol=2)
#for alabel in legend.get_texts():
#            alabel.set_fontsize('small')
  
# Save the result to png
plt.savefig("Sklearn_gbc.png")

