'''
Example code to evaluate the training results of
TMVA for multi-variate classification

2014/06/29 Yuan CHAO
'''

print(__doc__)

import ROOT
import sys
import os
import math

fin = ROOT.TFile("test.root")

# create a TTree object to pass to TMVA
testTree = fin.Get("TestTree")
trainTree = fin.Get("TrainTree")

hbdt = ROOT.TH1F("hbdt", "histogram", 100, -1., 1.)


# This are the final signal and background predictions
trainTree.Draw("BDT>>hbdt","BDT>=0.1", "goff")
Yhat_train = hbdt.GetSumOfWeights()
testTree.Draw("BDT>>hbdt","BDT>=0.1", "goff")
Yhat_valid = hbdt.GetSumOfWeights()

# To calculate the AMS data, first get the true positives and true negatives
# Scale the weights according to the r cutoff.
# Here the class ID is swapped!!!
# The training sample has been "un-weighted", needs extra treatments
# Scaling factor obtained from the TMVA log
trainTree.Draw("BDT>>hbdt","weight*(classID==0)", "goff");
TruePositive_train = hbdt.GetSumOfWeights()*(1.0/0.5)*(691.964/85667.)
trainTree.Draw("BDT>>hbdt","weight*(classID==1)", "goff");
TrueNegative_train = hbdt.GetSumOfWeights()*(1.0/0.5)*(411124./164333.)
testTree.Draw("BDT>>hbdt","weight*(classID==0)", "goff");
TruePositive_valid = hbdt.GetSumOfWeights()*(1.0/0.5)
testTree.Draw("BDT>>hbdt","weight*(classID==1)", "goff");
TrueNegative_valid = hbdt.GetSumOfWeights()*(1.0/0.5)
#print TruePositive_train, TrueNegative_train,
#print TruePositive_valid, TrueNegative_valid

# s and b for the training
# The training sample has been "un-weighted", needs extra treatments
# Scaling factor obtained from the TMVA log
# 0.27 is a cut point matching to the sk-learn example; 0.235 is optimal
trainTree.Draw("BDT>>hbdt","weight*(BDT>=0.27&&classID==0)", "goff");
s_train = hbdt.GetSumOfWeights()*(1.0/0.5)*(691.964/85667.)
trainTree.Draw("BDT>>hbdt","weight*(BDT>=0.27&&classID==1)", "goff");
b_train = hbdt.GetSumOfWeights()*(1.0/0.5)*(411124./164333.)
testTree.Draw("BDT>>hbdt","weight*(BDT>=0.27&&classID==0)", "goff");
s_valid = hbdt.GetSumOfWeights()*(1.0/0.5)
testTree.Draw("BDT>>hbdt","weight*(BDT>=0.27&&classID==1)", "goff");
b_valid = hbdt.GetSumOfWeights()*(1.0/0.5)
print s_train, b_train,
print s_valid, b_valid

# Now calculate the AMS scores
print 'Calculating AMS score for a probability cutoff pcut=', 0.1
def AMSScore(s,b): return  math.sqrt (2.*( (s + b + 10.)*math.log(1.+s/(b+10.))-s))
print '   - AMS based on 50% training   sample:',AMSScore(s_train,b_train)
print '   - AMS based on 50% validation sample:',AMSScore(s_valid,b_valid)
