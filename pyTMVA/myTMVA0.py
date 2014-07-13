# Example code for TMVA in pyROOT
# By Andrea Holzner
# http://aholzner.wordpress.com/2011/08/27/a-tmva-example-in-pyroot/

import ROOT
 
# create a TNtuple
ntuple = ROOT.TNtuple("ntuple","ntuple","x:y:signal")
 
# generate 'signal' and 'background' distributions
for i in range(10000):
    # throw a signal event centered at (1,1)
    ntuple.Fill(ROOT.gRandom.Gaus(1,1), # x
                ROOT.gRandom.Gaus(1,1), # y
                1)                      # signal
     
    # throw a background event centered at (-1,-1)
    ntuple.Fill(ROOT.gRandom.Gaus(-1,1), # x
                ROOT.gRandom.Gaus(-1,1), # y
                0)                       # background

# keeps objects otherwise removed by garbage collected in a list
gcSaver = []
 
# create a new TCanvas
gcSaver.append(ROOT.TCanvas())
 
# draw an empty 2D histogram for the axes
histo = ROOT.TH2F("histo","",1,-5,5,1,-5,5)
histo.Draw()
 
# draw the signal events in red
ntuple.SetMarkerColor(ROOT.kRed)
ntuple.Draw("y:x","signal > 0.5","same")
 
# draw the background events in blue
ntuple.SetMarkerColor(ROOT.kBlue)
ntuple.Draw("y:x","signal <= 0.5","same")

ROOT.c1.SaveAs("scatter.png")

factory.AddVariable("x","F")
factory.AddVariable("y","F") 
 
factory.AddSignalTree(ntuple)
factory.AddBackgroundTree(ntuple)
 
# cuts defining the signal and background sample
sigCut = ROOT.TCut("signal > 0.5")
bgCut = ROOT.TCut("signal <= 0.5")
 
factory.PrepareTrainingAndTestTree(sigCut,   # signal events
                                   bgCut,    # background events
                                   ":".join([
                                        "nTrain_Signal=0",
                                        "nTrain_Background=0",
                                        "SplitMode=Random",
                                        "NormMode=NumEvents",
                                        "!V"
                                       ]))

method = factory.BookMethod(ROOT.TMVA.Types.kBDT, "BDT",
                   ":".join([
                       "!H",
                       "!V",
                       "NTrees=850",
                       "nEventsMin=150",
                       "MaxDepth=3",
                       "BoostType=AdaBoost",
                       "AdaBoostBeta=0.5",
                       "SeparationType=GiniIndex",
                       "nCuts=20",
                       "PruneMethod=NoPruning",
                       ]))
 
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

