'''
Example code to read a CSV file and pass to
TMVA for multi-variate classification

2014/06/29 Yuan CHAO
'''

print(__doc__)

import ROOT
import sys
import os


# open the CSV file and extra the branch definition from header

file_lines = open("training.csv").readlines()

branch_def = ':'.join(file_lines[0].strip().split(','))

outfname = "test.root"
fout = ROOT.TFile(outfname, "recreate")

# create a TTree object to pass to TMVA
mvaTree = ROOT.TTree("mvaTree", "variables tree")

# replacing the {signal, background} to {1, 0}
csv_string = ''.join(file_lines[1:]).replace(',s',',1').replace(',b',',0')

csv_stream = ROOT.istringstream(csv_string)

# read the CSV content by calling the TTree helper function
mvaTree.ReadStream(csv_stream, branch_def, ',')

fout.cd()
#mvaTree.Write()

# keeps objects otherwise removed by garbage collected in a list
gcSaver = []
 
# create a new TCanvas
gcSaver.append(ROOT.TCanvas())
 
# draw an empty 2D histogram for the axes
histo = ROOT.TH2F("histo","",1,-5,5,1,-5,5)
histo.Draw()
 
ROOT.TMVA.Tools.Instance()
 
factory = ROOT.TMVA.Factory("TMVAClassification", fout,
                            ":".join([
                                "!V",
                                "!Silent",
                                "Color",
                                "DrawProgressBar",
#                                "Transformations=I;D;P;G,D",
                                "AnalysisType=Classification"]
                                     ))

for var in (branch_def.split(':'))[1:-2] :
  factory.AddVariable(var, "F")

factory.SetWeightExpression( "Weight" );

factory.AddSignalTree(mvaTree)
factory.AddBackgroundTree(mvaTree)


# cuts defining the signal and background sample
sigCut = ROOT.TCut("Label > 0.5")
bgCut = ROOT.TCut("Label <= 0.5")
 
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
                       "NTrees=200",
                       "MinNodeSize=5%",
                       "MaxDepth=5",
                       "BoostType=AdaBoost",
                       "AdaBoostBeta=0.5",
                       "SeparationType=GiniIndex",
                       "nCuts=50",
                       "PruneMethod=NoPruning",
                       ]))
 
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

fout.Close()
    
print "=== wrote root file %s\n" % outfname
print "=== TMVAClassification is done!\n"

## open the GUI for the result macros    
#ROOT.gROOT.LoadMacro("TMVAGui.C");
#ROOT.gROOT.ProcessLine( "TMVAGui(\"%s\")" % outfname )
#    
## keep the ROOT thread running
#ROOT.gApplication.Run() 

