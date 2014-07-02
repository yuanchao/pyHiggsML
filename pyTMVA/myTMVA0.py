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
