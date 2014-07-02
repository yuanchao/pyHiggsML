import ROOT
import sys
import os

file_lines = open("training.csv").readlines()

branch_def = ':'.join(file_lines[0].strip().split(','))

out_file = ROOT.TFile("test_out.root", "recreate")

mvaTree = ROOT.TTree("mvaTree", "variables tree")

# replacing the signal / background to 1 / 0
csv_string = ''.join(file_lines[1:]).replace(',s',',1').replace(',b',',0')

csv_stream = ROOT.istringstream(csv_string)

mvaTree.ReadStream(csv_stream, branch_def, ',')

out_file.cd()
mvaTree.Write()


