'''
Example code to evaluate the training results of
TMVA for multi-variate classification

2014/06/29 Yuan CHAO
'''

print(__doc__)

import ROOT
import sys
import os

# Print usage help
def usage():
    print " "
    print "Usage: python %s [input_root_file]" % sys.argv[0]
    print " "
    sys.exit(1)

if len(sys.argv) < 2:
    usage()
else:

    ROOT.gROOT.LoadMacro("TMVAGui.C");
    ROOT.gROOT.ProcessLine( "TMVAGui(\"%s\")" % sys.argv[1] )

# keep the ROOT thread running
    ROOT.gApplication.Run() 

