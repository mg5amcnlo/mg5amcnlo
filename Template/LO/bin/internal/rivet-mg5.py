#! /usr/bin/env python

"""
Usage: %(prog)s <cmdfile>

Configure and run the Rivet analysis library based on a command filex
"""

from __future__ import print_function

## Parse the command-line arguments
import argparse
ap = argparse.ArgumentParser(usage=__doc__)
ap.add_argument("CMDFILE")
args = ap.parse_args()

## Read the config file
import configparser
cp = configparser.ConfigParser()
cp.read_string(u"[RIVET]\n" + open(args.CMDFILE).read())
cmds = cp["RIVET"]
ignorebeams = cmds.getboolean("IgnoreBeams", fallback=False)
skipmw = cmds.getboolean("SkipMultiWeights", fallback=False)
selectmw = cmds.get("SelectMultiWeights", fallback="")
deselectmw = cmds.get("DeselectMultiWeights", fallback="")
nomweight = cmds.get("NominalWeightName", fallback="")
weightcap = cmds.get("WeightCap", fallback=None)
analyses = cmds.get("Analyses", fallback="EXAMPLE,MC_JETS").split(",")
hepmcfile = cmds.get("HepMCFile", fallback="events.hepmc")
histofile = cmds.get("HistoFile", fallback="rivet.yoda")
xsec_pb = cmds.get("CrossSectionPb", fallback=None)

## Make and configure the Rivet objects
import rivet, os, sys
ah = rivet.AnalysisHandler()
ah.setIgnoreBeams(ignorebeams)
ah.skipMultiWeights(skipmw)
ah.selectMultiWeights(selectmw)
ah.deselectMultiWeights(deselectmw)
ah.setNominalWeightName(nomweight)
if weightcap is not None:
    ah.setWeightCap(float(weightcap))
for ana in analyses:
    ah.addAnalysis(ana)
# if args.PRELOADFILE is not None:
#     ah.readData(args.PRELOADFILE)
# if args.DUMP_PERIOD:
#     ah.dump(args.HISTOFILE, args.DUMP_PERIOD)


## Run the init, event loop, and finalize stages
run = rivet.Run(ah)
if xsec_pb is not None:
    run.setCrossSection(float(xsec_pb))
run.init(hepmcfile, 1.0)
processed_ok = True
nevt = 0
while processed_ok:
    processed_ok = run.processEvent()
    if not processed_ok:
        raise Exception("Rivet processing error")
    nevt += 1
    if nevt % 1000 == 0:
        print("Processed {:d} events".format(nevt))
    run.readEvent()
run.finalize()

## Get output data objects
#yodas = ah.getYodaAOs()
ah.writeData(histofile)
