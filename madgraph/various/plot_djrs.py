#! /usr/bin/env python
################################################################################
#
# Copyright (c) 2010 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Example code to plot custom curves based on djrs.dat with matplotlib"""
import os
import sys 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab


################################################################################
#  TRY TO LINK TO HISTOGRAMS.PY
################################################################################
# We need to link to histograms.dat.
# You can put the path to MG5_aMC main directory here to link from any directory
#sys.path.append('PATH/TO/MADGRAPH')
#try to find relatively to this file
sys.path.append(os.path.basename(os.path.basename(__file__))) #../
sys.path.append(os.path.basename(os.path.basename(os.path.basename(__file__)))) #../../
# lets try the import.
try:
    import madgraph
except ImportError:
    try:
        import internal
    except ImportError:
        print "You need to specify the path to the MG5_aMC directory"    
        sys.exit(1)
    else:
        from internal.histograms import *
else:
    from madgraph.various.histograms import *

################################################################################
#  CHOICE OF INPUT FILE
################################################################################
# check if an argument is passed as inputfiles:
if len(sys.argv) >1:
    input_file = sys.argv[1]
else:
    #take the default path
    input_file = './Events/run_01/tag_1_djrs.dat'    
print "Reading information from: ", input_file


################################################################################
#  PARSING THE FILE AND ACCESS TO BASIC PROPERTY OF THE OBJECT
################################################################################
#parsing the data and create the object instance 
hwu_list = HwUList(input_file, raw_labels=True)
# raw label prevent modification of the weight label. They will stay at their inputfile value

# get the list of the plot names
names =  hwu_list.get_hist_names()
#print names
# get the list of the weight label
weights_name = hwu_list.get_wgt_names()
#print weights_name
# In this example, I want to plot the DJR1 -> select the histograms with d01 in their name:
selected_hist = [hwu_list.get(n) for n in names if 'd01' in n]

################################################################################
#  CREATE THE PLOT AND THE ASSOCIATE RATIO PLOT
################################################################################
# define a multi-plot frame for the plot
gs1 = gridspec.GridSpec(2, 1, height_ratios=[5,1])
gs1.update(wspace=0, hspace=0) # set the spacing between axes. 
main_frame = plt.subplot(gs1[0]) # main frame/plot
ratio_frame = plt.subplot(gs1[1]) # ratio frame/plot

main_frame.set_yscale('log')
#main_frame.yaxis.set_label_coords(-0.07, 0.90) 
main_frame.set_ylabel(r'$\frac{d\sigma_{LO}}{dDJR1} [pb]$')
main_frame.set_title('Differential Jet Rate')
main_frame.set_xticklabels([]) #remove x-axis in the main frame (due to the ratio frame)

#ratio_frame.xaxis.set_label_coords(0.90, -0.20) 
ratio_frame.set_xlabel(r'$log(DJR1/1[GeV])$')
ratio_frame.set_ylabel(r'$ME/PS$')


################################################################################
#  SETTING THE CURVE
################################################################################
# Adding the curves. Here I want to plot two curves:
# the curve with the maximum value of QCUT from the 0 jet sample
# the curve with the minimal value of QCUT from the highest multiplicity sample
qcut= [l for l in weights_name if l.startswith('MUF=1_MUR=1_PDF=247000_MERGING=')]
min_qcut,max_qcut = qcut[0],qcut[-1]

#get the histo
h_0j = [h for h in selected_hist if 'Jet sample 0' in h.get_HwU_histogram_name()][0]
h_1j = [h for h in selected_hist if 'Jet sample 1' in h.get_HwU_histogram_name()][0]

y_0j = h_0j.get(min_qcut)
y_1j = h_1j.get(max_qcut) 
l_0j, = main_frame.plot(h_0j.get('bins'), y_0j, label='0j', linestyle='steps') 
l_1j, = main_frame.plot(h_1j.get('bins'), y_1j, label='1j', linestyle='steps') 


################################################################################
#  ADDING UNCERTAINTY BAND
################################################################################
# Add the PDF uncertainty on the 0j sample
# the attribute of get_uncertainty_band can be a regular expression, a list of weight name, or a function returning 0/1
# Special attributes exists: PDF, QCUT, ALPSFACT, SCALE # this assumes standard name formatting
# For PDF you can force the type of uncertainty band by specifying mode='gaussian' or mode='hessian'
# if using 'PDF' attributes the correct type should be found automatically
pdfmin, pdfmax = h_0j.get_uncertainty_band('PDF')
fill_between_steps(h_0j.get('bins'), pdfmin, pdfmax, ax=main_frame, facecolor=l_0j.get_color(), alpha=0.5,
                        edgecolor=l_0j.get_color()
)
# use a second method for h_1j
pdfmin, pdfmax = h_1j.get_uncertainty_band(['MUF=1_MUR=1_PDF=%i_MERGING=30' % i for i in range(247000,247100)], mode='hessian')
fill_between_steps(h_1j.get('bins'), pdfmin, pdfmax, ax=main_frame, facecolor=l_1j.get_color(), alpha=0.5,
                        edgecolor=l_1j.get_color()
)


################################################################################
#  ADDING RATIO PLOT
################################################################################
ratio = [y_0j[i]/y_1j[i] if y_1j[i] else 0 for i in xrange(len(y_0j))]
ratio_frame.plot(h_0j.get('bins'), ratio, linestyle='steps')


################################################################################
#  SETTING SOME STYLE IMPROVMENT FOR THE PLOT
################################################################################
# Some final style processing of matplotlib
main_frame.legend(ncol=2, prop={'size':12}, loc=4)
ratio_frame.set_yticks(ratio_frame.get_yticks()[:-1]) # remove upper tick of the ratio plot
# Adding the MadGraph5_aMC@NLO flag on the plot (likely overcomplicated plot.text() should be better)
ax_c = main_frame.twinx()
ax_c.set_ylabel('MadGraph5_aMC@NLO')
ax_c.yaxis.set_label_coords(1.01, 0.25)
ax_c.set_yticks(main_frame.get_yticks())
ax_c.set_yticklabels([])


################################################################################
#  WRITE THE OUTPUT FILE
################################################################################
plt.savefig("DJR1.pdf") # many extension possible (jpg/png/...)






