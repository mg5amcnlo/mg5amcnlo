#!/usr/bin/env python
################################################################################
#
# Copyright (c) 2013 The MadGraph5_aMC@NLO Development team and Contributors
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
import glob
import os
import re
import subprocess

pjoin = os.path.join

class histograms:
    """ x and y values """
    def __init__(self):
        pass


class one_plot:
    """ All information about one plot """
    def __init__(self):
        self.histo = {}
        self.top = ""
        self.end = ""
        self.max = 0
        self.max_file =''
        
    def add_comment(self, top_comment, end_comment=""):
        
        self.top = top_comment
        self.end = end_comment       
        
    
    def add_histo(self, string_values, tag):
        """Add the histogram in the data base (string format) """
        if tag in self.histo.values():
            print "Warning: skyping the histogram with tag " + tag
            print "         since it is already present in the data base"
            return
        self.histo[tag] = {}
        self.histo[tag]["values"] = string_values
        old_max = self.max

        for line in string_values.split('\n'):
            split = line.split()
            if len(split) in [2,3]:
                self.max = max(self.max, float(line.split()[1]))
        if self.max != old_max:
            self.max_file = tag
        
    def get_histo(self, tag, norm):
        """return a string with the histogram values, and the normalization """
        if tag not in self.histo or self.histo[tag]["values"] == '':
            return ''
        histo = "SET ORDER X Y " + str(norm) + " \n"
        histo += self.histo[tag]["values"]
        return histo    
    

class load_data:
    """ Import all the plots from top drawer files """
    def __init__(self):

        self.plots = {}  # list of the tags associated with the plots
        self.normalization = 1.0
        self.order_plots = []
        self.cross = {}

    def import_data(self, file_name, file_nb, pretag=""):
        """ Read the file with the name file_name, and import the data for all the plots"""

        trappe = open(file_name, 'r')
        while 1:
            line = trappe.readline()
            if line == "": break
            if line.find("Cross-section") > -1:
                pos = line.find("is:")
                list = line[pos + 4:].split()
                self.cross1 = float(list[0])
            if line.find("NEW PLOT") > -1 or line.find("SET INTENSITY") > -1: 
                self.readplot(trappe, file_nb) 

    def print_plots(self, outputfile, file1, file2):
        """Write out histograms """
        trappe = open(outputfile, 'w')
        trappe.write("SET DEVICE POSTSCRIPT ORIENTATION=3 \n")
        trappe.write("set intensity 5 \n")



        for index, tag_plot in enumerate(self.order_plots):
            norm1 = 1 
            norm2 = 1
            
            if self.plots[tag_plot].max_file == file1:
                color1, color2 = 'WHITE','BLUE'
                histtype1, histtype2 = 'HIST SOLID\n', 'SET PATTERN .05 .07\n' 
            else:
                file1, file2 = file2, file1
                color1, color2 = 'BLUE', 'WHITE'
                histtype2, histtype1 = 'HIST SOLID\n', 'SET PATTERN .05 .07\n' 
            if self.plots[tag_plot].get_histo(file1, norm1) == '':
                continue
    
            trappe.write("NEW PLOT \n")
            top = self.plots[tag_plot].top
            if top.find('# particles') > -1:
                top = top.replace('LOG', 'LIN')
            trappe.write(top)
#            trappe.write("SET COLOR %s \n" % color2)
#            trappe.write("HIST PATTERNED %s \n" % color1)

                
            try:
                trappe.write(histtype1)                
                trappe.write(self.plots[tag_plot].get_histo(file1, norm1))
                trappe.write(histtype1)
                trappe.write("HIST PATTERNED %s \n" % color1)
#                trappe.write("SET COLOR %s \n" % color1)
            except:
                print "warning: cannot find histo in file 1 for tag " + tag_plot 
        
            try:
                trappe.write(self.plots[tag_plot].get_histo(file2, norm2))
                trappe.write(histtype2)
                trappe.write("HIST PATTERNED %s \n" % color2)
                trappe.write("SET COLOR WHITE \n")
            except Exception, error:
                print error
                print "warning: cannot find histo in file 2 for tag " + tag_plot 
                raise
            
            #trappe.write(self.plots[tag_plot].end)
    
            trappe.write("\n")
            trappe.write("\n")
        trappe.close()

    def readplot(self, trappe, file_nb):
        """ import the data of a single plot """
        top = ""
        end = ""
        histo = ""
        phase = 0
        plot_tag = ""
        newplot = 1
        while 1:
          line = trappe.readline()
          #print line
          if line == "": break
          if phase == 0:  # top of the histogram
             if line.find("TITLE TOP") > -1:
                index = line.find('''"''')
                if index < 0:
                  print "warning: unable to find the name of the plot in the title"
                  print "         skipping this plot (might not be a real plot)"
                  return
                else:
                  plot_tag = line[index + 1:]
                  plot_tag = plot_tag.replace('''"''', "")
                  plot_tag = plot_tag.replace("\n", "")
                  plot_tag = plot_tag.replace(" ", "")
             if line.find("SET LIMITS X") > -1:
                  tag = line.replace(".0 ", "")
                  tag = tag.replace(".00 ", "") 
                  tag = tag.replace(".000 ", "") 
                  tag = tag.replace(".0000 ", "") 
                  tag = tag.replace(".00000 ", "") 
                  tag = tag.replace(".0\n", "") 
                  tag = tag.replace(".00\n", "") 
                  tag = tag.replace(".000\n", "") 
                  tag = tag.replace(".0000\n", "") 
                  tag = tag.replace(".00000\n", "") 
                  tag = tag.replace(" ", "")
                  tag = tag.replace("\n", "")
                  tag = tag.replace("3.14160", "3.14159")
                  tag = tag.replace("9.42480", "9.42478")
                  tag = tag.replace("4.18880", "4.18879")
                  if plot_tag != "":
                    plot_tag += tag
                    if plot_tag not in self.plots:
                      self.plots[plot_tag] = one_plot()
                      self.order_plots.append(plot_tag)
                    else:
                      newplot = 0
                  else:
                    print "warning: unusual format, unable to extract the tag of the plot"
                    print "         skipping this plot (might not be a real plot)"
                    return
                  
             if line.find("SET ORDER") > -1 and plot_tag != "":
                line = trappe.readline()
                phase = 1
             else:
                top += line  # store the line 
          if phase == 1:  # histogram values
            if line.find("PLOT") < 0 and line.find("SET PATTERN") < 0 and line.find("HIST") < 0:
              histo += line  # store the line
            else:
              if line.find("HISTO") > -1: 
                histo_tag = file_nb
                self.plots[plot_tag].add_histo(histo, file_nb)
    
                if plot_tag.find("Weights") > -1:
                  pos = histo.find("1.0100")
                  if pos > -1:
                     list = histo[pos:].split()
                     self.cross[file_nb] = float(list[1])
                histo = ""
                phase = 2
    
    
          if phase == 2:
             if line.find("NEW PLOT") > -1:
              if (newplot):
                self.plots[plot_tag].add_comment(top, end_comment=end)
              self.readplot(trappe, file_nb)
             else:
                 if line != "":end += line

        if plot_tag and plot_tag in self.plots:
            self.plots[plot_tag].add_comment(top, end_comment=end)

def merge_all_plots(path1, path2, outputpath='/tmp', td='../../td/td', MA=None):
    """take a MA4 output and merge all the plots present in the HTML output"""
    
    #find which plots correspond to what
    pattern = re.compile(r'''TITLE TOP\s+\"\s*([^\"]*)\s*\"''')
    all_plot1 = {}
    for filepath in misc.glob('ma_*.top', path1):
        filename = os.path.basename(filepath)
        text = open(filepath).read()
        try:
            title = pattern.search(text).groups()[0].strip()
        except AttributeError:
            continue
        all_plot1[title] = filename
        
    # find the plot to add:
    for filepath in misc.glob('ma_*.top', path2):
        filename = os.path.basename(filepath)
        text = open(filepath).read()
        try:
            title = pattern.search(text).groups()[0].strip()
        except AttributeError:
            continue
        if title not in all_plot1:
            continue
        my_data = load_data()
        my_data.import_data(pjoin(path1, all_plot1[title]), 1)
        my_data.import_data(pjoin(path2, filename), 2)
        my_data.print_plots(pjoin(outputpath, filename), 1, 2)
        if 'DYLD_LIBRARY_PATH' not in os.environ:
            os.environ['DYLD_LIBRARY_PATH'] =  os.path.dirname(td)
        elif os.path.dirname(td) not in os.environ['DYLD_LIBRARY_PATH']:
            os.environ['DYLD_LIBRARY_PATH'] = '%s:%s' %( os.environ['DYLD_LIBRARY_PATH'], os.path.dirname(td))
        devnull = open(os.devnull,'w')
        subprocess.call([td, filename], cwd=outputpath, stdout=devnull)
        devnull.close()
        if MA:
            subprocess.call([pjoin(MA, 'epstosmth'),"--gsopt=\'-r60x60 -dGraphicsAlphaBits=4\'", 
                         "--gsdev=jpeg", filename.replace('.top', '.ps')], cwd=outputpath)
        #os.system('%s %s' % (td, path))      
        
        

if __name__ == "__main__":

    merge_all_plots('PROC_EWdim6_1/HTML/run_12/plots_parton_tag_1/',
                    'PROC_EWdim6_1/HTML/run_11_rw_15/plots_parton_reweight_tag_1/')

