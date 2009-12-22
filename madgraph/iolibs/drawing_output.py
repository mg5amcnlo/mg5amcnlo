################################################################################
#
# Copyright (c) 2009 The MadGraph Development team and Contributors
#
# This file is a part of the MadGraph 5 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph license which should accompany this 
# distribution.
#
# For more information, please visit: http://madgraph.phys.ucl.ac.be
#
################################################################################
from __future__ import division
import math
import os
import time
print os.path.realpath(__file__)
print os.path.dirname(os.path.realpath(__file__))
print os.path.split(os.path.dirname(os.path.realpath(__file__)))
_root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]+'/'
print _root_path

class draw_diagram:
    """ all generic routine in order to written diagram """

    def __init__(self,diagram='',file='diagram.eps'):
        """ load the data and assign the output file """
        
        self.text=''
        self.diagram=diagram
        self.file=file    
    
    def draw(self):
        """ draw the diagram """
        
        self.initialize()
        for line in self.diagram.LineList:
            self.draw_line(line)
        self.conclude()

    def initialize(self):
        """ start the initialization of the diagram """
        pass
    
    def conclude(self):
        """ last operation, writing the file
            default write the file
        """
        
        image_file=open(self.file,'w')
        image_file.writelines(self.text)
        image_file.close()
        return
    
    def draw_line(self,line):
        """ return the code associate tho this line """
        
        line_type=line.get_info('line')
        getattr(self,'draw_'+line_type)(line)
                        
        name=line.get_name()
        self.associate_name(line,name)

        #standard name for type: stragiht, dashed, wavy, curly
        #should be define in any true class based on this generic one
        
    def associate_name(self,line,name):
        """ place the name of the line at the correct position """
        pass
    
class draw_diagram_eps(draw_diagram):
    """ all the routine need to write a given diagram in eps format """
    
    width=450
    height=450
    npage=1
    x_min=150
    y_min=450
    x_max=450
    y_max=750
           
    def initialize(self):
        """ def the header of the file """

        self.text=file(_root_path+'iolibs/input_file/drawing_eps_header.inc').read()
        #replace variable in text put inside $ $
        self.text = self.text.replace('$x$',str(self.width))
        self.text = self.text.replace('$y$',str(self.height))
        self.text = self.text.replace('$npages$',str(self.npage))
        
    
    def conclude(self):
        """ def the footer of the file """
        
        self.text+='showpage\n'
        self.text+='%%trailer\n'

        #write the diagram.
        draw_diagram.conclude(self)
    

    def rescale(self, x, y):
        """ rescale the x, y coordinates of the point (belong to 0,1 interval)
            to the relative position of the image box
        """
        x=self.x_min+(self.x_max-self.x_min)*x
        y=self.y_min+(self.y_max-self.y_min)*y
        
        return x,y
         
           
    def line_format(self, x1, y1, x2, y2, name):
        """return the line in the correct format """
        x1, y1 = self.rescale(x1, y1)
        x2, y2 = self.rescale(x2, y2)
                
        return " %s %s %s %s %s \n" % (x1, y1, x2, y2, name)
        
    def draw_straight(self,line):
        """ return the code associate to this fermionic line """
        
        self.text += self.line_format(line.start['pos_x'], line.start['pos_y'], \
                         line.end['pos_x'], line.end['pos_y'], 'Ffermion')
        
        
    def draw_dashed(self,line):
        """ return the code associate to this spin 0 line """

        self.text += self.line_format(line.start['pos_x'], line.start['pos_y'], \
                         line.end['pos_x'], line.end['pos_y'], 'Fhiggs')
 
        
    def draw_wavy(self,line):
        """ return the code associate to the spin 1 line """

        self.text += self.line_format(line.start['pos_x'], line.start['pos_y'], \
                         line.end['pos_x'], line.end['pos_y'], '0 Fphotond')


    def draw_curly(self,line):
        """ return the code associate to the spin 1 line """
        
        #print line.start, line.end
        self.text += self.line_format(line.start['pos_x'], line.start['pos_y'], \
                         line.end['pos_x'], line.end['pos_y'], '0 Fgluon')
 
 
    def associate_name(self,line,name):
        """ place the name of the line at the correct position """

        x1, y1 = line.start['pos_x'], line.start['pos_y']
        x2, y2 = line.end['pos_x'], line.end['pos_y']

        print '(%s,%s) , (%s, %s)' %(x1,y1,x2,y2)
        d  = math.sqrt((x1-x2)**2+(y1-y2)**2)
        dx = (x1-x2)/d
        dy = (y1-y2)/d        
        
        if dy < 0:
            dx, dy = -dx, -dy
        
        x_pos = (x1 + x2) / 2 + 0.05  * dy
        y_pos = (y1 + y2) / 2 - 0.05 * dx      

        x_pos, y_pos =self.rescale(x_pos, y_pos)
        self.text += ' %s  %s moveto \n' % (x_pos,y_pos)  
        self.text += '(' + name+ ')   show\n'


class draw_diagrams_eps(draw_diagram_eps):
    """ all the routine need to write a set of diagrams in eps format """
    
    x_min=50
    y_min=600
    x_max=200
    y_max=750




if __name__ == '__main__':
    
    from madgraph.interface.cmd_interface import MadGraphCmd
    import drawing_lib as draw
    cmd = MadGraphCmd()
    cmd.do_import('v4 /Users/omatt/fynu/MadWeight/MG_ME_MW/Models/sm/particles.dat')
    cmd.do_import('v4 /Users/omatt/fynu/MadWeight/MG_ME_MW/Models/sm/interactions.dat')
    cmd.do_generate(' g g > g g')
    
    len(cmd.curr_amp['diagrams'])
    for i in range(0, len(cmd.curr_amp['diagrams'])):
        diagram = cmd.curr_amp['diagrams'][i]
        start=time.time()
        upgrade_diagram = draw.Feynman_Diagram(diagram,cmd.curr_model)
        upgrade_diagram.charge_diagram()
        print 'len', len(upgrade_diagram.LineList), len(upgrade_diagram.VertexList)
        upgrade_diagram.define_level()        
        print 'len', len(upgrade_diagram.LineList), len(upgrade_diagram.VertexList)
        #print diagram
        upgrade_diagram.find_initial_vertex_position()
        print upgrade_diagram._debug_position()
        #print time.time()-start, 'time to upgrade diagram'
        plot = draw_diagrams_eps(upgrade_diagram,'diagram_%s.eps' % (i) )

        plot.draw()
        print time.time()-start, 'full time to draw a diagram'
    print 'done'
