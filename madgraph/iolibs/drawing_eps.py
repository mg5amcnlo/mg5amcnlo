################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
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

"""This files contains class for creating files or object representing a 
diagram or a set of diagrams.

class structure:
 
DrawDiagram: 
    In principle ALL routines representing a diagram in Any format SHOULD derive
    from this class. This is a (nearly empty) frameworks to draw a diagram 
    in any type format.  

    This frameworks defines in particular 
        - function to convert the input diagram in the correct object. 
            [convert_diagram]
        - main loop to draw a diagram in a line-by-line method
            [draw - draw_diagram]
        
DrawDiagramEPS:
    This contains all the routines to represent one diagram in Encapsuled 
    PostScript (EPS)
    
DrawDiagramsEPS:
    This contains all the routines to represent a set of diagrams in Encapsuled 
    PostScript (EPS)."""

from __future__ import division

import os
import math
import madgraph.core.drawing as draw
import madgraph.core.base_objects as base_objects
import madgraph.loop.loop_base_objects as loop_objects
import logging

logger = logging.getLogger('madgraph.drawing_eps')

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'

#===============================================================================
# DrawDiagramEps
#===============================================================================
class EpsDiagramDrawer(draw.DiagramDrawer):
    """Class to write a EPS file containing the asked diagram 
    This class follows the DrawDiagram Frameworks.
    
    The main routine to draw a diagram is 'draw' which call
    1) initialize: setup things for the diagram (usually open a file)
    2) convert_diagram : Update the diagram in the correct format if needed
    3) draw_diagram : Perform diagram dependent operation
    4) conclude : finish the operation. 
    """

    #some page information
    width = 450
    height = 450
    npage = 1

    # Define where to put the diagram in the page. This is the coordinate of 
    #the two opposites point of the drawing area
    x_min = 150
    y_min = 450
    x_max = 450
    y_max = 750
    
    blob_size = 1.5

    def initialize(self):
        """Operation done before starting to create diagram specific EPS content
        First open the file in write mode then write in it the header and the 
        library of particle type."""

        # Open file 
        super(EpsDiagramDrawer, self).initialize()

        # File Header
        text = "%!PS-Adobe-2.0\n"
        text += "%%" + "BoundingBox: -20 -20 %s  %s \n" % \
                                                       (self.width, self.height)
        text += "%%DocumentFonts: Helvetica\n"
        text += "%%" + "Pages:  %s \n" % self.npage
        self.file.writelines(text)

        # Import the definition of the different way to represent a line
        self.file.writelines(open(os.path.join(_file_path, \
                        'iolibs/template_files/drawing_eps_header.inc')).read())


    def conclude(self):
        """Operation to perform when all code related to a specific diagram are
        finish. Operation :
        - Add the 'end of page' code
        - write unwritten text and close the file. [DrawDiagram.conclude]"""

        # Add an 'end of page statement'
        self.text = 'showpage\n'
        self.text += '%%trailer\n'

        #write the diagram file
        super(EpsDiagramDrawer, self).conclude()


    def rescale(self, x, y):
        """All coordinates belongs to [0,1]. So that in order to have a visible
        graph we need to re-scale the graph. This method distort the square in
        a oblong. Deformation are linear."""

        # Use the information for the graph position. 'self.x_???,self.y_????
        #are the coordinate of the two opposites point of the drawing area. 
        x = self.x_min + (self.x_max - self.x_min) * x
        y = self.y_min + (self.y_max - self.y_min) * y

        return x, y


    def line_format(self, x1, y1, x2, y2, name):
        """Specify the text format of a specific Particles.
        EPS format for Particle is either [X Y X Y NAME] or [X Y X Y NUM NAME].
        In this routine we will consider only the first format. The second can 
        be matched by redefining name in [NUM NAME]."""

        # Compute real position for starting/ending point
        x1, y1 = self.rescale(x1, y1)
        x2, y2 = self.rescale(x2, y2)

        #return the line in correct format
        return " %s %s %s %s %s \n" % (x1, y1, x2, y2, name)

    def draw_vertex(self, vertex, bypass = ['QED','QCD']  ):
        """Add blob in case on non QED-QCD information"""
        
        interaction = self.model.get_interaction(vertex.id)
        if interaction:
            order = interaction.get('orders')
            order = [key for key in order.keys() if order[key] and \
                                                     key not in bypass]

            if order:
                x1, y1 = self.rescale(vertex.pos_x, vertex.pos_y)
                self.text += " %s %s %s 1.0 Fblob \n" % (x1, y1, self.blob_size)



    def draw_straight(self, line):
        """ADD the EPS code for this fermion line."""

        #add the code in the correct format
        self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,
                         line.end.pos_x, line.end.pos_y, 'Ffermion')

    def draw_curved_dashed(self, line, cercle):
        """ADD the EPS code for this fermion line."""

        if not cercle:
            curvature = 0.4
        else:
            curvature = 1
        
        if (line.begin.pos_x, line.begin.pos_y) == self.curved_part_start:
            curvature *= -1
        
        #add the code in the correct format
        x1, y1 = self.rescale(line.begin.pos_x, line.begin.pos_y)
        self.text += ' %s  %s moveto \n' % (x1, y1)
        self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,
                         line.end.pos_x, line.end.pos_y, '%s Fhiggsl' %\
                         curvature)

    def draw_curved_straight(self, line, cercle):
        """ADD the EPS code for this fermion line."""

        if not cercle:
            curvature = 0.4
        else:
            curvature = 1
        
        if (line.begin.pos_x, line.begin.pos_y) == self.curved_part_start:
            curvature *= -1
        
        #add the code in the correct format
        self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,
                         line.end.pos_x, line.end.pos_y, '%s Ffermionl' %\
                         curvature)
    
    def draw_circled_straight(self, line, cercle):
        """ADD the EPS code for this fermion line."""

        if not cercle:
            curvature = 4
        else:
            curvature = 5
            
        is_tadpole = line.begin.pos_x==line.end.pos_x and \
                                                line.begin.pos_y==line.end.pos_y
                                                
        if is_tadpole:
            # Obtain the direction of the propagator supporting the tadpole
            direction = None
            for l in line.begin.lines:
                new_direction = (l.end.pos_x-l.begin.pos_x, l.end.pos_y-l.begin.pos_y)
                if new_direction==(0.0,0.0):
                    continue
                norm = math.sqrt(new_direction[0]**2+new_direction[1]**2)
                new_direction = (new_direction[0]/norm, new_direction[1]/norm)

                if not direction:
                    direction = new_direction
                else:
                    if direction not in \
                         [new_direction, (-new_direction[0],-new_direction[1])]:
                        pass
#                        logger.debug('The case of a five-point vertex'+
#                                          'yielding a tadpole is not supported')            
                continue

            #add the code in the correct format
            self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,
             line.end.pos_x+0.01*direction[0], line.end.pos_y+0.01*direction[1],
                                   '%s Ffermionl' % (curvature*7))
        else:
            #add the code in the correct format
            self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,
                         line.end.pos_x+0.01, line.end.pos_y+0.01, '%s Ffermionl' %\
                         curvature)
    def draw_dashed(self, line):
        """ADD the EPS code for this Higgs line."""

        #add the code in the correct format
        self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,
                         line.end.pos_x, line.end.pos_y, 'Fhiggs')


    def draw_circled_dashed(self, line,cercle):
        """ADD the EPS code for this Higgs line."""
        if not cercle:
            curvature = 4
        else:
            curvature = 5

        is_tadpole = line.begin.pos_x==line.end.pos_x and \
                                                line.begin.pos_y==line.end.pos_y
                                                
        if is_tadpole:
            # Obtain the direction of the propagator supporting the tadpole
            direction = None
            for l in line.begin.lines:
                new_direction = (l.end.pos_x-l.begin.pos_x, l.end.pos_y-l.begin.pos_y)
                if new_direction==(0.0,0.0):
                    continue
                norm = math.sqrt(new_direction[0]**2+new_direction[1]**2)
                new_direction = (new_direction[0]/norm, new_direction[1]/norm)

                if not direction:
                    direction = new_direction
                else:
                    if direction not in \
                         [new_direction, (-new_direction[0],-new_direction[1])]:
#                        logger.error('The case of a five-point vertex'+
#                                          'yielding a tadpole is not supported')            
                        pass

            #add the code in the correct format
            x, y = self.rescale(line.begin.pos_x, line.begin.pos_y)
            self.text += '%s %s moveto'%(x, y)
            self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,
             line.end.pos_x+0.01*direction[0], line.end.pos_y+0.01*direction[1],
                                   '%s Fhiggsl' % (curvature*7))
        else:
            #add the code in the correct format
            self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,
                         line.end.pos_x+0.01, line.end.pos_y+0.01, '%s Fhiggsl'% curvature)

    def draw_dotted(self,line):
        """ADD the EPS code for the ghost line."""

        #add the code in the correct format
        self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,\
                                      line.end.pos_x, line.end.pos_y, 'Fghost')

    def draw_curved_dotted(self, line, cercle):
        """ADD the EPS code for the ghost line."""
        if not cercle:
            curvature = 0.4
        else:
            curvature = 1

        if (line.begin.pos_x, line.begin.pos_y) == self.curved_part_start:
            curvature *= -1	
        #add the code in the correct format
        self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,\
                        line.end.pos_x, line.end.pos_y, '%s Fghostl'% curvature)

    def draw_circled_dotted(self, line, cercle):
        """ADD the EPS code for the ghost line."""
        if not cercle:
            curvature = 4
        else:
            curvature = 5

        is_tadpole = line.begin.pos_x==line.end.pos_x and \
                                                line.begin.pos_y==line.end.pos_y
                                                
        if is_tadpole:
            # Obtain the direction of the propagator supporting the tadpole
            direction = None
            for l in line.begin.lines:
                new_direction = (l.end.pos_x-l.begin.pos_x, l.end.pos_y-l.begin.pos_y)
                if new_direction==(0.0,0.0):
                    continue
                norm = math.sqrt(new_direction[0]**2+new_direction[1]**2)
                new_direction = (new_direction[0]/norm, new_direction[1]/norm)

                if not direction:
                    direction = new_direction
                else:
                    if direction not in \
                         [new_direction, (-new_direction[0],-new_direction[1])]:
#                        logger.error('The case of a five-point vertex'+
#                                          'yielding a tadpole is not supported')            
                        pass           
            
            #add the code in the correct format
            self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,
             line.end.pos_x+0.01*direction[0], line.end.pos_y+0.01*direction[1],
                                   '%s Fghostl' % (curvature*7))
        else:
            #add the code in the correct format
            self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,\
              line.end.pos_x+0.01, line.end.pos_y+0.01, '%s Fghostl'% curvature)

    def draw_wavy(self, line, opt=0, type=''):
        """ADD the EPS code for this photon line."""

        #add the code in the correct format
        self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,
                         line.end.pos_x, line.end.pos_y, '%d Fphoton%s' % (opt,type))

    def draw_curved_wavy(self, line, cercle, opt=0, type=''):
        """ADD the EPS code for this photon line."""
        if not cercle:
            curvature = 0.4
        else:
            curvature = 1
        if (line.begin.pos_x, line.begin.pos_y) == self.curved_part_start:
            curvature *= -1
        #add the code in the correct format
        self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,
                         line.end.pos_x, line.end.pos_y, '%d %s Fphotonl%s' % (opt,curvature,type))

    def draw_circled_wavy(self, line, cercle, opt=0, type=''):
        """ADD the EPS code for this photon line."""

        if not cercle:
            curvature = 4
        else:
            curvature = 5
        
        is_tadpole = line.begin.pos_x==line.end.pos_x and \
                                                line.begin.pos_y==line.end.pos_y
                                                
        if is_tadpole:
            # Obtain the direction of the propagator supporting the tadpole
            direction = None
            for l in line.begin.lines:
                new_direction = (l.end.pos_x-l.begin.pos_x, l.end.pos_y-l.begin.pos_y)
                if new_direction==(0.0,0.0):
                    continue
                norm = math.sqrt(new_direction[0]**2+new_direction[1]**2)
                new_direction = (new_direction[0]/norm, new_direction[1]/norm)

                if not direction:
                    direction = new_direction
                else:
                    if direction not in \
                         [new_direction, (-new_direction[0],-new_direction[1])]:
#                        logger.error('The case of a five-point vertex'+
#                                          'yielding a tadpole is not supported')            
                        pass            
            
            #add the code in the correct format
            self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,
             line.end.pos_x+0.01*direction[0], line.end.pos_y+0.01*direction[1],
                                   '%d %s Fphotonl%s' % (opt,curvature*7,type))
        else:
            #add the code in the correct format
            self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,
                         line.end.pos_x+0.01, line.end.pos_y+0.01, 
                                      '%d %s Fphotonl%s' % (opt,curvature,type))

    def draw_curly(self, line, type=''):
        """ADD the EPS code for this gluon line."""

        # Due to the asymmetry in the way to draw the gluon (everything is draw
        #upper or below the line joining the points). We have to put conditions
        #in order to have nice diagram.
        if (line.begin.pos_x < line.end.pos_x) or \
                                (line.begin.pos_x == line.end.pos_x and \
                                line.begin.pos_y > line.end.pos_y):
            self.text += self.line_format(line.begin.pos_x,
                        line.begin.pos_y, line.end.pos_x,
                        line.end.pos_y, '0 Fgluon%s' % type)
        else:
            self.text += self.line_format(line.end.pos_x,
                        line.end.pos_y, line.begin.pos_x,
                        line.begin.pos_y, '0 Fgluon%s' % type)
            
    def draw_curved_curly(self, line, cercle, type=''):
        """ADD the EPS code for this gluon line."""

        dist = math.sqrt((line.begin.pos_x-line.end.pos_x)**2 + \
                                           (line.begin.pos_y-line.end.pos_y)**2)
        if not cercle or dist > 0.3:
            curvature = 0.4
        else:
            curvature = 1


        # Due to the asymmetry in the way to draw the gluon (everything is draw
        #upper or below the line joining the points). We have to put conditions
        #in order to have nice diagram.
        
        if (line.begin.pos_x, line.begin.pos_y) == self.curved_part_start:
            curvature *= -1

        self.text += self.line_format(line.end.pos_x,
                        line.end.pos_y, line.begin.pos_x,
                        line.begin.pos_y, '0 %s Fgluonl%s' % (-1*curvature, type))
        
                    
    
    def draw_scurly(self, line):
        """ADD the EPS code for this gluino line."""
        self.draw_curly(line, type='r'
                        )
        self.draw_straight(line)
        
    def draw_swavy(self, line):
        """ADD the EPS code for this neutralino line."""
        self.draw_wavy(line, type='r')
        self.draw_straight(line)  
             
    def draw_double(self, line, type='r'):
        """ADD the EPS code for this neutralino line."""
        
        
        length = math.sqrt((line.end.pos_y - line.begin.pos_y)**2 + (line.end.pos_x - line.begin.pos_x) **2)
        c1 = (line.end.pos_x - line.begin.pos_x)/length
        c2 = (line.end.pos_y - line.begin.pos_y)/length
        
        gap = 0.013
        start2_x = line.begin.pos_x + gap * c1  
        start2_y = line.begin.pos_y + gap * c2
        stop1_x = line.end.pos_x - gap * c1
        stop1_y = line.end.pos_y - gap * c2
        
        
        self.text += self.line_format(line.begin.pos_x, line.begin.pos_y,
                         stop1_x, stop1_y, '0 Fphoton%s' % (type))
        #add the code in the correct format
        self.text += self.line_format(start2_x, start2_y,
                         line.end.pos_x, line.end.pos_y, '0 Fphoton%s' % (type))
        
        
    def put_diagram_number(self, number=0):
        """ADD the comment 'diagram [number]' just below the diagram."""

        # Position of the text in [0,1] square
        x = 0.2
        y = -0.17
        # Compute the EPS coordinate
        x, y = self.rescale(x, y)
        #write the text
        self.text += ' %s  %s moveto \n' % (x, y)
        
        if hasattr(self, 'diagram_type'):
            self.text += '(%s diagram %s )   show\n' % (self.diagram_type, number + 1) # +1 python
                                                            #starts to count at
                                                            #zero.
        else:
            self.text += '(diagram %s )   show\n' % (number + 1) # +1 python
                                                            #starts to count at
                                                            #zero.
            
        mystr = " (%s)" % ", ".join(["%s=%d" % (key, self.diagram.diagram['orders'][key]) \
                      for key in sorted(self.diagram.diagram['orders'].keys()) \
                      if key != 'WEIGHTED'])
        
        x = 0.6
        y = -0.17
        x, y = self.rescale(x, y)
        #write the text
        self.text += ' %s  %s moveto \n' % (x, y)
        self.text += '%s   show\n' % (mystr)                                                             
        
        

    def associate_number(self, line, number):
        """Write in the EPS figure the MadGraph5_aMC@NLO number associate to the line.
        Note that this routine is called only for external particle."""

        # find the external vertex associate to the line
        if line.begin.is_external():
            vertex = line.begin
        else:
            vertex = line.end

        # find the position of this vertex    
        x = vertex.pos_x
        y = vertex.pos_y

        # Move slightly the position to avoid overlapping
        if x == 0:
            x = -0.04
        else:
            x += 0.04
            y = line._has_ordinate(x)

        # Re-scale x,y in order to pass in EPS coordinate
        x, y = self.rescale(x, y)
        # Write the EPS text associate
        self.text += ' %s  %s moveto \n' % (x, y)
        self.text += '(%s)   show\n' % (number)

    def associate_name(self, line, name, loop=False, reverse=False):
        """ADD the EPS code associate to the name of the particle. Place it near
        to the center of the line.
        """

        is_tadpole = line.begin.pos_x==line.end.pos_x and \
                                                line.begin.pos_y==line.end.pos_y
        if is_tadpole:
            # Obtain the direction of the propagator supporting the tadpole
            direction = None
            for l in line.begin.lines:
                new_direction = (l.end.pos_x-l.begin.pos_x, l.end.pos_y-l.begin.pos_y)
                if new_direction==(0.0,0.0):
                    continue
                norm = math.sqrt(new_direction[0]**2+new_direction[1]**2)
                new_direction = (new_direction[0]/norm, new_direction[1]/norm)

                if not direction:
                    direction = new_direction
                else:
                    if direction not in \
                         [new_direction, (-new_direction[0],-new_direction[1])]:
#                        logger.error('The case of a five-point vertex'+
#                                          'yielding a tadpole is not supported')            
                        pass           
            
            # Compute the orthogonal the
            orthogonal = (-direction[1],direction[0])

        # Put alias for vertex positions
        x1, y1 = line.begin.pos_x, line.begin.pos_y
        x2, y2 = line.end.pos_x, line.end.pos_y

        d = line.get_length()
        if is_tadpole:
            scale = 0.08
            dx, dy = scale*orthogonal[0], scale*orthogonal[1]
        # compute gap from middle point
        elif abs(x1 - x2) < 1e-3:
            dx = 0.015
            dy = -0.01
        elif abs(y1 - y2) < 1e-3:
            dx = -0.01
            dy = 0.025
        elif ((x1 < x2) == (y1 < y2)):
            dx = -0.03 * len(name)
            dy = 0.02 * len(name) #d * 0.12
        else:
            dx = 0.01 #0.05
            dy = 0.02 #d * 0.12 
        if loop:
            dx, dy = 1.5* dx, dy
            if x1 == x2:
                if y1 < y2:
                    dx, dy = -dx, -dy
            elif y1 == y2:
                if x1 >x2:
                    dx, dy = -dx, -dy
            elif x1 < x2:
                dx, dy = -dx, -dy
        if reverse:
            dx, dy = -dx, -dy

            
                
        # Assign position
        x_pos = (x1 + x2) / 2 + dx
        y_pos = (y1 + y2) / 2 + dy
        
        # Pass in EPS coordinate
        x_pos, y_pos = self.rescale(x_pos, y_pos)
        #write EPS code
        self.text += ' %s  %s moveto \n' % (x_pos, y_pos)
        self.text += '(' + name + ')   show\n'


#===============================================================================
# DrawDiagramsEps
#===============================================================================
class MultiEpsDiagramDrawer(EpsDiagramDrawer):
    """Class to write a EPS file containing the asked set of diagram
    This class follows the DrawDiagram Frameworks.
    
    The main routine to draw a diagram is 'draw' which call
    1) initialize: setup things for the diagram (usually open a file)
    2) convert_diagram : Update the diagram in the correct format if needed
    3) draw_diagram : Perform diagram dependent operation
    4) conclude : finish the operation.
    """

    # Define where to put the diagrams in the page. This is the coordinate of 
    #the lower left corner of the drawing area of the first graph. and the 
    #dimension associate to this drawing area.
    
    x_min = 75
    x_size = 200
    y_min = 560
    y_size = 150
    # Define distances between two drawing area
    x_gap = 75
    y_gap = 70

    #define font
    font=9
    
    #Defines the number of line-column in a EPS page
    nb_line = 3
    nb_col = 2
    
    blob_size = 1.5
    
    lower_scale = 5
    second_scale ={'x_min': 40, 'x_size':150,'y_min':620,'y_size':100,
                   'x_gap':42,'y_gap':30,'font':6,'nb_line':5,'nb_col':3,
                   'blob_size':0.9}
    
    def __init__(self, diagramlist=None, filename='diagram.eps', \
                  model=None, amplitude=None, legend='',diagram_type=''):
        """Define basic variable and store some global information
        all argument are optional
        diagramlist : are the list of object to draw. item should inherit 
                from either  base_objects.Diagram  or drawing_lib.FeynmanDiagram
        filename: filename of the file to write
        model: model associate to the diagram. In principle use only if diagram
            inherit from base_objects.Diagram
        amplitude: amplitude associate to the diagram. NOT USE for the moment.
            In future you could pass the amplitude associate to the object in 
            order to adjust fermion flow in case of Majorana fermion."""

        #use standard initialization but without any diagram
        super(MultiEpsDiagramDrawer, self).__init__(None, filename , model, \
                                                                      amplitude)
        self.legend = legend
        #additional information
        self.block_nb = 0  # keep track of the number of diagram already written
        self.curr_page = 0 # keep track of the page position
        self.block_in_page = 0 #ckeep track of the block in a page
        #compute the number of pages
        self.npage = 1
        self.diagram_type = diagram_type

        diagramlist = [d for d in diagramlist if not (isinstance(d, loop_objects.LoopUVCTDiagram) or \
                (isinstance(d, loop_objects.LoopDiagram) and d.get('type') < 0))]
        diagramlist = base_objects.DiagramList(diagramlist)

        limit = self.lower_scale * self.nb_col * self.nb_line
        if len(diagramlist) < limit:
            self.npage += (len(diagramlist)-1) // (self.nb_col * self.nb_line)
        else:
            add = (len(diagramlist) - limit -1) // \
                     (self.second_scale['nb_col'] * self.second_scale['nb_line'])
            self.npage += self.lower_scale + add
            
        if diagramlist:
            # diagramlist Argument should be a DiagramList object
            assert(isinstance(diagramlist, base_objects.DiagramList))
            self.diagramlist = diagramlist
        else:
            self.diagramlist = None            
            
    def rescale(self, x, y):
        """All coordinates belongs to [0,1]. So that in order to have a visible
        graph we need to re-scale the graph. This method distort the square in
        a oblong. Deformation are linear."""

        # Compute the current line and column
        block_pos = self.block_in_page 
        line_pos = block_pos // self.nb_col
        col_pos = block_pos % self.nb_col

        # Compute the coordinate of the drawing area associate to this line
        #and column.
        x_min = self.x_min + (self.x_size + self.x_gap) * col_pos
        x_max = self.x_min + self.x_gap * (col_pos) + self.x_size * \
                                                                (col_pos + 1)
        y_min = self.y_min - (self.y_size + self.y_gap) * line_pos
        y_max = self.y_min - self.y_gap * (line_pos) - self.y_size * \
                                                                (line_pos - 1)

        # Re-scale the coordinate in that box
        x = x_min + (x_max - x_min) * x
        y = y_min + (y_max - y_min) * y

        return x, y

    def draw_diagram(self, diagram):
        """Creates the representation in EPS format associate to a specific 
        diagram."""

        # Standard method
        super(MultiEpsDiagramDrawer, self).draw_diagram(diagram, self.block_nb)
        # But keep track how many diagrams are already drawn
        
        self.block_nb += 1
        self.block_in_page +=1


    def draw(self, diagramlist='', opt=None):
        """Creates the representation in EPS format associate to a specific 
        diagram. 'opt' keeps track of possible option of drawing. Those option
        are used if we need to convert diagram to Drawing Object.
        opt is an DrawOption object containing all the possible option on how
        draw a diagram."""

        if diagramlist == '':
            diagramlist = self.diagramlist

        # Initialize some variable before starting to draw the diagram
        # This creates the header-library of the output file
        self.initialize()
        self.text += '/Helvetica findfont %s scalefont setfont\n' % self.font
        self.text += ' 50         770  moveto\n'
        self.text += ' (%s) show\n' % self.legend
        self.text += ' 525         770  moveto\n'
        self.text += ' (page %s/%s) show\n' % (self.curr_page + 1, self.npage)
        self.text += ' 260         50  moveto\n'
        self.text += ' (Diagrams made by MadGraph5_aMC@NLO) show\n'       
        # Loop on all diagram
        for i,diagram in enumerate(diagramlist):
            # Check if they need to be convert in correct format
            diagram = self.convert_diagram(diagram, self.model, self.amplitude, opt)
            if diagram==None:
                continue
            # Write the code associate to this diagram
            self.draw_diagram(diagram)

            # Check if the current page is full or not
            if self.block_in_page % (self.nb_col * self.nb_line) == 0:
                #if full initialize a new page
                self.pass_to_next_page()

        #finish operation
        self.conclude()

    def pass_to_next_page(self):
        """Insert text in order to pass to next EPS page."""

        self.curr_page += 1
        self.block_in_page = 0
        if self.curr_page == self.lower_scale:
            for key, value in self.second_scale.items():
                setattr(self, key, value)
        
        
        self.text += 'showpage\n'
        self.text += '%%' + 'Page: %s %s \n' % (self.curr_page+1, self.curr_page+1)
        self.text += '%%PageBoundingBox:-20 -20 600 800\n'
        self.text += '%%PageFonts: Helvetica\n'
        self.text += '/Helvetica findfont %s scalefont setfont\n' % self.font
        self.text += ' 50         770  moveto\n'
        self.text += ' (%s) show\n' % self.legend
        self.text += ' 525         770  moveto\n'
        self.text += ' (page %s/%s) show\n' % (self.curr_page + 1, self.npage)
        self.text += ' 260         40  moveto\n'
        self.text += ' (Diagrams made by MadGraph5_aMC@NLO) show\n'
        
