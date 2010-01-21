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

import math
import os
import time

import madgraph.iolibs.drawing_lib as Draw
import madgraph.core.base_objects as base_objects

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'


class DrawDiagram(object):
    """In principle ALL routines representing diagram in ANY format SHOULD 
    derive from this class.
     
    This is a (nearly empty) frameworks to draw a diagram in any type format  

    This frameworks defines in particular 
        - function to convert the input diagram (create by the generation step)
            in the correct object. [convert_diagram]
        - main loop to draw a diagram in a line-by-line method
            [draw - draw_diagram - draw_line] 
        - name of routine (routine are empty) in order to fit with the framework
            [ associate_name - associate_number - draw_straight ]
        - some basic definition of routines
            [conclude - initialize]
    
    This framework is base on the idea that we can create the diagram line after
    line. Indeed all line object (FeynmanLine) contains the full information 
    needed to be drawed independently of the rest of the diagram. 
    
    In order to create a class with this framework you should start to write the
    draw_straight, draw_curly, ... method which are called by the framework.
    
    If you want to write a file, you can store his content in self.text variable
    the routine conclude will then automatically write the file.
    
    The main routine to draw a diagram is 'draw' which call
    1) initialize: setup things for the diagram (usually open a file).
    2) convert_diagram : Update the diagram in the correct format if needed.
    3) draw_diagram : Build the diagram line after line.
    4) conclude : finish the operation. 
    """
    
    class DrawDiagramError(Exception):
        """Standard error for error occuring to create output of a Diagram."""

    def __init__(self, diagram='', file='', model='', amplitude=''):
        """Define basic variables and store some global information.
        All argument are optional:
        diagram : is the object to draw. 'diagram' should inherit from either 
                base_objects.Diagram  or drawing_lib.FeynmanDiagram.
        file: filename of the file to write.
        model: model associate to the diagram. In principle use only if diagram
            inherit from base_objects.Diagram (for conversion).
        amplitude: amplitude associates to the diagram. NOT USE for the moment.
            In future you could pass the amplitude associate to the object in 
            order to adjust fermion flow in case of Majorana fermion."""
                
        # Check the parameter value
        #No need to test Diagram class, it will be tested before using it anyway
        if model and not isinstance(model, base_objects.Model):
            raise self.DrawDiagramError('No valid model provide to convert ' + \
                                        'diagram in appropriate format')    
        
        if file and not isinstance(file, basestring):
            raise self.DrawDiagramError('No valid model provide to convert ' + \
                                        'diagram in appropriate format')
        
        # A Test of the Amplitude should be added when this one will be 
        #use.
        
        # Store the parameter in the object variable
        self.diagram = diagram
        self.filename = file
        self.model = model         # use for automatic conversion of graph
        self.amplitude = amplitude # will be use for conversion of graph
        
        # Set variable for storing text        
        self.text = ''
        # Do we have to write a file? -> store in self.file
        if file:
            self.file = True # Note that this variable will be overwritten. THis 
                             #will be the object file. [initialize]
        else:
            self.file = False  

    def draw(self, **opt):
        """Main routine to draw a single diagram.
        opt is the option for the conversion of the base_objects.Diagram in one 
        of the Draw.Diagram object. This is the list of recognize options:
            external [True] : authorizes external particles to finish on 
                horizontal limit of the square
            horizontal [True]: if on true use FeynmanDiagramHorizontal to 
                convert the diagram. otherwise use FeynmanDiagram (Horizontal 
                forces S-channel to be horizontal)
            non_propagating [True] : removes the non propagating particles 
                present in the diagram."""
        
        # Check if we need to upgrade the diagram.
        self.convert_diagram(**opt)
        # Initialize some variable before starting to draw the diagram
        # This is just for frameworks capabilities (default: open file in 
        #write mode if a filename was provide.
        self.initialize()
        # Call the instruction to draw the diagram line by line.
        self.draw_diagram(self.diagram)
        # Finish the creation of the file/object (default: write object if a 
        #filename was provide).
        self.conclude()

    
    def convert_diagram(self, diagram='', model='', amplitude='', **opt):
        """If diagram is a basic diagram (inherit from base_objects.Diagram)
        convert him to a FeynmanDiagram one. 'opt' keeps track of possible 
        option of drawing. 'amplitude' is not use for the moment. But, later,
        if defined will authorize to adjust the fermion-flow of Majorana 
        particles. This is the list of recognize options:
            external [True] : authorizes external particles to finish on 
                horizontal limit of the square
            horizontal [True]: if on true use FeynmanDiagramHorizontal to 
                convert the diagram. otherwise use FeynmanDiagram (Horizontal 
                forces S-channel to be horizontal)
            non_propagating [True] : removes the non propagating particles 
                present in the diagram."""
 
        if diagram == '':
            diagram = self.diagram
        
        #if already a valid diagram. nothing to do
        if isinstance(diagram, Draw.FeynmanDiagram):
            return
        
        # assign default for model and check validity (if not default)
        if model == '':
            model = self.model
        elif not isinstance(model, base_objects.Model):
            raise self.DrawDiagramError('No valid model provide to convert ' + \
                                        'diagram in appropriate format')

        # Test on Amplitude should be enter here, when we will use this 
        #information


        # Put default values for options
        authorize_options = ['external','horizontal','non_propagating']
        for key in authorize_options:
            if key not in opt:
                opt[key] = True
        
        # Upgrade diagram to FeynmanDiagram or FeynmanDiagramHorizontal 
        #following option choice
        if opt['horizontal']:
            diagram = Draw.FeynmanDiagramHorizontal(diagram, model, \
                                                drawing_mode=opt['external'])
        else:
            diagram = Draw.FeynmanDiagram(diagram, model, \
                                              drawing_mode=opt['external'])
            
        # Find the position of all vertex and all line orientation
        diagram.main(contract=opt['non_propagating'])

        # Store-return information
        self.diagram = diagram
        return diagram
        
    def initialize(self):
        """Initialization of object-file before starting in order to draw the
        diagram correctly. By default, we just check if we are in writing mode.
        And open the output file if we are."""
        
        # self.file is set on True/False in __init__. This defines if a filename
        #was provide in the __init__ step. 
        if self.file:
            self.file=open(self.filename,'w')        


    def draw_diagram(self, diagram='', number=0):
        """Building the diagram Line after Line. 
        This is the key routine of 'draw'."""
        
        for line in self.diagram.lineList:
            self.draw_line(line)
            
        # Finalize information related to the graph. First, associate a diagram  
        #position to the diagram representation.
        self.put_diagram_number(number)
                                    
        # Then If a file exist write the text in it                 
        if self.file:
            self.file.writelines(self.text)
            self.text = ""
    
    def conclude(self):
        """Final operation of the draw method. By default, this end to write the 
        file (if this one exist)
        """
        
        # self.file is set on True/False in __init__. If it is on True
        #the Initialize method change it in a file object
        if self.file:
            self.file.writelines(self.text)
            self.file.close()
        return
    
    def draw_line(self, line):
        """Draw the line information.
        First, call the method associate the line type [draw_XXXXXX]
        Then finalize line representation by adding his name and, if it's an 
        external particle, the MadGraph number associate to it."""
        
        # Find the type line of the particle [straight, wavy, ...]
        line_type = line.get_info('line')
        # Call the routine associate to this type [self.draw_straight, ...]
        getattr(self, 'draw_' + line_type)(line)
        
        # Finalize the line representation with adding the name of the particle
        name = line.get_name()
        self.associate_name(line, name)
        # And associate the MadGraph Number if it is an external particle
        if line.is_external():
            number = line.get('number')
            self.associate_number(line, number)
    
    def draw_straight(self,line):
        """Example of routine for drawing the line 'line' in a specific format.
        straight is an example and can be replace by other type of line as 
        dashed, wavy, curly, ..."""
        
        raise self.DrawDiagramError, 'DrawDiagram.draw_straight should be ' + \
                'overwritten by Inherited Class' 
        
    def associate_name(self, line, name):
        """Method to associate a name to a the given line. 
        The default action of this framework doesn't do anything"""
        pass

    
    def associate_number(self,line, number):
        """Method to associate a number to 'line'. By default this method is 
        call only for external particles and the number is the MadGraph number 
        associate to the particle. The default routine doesn't do anything"""
        pass

    
class DrawDiagramEps(DrawDiagram):
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
           
    def initialize(self):
        """Operation done before starting to create diagram specific EPS content
        First open the file in write mode then write in it the header and the 
        library of particle type."""

        # Open file 
        super(DrawDiagramEps, self).initialize()
        
        # File Header
        text="%!PS-Adobe-2.0\n"
        text+="%%"+"BoundingBox: -20 -20 %s  %s \n" % (self.width, self.height)
        text+="%%DocumentFonts: Helvetica\n"
        text+="%%"+"Pages:  %s \n" % self.npage
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
        super(DrawDiagramEps,self).conclude()
    

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
        
    def draw_straight(self, line):
        """ADD the EPS code for this fermion line."""
        
        #add the code in the correct format
        self.text += self.line_format(line.start.pos_x, line.start.pos_y, 
                         line.end.pos_x, line.end.pos_y, 'Ffermion')
        
        
    def draw_dashed(self, line):
        """ADD the EPS code for this Higgs line."""
        
        #add the code in the correct format
        self.text += self.line_format(line.start.pos_x, line.start.pos_y, 
                         line.end.pos_x, line.end.pos_y, 'Fhiggs')
 
        
    def draw_wavy(self, line):
        """ADD the EPS code for this photon line."""

        #add the code in the correct format
        self.text += self.line_format(line.start.pos_x, line.start.pos_y, 
                         line.end.pos_x, line.end.pos_y, '0 Fphotond')


    def draw_curly(self, line):
        """ADD the EPS code for this gluon line."""
        
        # Due to the asymmetry in the way to draw the gluon (everything is draw
        #upper or below the line joining the points). We have to put conditions
        #in order to have nice diagram.
        if (line.start.pos_x < line.end.pos_x) or \
                                (line.start.pos_x == line.end.pos_x and \
                                line.start.pos_y > line.end.pos_y):
            self.text += self.line_format(line.start.pos_x,
                        line.start.pos_y, line.end.pos_x,
                        line.end.pos_y, '0 Fgluon')
        else:
            self.text += self.line_format(line.end.pos_x,
                        line.end.pos_y, line.start.pos_x,
                        line.start.pos_y, '0 Fgluon')
            
            
    def put_diagram_number(self, number=0):
        """ADD the comment 'diagram [number]' just below the diagram."""                       
        
        # Postion of the text in [0,1] square
        x = 0.42
        y = -0.17
        # Compute the EPS coordinate
        x, y = self.rescale(x, y)
        #write the text
        self.text += ' %s  %s moveto \n' % (x, y)  
        self.text += '( diagram %s )   show\n' % (number+1) # +1 because python
                                                            #start to count at
                                                            #zero.
          

    def associate_number(self,line,number):
        """Write in the EPS figure the MadGraph number associate to the line.
        Note that this routine is called only for external particle."""

        # find the external vertex associate to the line
        if line.start.is_external():
            vertex = line.start
        else:
            vertex = line.end
        
        # find the position of this vertex    
        x = vertex.pos_x
        y = vertex.pos_y
        
        # Move slightly the position to avoid overlapping
        if x == 0:
            x = -0.04
        elif x == 1:
            x = 1.04
        if y == 0:
            y = -0.06
        elif y == 1:
            y = 1.04

        # Re-scale x,y in order to pass in EPS coordinate
        x, y = self.rescale(x, y)
        # Write the EPS text associate
        self.text += ' %s  %s moveto \n' % (x, y)  
        self.text += '(%s)   show\n' % (number)                 
 
    def associate_name(self, line, name):
        """ADD the EPS code associate to the name of the particle. Place it near
        to the center of the line.

         The position of the name follows the V4 routine.
         """

        # Put alias for vertex positions
        x1, y1 = line.start.pos_x, line.start.pos_y
        x2, y2 = line.end.pos_x, line.end.pos_y

        d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if d == 0:
            raise self.DrawDiagramError('Line can not have 0 length')
        
        # Compute gap
        dx = (x1 - x2) / d
        dy = (y1 - y2) / d        
        
        # Correct sign to avoid intersection between the name and the current 
        #line
        if dy < 0:
            dx, dy = -1 * dx, -1 * dy
        elif dy == 0:
            dx = 1.5

        # Assign position
        x_pos = (x1 + x2) / 2 + 0.04 * dy
        y_pos = (y1 + y2) / 2 - 0.055 * dx      

        # Pass in EPS coordinate
        x_pos, y_pos = self.rescale(x_pos, y_pos)
        #write EPS code
        self.text += ' %s  %s moveto \n' % (x_pos, y_pos)  
        self.text += '(' + name + ')   show\n'

################################################################################
class DrawDiagramsEps(DrawDiagramEps):
    """Class to write a EPS file containing the asked set of diagram
    This class follows the DrawDiagram Frameworks.
    
    The main routine to draw a diagram is 'draw' which call
    1) initialize: setup things for the diagram (usually open a file)
    2) convert_diagram : Udate the diagram in the correct format if needed
    3) draw_diagram : Perform diagram dependant operation
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

    #Defines the number of line-column in a EPS page
    nb_line = 3
    nb_col = 2
    
    def __init__(self, diagramlist='', file='diagram.eps', \
                  model='', amplitude=''):
        """Define basic variable and store some global information
        all argument are optional
        diagram : is the object to draw. 'diagram' should inherit from either 
                base_objects.Diagram  or drawing_lib.FeynmanDiagram
        file: filename of the file to write
        model: model associate to the diagram. In principle use only if diagram
            inherit from base_objects.Diagram
        amplitude: amplitude associate to the diagram. NOT USE for the moment.
            In future you could pass the amplitude associate to the object in 
            order to adjust fermion flow in case of Majorana fermion."""
        
        #use standard initialization but without any diagram
        super(DrawDiagramsEps,self).__init__('', file , model, amplitude)
        
        #additional information
        self.block_nb = 0  # keep track of the number of diagram already written
        self.npage = 1 + len(diagramlist) // (self.nb_col * self.nb_line)
        self.diagramlist = diagramlist
        
    def rescale(self, x, y):
        """All coordinates belongs to [0,1]. So that in order to have a visible
        graph we need to re-scale the graph. This method distort the square in
        a oblong. Deformation are linear."""
        
        # Compute the current line and column
        block_pos = self.block_nb % (self.nb_col * self.nb_line)
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
        super(DrawDiagramsEps,self).draw_diagram(diagram, self.block_nb)
        # But keep track how many diagrams are already drawn
        self.block_nb += 1
        
        
    def draw(self,diagramlist='', **opt):
        """Creates the representation in EPS format associate to a specific 
        diagram. 'opt' keeps track of possible option of drawing. Those option
        are used if we need to convert diagram to Drawing Object.
        This is the list of recognize options:
            external [True] : authorizes external particles to finish on 
                horizontal limit of the square
            horizontal [True]: if on true use FeynmanDiagramHorizontal to 
                convert the diagram. otherwise use FeynmanDiagram (Horizontal 
                forces S-channel to be horizontal)
            non_propagating [True] : removes the non propagating particles 
                present in the diagram."""
        
        if diagramlist == '':
            diagramlist = self.diagramlist
        
        # Initialize some variable before starting to draw the diagram
        # This creates the header-library of the output file
        self.initialize()
        # Loop on all diagram
        for diagram in diagramlist:
            # Check if they need to be convert in correct format
            diagram = self.convert_diagram(diagram, self.model, **opt)
            # Write the code associate to this diagram
            self.draw_diagram(diagram)
            
            # Check if the current page is full or not
            if self.block_nb % (self.nb_col * self.nb_line) == 0:
                #if full initialize a new page
                self.pass_to_next_page()
        
        #finish operation
        self.conclude()
        
    def pass_to_next_page(self):
        """Insert text in order to pass to next EPS page."""
        
        self.text += 'showpage\n'
        new_page = 1 + self.block_nb // (self.nb_col * self.nb_line)
        self.text += '%%'+'Page: %s %s \n' % (new_page, new_page)
        self.text += '%%PageBoundingBox:-20 -20 600 800\n'
        self.text += '%%PageFonts: Helvetica\n'
        self.text += '/Helvetica findfont 10 scalefont setfont\n'
        self.text += ' 240         770  moveto\n'
        self.text += ' (Diagrams by MadGraph) show\n'
