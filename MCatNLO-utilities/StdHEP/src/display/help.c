/*******************************************************************************
*									       *
* help.c -- help text for Phase Display tool				       *
*									       *
* Copyright (c) 1993, 1994 Universities Research Association, Inc.	       *
* All rights reserved.							       *
* 									       *
* This material resulted from work developed under a Government Contract and   *
* is subject to the following license:  The Government retains a paid-up,      *
* nonexclusive, irrevocable worldwide license to reproduce, prepare derivative *
* works, perform publicly and display publicly by or for the Government,       *
* including the right to distribute to other Government contractors.  Neither  *
* the United States nor the United States Department of Energy, nor any of     *
* their employees, makes any warranty, express or implied, or assumes any      *
* legal liability or responsibility for the accuracy, completeness, or         *
* usefulness of any information, apparatus, product, or process disclosed, or  *
* represents that its use would not infringe privately owned rights.           *
*                                        				       *
* Fermilab Nirvana GUI Library						       *
* February,  1994							       *
*									       *
* Written by Paul Lebrun             					       *
*									       *
*******************************************************************************/
#include <Xm/Xm.h>
#include "util/help.h"
#include "help.h"

static void helpButtonDialog(Widget parent, helpMenuInfo *help);
static void helpButtonDialogSpace(Widget parent, helpMenuInfo *helpS);
static void helpButtonDialogPara(Widget parent, helpMenuInfo *helpP);

/*
** Help item titles, menu information, and text
*/

static helpMenuInfo

gettingStarted = {"Getting Started", 'G', True,  HELP_TEXT,
"\n StdHep Phase Display is a tool to select, display, and browse StdHep \
events, in order to understand the kinematics, topology and the decay chain.\
Such Events can easily be created based on the StdHep framework, which \
interfaces to most HEP generators, such JETSET, ISAJET, HERWIG  \n\n\
After invoking Phase Display , you'll want to do one of two  \
things:\n\n\
(--> Use scroll bar at right to see rest of this message.)\n\n\
     Open an Native StdHep file\n\
     Open a FZ (XL format) StdHep file\n\n\
Use the File pull-down menu to perform any of these functions.  \n\n\
After specifying the file you which to browse, you need to specify \
the event to view.  You can use the arrow buttons on the top of this \
panel to select the event you whish to see. Currently, you can only  \
browse through 100 events of this file (or less). Since StdHep files are \
sequential, reaching further downstream would probably be too slow. \
At the outset, the first event in the file is shown. \
By default, the length of the track is mapped to the norm of the 3-mom. \
vector of the particle. You can change that on the panel. \
The color is mapped to the particle identification. The Color Code is \
available in the Event pull down menu\n\n\
The next step is to relate the kinematics to the decay chain. Under the \
Event pull down menu, selected the Event Tree option. More help on \
node or track selection in those display is available in the Help pull down \
menu.", 0, 0},

mainP = {"Main Panel", 'P', True, HELP_TEXT, 
"\n The main panel of this window graphically describes the kinematics \
of the event. On the left, a few buttons to select which type of tracks \
are on display and to control the appearance of the plot. On the right, \
the drawing area, displaying the selected track. The length of a track \
can be mapped to the momentum, transverse momentum, rapidity or \
 pseudorapidity. At the beginning, it is mapped to the momentum, you \
can changed that by depressing (left mouse button) one of the radio buttons \
on  the upper right corner of the panel.  It would be not too difficult \
to relink this application with an other user-defined metric, that's \
why an de-sensitized button has been placed. \n\n\
   At the outset, all tracks are visible. This can be changed by clicking \
on the left-center of the panel. This should allow you to get rid of unwanted \
information. The selection is based either on particle stability or particle \
type. \n\n\
  The left-bottom part of the panel is devoted to handle the control of the \
viewing angle of the 3D display. The top part sets various rotation angles, \
the left-bottom part, zooming. The right-bottom button allows you to get \
back to the default viewing angle ( that is, a bird's eye view of the X-Y \
plane.\n\n\
  Known deficiency : No perspective is currently available.",0,0}, 
/*        1         2         3         4         5         6 */
/* 3456789012345678901234567890123456789012345678901234567890 */
  
file = {"File", 'F', False, HELP_TEXT,
"\nThe File pull-down menu allows you to get the input data and create \
new anciliary window showing the selected event. It also has the print \
and exit options. \n\n\
New Window:\n\
    Once you have an event on display, it might be \n\
    interesting to see it simultaneouly under different\n\
    angles, or mapped to an other kinematical variables.\n\
    These new windows do not allow you to see an other \n\
    eventnor open an other file. \n\
    If you whish to compare events across different files\n\
    you will have to start an other Phase Display process.\n\
Close: \n\
    Closing the last window will terminate the application \n\
Open Native StdHep file: \n\
    This type of file has been written using the HERWRT call\n\
    in the StdHep library. This application will scan up to\n\
    100 events, in order to check the self consistency of \n\
    the file.\n\
    If the file contains less 100 events, an non-fatal\n\
    error message from StdLib will be printed on stdout, and\n\
    the real number of events available for scanning  will \n\
    be printed in the top right corner of the main panel. \n\
    By default, the first event is selected, and read in.\n\
    The file is closed after each read. \n\
Open Xdr StdHep file: \n\
    This type of file must have been written through the \n\
    mcfio xdr interface, usually by calls to STDXWRT. \n\
    The procedure to scan, read or select events is the \n\
    same as for Native StdHep files. \n\
Open Zebra (XL) StdHep file:\n\
    This type of file must have been written through a \n\
    HEPZWRT call, where the file has been opened with the\n\
    XL option, which means that the file is in exchange\n\
    mode and has been written using c-interface mode.\n\
    The procedure to scan, read or select events is the \n\
    same as for Native StdHep files. \n\
Print: \n\
     To print the main panel drawing area to a PostScript\n\
     printer.  Phase Display will pop up a Printer Options\n\
     window for specifying the printer queue name and the\n\
     number of copies to print.  The last field in this\n\
     dialog is the Unix (or VMS) command that will be \n\
     used to queue the plot for printing.  Since this \n\
     command is generated from the other options in the \n\
     dialog, there is normally no reason to change it, but\n\
     you can use the field to add options and change the \n\
     command in ways not otherwise supported by the print\n\
     dialog.", 0,0 },
/*        1         2         3         4         5         6 */
/* 3456789012345678901234567890123456789012345678901234567890 */
event = {"Event", 'E', False, HELP_TEXT,
"\nThe Event pull-down menu allows you to understand the topology of the\
event trough color mapping, 3D manipulation and the Tree Display\n\n\
Show Rotation Angle:\n\
     You can rotate the Event by direct manipulation, \n\
     (click/drag on the graph, using the left button), or by\n\
     using the push button on the main panel, or, if you\n\
     like accuracy and remember what the Euler angles are,\n\
     you can use the Rotation angle panel \n\
     to set these angles. \n\n\
Show Event Tree:\n\
     The StdHep data structure is hierarchical, e.g., the \n\
     parent to daughter relationship is kept in the \n\
     generation/decay process. Such a genealogical tree can\n\
     be display. In order to simplify the picture, \n\
     a standard tree structure is used, e.g., a particle \n\
     has one and only  parent ( a ficticious node being \n\
     placed on the top). \n\n\
     This option display such a tree, where a node\n\
     represented a single particle, or collection of\n\
     them. The algorithm goes as follow: if  particle(s)\n\
     are stable and have no daughter (leave node), and if\n\
     other particles are sharing the same id and have \n\
     the same mother, they are lumepd together in a single \n\
     node. This allows to simply high multiplicity trees. \n\n\
     The Tree display is made of buttons, pressing them \n\
     ( left buttons)  will give more information about the\n\
     given node. Also, the tracks inluded in that node will \n\
     be highlighted on the main panel \n\n\
     Known deficiency : The Tree widget ( or Motif) has a\n\
     bug in the DestroyWidget method, which can (and usually\n\
     will) induce a crash of this application upon removal \n\
     of the a Tree Display if the number of node exceeds a \n\
     few hundreds. This bug is currently under study. \n\
     In any event, exploring a ~500-node tree can be time \n\
     consuming and  not very useful, one can suggest various\n\
     algorithm to simplify a tree, for instance, collapse \n\
     all particles to the top, if they do not have an\n\
     interesting ancestry ( e.g., they don't come from \n\
     a top or bottom quark.)\n\n\
Show Eta-Pt Display:\n\
     Once you have an event on display, it might be \n\
     interesting to see it simultaneouly in a Phase Space \n\
     appropriate to Hadronic physics: rapidity an Pt Space.\n\
     This new window does not allow you \n\
     to see an other event nor open an other file. \n\
     If you whish to compare events across different files\n\
     you will have to start an other Phase or Space Display.\n\n\
Show Color Code: \n\
     A fictious Tree display can be used to understand the\n\
     crummy color chosen in this application.  Currently, \n\
     this coding is fixed. If you have a better idea and \n\
     are willing to do the coding, let's talk !",0,0,},
/*        1         2         3         4         5         6 */
/* 3456789012345678901234567890123456789012345678901234567890 */

preferences = {"Preferences", 'P', True, HELP_TEXT,
"\nThe preferences pull-down menu allows you set/unset various options \
affecting the PhaseDisplay session. \n\n\
Show Axes:\n\
     By default, the axis in the Phase space are shown. The\n\
     scale is identical for all three axis, and is \n\
     represented in the main panel.  To make the event\n\
     more visible, it might be occasionaly useful not to\n\
     draw these axis.  This option lets you do this. \n\n\
Buffer Graphics:\n\
     Graphics buffering changes the way Histo-Scope draws\n\
     moving graphics in X-Windows.  Depending on your X-Win-\n\
     dows server or X-Terminal, turning on buffered graphics\n\
     may either improve or degrade animation quality - it\n\
     eliminates flicker at the expense of drawing speed and\n\
     X-server memory.\n\
     Initially Buffer Graphics mode is off.\n\n\
Percent Vector Shown: \n\
     In order to avoid cluttering the center of the graph,\n\
     and make more visible the small momentum tracks, the\n\
     tracks are represented by segments pointing to the \n\
     center:  a certain percentage of the track has been\n\
     erased starting from the vertex.\n\
     You can change this parameter. \n\n\
Rotation Increment: \n\
     You can rotate the Event by direct manipulation, \n\
     (click/drag on the graph, using the left button), or,\n\
     by using the push button on the main panel. This\n\
     increment the corresponding rotation angle, you can\n\
     easily change the increment size using this option.\n\n\
Track Highlights: \n\
     As you click on a given node of the Event Tree \n\
     Display, tracks related to that particular node are \n\
     highlighted. This relation ship goes as folow: \n\
        - Node : All tracks belonging to the selected node \n\
                 are highlighted. This is the default.\n\
        - Daughters : All daughter tracks are highlighted\n\
        - Descendants : All descendants are highlighted\n\
        - Mother : The mother track of the selected node \n\
        	  is highlighted. \n\
        - Ancestors : Up to the tree top, all ancestors of \n\
        	  the selected node are highlighted. ",0,0},
     
trackSel = {"Track & Node Selection",'S', False, HELP_TEXT,
"\n Further information about tracks or nodes in the tree is available. \
In fact, all the information in the HEPEVT COMMON block, written in the \
StdHep files, can be gotten from interacting with this display. \
(Except the position of the vertices in real space, which will be \
available in the Spatial Display, soon to be released.\n\n\
You can pick a selected track on the drawing area by pointing the cursor to \
it and depressing the central button ( the left button controls the motion \
of the display). The selected track is highlighted on the display, \
and a small text box should appear if the pointing is succesful, \
indicating the particle identity, it's place in the HEPEVT block, and various \
kinematical quantities. If the Event Tree is also displayed, the node to \
which the tracks belongs to is highlighted, indicating the position of the \
selected particle in the hierarchy.\n\n\
You can also get more information about a given node in the tree, by pressing \
(left mouse button) on a tree element. The corresponding tracks are \
 highlighted on the event display.\n\n\
 If Greek fonts are available on yourplatform ( usualy refered as a symbol \
 font, Phase Display will display the name in greek, if applicable. \
 This is done by adding a few line to your .Xdefaults file, specifying a \
 fontlist. The name of this application is phase. \
 For instance, on the SGI, you can include:\n\n\
phase*fontList: \n\
 -adobe-helvetica-medium-r-normal--12-120-75-75-m-70\n\
 -iso8859-1, \n\
 -adobe-symbol-medium-r-normal--14-100-100-100-p-85\n\
 -adobe-fontspecific=greek\n\n\
 Also, the color of the background of the Phase Display using the spin widget \
 can be changed :\n\n\
phase*Spin.background:black\n\n\
Known deficiency : the highlight may no work properly upon resizing the \
main panel. If so, resize the window to readjust graphical context ",0,0};

/*
** Menu lists
**
** Some of the menu list arrays have the number of items specified because
** the IBM C compiler complains: "Number of initializers cannot be greater
** than the number of aggregate members"
*/
helpMenuInfo

*MainMenuHelp[] = {&gettingStarted,&mainP, &file, &event, &preferences, 
        &trackSel, NULL};
        
/*
**
** Similar prose for the Space Display
**
*/

static helpMenuInfo

gettingStartedSpace = {"Getting Started", 'G', True,  HELP_TEXT,
"\n StdHep Space Display is a tool to select, display, and browse StdHep \
tracks and vertices, in order to understand the kinematics, spatial \
relationships among these vertices \
and the decay chain.\
Such Events can easily be created based on the StdHep framework, which \
interfaces to most HEP generators, such JETSET, ISAJET, HERWIG  \n\n\
After invoking Space Display , you'll want to do one of two  \
things:\n\n\
(--> Use scroll bar at right to see rest of this message.)\n\n\
     Open an Native StdHep file\n\
     Open a FZ (XL format) StdHep file\n\n\
Use the File pull-down menu to perform any of these functions.  \n\n\
After specifying the file you which to browse, you need to specify \
the event to view.  You can use the arrow buttons on the top of this \
panel to select the event you whish to see. Currently, you can only  \
browse through 100 events of this file (or less). Since StdHep files are \
sequential, reaching further downstream would probably be too slow. \
At the outset, the first event in the file is shown. \
The length of the track is mapped to the norm of the 3-momentum vector, \
sliders can help \
you to manipulate the event in 3-D space. Note that there is a transverse \
magnification factor ( most, if not all, event generators have the beam axis \
along the Z,  e.g. longitudinal direction.) \
The color is mapped to the particle identification. The Color Code is \
available in the Event pull down menu\n\n\
The next step is to relate the kinematics to the decay chain. Under the \
Event pull down menu, selected the Event Tree option. More help on \
node or track selection in those display is available in the Help pull down \
menu.  The Phase space display is also available from this application.",
 0, 0},

mainSpace = {"Main Panel", 'P', True, HELP_TEXT, 
"\n The main panel of this window graphically describes the kinematics in real \
space. On the left, a few buttons to select which type of tracks \
are on display and to control the appearance of the plot. On the right, \
the drawing area, displaying the selected track. The length of a track \
is always mapped to the size of the 3-momentum. \n\n\
   At the outset, all tracks are visible. This can be changed by clicking \
on the left-center of the panel. This should allow you to get rid of unwanted \
information. The selection is based either on particle stability or particle \
type. \n\n\
  Six sliders lets you control the positions of the tracks and vertices. The \
first one controls the over scale of the problem: B to D to strangeness decays \
do covers \
easily over 5 order of magnitudes, a logarythmic sliders allows to set the \
range (in cm.) covered by the display window. The next \
three sliders controls the translation ( or, effective panning) of the entire \
event. The next one controls the conversion factor from momentum to real \
space.  This can be used to easily match a mother particle to it's decay \
location. The last one controls the longitudinal to transverse ratio. \n\n\
  The left-bottom part of the panel is devoted to handle the control of the \
viewing angle of the 3D display. The top part sets various rotation angles, \
the left-bottom part, zooming. The right-bottom button allows you to get \
back to the default viewing angle ( that is, a bird's eye view of the X-Y \
plane.\n\n\
  Known deficiency : No perspective is currently available.",0,0}, 
/*        1         2         3         4         5         6 */
/* 3456789012345678901234567890123456789012345678901234567890 */
  
fileSpace = {"File", 'F', False, HELP_TEXT,
"\nThe File pull-down menu allows you to get the input data and create \
new anciliary window showing the selected event. It also has the print \
and exit options. \n\n\
Close: \n\
    Closing the last window will terminate the application \n\
Open Native StdHep file: \n\
    This type of file has been written using the HERWRT call\n\
    in the StdHep library. This application will scan up to\n\
    100 events, in order to check the self consistency of \n\
    the file.\n\
    If the file contains less 100 events, an non-fatal\n\
    error message from StdLib will be printed on stdout, and\n\
    the real number of events available for scanning  will \n\
    be printed in the top right corner of the main panel. \n\
    By default, the first event is selected, and read in.\n\
    The file is closed after each read. \n\
Open Zebra (XL) StdHep file:\n\
    This type of file must have been written through a \n\
    HEPZWRT call, where the file has been opened with the\n\
    XL option, which means that the file is in exchange\n\
    mode and has been written using c-interface mode.\n\
    The procedure to scan, read or select events is the \n\
    same as for Native StdHep files. \n\
Print: \n\
     To print the main panel drawing area to a PostScript\n\
     printer.  Phase Display will pop up a Printer Options\n\
     window for specifying the printer queue name and the\n\
     number of copies to print.  The last field in this\n\
     dialog is the Unix (or VMS) command that will be \n\
     used to queue the plot for printing.  Since this \n\
     command is generated from the other options in the \n\
     dialog, there is normally no reason to change it, but\n\
     you can use the field to add options and change the \n\
     command in ways not otherwise supported by the print\n\
     dialog.", 0,0 },
/*        1         2         3         4         5         6 */
/* 3456789012345678901234567890123456789012345678901234567890 */
eventSpace = {"Event", 'E', False, HELP_TEXT,
"\nThe Event pull-down menu allows you to understand the topology of the\
event trough color mapping, 3D manipulation and the Tree Display.\n\n\
Show Rotation Angle:\n\
     You can rotate the Event by direct manipulation, \n\
     (click/drag on the graph, using the left button), or by\n\
     using the push button on the main panel, or, if you\n\
     like accuracy and remember what the Euler angles are,\n\
     you can use the Rotation angle panel \n\
     to set these angles. \n\n\
Show Eta-Pt Display:\n\
    Once you have an event on display, it might be \n\
    interesting to see it simultaneouly in a Phase Space\n\
    appropriate to Hadronic physics: rapidity an Pt Space.\n\
    This new window does not allow you \n\
    to see an other event nor open an other file. \n\
    If you whish to compare events across different files\n\
    you will have to start an other Phase or Space \n\
    Display process. \n\n\
Show Phase Display:\n\
    Once you have an event on display, it might be \n\
    interesting to see it simultaneouly in real space \n\
    in Phase Space. This new window has the same \n\
    constraints as the Eta-Pt one: no new events \n\n\
Show Event Tree:\n\
     The StdHep data structure is hierarchical, e.g., the \n\
     parent to daughter relationship is kept in the \n\
     generation/decay process. Such a genealogical tree can\n\
     be display. In order to simplify the picture, \n\
     a standard tree structure is used, e.g., a particle \n\
     has one and only  parent ( a ficticious node being \n\
     placed on the top). \n\n\
     This option display such a tree, where a node\n\
     represented a single particle, or collection of\n\
     them. The algorithm goes as follow: if  particle(s)\n\
     are stable and have no daughter (leave node), and if\n\
     other particles are sharing the same id and have \n\
     the same mother, they are lumepd together in a single \n\
     node. This allows to simply high multiplicity trees. \n\n\
     The Tree display is made of buttons, pressing them \n\
     ( left buttons)  will give more information about the\n\
     given node. Also, the tracks inluded in that node will \n\
     be highlighted on the main panel \n\n\
     Known deficiency : The Tree widget ( or Motif) has a\n\
     bug in the DestroyWidget method, which can (and usually\n\
     will) induce a crash of this application upon removal \n\
     of the a Tree Display if the number of node exceeds a \n\
     few hundreds. This bug is currently under study. \n\
     In any event, exploring a ~500-node tree can be time \n\
     consuming and  not very useful, one can suggest various\n\
     algorithm to simplify a tree, for instance, collapse \n\
     all particles to the top, if they do not have an\n\
     interesting ancestry ( e.g., they don't come from \n\
     a top or bottom quark. \n\n\
Show Detector Sketch:\n\
     A cartoon detector can be drawn on this display, to \n\
     merely indicate the scale and related track directions \n\
     and geometrical acceptances. You'll have to create a \n\
     simple ASCII file, each line containing the \n\
     coordinates of a line segment : \n\
           (x1, y1, z1, x2, y2, z2), expressed in cm.\n\
     Comments line are allowed. Free format ( %f) is used\n\
     An example file sketch1.det is available with the \n\
     product. Once you activate this option, you will be\n\
     prompted to open such a file, and the display will be\n\
     refreshed with these additional line segments,\n\
     all of them black. ( Sorry, no color!). \n\
     To remove this sketch, ask for this option again, \n\
     and cancel the file selection. \n\n\
Show Color Code: \n\
     A fictious Tree display can be used to understand the\n\
     crummy color chosen in this application.  Currently, \n\
     this coding is fixed. If you have a better idea and \n\
     are willing to do the coding, let's talk! ", 0, 0},     
/*        1         2         3         4         5         6 */
/* 3456789012345678901234567890123456789012345678901234567890 */

preferencesSpace = {"Preferences", 'P', True, HELP_TEXT,
"\nThe preferences pull-down menu allows you set/unset various options \
affecting the Space Display session. \n\n\
Show Axes:\n\
     By default, the axis in real space are shown. The\n\
     scale is identical for all three axis, and is \n\
     represented in the main panel.  To make the event\n\
     more visible, it might be occasionaly useful not to\n\
     draw these axis.  This option lets you do this. \n\n\
Show Vertices:\n\
     By default, the vertices are represented by a \n\
     little \"pin-balls made\" of short straight segments.\n\
     This representation helps you locate the vertices,\n\
     but in some case can be confusing in presence of \n\
     many tracks of small momentum. You can toggle this\n\
     button and the vertices will disappear.\n\
Buffer Graphics:\n\
     Graphics buffering changes the way Histo-Scope draws\n\
     moving graphics in X-Windows.  Depending on your X-Win-\n\
     dows server or X-Terminal, turning on buffered graphics\n\
     may either improve or degrade animation quality - it\n\
     eliminates flicker at the expense of drawing speed and\n\
     X-server memory.\n\
     Initially Buffer Graphics mode is off.\n\n\
Rotation Increment: \n\
     You can rotate the Event by direct manipulation, \n\
     (click/drag on the graph, using the left button), or,\n\
     by using the push button on the main panel. This\n\
     increment the corresponding rotation angle, you can\n\
     easily change the increment size using this option. \n\
Track Highlights: \n\
     As you click on a given node of the Event Tree \n\
     Display, tracks related to that particular node are \n\
     highlighted. This relation ship goes as folow: \n\
        - Node : All tracks belonging to the selected node \n\
                 are highlighted. This is the default.\n\
        - Daughters : All daughter tracks are highlighted\n\
        - Descendants : All descendants are highlighted\n\
        - Mother : The mother track of the selected node \n\
        	  is highlighted. \n\
        - Ancestors : Up to the tree top, all ancestors of \n\
        	  the selected node are highlighted. ",0,0},
     
trackSelSpace = {"Track & Node Selection",'S', False, HELP_TEXT,
"\n Further information about tracks or nodes in the tree is available. \
In fact, all the information in the HEPEVT COMMON block, written in the \
StdHep files, can be gotten from interacting with this display.\n\n\
You can pick a selected track on the drawing area by pointing the cursor to \
the middle of the track segment and depressing the central button \
( the left button controls the motion \
of the display). The selected track ( or a portion of it )\
 is highlighted on the display, \
and a small text box should appear if the pointing is succesful, \
indicating the particle identity, it's place in the HEPEVT block, and various \
kinematical quantities. If the Event Tree is also displayed, the node to \
which the tracks belongs to is highlighted, indicating the position of the \
selected particle in the hierarchy.\n\n\
You can also get more information about a given node in the tree, by pressing \
(left mouse button) on a tree element. The corresponding tracks are \
 highlighted on the event display.\n\n\
 If Greek fonts are available on yourplatform ( usualy refered as a symbol \
 font, Phase Display will display the name in greek, if applicable. \
 This is done by adding a few line to your .Xdefaults file, specifying a \
 fontlist. The name of this application is phase. \
 For instance, on the SGI, you can include:\n\n\
phase*fontList: \n\
 -adobe-helvetica-medium-r-normal--12-120-75-75-m-70\n\
 -iso8859-1, \n\
 -adobe-symbol-medium-r-normal--14-100-100-100-p-85\n\
 -adobe-fontspecific=greek\n\n\
 Also, the color of the background of the Phase Display using the spin widget \
 can be changed :\n\n\
phase*Spin.background:black\n\n\
Known deficiency : the highlight may no work properly upon resizing the \
main panel. If so, resize the window to readjust graphical context ",0,0};

helpMenuInfo

*MainMenuHelpSpace[] = {&gettingStartedSpace, &mainSpace, &fileSpace,
                         &eventSpace, &preferencesSpace, &trackSelSpace, NULL};
/*
**
** Short prose for the Para Display
**
*/

static helpMenuInfo
                         
gettingStartedPara = {"Getting Started", 'G', True,  HELP_TEXT,
"\n The Eta-Pt ( e.g. rapidity - transverse plane ) 3-D plot \
of the StdHep Display provides an effective way to represent the parton  \
collision.  It is assumed that the user is familiar with other StDhep \
windows, this help menu will not repeat all the details available in the \
Phase Space display or the real space display exhibiting detached vertices.  \
More details on this particular representation is available in the next \
menu. Thanks to A. Para for suggesting this window.", 0, 0},

mainPPara = {"Main Panel", 'P', True, HELP_TEXT, 
"\n The main panel of this window graphically describes the kinematics \
of the event, in the context of the naive parton model. Along the \
Z (beam) axis, the rapidity, on the X-Y plane, the transverse momentum. \
On the left, a few buttons to select which type of tracks \
are on display and to control the appearance of the plot. On the right, \
the drawing area, displaying the selected track. The length of a track \
is always mapped to its transvere momentum, in GeV/c. The origin always \
intrersects the Z ( rapidity or beam axis) and all tracks are perpendicular \
to the Z axis. \n\n\
  At the outset, all tracks are visible. This can be changed by clicking \
on the left-center of the panel. This should allow you to get rid of unwanted \
information. The selection is based either on particle stability or particle \
type. \n\n\
  In the middle of the left-hand side, you will see two sliders, controlling \
 (i) A translation along the rapidity axis Z and \
(ii) the aspect ratio longitudinal vs transverse ( in other words, setting an \
arbitrary conversion factor from eta to Gev to match units). \n\n\
  The left-bottom part of the panel is devoted to handle the control of the \
viewing angle of the 3D display. The top part sets various rotation angles, \
the left-bottom part, zooming. The right-bottom button allows you to get \
back to the default viewing angle ( that is, a bird's eye view of the X-Y \
plane.\n\n\
  Known deficiency : No perspective is currently available.",0,0}; 

helpMenuInfo

*MainMenuHelpPara[] = {&gettingStartedPara, &mainPPara, NULL};
  
