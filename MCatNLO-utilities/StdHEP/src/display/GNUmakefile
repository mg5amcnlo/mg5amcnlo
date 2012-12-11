#
# Platform independent part of make procedure for Nirvana phase space 
# event display, included by machine specific makefiles.
#
include ../stdhep_arch

NLIBDIR = ../../lib
NBINDIR = ../../bin

ifdef CRNLIB
   EXECS = $(NBINDIR)/Phase$(SFX) $(NBINDIR)/Space$(SFX)
endif


OBJS  =	phase$(SFX).o panel$(SFX).o menu$(SFX).o drawEvent$(SFX).o pick$(SFX).o pixmaps$(SFX).o settings$(SFX).o \
	rotation$(SFX).o hepnmg$(SFX).o stdHepFortranUtil$(SFX).o \
	Tree$(SFX).o dispTree$(SFX).o listNodeHep$(SFX).o stdHepFiles$(SFX).o help$(SFX).o \
	panelSpace$(SFX).o space$(SFX).o panelPara$(SFX).o Space_main$(SFX).o
SRCS  =	phase.c panel.c menu.c drawEvent.c pick.c pixmaps.c settings.c \
	rotation.c help.c hepnmg.f stdHepFortranUtil.f \
	Tree.c dispTree.c listNodeHep.c stdHepFiles.c \
	panelSpace.c space.c panelPara.c Space_main.c
INCLS =	drawEvent.h panel.h phaseP.h pixmaps.h settings.h menu.h phase.h \
	pick.h rotation.h stdHepFiles.h tree.h treeP.h dispTree.h \
	listNodeHep.h help.h \
	panelSpace.h space.h panelPara.h

INCS = -I../inc

all: $(NLIBDIR)/libPhaseStdHep$(SFX).a $(EXECS)

$(NLIBDIR)/libPhaseStdHep$(SFX).a: $(OBJS)
	$(AR) -urs $(NLIBDIR)/libPhaseStdHep$(SFX).a $(OBJS)

$(NBINDIR)/Phase$(SFX): Phase_main$(SFX).o TestParticles$(SFX).o $(NLIBDIR)/libPhaseStdHep$(SFX).a
	$(CC) $(CFLAGS)  Phase_main$(SFX).o  TestParticles$(SFX).o \
	      $(NLIBDIR)/libPhaseStdHep$(SFX).a  \
	      $(NLIBDIR)/libSpin$(SFX).a $(NLIBDIR)/libNUtil$(SFX).a \
	      ../../lib/libstdhep$(SFX).a \
	      ../../lib/libstdhepC$(SFX).a \
	      ../../lib/libFmcfio$(SFX).a \
	      $(CRNLIB)/libpacklib.a \
	      $(LIBS) -o $@
      
$(NBINDIR)/Space$(SFX): Space_main$(SFX).o $(NLIBDIR)/libPhaseStdHep$(SFX).a
	$(CC) $(CFLAGS) Space_main$(SFX).o \
	      $(NLIBDIR)/libPhaseStdHep$(SFX).a \
	      $(NLIBDIR)/libSpin$(SFX).a $(NLIBDIR)/libNUtil$(SFX).a \
	      ../../lib/libstdhep$(SFX).a \
	      ../../lib/libstdhepC$(SFX).a \
	      ../../lib/libFmcfio$(SFX).a \
	      $(CRNLIB)/libpacklib.a \
	      $(LIBS) -o $@

help$(SFX).o: help.c
	$(CC) $(CFLAGS) $(CXFLAGS) $(BIGGER_STRINGS) -c help.c -o $@

install:
	-mv libPhaseStdHep$(SFX).a $(NLIBDIR)/.
	-mv Phase$(SFX) $(NBINDIR)/.
	-mv Space$(SFX) $(NBINDIR)/.

clean:
	rm -f *$(SFX).o
	
realclean:
	rm -f *.o
	rm -f $(NBINDIR)/Phase* $(NBINDIR)/Space* $(NLIBDIR)/libPhaseStdHep*.a
	
depend:
	$(MAKEDEPEND) -fGNUmakefile -- $(CFLAGS) $(CXFLAGS) -- $(SRCS) Phase.c \
		     space.c TestParticles.c

#------------------------------------------

.SUFFIXES : 		# cancel implicit rules
.SUFFIXES : .o .c .F .f

%.o: %.F
	$(F77) $(INCS) $(FFLAGS) -DUNIX -c $<

%_g.o: %.F
	$(F77) $(INCS) $(FFLAGS) -DUNIX -c -o $@ $<

%.o: %.f
	$(F77) $(INCS) $(FFLAGS) -c $*.f

%_g.o: %.f
	$(F77) $(INCS) $(FFLAGS) -c -o $@ $*.f

%.o: %.c
	$(CC) $(INCS) $(CFLAGS) $(CXFLAGS) -c $<

%_g.o: %.c
	$(CC) $(INCS) $(CFLAGS) $(CXFLAGS) -c -o $@ $<

# DO NOT DELETE THIS LINE -- make depend depends on it.

phase.o: spin/Spin.h
phase.o: phase.h
phase.o: space.h
phase.o: phaseP.h
phase.o: panel.h
phase.o: pixmaps.h
phase.o: pick.h
phase.o: drawEvent.h
phase.o: dispTree.h
panel.o: spin/Spin.h
panel.o: util/DialogF.h
panel.o: phase.h
panel.o: space.h
panel.o: phaseP.h
panel.o: panel.h
panel.o: menu.h
panel.o: pick.h
panel.o: rotation.h
menu.o: phase.h
menu.o: space.h
menu.o: phaseP.h
menu.o: menu.h
menu.o: panel.h
menu.o: settings.h
menu.o: rotation.h
menu.o: stdHepFiles.h
menu.o: pick.h
menu.o: dispTree.h
menu.o: util/printUtils.h
menu.o: util/getfiles.h
menu.o: util/DialogF.h
menu.o: util/misc.h
menu.o: spin/Spin.h
menu.o: util/help.h
menu.o: help.h
drawEvent.o: spin/Spin.h
drawEvent.o: util/stringUtils.h
drawEvent.o: phase.h
drawEvent.o: space.h
drawEvent.o: phaseP.h
drawEvent.o: drawEvent.h
pick.o: spin/Spin.h
pick.o: util/stringUtils.h
pick.o: phase.h
pick.o: space.h
pick.o: phaseP.h
pick.o: pick.h
pick.o: drawEvent.h
pick.o: dispTree.h
pixmaps.o: pixmaps.h
settings.o: phase.h
settings.o: space.h
settings.o: phaseP.h
rotation.o: phase.h
rotation.o: space.h
rotation.o: phaseP.h
help.o: util/help.h
help.o: help.h
Tree.o: tree.h
Tree.o: treeP.h
dispTree.o: tree.h
dispTree.o: phase.h
dispTree.o: space.h
dispTree.o: phaseP.h
dispTree.o: dispTree.h
dispTree.o: pick.h
dispTree.o: spin/Spin.h
dispTree.o: drawEvent.h
dispTree.o: listNodeHep.h
dispTree.o: util/stringUtils.h
listNodeHep.o: phase.h
listNodeHep.o: space.h
listNodeHep.o: phaseP.h
listNodeHep.o: dispTree.h
listNodeHep.o: spin/Spin.h
listNodeHep.o: drawEvent.h
listNodeHep.o: listNodeHep.h
stdHepFiles.o: phase.h
stdHepFiles.o: space.h
stdHepFiles.o: phaseP.h
stdHepFiles.o: spin/Spin.h
stdHepFiles.o: drawEvent.h
panelSpace.o: spin/Spin.h
panelSpace.o: util/DialogF.h
panelSpace.o: phase.h
panelSpace.o: space.h
panelSpace.o: phaseP.h
panelSpace.o: panelSpace.h
panelSpace.o: menu.h
panelSpace.o: pick.h
panelSpace.o: rotation.h
panelPara.o: spin/Spin.h
panelPara.o: util/DialogF.h
panelPara.o: phase.h
panelPara.o: space.h
panelPara.o: phaseP.h
panelPara.o: panelPara.h
panelPara.o: menu.h
panelPara.o: pick.h
panelPara.o: rotation.h
space.o: spin/Spin.h
space.o: phase.h
space.o: space.h
space.o: phaseP.h
space.o: panelSpace.h
space.o: pixmaps.h
space.o: pick.h
space.o: drawEvent.h
space.o: dispTree.h
Phase_main.o: phase.h
Phase_main.o: space.h
Phase_main.o: phaseP.h
Phase_main.o: spin/Spin.h
Phase_main.o: drawEvent.h
TestParticles.o: phase.h
phase_g.o: spin/Spin.h
phase_g.o: phase.h
phase_g.o: space.h
phase_g.o: phaseP.h
phase_g.o: panel.h
phase_g.o: pixmaps.h
phase_g.o: pick.h
phase_g.o: drawEvent.h
phase_g.o: dispTree.h
panel_g.o: spin/Spin.h
panel_g.o: util/DialogF.h
panel_g.o: phase.h
panel_g.o: space.h
panel_g.o: phaseP.h
panel_g.o: panel.h
panel_g.o: menu.h
panel_g.o: pick.h
panel_g.o: rotation.h
menu_g.o: phase.h
menu_g.o: space.h
menu_g.o: phaseP.h
menu_g.o: menu.h
menu_g.o: panel.h
menu_g.o: settings.h
menu_g.o: rotation.h
menu_g.o: stdHepFiles.h
menu_g.o: pick.h
menu_g.o: dispTree.h
menu_g.o: util/printUtils.h
menu_g.o: util/getfiles.h
menu_g.o: util/DialogF.h
menu_g.o: util/misc.h
menu_g.o: spin/Spin.h
menu_g.o: util/help.h
menu_g.o: help.h
drawEvent_g.o: spin/Spin.h
drawEvent_g.o: util/stringUtils.h
drawEvent_g.o: phase.h
drawEvent_g.o: space.h
drawEvent_g.o: phaseP.h
drawEvent_g.o: drawEvent.h
pick_g.o: spin/Spin.h
pick_g.o: util/stringUtils.h
pick_g.o: phase.h
pick_g.o: space.h
pick_g.o: phaseP.h
pick_g.o: pick.h
pick_g.o: drawEvent.h
pick_g.o: dispTree.h
pixmaps_g.o: pixmaps.h
settings_g.o: phase.h
settings_g.o: space.h
settings_g.o: phaseP.h
rotation_g.o: phase.h
rotation_g.o: space.h
rotation_g.o: phaseP.h
help_g.o: util/help.h
help_g.o: help.h
Tree_g.o: tree.h
Tree_g.o: treeP.h
dispTree_g.o: tree.h
dispTree_g.o: phase.h
dispTree_g.o: space.h
dispTree_g.o: phaseP.h
dispTree_g.o: dispTree.h
dispTree_g.o: pick.h
dispTree_g.o: spin/Spin.h
dispTree_g.o: drawEvent.h
dispTree_g.o: listNodeHep.h
dispTree_g.o: util/stringUtils.h
listNodeHep_g.o: phase.h
listNodeHep_g.o: space.h
listNodeHep_g.o: phaseP.h
listNodeHep_g.o: dispTree.h
listNodeHep_g.o: spin/Spin.h
listNodeHep_g.o: drawEvent.h
listNodeHep_g.o: listNodeHep.h
stdHepFiles_g.o: phase.h
stdHepFiles_g.o: space.h
stdHepFiles_g.o: phaseP.h
stdHepFiles_g.o: spin/Spin.h
stdHepFiles_g.o: drawEvent.h
panelSpace_g.o: spin/Spin.h
panelSpace_g.o: util/DialogF.h
panelSpace_g.o: phase.h
panelSpace_g.o: space.h
panelSpace_g.o: phaseP.h
panelSpace_g.o: panelSpace.h
panelSpace_g.o: menu.h
panelSpace_g.o: pick.h
panelSpace_g.o: rotation.h
panelPara_g.o: spin/Spin.h
panelPara_g.o: util/DialogF.h
panelPara_g.o: phase.h
panelPara_g.o: space.h
panelPara_g.o: phaseP.h
panelPara_g.o: panelPara.h
panelPara_g.o: menu.h
panelPara_g.o: pick.h
panelPara_g.o: rotation.h
space_g.o: spin/Spin.h
space_g.o: phase.h
space_g.o: space.h
space_g.o: phaseP.h
space_g.o: panelSpace.h
space_g.o: pixmaps.h
space_g.o: pick.h
space_g.o: drawEvent.h
space_g.o: dispTree.h
Phase_main_g.o: phase.h
Phase_main_g.o: space.h
Phase_main_g.o: phaseP.h
Phase_main_g.o: spin/Spin.h
Phase_main_g.o: drawEvent.h
TestParticles_g.o: phase.h
