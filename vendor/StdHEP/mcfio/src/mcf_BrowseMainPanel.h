/*******************************************************************************
*									       *
* mcf_BrowseMainPanel.h -- Main panel for the Mcfio Data Browser		       *			       *
*									       *
* Copyright (c) 1995, 1996 Universities Research Association, Inc.	       *
* All rights reserved.							       *
* 									       *
*******************************************************************************/extern Widget MainPanelW;
enum panelState {NO_FILE_OPEN, FILE_OPEN, SCANNING,
                 STEPPING, STOPPED_FILE_OPEN};
extern Widget McfioMainPanelW;

Widget mcfioC_CreateBrWindow(Display *display);
void mcfioC_OpenNewFile();
int mcfioC_OpenInitialFile(char *filename);
void mcfioC_CloseBrowser();
int mcfioC_SelectedNTuBrListPos(); 
int mcfioC_SetSpecificNTupleBr(nTuBrowserInfo *nTuBr);
