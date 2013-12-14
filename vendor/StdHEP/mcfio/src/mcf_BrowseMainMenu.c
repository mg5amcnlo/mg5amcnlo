/*******************************************************************************
*									       *
* mcf_BrowseMainMenu.c -- Main Menu bar for the Mcfio Data Browser	       *
*									       *
* Copyright (c) 1995, 1996 Universities Research Association, Inc.	       *
* All rights reserved.							       *
* 									       *
*******************************************************************************/
#include <sys/param.h>
#include <stdio.h>
#include <stdlib.h>
#include <rpc/types.h>
#include <sys/types.h>
#include <rpc/xdr.h>
#include <ctype.h>
#include <limits.h>
#include <X11/Intrinsic.h>
#include <X11/Shell.h>
#include <Xm/Xm.h>
#include <Xm/MainW.h>
#include <Xm/Form.h>
#include <Xm/List.h>
#include <Xm/LabelG.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/ToggleB.h>
#include <Xm/ToggleBG.h>
#include <Xm/Separator.h>
#include <Xm/SeparatoG.h>
#include <Xm/FileSB.h>
#include <Xm/TextF.h>
#include <Xm/DialogS.h>
#include "DialogF.h"
#include "getfiles.h"
#include "misc.h"
#include "help.h"
#include "mcf_nTupleDescript.h"
#include "mcf_xdr.h"
#include "mcfio_Util1.h"
#include "mcfio_Direct.h"
#include "mcfio_Dict.h"
#include "mcf_nTupleBuild.h"
#include "mcf_BrowseUtil1.h"
#include "mcf_BrowseMainPanel.h"
#include "mcf_BrowseMainMenu.h"
#include "mcf_BrowseHelp.h"

extern nTuDDL **NTuDDLList;
extern int NumOfNTuples;
extern int McfioSetMemoryMappedIO;
extern int McfioPanelState;
extern Widget McfioMainPanelW;
extern Widget McfioMenuShowHeaderW;
extern Widget McfioMenuShowEvtHeaderW;
extern Widget McfioMenuShowOneDHistMainW;
extern nTuBrowserInfo *CurrentNTupleBrowserSelected;

static void exitCB(Widget w, Widget parent, caddr_t callData);
static void closeCB(Widget w,  caddr_t clientData, caddr_t callData);
static void fileOpenCB(Widget w, caddr_t clientData, caddr_t callData);
static void showHeaderCB(Widget w, caddr_t clientData, caddr_t callData);
static void showEvtHeaderCB(Widget w, caddr_t clientData, caddr_t callData);
#ifdef HISTO
static void showOneDHistCB(Widget w, caddr_t clientData, caddr_t callData);
#endif
static void setDecodingCB(Widget w, caddr_t clientData, caddr_t callData);

static void setMemMappedCB(Widget w, caddr_t clientData, caddr_t callData);

Widget mcfioC_CreateBrMenu(Widget parent)
{
    Widget menuBar, menuPane;

    /* create the menu bar widget */
    menuBar = XmCreateMenuBar(parent, "menuBar", NULL, 0);
     
    /* Create the File menu */
    menuPane = AddSubMenu(menuBar, "fileMenu", "File", 'F');
    AddMenuItem(menuPane, "open", "Open Mcfio File...", 'O', "Ctrl<Key>o",
    	    "Ctrl O", (XtCallbackProc)fileOpenCB, NULL);
    AddMenuItem(menuPane, "close", "Close", 'C',
    	    "Ctrl<Key>w", "Ctrl W", (XtCallbackProc)closeCB, NULL);
    AddMenuItem(menuPane, "exit", "Exit", 'E', "<Key>F3:",
    	    "F3", (XtCallbackProc)exitCB,  NULL);
    
    /* Create the Create Analysis menu */
    menuPane = AddSubMenu(menuBar, "analysisMenu", "Analysis", 'A');
    McfioMenuShowHeaderW =  AddMenuItem(menuPane, "showHeader", 
             "Show header File...", 'H', "Ctrl<Key>h",
    	    "Ctrl H", (XtCallbackProc)showHeaderCB, NULL);
    XtSetSensitive(McfioMenuShowHeaderW, False);
    McfioMenuShowEvtHeaderW =  AddMenuItem(menuPane, "showEvtHeader", 
             "Show Event/File header...", 'A', NULL,
    	    "", (XtCallbackProc)showEvtHeaderCB, NULL);
    XtSetSensitive(McfioMenuShowEvtHeaderW, False);
    AddMenuSeparator(menuPane, "s0");
#ifdef HISTO    
    McfioMenuShowOneDHistMainW =  AddMenuItem(menuPane, "on1DHistMainM", 
             "Define 1D histograms...", 'A', NULL,
    	    "", (XtCallbackProc)showOneDHistCB, NULL);
    XtSetSensitive(McfioMenuShowOneDHistMainW, False);
#endif     	        
    /* Create the Preferences menu */
    menuPane = AddSubMenu(menuBar, "preferencesMenu", "Preferences", 'P');
    AddMenuItem(menuPane, "MemMappedIO", "Memory Mapped I/O... ",
             'A', NULL,
    	    "", (XtCallbackProc)setMemMappedCB, NULL);
    
    AddMenuItem(menuPane, "saveDecoding", "Set For Decoding... ",
             'A', NULL,
    	    "", (XtCallbackProc)setDecodingCB, NULL);
    
    /* Create the Help menu */
    CreateHelpPulldownMenu(menuBar, NTuBrowseHelp);

    XtManageChild(menuBar);
    XtSetSensitive(McfioMenuShowHeaderW, False);
    
    return menuBar;
}

static void exitCB(Widget w, Widget parent, caddr_t callData)
{
	mcfioC_CloseBrowser();
	exit(0);
}
    
static void closeCB(Widget w, caddr_t clientData, caddr_t callData)
{
	mcfioC_CloseBrowser();
}
    
static void fileOpenCB(Widget w,  caddr_t clientData, caddr_t callData)
{
	mcfioC_OpenNewFile();
} 
   
static void showHeaderCB(Widget w,  caddr_t clientData, caddr_t callData)
{
      mcfioC_ShowBrowserHeadDump();
}

static void showEvtHeaderCB(Widget w,  caddr_t clientData, caddr_t callData)
{
      mcfioC_ShowBrowserEvtHeadDump();
}
#ifdef HISTO
static void showOneDHistCB(Widget w,  caddr_t clientData, caddr_t callData)
{
     if (CurrentNTupleBrowserSelected == NULL) {
        if (mcfioC_SelectedNtuBrListPos() == 0) return;
     }
     mcfioC_ShowBrowserOneDHist();
}
#endif
static void setMemMappedCB(Widget w,  caddr_t clientData, caddr_t callData)
{
   
    char *fn;
    int ll;
    int old_set = McfioSetMemoryMappedIO;
    
    if (DialogF(DF_QUES, McfioMainPanelW, 2, 
"For Improved I/O performance, on can map the entire file to virtual memory.\n\
 However, potentially sever memory leak could occur if the I/O processing \n\
 is un-intentionaly interrupted",
     "Memory Mapped I/O ", "Standard I/O ") == 1)
         McfioSetMemoryMappedIO = True; 
    else  McfioSetMemoryMappedIO = False; 
    if ((old_set != McfioSetMemoryMappedIO) && 
         (McfioPanelState != NO_FILE_OPEN)) {
             if (McfioPanelState == SCANNING) {
                 DialogF(DF_WARN, McfioMainPanelW, 1,
"You can not changed I/O acces mode while scanning the file. \n\
  Stop running first !.... ", "Acknowledged");
                 McfioSetMemoryMappedIO  = old_set;
                 return;
              }
         if (DialogF(DF_QUES, McfioMainPanelW, 2, 
"You changed the Input mode. \n The file must be reopened.",
     "Re-open", "Cancel") != 1) {
             McfioSetMemoryMappedIO = old_set; 
             return;
         }
         fn = (char *) malloc(sizeof(char) * FILENAME_MAX);
         mcfioC_InfoStreamChar(1, MCFIO_FILENAME, fn, &ll);
         mcfioC_Close(1);
         if (mcfioC_OpenInitialFile(fn) == False) { 
                 DialogF(DF_ERR, McfioMainPanelW, 1,
" Sorry, I am utterly confused while re-opening the file\n\
  Better give up now...", "Acknowledged");
                 exit(0);
              }
    }
}

static void setDecodingCB(Widget w,  caddr_t clientData, caddr_t callData)
{
      if (DialogF(DF_QUES, McfioMainPanelW, 2, 
" Upon decrypting the Ntuple header, you might need to re-create \n\
  internals pointer lists in case the file was geenrated on a different UNIX\n\
  flavor. By default, this step is skipped.",
     "Re-Created Pointers", "Skip ") == 1) mcfioC_SetForSaveDecoding(1);
     else  mcfioC_SetForSaveDecoding(0); 
}
