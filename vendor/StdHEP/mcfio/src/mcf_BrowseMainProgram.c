/*******************************************************************************
*									       *
* mcf_BrowseMainProgram -- Browser for MCFIO file, Main program.               *
*									       *
*	P. Lebrun, November 1995.					       *
*									       *
*******************************************************************************/
/*
** This is a simple GUI front end to create or modify, read, save.. a 
** generalized Ntuple Descriptor. The NFit Parameter panel has been
** inspirational in building this one. 
*/
#include <stdio.h>
#include <limits.h>
#include <X11/Intrinsic.h>
#include <X11/IntrinsicP.h>
#include <X11/StringDefs.h>
#include <Xm/Xm.h>
#include <Xm/XmP.h>
#include <rpc/types.h>
#include <sys/types.h>
#include <rpc/xdr.h>
#include "mcf_nTupleDescript.h"
#include "mcf_xdr.h"
#include "mcfio_Dict.h"
#include "mcfio_Util1.h"
#include "mcf_nTupleBuild.h"
#include "mcf_BrowseUtil1.h"
#include "mcf_BrowseMainPanel.h"

static char *fallbackResources[] = {
    "*background:  wheat", 
    "*foreground: black",
    "*fileMenu.tearOffModel: XmTEAR_OFF_ENABLED",
    "*preferencesMenu.tearOffModel: XmTEAR_OFF_ENABLED",
    "*fontList:-adobe-courier-bold-r-normal-*-14-*-*-*-*-*-*-*",
    "*XmList.fontList:-adobe-courier-bold-r-normal-*-14-*-*-*-*-*-*-*",
    0};
/*
** Global, to be use throughout this application, 
**        to pick up available NTuBuildWindows... 
*/
extern nTuBuildWindow *NTuBuildWindowList;

main(int argc, char **argv)
{
    XtAppContext context;
    Display *display;
    Widget mainPanel;
    char *fileName, *env, *line;
    XrmDatabase prefDB;
    FILE *Ffp;
    
    /*
    ** Initialize mcfio
    */
    mcfioC_Init();
    mcfioC_SetForSaveDecoding(0);
     
    /* Initialize toolkit and create the obligatory application context */
    XtToolkitInitialize();
    context = XtCreateApplicationContext();
    
    /* Motif still has problems with releasing non-existent passive grabs */
    SuppressPassiveGrabWarnings();

    /* Set up default resources */
    XtAppSetFallbackResources(context, fallbackResources);
    
#if !AIX
    /* Allow users to change tear off menus with X resources */
    XmRepTypeInstallTearOffModelConverter();
#endif
  
   /* Open the Display */
    display = XtOpenDisplay(context, NULL, APP_NAME, APP_CLASS, NULL,
            0, &argc, argv );
    if (!display) {
        fprintf(stderr,"NtuBrowser: Can't open display\n");
        exit(0);
    }
    
    /* Open an empty fit window to give users a menu bar */
   mcfioC_CreateBrWindow(display);
    
    /* Process inline arguments.  For now, only one MCFIO Input file. */
    
    if (argc > 1) {
       fileName = argv[1];
       mcfioC_OpenInitialFile(fileName);
    }    
    /* Process events */
    XtAppMainLoop (context);
}
