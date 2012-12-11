/*******************************************************************************
*									       *
* mcf_nTupleBuild.c -- Program for mcfast generalized nTuple Builder GUI       *
*									       *
*	P. Lebrun, September 1995.					       *
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

Widget McfioMainPanelW = NULL; /* Not use in this context, but linked in */

static char *fallbackResources[] = {
    "*background:  white", 
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
    
    env = NULL;
    env = getenv("MCFIO_DIR");
    /*
    ** Check now that the master template exist.
    ** no longer needed, as this master template is used only to create
    ** the code that read our db..
    line = (char *) malloc(sizeof(char) * (FILENAME_MAX+1));
    sprintf(line,"%s/mcf_NTuBld.db", env);
    Ffp = fopen(line, "r");
    if (Ffp == NULL) {
       printf ("The file %s could not be opened. \n", line);
       printf (" Please check MCFIO installation. \n"); 
       exit(0);
    }
    fclose(Ffp);
    free(line);
    */
    /*
    ** Initialize mcfio
    */
    mcfioC_Init();
     
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
            0, &argc, argv);
    if (!display) {
        fprintf(stderr,"NtuBuild: Can't open display\n");
        exit(0);
    }
    
    /* Open an empty fit window to give users a menu bar */
    CreateNTuBuildWindow(display, "NTuple Builder, first window", False);
    
    /* Process inline arguments.  For now, only implement file Read */
    
    if (argc > 1) {
       fileName = argv[1];
       GetFileNTuBuildWindow(NTuBuildWindowList, fileName);
       XtVaSetValues(NTuBuildWindowList->shell, XmNtitle, fileName, 0);
    }    
    /* Process events */
    XtAppMainLoop (context);
}
/*
** A bunch of dummy function that are never called from this program, 
** but neede to complete the link of the executable (one panel beeing 
** shared between the Browser and the Builder modules. )
*/
void mcfioC_ShowBrowserOneDHist()
{
      return;
}
