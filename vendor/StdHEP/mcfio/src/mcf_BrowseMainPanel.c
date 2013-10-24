/*******************************************************************************
*									       *
* mcf_BrowseMainPanel.c -- Main panel for the Mcfio Data Browser	       *			       *
*									       *
* Copyright (c) 1995, 1996 Universities Research Association, Inc.	       *
* All rights reserved.							       *
* 									       *
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <rpc/types.h>
#include <sys/types.h>
#include <sys/param.h>
#include <rpc/xdr.h>
#include <ctype.h>
#include <limits.h>
#include <X11/Intrinsic.h>
#include <X11/Shell.h>
#include <X11/Xatom.h>
#include <Xm/Xm.h>
#include <Xm/MainW.h>
#include <Xm/Form.h>
#include <Xm/List.h>
#include <Xm/LabelG.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>
#include <Xm/RowColumn.h>
#include <Xm/ToggleB.h>
#include <Xm/ToggleBG.h>
#include <Xm/Separator.h>
#include <Xm/SeparatoG.h>
#include <Xm/FileSB.h>
#include <Xm/TextF.h>
#include <Xm/DialogS.h>
#include <Xm/Frame.h>
#include <Xm/Scale.h>
#include "DialogF.h"
#include "getfiles.h"
#include "misc.h"
#include "mcf_nTupleDescript.h"
#include "mcf_xdr.h"
#include "mcfio_Dict.h"
#include "mcfio_Util1.h"
#include "mcfio_Direct.h"
#include "mcf_nTupleBuild.h"
#include "mcf_NTuIOUtils.h"
#include "mcf_NTuBuildWindow.h"
#include "mcf_BrowseUtil1.h"
#include "mcf_BrowseMainPanel.h"
#include "mcf_BrowseMainMenu.h"
#include "mcf_BrowseUtil2.h"
#ifdef HISTO
#include "histoscope.h"
#endif

#define INITIAL_LIST_HEIGHT 14	/* number of rows in the category list */
#define OPT_MENU_PAD 4		/* added to align other widgets with option
				   menus to compensate for space on left */
#define SLIDER_MAX 1000 
#define TIME_BETWEEN_STATUS 2000

extern nTuDDL **NTuDDLList;
extern int NumOfNTuples;
extern int HistoIsBrowserNtuInit;
int McfioSetMemoryMappedIO = False;
int McfioPanelState = NO_FILE_OPEN; /* The status of the Main Panel */

/* Global Variables */
Widget McfioMainPanelW; /* The Main panel widget */
Widget McfioMenuShowHeaderW;
Widget McfioMenuShowEvtHeaderW;
Widget McfioMenuShowOneDHistMainW;
#ifdef HISTO
Widget McfioHsResetBtnW; /* To reset all existing histograms */
#endif
nTuBrowserInfo *CurrentNTupleBrowserSelected;



/* Module Global Variables */
static Widget MainContentsW, CategoryListW, NTupleListW;
static Widget CategoryMenuW, CategoryOptMenuW, OpenBtnW;
static Widget  ViewTemplateBtnW, ViewDumpBtnW, ShowIdBtnW;
static Widget StreamTitleLabelW;
static Widget RewindBtnW, NextEventBtnW, RunBtnW, StopBtnW, SelectEvtBtnW;
static Widget InputRateSliderW, InputRateLabelW;
static char McfioFilename[MAXPATHLEN];
static int *ListedIDs;
static char CurrentCategory[NTU_MAX_CATEGORY_LENGTH] = "   Uncategorized   ";
static char TopCategory[NTU_MAX_CATEGORY_LENGTH] = "   Uncategorized   ";
static int TotalNumberOfEvts;
static int CurrentEventNumber;
static int CurrentSeqEvtNum;
static int CurrentRunNumber;
static int CurrentStoreNumber;
static int CurrentTriggerMask, EventCountAtLastUpdate;
static int CurrentWordCount, PreviousWordCount, InitialWordCount;
static float TimeLeftBeforeNextUptade;
static struct timeval TimeAtPreviousItteration;

/* Function Prototypes */

/* Function Prototypes */
static void createContents(Widget parent);
void redisplayNTupleList(int resetCurrentCategory);
static void closeCB(Widget w, caddr_t clientData, caddr_t callData);
static void categoryListCB(Widget w, caddr_t clientData, caddr_t callData); 
static void categoryMenuCB(Widget w, int catNum, caddr_t callData); 
static void nTupleListCB(Widget w, caddr_t clientData, caddr_t callData);
static void nTupleBrowseListCB(Widget w, caddr_t clientData, caddr_t callData);
static void openCB(Widget w, caddr_t clientData, caddr_t callData);
static void viewTemplateCB(Widget w, caddr_t clientData, caddr_t callData);
static void viewDumpCB(Widget w, caddr_t clientData, caddr_t callData);
static void showIdCB(Widget w, caddr_t clientData, caddr_t callData); 
static void sliderTuneCB(Widget w, caddr_t clientData, caddr_t callData);
static void resetTopCategory(void);
static void redisplayItems(void);
static void redisplaySubcategories(void);
static void redisplayCategoryMenu(void);
static void setMainPanelState(int state, char * title);
static char *nextCategory(char *toString, char *fromString);
static int subCategory(char *category, char *categories, char *subCategory);
static int itemInCategory(nTuDDL *ddl, char *category);
static void setListItems(Widget w, XmString *strings, int nStrings);
static void rewindCB(Widget w, caddr_t clientData, caddr_t callData);
static void nextEventCB(Widget w, caddr_t clientData, caddr_t callData);
static void runCB(Widget w, caddr_t clientData, caddr_t callData);
static void stopCB(Widget w, caddr_t clientData, caddr_t callData);
static void selectEvtCB(Widget w, caddr_t clientData, caddr_t callData);
#ifdef HISTO
static void hsResetCB(Widget w, caddr_t clientData, caddr_t callData);
#endif
static void sliderTuneCB(Widget w, caddr_t clientData, caddr_t callData);
static Boolean anyUpper(char *string);
static void updateStatusLabel(float speed);
static int decodeNTuples();
#ifdef HISTO         
static void fillOneDHists(nTuBrowserInfo *nTuBr);
static void fillOneDHistValue(int hs_id, long *lDat, int nskip, int type);
#endif
/*
** Create the NTuple Browser Main Panel. 
*/
Widget mcfioC_CreateBrWindow(Display *display)
{
    Widget appShell, menuBar;
    Arg al[30];
    int ac;
    
    /* Create an toplevel shell to hold the window */
    ac = 0;
    XtSetArg(al[ac], XmNtitle, "Mcfio Data Browser"); ac++;
    XtSetArg(al[ac], XmNdeleteResponse, XmDO_NOTHING); ac++;
    XtSetArg(al[ac], XmNiconName, "McfioBrowser"); ac++;
    appShell = XtAppCreateShell("mcfio_Browse", "MCFIO_Browse",
		applicationShellWidgetClass, display, al, ac);
    	    
    /* Create a main window holding a menu bar and a form with the rest of
       the window contents. */
    McfioMainPanelW = XmCreateMainWindow(appShell, "main", NULL, 0);
    XtManageChild(McfioMainPanelW);
    menuBar = mcfioC_CreateBrMenu(McfioMainPanelW);
    XtManageChild(menuBar);
    createContents(McfioMainPanelW);
    setMainPanelState(NO_FILE_OPEN, "Mcfio Browser");
    
    /* realize all of the widgets in the new window */
    XtRealizeWidget(appShell);

    /* set up closeCB to be called when the user selects close from the
       window frame menu */
    AddMotifCloseCallback(appShell,(XtCallbackProc) closeCB, NULL);
   HistoIsBrowserNtuInit = 0; 
    return McfioMainPanelW;
}
/*
** Create the contents area of the Histoscope main window
*/
static void createContents(Widget parent)
{
    Arg args[50];
    int ac;
    XmString s1, *st1;
    Widget categoryForm, categoryLabel, subCatLabel;
    Widget nTupleLabel, fakeBtn, frame;

    /* Create the form onto which everything goes */
    ac = 0;
    MainContentsW = XmCreateForm(parent, "form", args, ac);

    ac = 0;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNrightAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNshadowType, XmSHADOW_ETCHED_OUT); ac++;
    frame = XmCreateFrame(MainContentsW, "titleFrame", args, ac);
    XtManageChild(frame);
    
    
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, 
     (s1=XmStringCreateSimple(
     "Stream Title: Blank                                          "))); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftOffset,1); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNrecomputeSize, False); ac++;
    XtSetArg(args[ac], XmNtopOffset,1); ac++;
    StreamTitleLabelW = 
       XmCreateLabelGadget(frame, "streamTitleLabel", args,ac);
    XmStringFree(s1);
    XtManageChild(StreamTitleLabelW);
/*
** A row of buttons to go into this stream..
*/
   RewindBtnW = XtVaCreateManagedWidget("rewindBtn",
    	    xmPushButtonGadgetClass, MainContentsW ,
    	    XmNlabelString, s1=XmStringCreateSimple("Rewind"),
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopWidget, frame ,
    	    XmNtopOffset, 7,
    	    XmNleftAttachment, XmATTACH_FORM,
    	    XmNleftOffset, 5, 0);
    XmStringFree(s1);
    XtAddCallback(RewindBtnW, XmNactivateCallback,
    	    (XtCallbackProc)rewindCB, NULL);
    XtSetSensitive(RewindBtnW, False);
    
         
   NextEventBtnW = XtVaCreateManagedWidget("nextEvtBtn",
    	    xmPushButtonGadgetClass, MainContentsW ,
    	    XmNlabelString, s1=XmStringCreateSimple("Next Event ->"),
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopWidget, frame,
    	    XmNtopOffset, 7,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftWidget, RewindBtnW,
    	    XmNleftOffset, 3, 0);
    XmStringFree(s1);
    XtAddCallback(NextEventBtnW, XmNactivateCallback,
    	    (XtCallbackProc)nextEventCB, NULL);
    XtSetSensitive(NextEventBtnW, False);


   SelectEvtBtnW = XtVaCreateManagedWidget("selectEvtBtn",
    	    xmPushButtonGadgetClass, MainContentsW ,
    	    XmNlabelString, s1=XmStringCreateSimple("Select Event..."),
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopWidget, frame,
    	    XmNtopOffset, 7,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftWidget, NextEventBtnW,
    	    XmNleftOffset, 3, 0);
    XmStringFree(s1);
    XtAddCallback(SelectEvtBtnW, XmNactivateCallback,
    	    (XtCallbackProc)selectEvtCB, NULL);
    XtSetSensitive(SelectEvtBtnW, False);
    
   RunBtnW = XtVaCreateManagedWidget("RunBtn",
    	    xmPushButtonGadgetClass, MainContentsW ,
    	    XmNlabelString, s1=XmStringCreateSimple("  Run ->>"),
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopWidget, frame,
    	    XmNtopOffset, 7,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftWidget, SelectEvtBtnW,
    	    XmNleftOffset,3, 0);
    XmStringFree(s1);
    XtAddCallback(RunBtnW, XmNactivateCallback,
    	    (XtCallbackProc)runCB, NULL);
    XtSetSensitive(RunBtnW, False);
    
   StopBtnW = XtVaCreateManagedWidget("StopBtn",
    	    xmPushButtonGadgetClass, MainContentsW ,
    	    XmNlabelString, s1=XmStringCreateSimple("Stop"),
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopWidget, frame,
    	    XmNtopOffset, 7,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftWidget,RunBtnW,
    	    XmNleftOffset, 5, 0);
    XmStringFree(s1);
    XtAddCallback(StopBtnW, XmNactivateCallback,
    	    (XtCallbackProc)stopCB, NULL);
    XtSetSensitive(StopBtnW, False);
    
#ifdef HISTO
   McfioHsResetBtnW = XtVaCreateManagedWidget("HsRestBtn",
    	    xmPushButtonGadgetClass, MainContentsW ,
    	    XmNlabelString, s1=XmStringCreateSimple("Histo Reset"),
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopWidget, frame,
    	    XmNtopOffset, 7,
    	    XmNrightAttachment, XmATTACH_FORM,
    	    XmNrightOffset, 3, 0);
    XmStringFree(s1);
    XtAddCallback(McfioHsResetBtnW, XmNactivateCallback,
    	    (XtCallbackProc)hsResetCB, NULL);
    XtSetSensitive(McfioHsResetBtnW, False);
#endif
/*
** A label, taking 2lines, the entire length, almost, stating the evt count:
** At event xxxxxx, Run yyyy, Spill yyyy, Event Size = xx kB
** Scanning at yyy Kb/sec, expecting to complete in xx minutes  
*/
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, 
     (s1=XmStringCreateLtoR(
"Stream is uninitialized...                                                  \n\
Idle....                                                                  ",
 XmSTRING_DEFAULT_CHARSET))); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftOffset,1); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopWidget, StopBtnW); ac++;
    XtSetArg(args[ac], XmNrecomputeSize, False); ac++;
    XtSetArg(args[ac], XmNtopOffset,5); ac++;
    InputRateLabelW = 
       XmCreateLabelGadget(MainContentsW, "statusLabel", args,ac);
    XmStringFree(s1);
    XtManageChild(InputRateLabelW);
/*
** A Slider, show graphically where we are in the stream.
*/
    InputRateSliderW =
        XtVaCreateManagedWidget("Rate", xmScaleWidgetClass, MainContentsW ,
    		 XmNorientation, XmHORIZONTAL,
    		 XmNminimum, 0,
    		 XmNmaximum, SLIDER_MAX,
    		 XmNvalue, 0,
    		 XmNrightAttachment, XmATTACH_POSITION,
    		 XmNleftAttachment, XmATTACH_FORM,
    		 XmNtopAttachment,  XmATTACH_WIDGET,
    		 XmNrightPosition, 99,
    		 XmNtopWidget,InputRateLabelW , 
    		 XmNleftOffset, 1,
    		 XmNtopOffset, 1, 0);
	XtAddCallback(InputRateSliderW , XmNdragCallback,
	    	    (XtCallbackProc)stopCB, NULL);
	XtAddCallback(InputRateSliderW, XmNvalueChangedCallback,
	    	    (XtCallbackProc)sliderTuneCB, NULL);
    XtSetSensitive(InputRateSliderW, False);	    	    
	    	    
         
    /* Create a form to hold the first column of information (category list) */
    ac = 0;
    XtSetArg(args[ac], XmNmarginHeight, 0); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftOffset, 6); ac++;
    XtSetArg(args[ac], XmNtopOffset, 10); ac++;
    XtSetArg(args[ac], XmNtopWidget,InputRateSliderW); ac++;
    categoryForm = XmCreateForm(MainContentsW, "categoryForm", args, ac);
    XtManageChild(categoryForm);

    ac = 0;
    XtSetArg(args[ac], XmNlabelString,
     (s1=XmStringCreateSimple("Current Category"))); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftOffset, OPT_MENU_PAD); ac++;
    categoryLabel = XmCreateLabelGadget(categoryForm, "categoryLabel", args,ac);
    XmStringFree(s1);
    XtManageChild(categoryLabel);
 
    ac = 0;
    CategoryMenuW = XmCreatePulldownMenu(categoryForm, "category", args, ac);

    ac = 0;
    XtSetArg(args[ac], XmNspacing, 0); ac++;
    XtSetArg(args[ac], XmNmarginWidth, 0); ac++;
    XtSetArg(args[ac], XmNresizeWidth, False); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNrightAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNtopWidget, categoryLabel); ac++;
    XtSetArg(args[ac], XmNsubMenuId, CategoryMenuW); ac++;
    CategoryOptMenuW = XmCreateOptionMenu(categoryForm, "categoryMenu",
    					  args, ac);
    XtManageChild(CategoryOptMenuW);
 
    /* Option menus have several serious problems, particularly in the version
       of Motif on SGI. Two separate methods are used to keep the option
       menu at a constant width.  For the current SGI version, we create a
       fake, zero height, menu item to keep the option menu widget from
       narrowing when the menu items are changed.  The string in the category
       option menu (and of course the font) determines the width for both the
       category menu and for the sub-category list. */
    /*
    **  This was the original comment from Mark, from HistoScope. 
    **  It might not be really relevant with Motif 1.2, but we keep it..
    */  
    fakeBtn = XtVaCreateManagedWidget("fakeCategory", xmPushButtonWidgetClass,
    	    CategoryMenuW,
    	    XmNlabelString, s1=XmStringCreateSimple(CurrentCategory),
    	    XmNshadowThickness, 0,
    	    XmNmarginHeight, 0,
    	    XmNsensitive, False,
    	    XmNrecomputeSize, False,
    	    XmNheight, 0 , 0);
    XmStringFree(s1);
    XtVaSetValues(fakeBtn, XmNlabelString, s1=XmStringCreateSimple(""),
    	    XmNheight, 0, 0);
    /* On systems other than the SGI, just displaying the "   Uncategorized   "
       initial category item and setting XmNrecomputeSize to false is enough */
    redisplayCategoryMenu();
    XmStringFree(s1);				
    XtVaSetValues(XmOptionButtonGadget(CategoryOptMenuW),
    	    XmNrecomputeSize, False, 0);
    
    ac = 0;
    XtSetArg(args[ac], XmNlabelString,
     (s1=XmStringCreateSimple("Sub Categories"))); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNtopWidget, CategoryOptMenuW); ac++;
    XtSetArg(args[ac], XmNleftOffset, OPT_MENU_PAD); ac++;
    subCatLabel = XmCreateLabelGadget(categoryForm, "subCatLabel", args,ac);
    XmStringFree(s1);
    XtManageChild(subCatLabel);

    ac = 0;
    XtSetArg(args[ac], XmNlabelString, (s1=XmStringCreateSimple("Open"))); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNbottomOffset, 4); ac++;
    XtSetArg(args[ac], XmNleftOffset, 6 + OPT_MENU_PAD); ac++;
    OpenBtnW = XmCreatePushButton(categoryForm, "openBtn", args,ac);
    XtAddCallback(OpenBtnW, XmNactivateCallback,
                  (XtCallbackProc)  openCB, NULL);
    XmStringFree(s1);
    XtManageChild(OpenBtnW);
 
    ac = 0;
    XtSetArg(args[ac], XmNlabelString,
     (s1=XmStringCreateSimple("View Template"))); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftWidget, OpenBtnW); ac++;
    XtSetArg(args[ac], XmNbottomOffset, 4); ac++;
    XtSetArg(args[ac], XmNleftOffset, 10); ac++;
    XtSetArg(args[ac], XmNrightOffset, 10); ac++;
    ViewTemplateBtnW = XmCreatePushButton(categoryForm,
     "viewTemplateBtn", args,ac);
    XtAddCallback(ViewTemplateBtnW, XmNactivateCallback,
                  (XtCallbackProc)  viewTemplateCB, NULL);
    XmStringFree(s1);
    XtManageChild(ViewTemplateBtnW);
  
    ac = 0;
    XtSetArg(args[ac], XmNlabelString,
     (s1=XmStringCreateSimple("View Data"))); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftWidget, ViewTemplateBtnW); ac++;
    XtSetArg(args[ac], XmNbottomOffset, 4); ac++;
    XtSetArg(args[ac], XmNleftOffset, 10); ac++;
    XtSetArg(args[ac], XmNrightOffset, 10); ac++;
    ViewDumpBtnW = XmCreatePushButton(categoryForm,
     "viewDumpBtn", args,ac);
    XtAddCallback(ViewDumpBtnW, XmNactivateCallback,
                  (XtCallbackProc)  viewDumpCB, NULL);
    XmStringFree(s1);
    XtManageChild(ViewDumpBtnW);
    XtSetSensitive(ViewDumpBtnW, False);
    
    ac = 0;
    XtSetArg(args[ac], XmNitems, (st1=StringTable(1, " "))); ac++;
    XtSetArg(args[ac], XmNitemCount, 1); ac++;
    XtSetArg(args[ac], XmNscrollBarDisplayPolicy, XmAS_NEEDED); ac++;
    XtSetArg(args[ac], XmNvisibleItemCount, INITIAL_LIST_HEIGHT); ac++;
    XtSetArg(args[ac], XmNselectionPolicy, XmBROWSE_SELECT); ac++;
    XtSetArg(args[ac], XmNlistSizePolicy, XmCONSTANT); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNrightAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNtopWidget, subCatLabel); ac++;
    XtSetArg(args[ac], XmNbottomWidget, ViewDumpBtnW); ac++;
    XtSetArg(args[ac], XmNtopOffset, 0); ac++;
    XtSetArg(args[ac], XmNleftOffset, 3 + OPT_MENU_PAD); ac++;
    XtSetArg(args[ac], XmNbottomOffset, 4); ac++;
    CategoryListW = XmCreateScrolledList(categoryForm, "categoryList", args,ac);
    FreeStringTable(st1);
    XtAddCallback(CategoryListW, XmNdefaultActionCallback,
                  (XtCallbackProc)  categoryListCB,NULL);
    XtManageChild(CategoryListW);
 
    ac = 0;
    XtSetArg(args[ac], XmNlabelString,
    (s1=XmStringCreateSimple("Ntuple List"))); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftWidget, categoryForm); ac++;
    XtSetArg(args[ac], XmNtopWidget, InputRateSliderW); ac++;
    XtSetArg(args[ac], XmNtopOffset, 10); ac++;
    XtSetArg(args[ac], XmNleftOffset, 10); ac++;
    XtSetArg(args[ac], XmNrightOffset, 10); ac++;
    nTupleLabel = XmCreateLabelGadget(MainContentsW, "NtupleLabel",
    					 args, ac);
    XmStringFree(s1);
    XtManageChild(nTupleLabel);
 
    ac = 0;
    XtSetArg(args[ac], XmNlabelString,
    	     (s1=XmStringCreateSimple("Show User Id Numbers"))); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNrightAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftWidget, categoryForm); ac++;
    XtSetArg(args[ac], XmNbottomOffset, 4); ac++;
    XtSetArg(args[ac], XmNleftOffset, 10); ac++;
    XtSetArg(args[ac], XmNrightOffset, 10); ac++;
    ShowIdBtnW = XmCreateToggleButtonGadget(MainContentsW, "showIdBtn", args, ac);
    XtAddCallback(ShowIdBtnW, XmNvalueChangedCallback,
                 (XtCallbackProc) showIdCB, NULL);
    XmStringFree(s1);
    XtManageChild(ShowIdBtnW);

    ac = 0;
    XtSetArg(args[ac], XmNitems,
    	     (st1=StringTable(1, "(No file open)"))); ac++;
    XtSetArg(args[ac], XmNitemCount, 1); ac++;
    XtSetArg(args[ac], XmNscrollBarDisplayPolicy, XmAS_NEEDED); ac++;
    XtSetArg(args[ac], XmNvisibleItemCount, INITIAL_LIST_HEIGHT); ac++;
    XtSetArg(args[ac], XmNselectionPolicy, XmBROWSE_SELECT); ac++;
    XtSetArg(args[ac], XmNlistSizePolicy, XmCONSTANT); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNrightAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftWidget, categoryForm); ac++;
    XtSetArg(args[ac], XmNtopWidget, nTupleLabel); ac++;
    XtSetArg(args[ac], XmNbottomWidget, ShowIdBtnW); ac++;
    XtSetArg(args[ac], XmNtopOffset, 0); ac++;
    XtSetArg(args[ac], XmNbottomOffset, 4); ac++;
    XtSetArg(args[ac], XmNleftOffset, 10); ac++;
    XtSetArg(args[ac], XmNrightOffset, 10); ac++;
    NTupleListW = XmCreateScrolledList(MainContentsW, "NTupleList", 
    					  args, ac);
    XtAddCallback(NTupleListW, XmNdefaultActionCallback,
                  (XtCallbackProc)  nTupleListCB, NULL);
                  
    XtAddCallback(NTupleListW, XmNbrowseSelectionCallback,
                  (XtCallbackProc)  nTupleBrowseListCB, NULL);
    FreeStringTable(st1);
    XtManageChild(NTupleListW);
    
    /* The panel is set insensitive (dimmed) after being created, rather
       than being created that way, because option menus stick permanently
       insensitive if they are created with an insensitive parent! */
    XtSetSensitive(MainContentsW, False);

    XtManageChild(MainContentsW);
}
void mcfioC_OpenNewFile()
{
   char *ffname, title[80];
   int istat;
   
  ffname = (char *) malloc(FILENAME_MAX*sizeof(char));
  if (McfioPanelState != NO_FILE_OPEN) {
	if (DialogF(DF_QUES, MainContentsW , 2, 
"Opening a new file implies Closing the current one \n\
Do you wish to open this new file ? ",
		"Open", "Cancel") == 2) {
                    free(ffname);
                    return;
                }
         mcfioC_Close(1);
         mcfioC_DestroyBrowserAnalysis();
         mcfioC_Init(); TotalNumberOfEvts =0;
         setMainPanelState(NO_FILE_OPEN, "nTuple Browser"); 
    }            
  if (GetExistingFilename (MainContentsW,
                       "Mcfio File Name:",ffname ) 
                         == GFN_CANCEL) {
                            free(ffname);
                            return;
                         }
   if (McfioSetMemoryMappedIO) istat = mcfioC_OpenReadMapped(ffname);
      else  istat = mcfioC_OpenReadDirect(ffname);           
   if (istat == 1) {
         mcfioC_InfoStreamInt(1, MCFIO_NUMWORDS, &CurrentWordCount);	
         InitialWordCount = CurrentWordCount;	
         setMainPanelState(FILE_OPEN, ffname);
   }        
   free(ffname);
   return;	
}
	
int mcfioC_OpenInitialFile(char *filename)
{
      int istat;
      
      strcpy(McfioFilename, filename);
      if (McfioSetMemoryMappedIO) istat = mcfioC_OpenReadMapped(filename);
      else  istat = mcfioC_OpenReadDirect(filename);           
      if (istat != 1) return FALSE;
      mcfioC_InfoStreamInt(1, MCFIO_NUMWORDS, &CurrentWordCount);
      InitialWordCount = CurrentWordCount;	
      setMainPanelState(FILE_OPEN, filename);
      return True;
}
 
void mcfioC_CloseBrowser() 
{
    if (McfioPanelState == NO_FILE_OPEN) return;
    mcfioC_Close(1); TotalNumberOfEvts =0;
    mcfioC_DestroyBrowserAnalysis(); mcfioC_Init();
    setMainPanelState(NO_FILE_OPEN, "nTuple Browser");
    return;
}
/*
** Redisplay the nTuple list after changes have been made to
** CurrentCategory or NtupleList
*/
void redisplayNTupleList(int resetCurrentCategory)
{
    resetTopCategory();
    if (resetCurrentCategory)
    	strcpy(CurrentCategory, TopCategory);
    redisplayItems();
    redisplaySubcategories();
    redisplayCategoryMenu();
}

/*
** Callback procedures for the widgets on the Histoscope main panel
*/
static void closeCB(Widget w, caddr_t clientData, caddr_t callData) 
{
    mcfioC_CloseBrowser();
}
static void categoryListCB(Widget w, caddr_t clientData, caddr_t callData) 
{
    SimulateButtonPress(OpenBtnW);
}

 
static void categoryMenuCB(Widget w, int catNum, caddr_t callData) 
{
    char *cats, cat[NTU_MAX_CATEGORY_LENGTH];
    int i;
    
    /* Move upwards in the category hierarchy by truncating the
       CurrentCategory string with a null after catNum categories */
    cats = CurrentCategory;
    if (catNum == 0) {
    	*CurrentCategory = '\0';
    } else {
	for (i=1; i<=catNum; i++) {
    	    cats = nextCategory(cat, cats);
    	    if (cats == NULL) {
    		/* safety valve in case CurrentCategory has changed between
    		   callback setup and invocation.  Shouldn't happen often */
    		return;
    	    }
	}
	*cats = '\0';
    }
    redisplayNTupleList(False);
}
static void nTupleListCB(Widget w, caddr_t clientData, caddr_t callData)
{
    if ((McfioPanelState == SCANNING) || (McfioPanelState == STEPPING)) {
        if (McfioPanelState == SCANNING) SimulateButtonPress(StopBtnW);
        SimulateButtonPress(ViewDumpBtnW);
    } else if (McfioPanelState == FILE_OPEN)
        SimulateButtonPress(ViewTemplateBtnW);
}
static void nTupleBrowseListCB(Widget w, caddr_t clientData, caddr_t callData)
{
    int i, *posList, count, id;
    
    XmListGetSelectedPos(NTupleListW, &posList, &count);
    if (count < 1) return;
    id = ListedIDs[(*posList) -1];
    CurrentNTupleBrowserSelected = NTupleBrowserList[id-1];
#ifdef HISTO    
    mcfioC_OneDHistUpdateNTupleContent();
#endif
}

static void openCB(Widget w, caddr_t clientData, caddr_t callData)
{
    int *posList, posCount;
    XmString *selectedItems;
    char *category, *lastSlash;
    
    /* Make sure a category is selected */
    posList = NULL;
    posCount =0;
    if (!XmListGetSelectedPos(CategoryListW, &posList, &posCount)) {
	DialogF(DF_WARN, w, 1, "Please select a\ncategory to open",
		"Acknowledged");
	return;
    }
    XtVaGetValues(CategoryListW, XmNselectedItems, &selectedItems, 0);
    category = NULL;
    XmStringGetLtoR(*selectedItems, XmSTRING_DEFAULT_CHARSET, &category);
    
    /* Change the current category to the category selected */
    if (!strncmp(category, "<-- ", 4)) {
    	/* move up in the category hierarchy */
    	lastSlash = strrchr(CurrentCategory, '/');
    	if (lastSlash == NULL)
    	    *CurrentCategory = '\0';
    	else
    	    *lastSlash = '\0';
    } else {
	/* move down the category hierarchy */
	if (strcmp(CurrentCategory, ""))
	    strcat(CurrentCategory, "/");
	strcat(CurrentCategory, category);
    }
    redisplayNTupleList(False);
    XtFree((char *) posList);
    XtFree(category);			
}

static void showIdCB(Widget w, caddr_t clientData, caddr_t callData)
{
       redisplayNTupleList(False);
}
static void viewTemplateCB(Widget w, caddr_t clientData, caddr_t callData)
{
      int id;
      nTuDDL *ddl;
      descrGenNtuple *dNTu;
      nTuBuildWindow *window;      
      
      int listPos = mcfioC_SelectedNtuBrListPos();
      if (listPos == 0) return;
      id = ListedIDs[listPos -1];
      if (NTupleBrowserList[id-1]->templateW == NULL) {
         ddl = mcf_GetNTuByPtrID(id);
         window = 
            CreateNTuBuildWindow(XtDisplay(McfioMainPanelW), ddl->title,
	    					 True); 
         NTupleBrowserList[id-1]->templateW = window; 
	 if (ddl->descrNtu == NULL) dNTu = ddl->reference->descrNtu;
	    else dNTu = ddl->descrNtu;
	 window->descrNtu = dNTu;
         XmTextSetString(window->descriptW, dNTu->description);
         XmTextSetString(window->titleW, dNTu->title);
         XmTextSetString(window->versionW, dNTu->version);
         XmTextSetString(window->nameIndexW, dNTu->nameIndex);
         if (dNTu->maxMultiplicity > 0) 
         SetIntText(window->multiplicityW, dNTu->maxMultiplicity);
         UpdateVariableList(window, 1);
         UpdateDialogVFields(window);
         if (VerifyStruct(window) == True) 
             XtSetSensitive(window->saveAsItem, True);
      }
      else { /* Assume the thing is managed at this point */
         XMapRaised(
             XtDisplay(NTupleBrowserList[id-1]->templateW->shell), 
                       XtWindow(NTupleBrowserList[id-1]->templateW->shell));
      }                 
      
}
static void viewDumpCB(Widget w, caddr_t clientData, caddr_t callData)
{
      int id;
      nTuDDL *ddl;
      int listPos = mcfioC_SelectedNtuBrListPos();
      if (listPos == 0) return;
      id = ListedIDs[listPos -1];
      ddl = mcf_GetNTuByPtrID(CurrentNTupleBrowserSelected->id);
      /*
      ** If first time around, we need to create and managed the Text 
      ** widget... 
      */
      if (NTupleBrowserList[id-1]->dumpDataShellW == NULL) 
        mcfioC_ShowBrowserDataDump(NTupleBrowserList[id-1]);
      if (decodeNTuples() == False) {
	DialogF(DF_ERR, w, 1, "Can not decode request NTuple",
		"Acknowledged");
        McfioPanelState = STOPPED_FILE_OPEN;
        return;
      }
      mcfioC_ShowBrowserDataDump(NTupleBrowserList[id-1]);
}
int mcfioC_SelectedNtuBrListPos()
{
    int listPos, id;
    int *posList = NULL, posCount = 0;

    if (!XmListGetSelectedPos(NTupleListW, &posList, &posCount)) {
        DialogF(DF_WARN, MainContentsW, 1,  
                "Please Select an NTuple from the List in Main Panel\n\
 Histograms are based on Ntuple Descriptors ",
                "Acknowledged");
    	return 0;
    }
    listPos = *posList;
    if (listPos != 0) {
      id = ListedIDs[listPos -1];
       CurrentNTupleBrowserSelected = NTupleBrowserList[id-1];
#ifdef HISTO       
       mcfioC_OneDHistUpdateNTupleContent();
#endif       
    } else  CurrentNTupleBrowserSelected = NULL;
    XtFree((char *)posList);
    return listPos;
}
int  mcfioC_SetSpecificNTupleBr(nTuBrowserInfo *nTuBr) 
{
/*
** Because the user can defined HistoScope items from other panel, 
** for consistency, we may have to set the Ntuple List from 
** outside this Module. This routines attemps to do this
*/
   int i, iPos, nItems;
   nTuDDL *ddl;
   Arg args[2];
   
   ddl = mcf_GetNTuByPtrID(nTuBr->id);
   if (itemInCategory(ddl, CurrentCategory) != True) {
       resetTopCategory();
       DialogF(DF_WARN, MainContentsW, 1,
" You are defining HistoScope items from category to an other \n Please\
 reselect the target category in the Main panel.\nSorry for the inconvenience.",
"Acknowledged");
       return False;
   }
   redisplayItems();
   XtSetArg (args[0], XmNitemCount, &nItems); 
   XtGetValues(NTupleListW, args, 1);
   for (i=0, iPos=-1; i<nItems; i++) 
      if(ListedIDs[i] == ddl->id) iPos = i+1;
   if (iPos == -1) {
       DialogF(DF_WARN, MainContentsW, 1,
"Internal error in mcfioC_SetSpecificNTupleBr\n Please Report. ",
"Acknowledged");
       return False;
    }
    XmListSelectPos(NTupleListW, iPos, False);
    return True;
}

static void sliderTuneCB(Widget w, caddr_t clientData, caddr_t callData)
{
     int requestedEvt, sliderValue ;
     
     if(McfioPanelState == NO_FILE_OPEN) return;
     if (McfioPanelState == SCANNING) SimulateButtonPress(StopBtnW);
     XtVaGetValues(w, XmNvalue, &sliderValue, 0);
     requestedEvt = sliderValue/SLIDER_MAX * TotalNumberOfEvts;
     CurrentEventNumber = requestedEvt; 
     mcfioC_SpecificEvent(1,  CurrentEventNumber, CurrentStoreNumber,
                              CurrentRunNumber,  CurrentTriggerMask);
     
}
static void resetTopCategory(void)
{
    char category[NTU_MAX_CATEGORY_LENGTH+1];
    char firstTopCategory[NTU_MAX_CATEGORY_LENGTH+1];
    int i;

    /* Empty histogram list: top category is the empty string */
    if ((NTuDDLList == NULL) || (NumOfNTuples < 1)) {
    	TopCategory[0] = '\0';
    	return;
    }

    /* Compare the top category of the first item with the top category of all
       of the others.  If any don't match, top category is the empty string */
    nextCategory(firstTopCategory, NTuDDLList[0]->category);
    for (i=1; i<NumOfNTuples; i++) {
    	nextCategory(category, NTuDDLList[i] ->category);
    	if (strcmp(category, firstTopCategory)) {
    	    TopCategory[0] = '\0';
    	    return;
	}
    }
    
    /* Top category for all items is the same, use it as the new top category */
    strcpy(TopCategory, firstTopCategory);

    /* If the current category does not descend from the new top category,
       change it to keep the category menu routine from getting confused.
       Otherwise, leave the current category as is even if no items match */
    nextCategory(category, CurrentCategory);
    if (strcmp(category, firstTopCategory))
	strcpy(CurrentCategory, firstTopCategory);
}
/*
** Redisplay the histogram/ntuple/indicator/control list
** from CurrentCategory and HistoList
*/
static void redisplayItems(void)
{
    int i, nItems = 0;
    XmString *stringTable;
    char entryText[NTU_MAX_TITLE_LENGTH + 16];
    nTuDDL *ddl;

    if (NumOfNTuples == 0) {
       stringTable = (XmString *)XtMalloc((2) * sizeof(XmString));
       stringTable[0] = XmStringCreateSimple("No NTuples in Stream");
       stringTable[1] = (XmString)0;
       setListItems(NTupleListW, stringTable, 1);
      /* free the string table */
       FreeStringTable(stringTable);
       XtSetSensitive(ViewTemplateBtnW, False);
       XtSetSensitive(ViewDumpBtnW, False);
       XtSetSensitive(OpenBtnW, False);
       return;
    } else {
       XtSetSensitive(ViewTemplateBtnW, True);
       XtSetSensitive(OpenBtnW, True);
    }   

    /* Count the number of matching items on the histogram list */
    for (i=0; i<NumOfNTuples; i++)
    	if (itemInCategory(NTuDDLList[i], CurrentCategory))
    	    nItems++;

    /* Allocate enough memory for a string table to supply to the list
       widget, and an id table to map the list positions to ntuple ids */
    if (ListedIDs != NULL)
    	free(ListedIDs);
    if (nItems == 0)			/* JMK - 6/28/93	*/
    	ListedIDs = NULL;
    else
    	ListedIDs = (int *)XtMalloc(sizeof(int) * nItems);
    stringTable = (XmString *)XtMalloc((nItems+1) * sizeof(XmString));
       
    /* Build the id table and the string table */
    nItems = 0;
    for (i=0; i<NumOfNTuples; i++) {
        ddl = NTuDDLList[i];
    	if (itemInCategory(ddl, CurrentCategory)) {
    	    ListedIDs[nItems] = ddl->id;
    	    if (XmToggleButtonGetState(ShowIdBtnW)) {
    	    	sprintf(entryText, "%-11d %s", ddl->uid, ddl->title);
    	    	stringTable[nItems++] = XmStringCreateSimple(entryText);
    	    } else {
    	    	stringTable[nItems++] =  XmStringCreateSimple(ddl->title);
    	    }
    	}
    }
    stringTable[nItems] = (XmString)0; /* end of table is marked with a 0 */
    
    /* Display the items in the histogram list widget */
    setListItems(NTupleListW, stringTable, nItems);

    /* free the string table */
    FreeStringTable(stringTable);
}
/*
** Redisplay the subcategory list from CurrentCategory and Ntuple List
*/
static void redisplaySubcategories(void)
{
    int i, c, duplicate, nCategories = 0;
    char subCat[NTU_MAX_CATEGORY_LENGTH];
    XmString mSubCat, subCategories[NTU_MAX_CATEGORY_DEPTH+2];
    nTuDDL *ddl;
    
    /* Make the entry for moving up in the category hierarchy */
    if (strcmp(CurrentCategory, TopCategory))
    	subCategories[nCategories++] =
    	      XmStringCreateSimple("<-- Up One Level");
    
    /* Fill in the sub-categories from the histogram list */
    for (i=0; i<NumOfNTuples; i++) {
        ddl = NTuDDLList[i];
    	if (subCategory(CurrentCategory, ddl->category, subCat)) {
    	    mSubCat = XmStringCreateSimple (subCat);
    	    /* Check for existing duplicate */
    	    duplicate = False;
    	    for (c = 0; c < nCategories; c++) {
    	    	if (XmStringCompare(subCategories[c], mSubCat)) {
    	    	    duplicate = True;
    	    	    XmStringFree(mSubCat);
    	    	    break;
    	    	}
    	    }
    	    if (!duplicate) {
    	    	subCategories[nCategories++] = mSubCat;
    	    }
    	}
    }
    subCategories[nCategories] = NULL;
    
    /* Display the list of categories in the category list widget */
    setListItems(CategoryListW, subCategories, nCategories);

    /* Free the Xm strings */
    for(i = 0; i<nCategories; i++)
	XmStringFree(subCategories[i]);
}

/*
** Redisplay the category menu from CurrentCategory and Ntuple List
*/
static void redisplayCategoryMenu(void)
{
    char *cats, categories[NTU_MAX_CATEGORY_LENGTH], cat[NTU_MAX_CATEGORY_LENGTH];
    static Widget categoryBtns[NTU_MAX_CATEGORY_DEPTH];
    Widget oldBtns[NTU_MAX_CATEGORY_DEPTH];
    static int nBtns = 0;
    int oldNBtns;
    static Arg args[1] = {{XmNlabelString, (XtArgVal)0}};
    Widget button;
    int i, catNum, nCats = 0;

    /* leave the old buttons in the menu until after the new ones are added.
       (Motif option menus are tempramental, this is necessary on SGI) */
    for (i=0; i<nBtns; i++)
    	oldBtns[i] = categoryBtns[i];
    oldNBtns = nBtns;
    nBtns = 0;
    
    /* If the top category is "", add a fake one called "Uncategorized",
       start the category numbers: 0 == "" 1 == "cat1" 2 == "cat1/cat2" */
    if (!strcmp(TopCategory, "")) {
    	sprintf(categories, "Uncategorized/%s", CurrentCategory);
    	catNum = 0;
    } else {
    	strcpy(categories, CurrentCategory);
    	catNum = 1;
    }
    cats = categories;

    /* loop through the categories creating a new menu button for each */
    while (True) {
    	cats = nextCategory(cat, cats);
    	if (cats == NULL)
    	    break;
    	nCats++;
    	if (args[0].value != 0)
    	    XmStringFree((XmString)args[0].value);
    	args[0].value = (XtArgVal)XmStringCreateSimple(cat);
    	button =  XmCreatePushButton(CategoryMenuW, "btn", args, 1);
    	XtAddCallback(button, XmNactivateCallback,
    		      (XtCallbackProc)categoryMenuCB, (caddr_t)catNum++);
    	XtManageChild(button);
    	categoryBtns[nBtns++] = button;
    }
    /* Safety check, stuff below will bomb if no categories were processed */
    if (nCats == 0)
    	fprintf(stderr, "Internal Error, unreadable current category\n");
    
    /* Set the option menu to show the last category in the list.  Use
       the button widget and same argument list used in the last iteration
       of the loop above to change the position of the option menu and to
       set the label on the label gadget associated with the option menu.
       (In some cases, setting XmNmenuHistory for the option menu is
       is not sufficient to make motif change the label.)	     */
    XtVaSetValues(CategoryOptMenuW, XmNmenuHistory, button, 0);
    XtSetValues(XmOptionButtonGadget(CategoryOptMenuW), args, 1);
    XmStringFree((XmString)args[0].value);
    args[0].value = (XtArgVal)0;

    /* destroy all of the old buttons */
    for (i=0; i<oldNBtns; i++)
    	XtDestroyWidget(oldBtns[i]);
}    

static int itemInCategory(nTuDDL *ddl, char *category)
{
    return !strcmp(category, ddl->category);
}

static char *nextCategory(char *toString, char *fromString)
{
    char *to = toString, *from = fromString;
    
    while (TRUE) {
        if (*from =='\0') {
            *to = '\0';
            if (to == toString)
            	return NULL;		/* No categories left in string */
            else
            	return from;		/* will return NULL on next call */
        } else if (*from=='/') {
	    *to = '\0';
	    if (to == toString)		/* ignore leading '/' and // */
	    	from++;
	    else
	    	return from;
	} else {
	    *to++ = *from++;
	}
    }
}

static int subCategory(char *category, char *categories, char *subCategory)
{
    int catLen;
    
    catLen = strlen(category);
    if (!strncmp(category, categories, catLen))
    	if (nextCategory(subCategory, &categories[catLen]) != NULL)
    	    return True;
    return False;
}
static void setMainPanelState(int state, char * title)
{
    char streamTitle[MCF_XDR_F_TITLE_LENGTH+1];
    int nn, lt;
    XmString s1;
    
    McfioPanelState = state;
    if (state == NO_FILE_OPEN) {
        XtVaSetValues(StreamTitleLabelW, XmNlabelString, 
           s1 = XmStringCreateSimple("No stream Open"), 0);
        XmStringFree(s1);
       resetTopCategory ();
       redisplayNTupleList(True);
       updateStatusLabel(0.);
        XtSetSensitive(McfioMenuShowHeaderW, False);
        XtSetSensitive(McfioMenuShowEvtHeaderW, False);
    }     
    
    XtSetSensitive(MainContentsW, state!=NO_FILE_OPEN);
    if (state == FILE_OPEN) {
        XtSetSensitive(StopBtnW, False);
        XtSetSensitive(RewindBtnW, True);
        XtSetSensitive(NextEventBtnW, True);
        XtSetSensitive(SelectEvtBtnW, True);
        XtSetSensitive(RunBtnW, True);
        XtSetSensitive(McfioMenuShowHeaderW, True);
        mcfioC_InfoStreamChar(1, MCFIO_TITLE, streamTitle, &lt);
        XtVaSetValues(StreamTitleLabelW, XmNlabelString, 
           s1 = XmStringCreateSimple(streamTitle), 0);
        XmStringFree(s1);
       resetTopCategory ();
       mcfioC_InfoStreamInt(1, MCFIO_NUMNTUPLES, &nn);
#ifdef HISTO      
       if (nn > 0) XtSetSensitive(McfioMenuShowOneDHistMainW, True);
#endif       
       redisplayNTupleList(True);
       updateStatusLabel(0.);
       XtSetSensitive(ViewDumpBtnW, False);
       mcfioC_CreateBrowserAnalysis();
    }     
    XtVaSetValues(XtParent(McfioMainPanelW), XmNtitle, title, 0);
}
/*
** Set the contents of a list and compensate for various motif problems
** associated with this apparently bizarre act.  (It might be worth
** further investigation to better understand why these workarounds
** are necessary).
*/
static void setListItems(Widget w, XmString *strings, int nStrings)
{
    XmString *st1;
    
    /* Motif doesn't reset the selection when items are changed */
    XmListDeselectAllItems(w);

    /* .. and sometimes leaves stray scroll bars if nothing is in the list */
    if (nStrings == 0) {
    	XtVaSetValues(w, XmNitems, st1=StringTable(1, " "),
    		XmNitemCount, 1, XmNsensitive, False, 0);
    	FreeStringTable(st1);
    } else {
    	XtVaSetValues(w, XmNitems, strings, XmNitemCount, nStrings,
    		XmNsensitive, True, 0);
    }
}
static void rewindCB(Widget w, caddr_t clientData, caddr_t callData)
{
      if (McfioPanelState == NO_FILE_OPEN) return;
      mcfioC_Rewind(1);
      McfioPanelState = FILE_OPEN;
      CurrentWordCount = InitialWordCount;
      XtSetSensitive(ViewDumpBtnW, False);
      XtSetSensitive(McfioMenuShowEvtHeaderW, False);
      CurrentSeqEvtNum = 0;
      updateStatusLabel(0.);
      
}
static void nextEventCB(Widget w, caddr_t clientData, caddr_t callData)
{
     int i;
      if (McfioPanelState == NO_FILE_OPEN) return;
      PreviousWordCount = CurrentWordCount;
      if (mcfioC_NextEvent(1) != MCFIO_RUNNING) {
         DialogF(DF_WARN, McfioMainPanelW, 1, "Can Not reach Next Event\n\
Please Rewind the stream, or open an other file", "Acknowledged");
         McfioPanelState = FILE_OPEN;
      } else {
         McfioPanelState = STEPPING;
         CurrentSeqEvtNum++;
         updateStatusLabel(0.);
         for (i=0; i<NumOfNTuples; i++) 
             NTupleBrowserList[i]->currentData = False; 
         XtSetSensitive(ViewDumpBtnW, True);
         XtSetSensitive(McfioMenuShowEvtHeaderW, True);
         if (decodeNTuples() == FALSE)  McfioPanelState = STOPPED_FILE_OPEN;
     } 
}

static void selectEvtCB(Widget w, caddr_t clientData, caddr_t callData)
{
     int i, evt, iNum[3];
     char str1[255], *endPtr, *t1;
     
     if (McfioPanelState == NO_FILE_OPEN) return;
     PreviousWordCount = CurrentWordCount;
     McfioPanelState = STEPPING;
     DialogF(DF_PROMPT, McfioMainPanelW, 0, 
 " Please type the event #, store #, run #, trigger Mask # \n\
 You can ommit any of these quantity, but you must type at least the event #\n\
 0 means any, e.g. to get the first event run # 5, type: \n\
 0, 5 ", str1);
     mcfioC_RemoveWhiteSpace(str1);
     evt = strtol(str1, &endPtr, 10);
     if (strlen(str1) == 0) {
	   DialogF(DF_ERR,  McfioMainPanelW, 1,
	    "Please Try again, we need an integer number! ","Acknowledged");
           return;
     }
     t1 = endPtr;
    for (i=0; i<3; i++) iNum[i]=0; i=0;
     while ((strlen(t1) != 0) && (*endPtr != '\0') && (i<3)) {
         if (*t1 != ',' ) {
	   DialogF(DF_ERR, McfioMainPanelW, 1,
	    "Integer number must be separated by , ! ","Acknowledged");
           return;
         }
         t1++;
         iNum[i] = strtol(t1, &endPtr, 10);
         t1 = endPtr; i++; 
     }     
     if (mcfioC_NextSpecificEvent(1,  evt, iNum[0],iNum[1], iNum[2]) !=
           MCFIO_RUNNING) { 
	   DialogF(DF_WARN,  McfioMainPanelW, 1,
	    "The event %s  , is not found! ","Acknowledged", str1);
	   SimulateButtonPress(RewindBtnW);
           return;
     }      
     CurrentSeqEvtNum++;
     updateStatusLabel(0.);
     for (i=0; i<NumOfNTuples; i++) 
             NTupleBrowserList[i]->currentData = False; 
     XtSetSensitive(ViewDumpBtnW, True);
     XtSetSensitive(McfioMenuShowEvtHeaderW, True);
     if (decodeNTuples() == FALSE)  McfioPanelState = STOPPED_FILE_OPEN;
}

static void runCB(Widget w, caddr_t clientData, caddr_t callData)
{
    int i, nSecs, nMuSecs;
    float delta, speed;
    struct timeval timeNow;
    int numProcX, nn = 0;
    XtAppContext context = 
          XtWidgetToApplicationContext(McfioMainPanelW);
    numProcX = 0;  
    gettimeofday(&timeNow, NULL);
    if (McfioPanelState == NO_FILE_OPEN) return;
    McfioPanelState = SCANNING;
    XtSetSensitive(StopBtnW, True);
    EventCountAtLastUpdate = 0;
    TimeLeftBeforeNextUptade = TIME_BETWEEN_STATUS;
    XtSetSensitive(RewindBtnW, False);
    XtSetSensitive(NextEventBtnW, False);
    XtSetSensitive(SelectEvtBtnW, False);
    XtSetSensitive(RunBtnW, False);
    XtSetSensitive(ViewDumpBtnW, True);
    while((McfioPanelState == SCANNING) &&
            (nn < TotalNumberOfEvts)) {
            while (XtAppPending(context)) {
                numProcX++;
    	        XtAppProcessEvent(context, XtIMAll);
    	    }    
          PreviousWordCount = CurrentWordCount;
          if (mcfioC_NextEvent(1) != MCFIO_RUNNING) {
              DialogF(DF_WARN, McfioMainPanelW, 1, "Can Not reach Next Event\n\
Please Rewind the stream, or open an other file", "Acknowledged");
              McfioPanelState = STOPPED_FILE_OPEN;
              XtSetSensitive(StopBtnW, False);
              XtSetSensitive(ViewDumpBtnW, False);
          }
          CurrentSeqEvtNum++;
          nn++;
          /*
          ** Update the Current event, and so forth..
          */
         gettimeofday(&timeNow, NULL);
         nSecs = timeNow.tv_sec - TimeAtPreviousItteration.tv_sec;
         nMuSecs = timeNow.tv_usec - TimeAtPreviousItteration.tv_usec;
         delta = (float) nMuSecs/1000. + (float) nSecs * 1000.;
         TimeLeftBeforeNextUptade -= delta;
         if (TimeLeftBeforeNextUptade <= 0) {
            speed = ((nn - EventCountAtLastUpdate) * 1000.)/ 
                     TIME_BETWEEN_STATUS; 
            updateStatusLabel(speed);
            EventCountAtLastUpdate = nn;
            TimeLeftBeforeNextUptade = TIME_BETWEEN_STATUS;
         }   
         for (i=0; i<NumOfNTuples; i++) 
             NTupleBrowserList[i]->currentData = False; 
         if (decodeNTuples() == FALSE)  McfioPanelState = STOPPED_FILE_OPEN;
         TimeAtPreviousItteration.tv_sec = timeNow.tv_sec;
         TimeAtPreviousItteration.tv_usec = timeNow.tv_usec;
      }
#ifdef HISTO
      if (HistoIsBrowserNtuInit == 1) {
         hs_update();
         hs_update();
      }
#endif            
      XtSetSensitive(StopBtnW, False);
      XtSetSensitive(RewindBtnW, True);
      XtSetSensitive(NextEventBtnW, True);
      XtSetSensitive(SelectEvtBtnW, True);
      XtSetSensitive(RunBtnW, True);
      XtSetSensitive(ViewDumpBtnW, False);
      updateStatusLabel(speed);
}         
      
static void updateStatusLabel(float speed)
{
      char line[180], *tc;
      XmString s1;
      int nc, nn, l, k;
     
     
      if (McfioPanelState == NO_FILE_OPEN) { 
          tc = line;
          sprintf(tc, "------------------------------------ %n", &l); 
      } else if (McfioPanelState == FILE_OPEN) {
          mcfioC_InfoStreamInt(1, MCFIO_NUMEVTS, &TotalNumberOfEvts);
          tc = line;
          sprintf(tc,
 "At Beginning of file.  Total number of Events = %d  %n", 
      TotalNumberOfEvts, &l); 
      if (l < 76)  for (k=l; k<76; k++) line[k] = ' ';
      line[76] = '\n';
      sprintf(&line[77],
 "----------------------------------------------------------------------   ");
      CurrentEventNumber = 0;
      CurrentSeqEvtNum = 0;
     } else {     
        mcfioC_InfoEventInt(1, MCFIO_EVENTNUMBER, &CurrentEventNumber);
        mcfioC_InfoEventInt(1, MCFIO_STORENUMBER, &CurrentStoreNumber);
        mcfioC_InfoEventInt(1, MCFIO_RUNNUMBER, &CurrentRunNumber);
        mcfioC_InfoEventInt(1, MCFIO_TRIGGERMASK, &CurrentTriggerMask);
        mcfioC_InfoStreamInt(1, MCFIO_NUMWORDS, &CurrentWordCount);
        nn = (CurrentWordCount - PreviousWordCount)*4;	
/*
** A label, taking 2lines, the entire length, almost, stating the evt count:
** At event xxxxxx, Run yyyy, Spill yyyy, Trigger Mask = xx 
** Scanning at yyy Kb/sec, expecting to complete in xx minutes  
*/
        tc = line;
        sprintf(tc, "At event %d, store  %d, run %d, Trigger Mask %d %n", 
         CurrentEventNumber, CurrentStoreNumber,
         CurrentRunNumber, CurrentTriggerMask , &l);
        if (l < 76)  for (k=l; k<76; k++) line[k] = ' ';
        line[76] = '\n';
        line[77] = '\0';
        if (McfioPanelState == SCANNING) {
           if (speed < 1.)    sprintf(&line[77], 
   "Read an additional  %d bytes. Crawling at %f evts/sec ",
    nn, speed );
           else if (speed < 100.)    sprintf(&line[77], 
"Read an additional  %d bytes. Grinding at %f evts/sec ",
    nn, speed );
    
           else if (speed < 1000.)    sprintf(&line[77], 
"Read an additional  %d bytes.  Moving at %f evts/sec ",
    nn, speed );
    
           else  sprintf(&line[77], 
"Read an additional  %d bytes.  Speeding at %f evts/sec. ",
    nn, speed );
        } 
        if (McfioPanelState == STEPPING) 
              sprintf(&line[77], 
   "# Read an additional  %d. Byte count into stream = %d ",
    nn, (CurrentWordCount * 4));
     }
        s1 = XmStringCreateLtoR(line, XmSTRING_DEFAULT_CHARSET);
        XtVaSetValues(InputRateLabelW, XmNlabelString, s1, 0);
        XmStringFree(s1);
 /*
 ** Set the Slider bar ..
 */
 	if (TotalNumberOfEvts > 0) {
 	   nn = (CurrentSeqEvtNum * SLIDER_MAX) / TotalNumberOfEvts;
 	   XmScaleSetValue(InputRateSliderW, nn);
 	} else  XmScaleSetValue(InputRateSliderW, 0);  
} 
static int decodeNTuples()
{
      int i;
      for (i=0; i<NumOfNTuples; i++) {
           if (NTupleBrowserList[i]->currentData == False) {
	     if (((NTupleBrowserList[i]->dumpDataW != NULL)  && 
	         (XtIsManaged(NTupleBrowserList[i]->dumpDataFormW))) ||
	         (NTupleBrowserList[i]->nHistoItems > 0)) {
	           if (NTupleBrowserList[i]->data == NULL)
	               mcfioC_createBrowserData(NTupleBrowserList[i]);
	           if (mcfioC_NTuple(1, (i+1), 
	              (char *) NTupleBrowserList[i]->data) == False)
	                   return False;
	           NTupleBrowserList[i]->currentData = True;
	           if((NTupleBrowserList[i]->dumpDataW != NULL) &&
	              (XtIsManaged(NTupleBrowserList[i]->dumpDataFormW)))
	              mcfioC_ShowBrowserDataDump(NTupleBrowserList[i]);
	           /* Here comes the filling  */
#ifdef HISTO	           
	           if(NTupleBrowserList[i]->nHisto1D > 0)
	                 fillOneDHists(NTupleBrowserList[i]); 
#endif	                         
	            }
	      } /* Here comes partial decoding of an NTuple
	           Skip for now, even if we are histogramming only 
	           onely one variable decode the whole block */
      }                    
	
	return True;
}	        
static void stopCB(Widget w, caddr_t clientData, caddr_t callData)
{
      McfioPanelState = STOPPED_FILE_OPEN;
      XtSetSensitive(StopBtnW, False);
      XtSetSensitive(RewindBtnW, True);
      XtSetSensitive(NextEventBtnW, True);
      XtSetSensitive(SelectEvtBtnW, False);
      XtSetSensitive(RunBtnW, True);
} 
#ifdef HISTO
static void fillOneDHists(nTuBrowserInfo *nTuBr)
{

    int i, j, ivar, id, isub, lastFixed, nn, ln, nskip, lll, nsk, *indices;
    float val;
    nTuDDL *ddl;
    descrGenNtuple *dNTu;
    char *cDat;
    long pp, *lDat;
    varGenNtuple *var;
    float weight = 1;
    nTuBroHs1D *nTuH1;

    
    if (nTuBr->data == NULL) return;	
    ddl = mcf_GetNTuByPtrID(nTuBr->id);
    if (ddl->descrNtu == NULL) dNTu = ddl->reference->descrNtu;
	    else dNTu = ddl->descrNtu;
    if(dNTu->firstIndexed == -1) lastFixed = dNTu->numVariables;
           else lastFixed = dNTu->firstIndexed;
    for (i=0; i<nTuBr->sizeOfLists; i++) {
       nTuH1 = (nTuBroHs1D *) nTuBr->hsItemList[i];
       if (nTuH1 == NULL) continue;
       id = nTuH1->id;
       if (nTuH1->type != HS_1D_HISTOGRAM) continue;
       ivar = nTuH1->varNumber;
       cDat = (char *) nTuBr->data;
       if (ivar == 0) {  /* Histogram the multiplicity */
          nn = *((int *)(nTuH1->lDat));
          val =  (float) nn;
          hs_fill_1d_hist(id, val , weight);
          hs_update();
       } else {
          ivar --;  
          var = dNTu->variables[ivar];
          if (var->numDim < 1)  nskip=0; 
          else { /* Assume C arrangement of arrays. 
                       This is a semi-serious deficiency of the ddl */
              indices =  nTuH1->varIndices;            
              for (ln=0, nskip=0; ln <var->numDim; ln++) { 
               for (lll=ln+1, nsk=1; lll<var->numDim; lll++) 
                            nsk = nsk*var->dimensions[lll];
               nskip += nsk * indices[ln];
              } 
          }
          if (ivar < lastFixed)  
            fillOneDHistValue(id, nTuH1->lDat, nskip, var->type);
          else { /* A specific subVariable (leaf) */
            cDat += dNTu->multOffset; 
            lDat = (long *) cDat;
             nn = *lDat;
            if (nTuH1->subBlock < 0) {
               if (dNTu->orgStyle == PARALLEL_ARRAY_NTU) {
                  pp = ((long) nTuBr->data) +  var->offset;
                  lDat = (long *) pp;
                  for (j=0; j<nn; j++) {
                         fillOneDHistValue(id, lDat, nskip, var->type);
                         lDat += var->lengthW;
                  }
	      } else  
                  for (j=0; j<nn; j++) {
                       pp = ((long) nTuBr->data) +  var->offset +
                               dNTu->subOffset[j];
                        lDat = (long *) pp;
                         fillOneDHistValue(id, lDat, nskip, var->type);
                  }
	   } else { /* a specific instance.. */
               if (dNTu->orgStyle == PARALLEL_ARRAY_NTU) {
                  pp = ((long) nTuBr->data) +  var->offset;
                  lDat = (long *) pp; j=0;
	          while(j <(nTuH1->subBlock - 1)) { 
	               lDat += var->lengthW; j++;
	           } 
	       } else {
                    pp = ((long) nTuBr->data) +  var->offset +
                               dNTu->subOffset[nTuH1->subBlock]; 
                    lDat = (long *) pp;
	        } /* End of specific organization */
                fillOneDHistValue(id, lDat, nskip, var->type);
            } /* End og a specific subblock instance */
          } /*  End of specific SubVariable  (leaf)*/
       } /* End of a specific variable, Fixed size or Variable size */
    } /* End over Histograms list within the Ntuple */
}

static void hsResetCB(Widget w, caddr_t clientData, caddr_t callData)
{
    int jn, i, id;
    nTuBrowserInfo *nTuBr;
    nTuBroHsGeneral *nTuHs;
    
    for (jn=0; jn<NumOfNTuples; jn++) {
        nTuBr = NTupleBrowserList[jn];
        for (i=0; i<nTuBr->sizeOfLists; i++) {
           nTuHs  = nTuBr->hsItemList[i];
           if( nTuHs != NULL)  hs_reset(nTuHs->id);
           hs_update();
        }
    }
}           
    
static void fillOneDHistValue(int hs_id, long *lDat, int nskip, int type)
{
   int *iDat;
   float val, *fDat; 
   double *gDat;
   char *cDat;
   short *sDat;
   float weight = 1.;

    switch(type) {
       case BYTE_NTU: case CHARACTER_NTU:
          cDat = (char *) lDat;
          if (nskip > 0) cDat += nskip;
          val = (float) *cDat;
          break;
       case INTEGER2_NTU:
          sDat = (short *) lDat;
          if (nskip > 0) sDat += nskip;
          val = (float) *sDat;
          break;
       case INTEGER_NTU:
          iDat = (int *) lDat;
          if (nskip > 0) iDat += nskip;
          val = (float) *iDat;
          break;
       case REAL_NTU:
          fDat = (float *) lDat;
          if (nskip > 0) fDat += nskip;
          val = *fDat;
          break;
       case DBL_PRECISION_NTU:
          gDat = (double *) lDat;
          if (nskip > 0) gDat += nskip;
          val = (float) *gDat;
          break;
       default:
          val = 0.; /* Complex and double precision complex not supported */
    }
    hs_fill_1d_hist(hs_id, val, weight);
    hs_update();

}
#endif     
static Boolean anyUpper(char *string)
{
    int i, stringLen = strlen(string);
    
    for (i = 0; i < stringLen; ++i)
    	if (isalpha(string[i]) && isupper(string[i]))
    	    return True;
    return False;
}
