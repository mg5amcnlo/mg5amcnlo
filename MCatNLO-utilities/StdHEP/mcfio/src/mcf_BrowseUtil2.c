/*******************************************************************************
*									       *
* mcf_BrowseUtil2.c -- Utilities and auxillary Panels for NTuple Browser.      *
*	Gives a short listing of the characteristic of the file header         *
*									       *
* Copyright (c) 1995, 1996 Universities Research Association, Inc.	       *
* All rights reserved.							       *
* 									       *
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/param.h>
#include <rpc/types.h>
#include <sys/types.h>
#include <rpc/xdr.h>
#include <ctype.h>
#include <limits.h>
#include <X11/Intrinsic.h>
#include <X11/Shell.h>
#include <Xm/Xm.h>
#include <Xm/DialogS.h>
#include <Xm/SelectioB.h>
#include <Xm/RowColumn.h>
#include <Xm/ToggleB.h>
#include <Xm/Form.h>
#include <Xm/List.h>
#include <Xm/Label.h>
#include <Xm/LabelG.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>
#include <Xm/Text.h>
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
#include "mcf_BrowseUtil1.h"
#include "mcf_BrowseMainPanel.h"
#include "mcf_BrowseUtil2.h"
#ifdef HISTO
#include "histoscope.h"
#endif

extern nTuDDL **NTuDDLList;
extern int NumOfNTuples;
extern nTuBrowserInfo *CurrentNTupleBrowserSelected;
extern Widget McfioHsResetBtnW; /* To reset all existing histograms */

int HistoIsBrowserNtuInit;
                /* A flag to state if HistoScope is set */ 

static Widget DumpHeadTextW = NULL;
		 /* The text widget to view the header content */
static Widget DumpHeadFormW = NULL;
	        /*  The form widget to view the header content */
static Widget DumpHeadShellW = NULL;
	        /*  The shell widget to view the header content */
static struct timeval TimeSetTimeOut = {0, 0};
static int TimeOutTime = 0;
static XtIntervalId TimeOutID = 0;

/*
* Widgets and stuff for 1D histogram utilities
*/
static Widget OneDHistFormW  = NULL;
      /* The form widget associated with the panel for 1D histo utilities */       
static Widget OneDHistShellW  = NULL;
      /* The form widget associated with the panel for 1D histo utilities */ 
static Widget OneDHistHistoListW = NULL;
      /* A list to manipulate to manipulate these histograms */
static Widget OneDHistNtupleCategoryLabelW;
      /* To display the Ntuple Category Label */
static Widget OneDHistNtupleTitleLabelW;
      /* To display the Ntuple Title  Label */
static Widget OneDHistVariableListW = NULL;
      /* The list of variable corresponding to the selected Ntuple */
static Widget OneDHistCreateW = NULL;
      /* The button to create a 1D histogram */
static Widget OneDHistModifyW = NULL; /* To modify a 1D histogram */
static Widget OneDHistDeleteW = NULL; /* To create a 1D histogram */
static Widget OneDHistTitleW = NULL; /* The title field for OneDhist */
static Widget OneDHistNumBinsW = NULL; /* The title field for OneDhist */
static Widget OneDHistLowBinW = NULL; /* The title field for OneDhist */
static Widget OneDHistHighBinW = NULL; /* The title field for OneDhist */
static Widget OneDHistMultW = NULL; /* The title field for OneDhist */
static int FirstOneDHistogram;
static int OneDHistNumberListed = 0; 
	/* The number of 1D histogram managed by this panel */
	
static int OneDHistSizeOfList = 0; 
	
static int *OneDHistListedIDs = NULL; 
        /* The hs ids of the listed histograms */
static int CurrentHistoUID = 0 ; 
	/* A holding place to kee ptrack of the modified histo UID */
static void showHeaderData(int all);
static void createBrowserHeadDump();
static void dismissHeadDumpCB(Widget w, nTuBrowserInfo *nTuBr,
                               caddr_t call_data);
                               
static void createBrowserHeadDump();
static void dismissHeadDumpCB(Widget w, nTuBrowserInfo *nTuBr,
                               caddr_t call_data);
#ifdef HISTO                               
static void createOneDHistCB(Widget w, nTuBrowserInfo *nTuBr, 
				caddr_t call_data);
				
static void createOneDHistActual(int newHisto);
static void createBrowserOneDHistPanel(nTuBrowserInfo *nTuBr);
static void dismissOneDHistCB(Widget w, nTuBrowserInfo *nTuBr,
                               caddr_t call_data);
static void modifyOneDHistCB(Widget w, nTuBrowserInfo *nTuBr,
                               caddr_t call_data);
                               
static void deleteOneDHistCB(Widget w, nTuBrowserInfo *nTuBr,
                               caddr_t call_data);
static void redisplay1DHistoList();
static void oneDHistListCB(Widget w, nTuBrowserInfo *nTuBr,
                               caddr_t call_data);
static nTuBrowserInfo *oneDHistNTupleSource(int hs_id);
#endif                               
static void getVarIndexDialog(int *loc, int nDim, int *maxVal);
static void setListItems(Widget w, XmString *strings, int nStrings);
static void      setTimer(int timeInMillis);
static void      cancelTimer(void);
static void      timerUpCB(void);
/*
** Some scrap space to define histograms
*/
static char BrowseUtil2htitle[255];
                                                              
/*
** Remove the white space (blanks and tabs) from a string
*/
void mcfioC_RemoveWhiteSpace(char *string)
{
    char *outPtr = string;
    
    while (TRUE) {
    	if (*string != ' ' && *string != '\t')
	    *(outPtr++) = *(string++);
	else
	    string++;
    	if (*string == 0) {
	    *outPtr = 0;
	    return;
	}
    }
}

void mcfioC_ShowBrowserHeadDump() {

   if (DumpHeadShellW == NULL) createBrowserHeadDump();
   showHeaderData(False);
}

       
void mcfioC_ShowBrowserEvtHeadDump() {

   if (DumpHeadShellW == NULL) createBrowserHeadDump();
   showHeaderData(True);
}
#ifdef HISTO
void mcfioC_ShowBrowserOneDHist()
{

   char *text, *t1, *t2;
   int i, nChar, len, n1, n2, iln,  *blockList;
   nTuBrowserInfo *nTuBr;
   
   nTuBr = CurrentNTupleBrowserSelected;
   if (OneDHistShellW == NULL) createBrowserOneDHistPanel(nTuBr);
   if (!XtIsManaged(OneDHistFormW)) XtManageChild(OneDHistFormW);
   /*
   ** We have to set the selection in the Main Panel to this requested 
   ** Ntuple.
   */
   if (CurrentNTupleBrowserSelected != nTuBr) {
      if  (mcfioC_SetSpecificNTupleBr(nTuBr) != True) return;
   }
   mcfioC_OneDHistUpdateNTupleContent();
}
void mcfioC_OneDHistUpdateNTupleContent()
{
/*
** Update the Data Descriptioon in the 1D Histogram panel
*/
   int i, id, nItems, l, uid, k;
   XmString *stringTable;
   nTuDDL *ddl;
   descrGenNtuple *dNTu;
   varGenNtuple *var;
   Arg args[2];
   XmString s1;
   
   if (OneDHistShellW == NULL) return;
   if (!XtIsManaged(OneDHistFormW)) return;
   ddl = mcf_GetNTuByPtrID(CurrentNTupleBrowserSelected->id);
   if (ddl->descrNtu == NULL) dNTu = ddl->reference->descrNtu;
	    else dNTu = ddl->descrNtu;
   /*
   ** Update the Label fields 
   */
   XtSetArg(args[0], XmNlabelString, s1 = XmStringCreateSimple(ddl->category));
   XtSetValues(OneDHistNtupleCategoryLabelW, args, 1);
   XmStringFree(s1);
   
   XtSetArg(args[0], XmNlabelString, s1 = XmStringCreateSimple(ddl->title));
   XtSetValues(OneDHistNtupleTitleLabelW, args, 1);
   XmStringFree(s1);
   
   nItems = dNTu->numVariables + 1; 
   stringTable = (XmString *)XtMalloc((nItems+1) * sizeof(XmString));
   stringTable[0] = XmStringCreateSimple ("Multiplicity");
   for (k=0; k<dNTu->numVariables; k++) {
      var = dNTu->variables[k];
      stringTable[k+1] = XmStringCreateSimple (var->name);
    }
    stringTable[nItems] = (XmString ) NULL;
    setListItems(OneDHistVariableListW, stringTable, nItems);
    FreeStringTable(stringTable);
    return;
}
#endif
   
static void showHeaderData(int all)
{

   char *text, *t1, *t2;
   int i, nChar, len, n1, n2, n11, n12, iln,  *blockList, i1, i2, i3;
   nTuDDL *ddl;
   descrGenNtuple *dNTu;
   nTuBrowserInfo *nTuBr;


   /*
   ** set the text widget From, To, to go for a specific track  
   ** or substructures
   */
   mcfioC_InfoStreamInt(1, MCFIO_NUMBLOCKS, &n1);
   nChar = 2*MCF_XDR_F_TITLE_LENGTH + 81*6 + 80*( n1/10);
   if (all) {
       mcfioC_InfoEventInt(1, MCFIO_NUMBLOCKS, &n11);
       mcfioC_InfoEventInt(1, MCFIO_NUMNTUPLES, &n12);
      nChar += 800 + 80*n12 + 80*(n11/10);
   }    
   text = (char *) malloc(sizeof(char) * nChar);
   t1 = text;
   sprintf(t1, "Title : %n",&len); t1+=len;
   mcfioC_InfoStreamChar(1,MCFIO_TITLE, t1, &len); t1 +=len;
   sprintf (t1, " \n%n",&len); t1+=len;
   sprintf (t1, " \n%n",&len); t1+=len;
   
   sprintf(t1, "Comment : %n",&len); t1+=len;
   mcfioC_InfoStreamChar(1,MCFIO_COMMENT, t1, &len); t1 +=len;
   sprintf (t1, " \n%n",&len); t1+=len;
   sprintf (t1, " \n%n",&len); t1+=len;
   
   sprintf(t1, "Creation Date : %n",&len); t1+=len;
   mcfioC_InfoStreamChar(1,MCFIO_CREATIONDATE, t1, &len); t1 +=len;
   sprintf (t1, " \n%n",&len); t1+=len;
   
   sprintf(t1, "Closing Date : %n",&len); t1+=len;
   mcfioC_InfoStreamChar(1,MCFIO_CLOSINGDATE, t1, &len); t1 +=len;
   sprintf (t1, " \n%n",&len); t1+=len;
   
   mcfioC_InfoStreamInt(1, MCFIO_NUMEVTS, &n2);
   if (n2 <= 0) {
     sprintf(t1,
     "The number of events found on the file is 0 or unknown \n%n",&len);
     t1+=len;
     sprintf(t1,
     "  It is likely that this file hasn't been closed properly \n%n",&len);
     t1+=len;
    } else 
     sprintf(t1, "Number of events : %d \n%n", n2, &len); t1+=len;
   sprintf (t1, " \n%n",&len); t1+=len;
   
   sprintf(t1, "Number of private blocks : %d \n%n",n1, &len); t1+=len;
   blockList = (int *) malloc(sizeof(int) * n1);
   mcfioC_InfoStreamInt(1, MCFIO_BLOCKIDS, blockList);
   sprintf(t1, " List of blocks I.D. : %n", &len); t1 +=len;
   iln = 2;
   for (i=0; i<n1; i++, iln++) {
     if (i == (n1-1))
         sprintf(t1," %d %n", blockList[i], &len); 
     else 
         sprintf(t1," %d, %n", blockList[i], &len);
     t1 +=len; 
     if (iln % 8 == 0) {
          sprintf(t1, " \n%n", &len);  t1 +=len;
     }
   } 
   sprintf(t1, " \n%n", &len);  t1 +=len;
   free(blockList);
   if (all) {
     sprintf(t1,
" ==============================================================\n\n%n",&len);
     t1 +=len;
     mcfioC_InfoEventInt(1, MCFIO_EVENTNUMBER, &i1);
     mcfioC_InfoEventInt(1, MCFIO_STORENUMBER, &i2);
     mcfioC_InfoEventInt(1, MCFIO_RUNNUMBER, &i3);
     sprintf(t1,
  "Header Information for event %d, store %d, run %d \n%n", i1, i2, i3, &len);
     t1 +=len;
     mcfioC_InfoEventInt(1, MCFIO_TRIGGERMASK, &i1);
     sprintf(t1, "Trigger Mask: %d \n%n", i1, &len); t1 +=len;
     sprintf(t1, "Number of private blocks : %d \n%n", n11, &len); t1+=len;
     blockList = (int *) malloc(sizeof(int) * n11);
     mcfioC_InfoEventInt(1, MCFIO_BLOCKIDS, blockList);
     sprintf(t1, " List of blocks I.D. : %n", &len); t1 +=len;
     iln = 2;
     for (i=0; i<n11; i++, iln++) {
       if (i == (n1-1))
           sprintf(t1," %d %n", blockList[i], &len); 
       else 
           sprintf(t1," %d, %n", blockList[i], &len);
       t1 +=len; 
       if (iln % 8 == 0) {
          sprintf(t1, " \n%n", &len);  t1 +=len;
       }
     }
     free(blockList);
     
     sprintf(t1, "Number of Ntuples : %d \n%n", n12, &len); t1+=len;
     blockList = (int *) malloc(sizeof(int) * n12);
     mcfioC_InfoEventInt(1, MCFIO_NTUPLESLIST, blockList);
     sprintf(t1, 
     " List of Ntuple Category and titles in this event: \n%n", &len); 
     t1 +=len;
     for (i=0; i<n12; i++) {
       ddl = mcf_GetNTuByPtrID(blockList[i]);
       if (ddl->descrNtu == NULL) dNTu = ddl->reference->descrNtu;
       sprintf(t1,"      %s   :   %s\n%n", ddl->category, ddl->title, &len); 
       t1 += len;     
     }
     free(blockList);
   }
    
   XmTextSetString(DumpHeadTextW,text);
   XtManageChild(DumpHeadFormW);
   free(text);
}
   static void createBrowserHeadDump()
{
    Arg args[50];
    int ac;
    XmString s1;
    Widget dismissBtn;

    ac = 0;
    XtSetArg(args[ac], XmNautoUnmanage, False); ac++; 
    XtSetArg(args[ac], XmNresizePolicy, XmRESIZE_NONE); ac++; 
    DumpHeadFormW = XmCreateFormDialog(McfioMainPanelW, "form", args, ac);
    DumpHeadShellW = XtParent(DumpHeadFormW);
    XtVaSetValues(DumpHeadShellW, XmNtitle,
                                "File Header Information", 0);
    AddMotifCloseCallback(XtParent(DumpHeadFormW),
                         (XtCallbackProc)dismissHeadDumpCB, NULL);
                         
                         
    dismissBtn = XtVaCreateManagedWidget("dismissBtn",
    	    xmPushButtonGadgetClass, DumpHeadFormW,
    	    XmNlabelString, s1=XmStringCreateSimple("Dismiss"),
    	    XmNbottomAttachment, XmATTACH_FORM,
    	    XmNbottomOffset, 4,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 25,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 75, 0);
    XmStringFree(s1);
    XtAddCallback(dismissBtn, XmNactivateCallback,
    	    (XtCallbackProc)dismissHeadDumpCB, NULL); 
    
    ac = 0;
    XtSetArg(args[ac], XmNrows, 8); ac++;
    XtSetArg(args[ac], XmNcolumns, 80); ac++;
    XtSetArg(args[ac], XmNeditMode, XmMULTI_LINE_EDIT); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_POSITION); ac++;
    XtSetArg(args[ac], XmNrightAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomWidget, dismissBtn); ac++;
    XtSetArg(args[ac], XmNdeleteResponse, XmDO_NOTHING); ac++;
    DumpHeadTextW = XmCreateScrolledText(DumpHeadFormW,
                                          "HeaderText", args, ac);
    XtManageChild(DumpHeadTextW);
    XtManageChild(DumpHeadFormW);
}
    
static void dismissHeadDumpCB(Widget w, nTuBrowserInfo * nTuBr,
                               caddr_t call_data)
{
    XtUnmanageChild(DumpHeadFormW);
}
#ifdef HISTO
static void createBrowserOneDHistPanel(nTuBrowserInfo *nTuBr)
{
    Arg args[50];
    int ac;
    XmString s1, *st1;
    Widget dismissBtn, nbinLabel, lowEdgeLabel, highEdgeLabel, multLabel;
    Widget histForm, ntupleForm, titleLabel, histLabel, ntupleLabel;
    Widget ntupleCatLabel, ntupleTitleLabel;

    ac = 0;
    XtSetArg(args[ac], XmNautoUnmanage, False); ac++; 
    XtSetArg(args[ac], XmNresizePolicy, XmRESIZE_NONE); ac++; 
    OneDHistFormW = XmCreateFormDialog(McfioMainPanelW, "form", args, ac);
    OneDHistShellW = XtParent(OneDHistFormW);
    XtVaSetValues(OneDHistShellW, XmNtitle,
                                "One Dimensional Histogram Utility", 0);
    AddMotifCloseCallback(XtParent(OneDHistFormW),
                         (XtCallbackProc)dismissOneDHistCB, NULL);
                         
   OneDHistCreateW = XtVaCreateManagedWidget("create1DBtn",
    	    xmPushButtonGadgetClass, OneDHistFormW,
    	    XmNlabelString, s1=XmStringCreateSimple("Create"),
    	    XmNbottomAttachment, XmATTACH_FORM,
    	    XmNbottomOffset, 4,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 2,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 18, 0);
    XmStringFree(s1);
    XtAddCallback( OneDHistCreateW, XmNactivateCallback,
    	    (XtCallbackProc)createOneDHistCB, (nTuBrowserInfo *) nTuBr );
                         
                         
   OneDHistModifyW = XtVaCreateManagedWidget("modify1DBtn",
    	    xmPushButtonGadgetClass, OneDHistFormW,
    	    XmNlabelString, s1=XmStringCreateSimple("Modify"),
    	    XmNbottomAttachment, XmATTACH_FORM,
    	    XmNbottomOffset, 4,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 20,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 40, 0);
    XmStringFree(s1);
    XtAddCallback( OneDHistModifyW, XmNactivateCallback,
    	    (XtCallbackProc)modifyOneDHistCB, NULL);
    	    
   OneDHistDeleteW = XtVaCreateManagedWidget("delete1DBtn",
    	    xmPushButtonGadgetClass, OneDHistFormW,
    	    XmNlabelString, s1=XmStringCreateSimple("Delete"),
    	    XmNbottomAttachment, XmATTACH_FORM,
    	    XmNbottomOffset, 4,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 42,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 60, 0);
    XmStringFree(s1);
    XtAddCallback( OneDHistDeleteW, XmNactivateCallback,
    	    (XtCallbackProc)deleteOneDHistCB, NULL);
    	    
    dismissBtn = XtVaCreateManagedWidget("dismissBtn",
    	    xmPushButtonGadgetClass, OneDHistFormW,
    	    XmNlabelString, s1=XmStringCreateSimple("Dismiss"),
    	    XmNbottomAttachment, XmATTACH_FORM,
    	    XmNbottomOffset, 4,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 80,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 98, 0);
    XmStringFree(s1);
    XtAddCallback(dismissBtn, XmNactivateCallback,
    	    (XtCallbackProc)dismissOneDHistCB, NULL);

    ac = 0;
    XtSetArg(args[ac], XmNlabelString, 
     (s1=XmStringCreateSimple("Number of bins:"))); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftOffset,5); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomWidget,OneDHistCreateW); ac++;
    XtSetArg(args[ac], XmNbottomOffset,3); ac++;
    nbinLabel = 
       XmCreateLabelGadget(OneDHistFormW, "num1Dbin", args, ac);
    XmStringFree(s1);
    XtManageChild(nbinLabel);
    
    OneDHistNumBinsW = XtVaCreateManagedWidget("numBin1D",
    	    xmTextWidgetClass, OneDHistFormW,
            XmNcolumns, 5,
            XmNmaxLength, 5,
            XmNeditMode, XmSINGLE_LINE_EDIT,
    	    XmNbottomAttachment, XmATTACH_WIDGET,
    	    XmNbottomWidget,OneDHistCreateW,
    	    XmNbottomOffset,3,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftWidget,nbinLabel, 0);
    RemapDeleteKey(OneDHistNumBinsW);
    XmTextSetString(OneDHistNumBinsW, "100");
    
    
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, 
     (s1=XmStringCreateSimple("Low Edge:"))); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftWidget,OneDHistNumBinsW); ac++;
    XtSetArg(args[ac], XmNleftOffset,3); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomWidget,OneDHistCreateW); ac++;
    XtSetArg(args[ac], XmNbottomOffset,3); ac++;
    lowEdgeLabel = 
       XmCreateLabelGadget(OneDHistFormW, "low1Dbin", args, ac);
    XmStringFree(s1);
    XtManageChild(lowEdgeLabel);
    
    OneDHistLowBinW = XtVaCreateManagedWidget("lowEdge1D",
    	    xmTextWidgetClass, OneDHistFormW,
            XmNcolumns, 10,
            XmNmaxLength, 20,
            XmNeditMode, XmSINGLE_LINE_EDIT,
    	    XmNbottomAttachment, XmATTACH_WIDGET,
    	    XmNbottomWidget,OneDHistCreateW,
    	    XmNbottomOffset,3,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftWidget,lowEdgeLabel, 0);
    RemapDeleteKey(OneDHistLowBinW);
    XmTextSetString(OneDHistLowBinW, "0.");
        	    
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, 
     (s1=XmStringCreateSimple("High Edge:"))); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftWidget,OneDHistLowBinW); ac++;
    XtSetArg(args[ac], XmNleftOffset,3); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomWidget,OneDHistCreateW); ac++;
    XtSetArg(args[ac], XmNbottomOffset,3); ac++;
    highEdgeLabel = 
       XmCreateLabelGadget(OneDHistFormW, "high1Dbin", args, ac);
    XmStringFree(s1);
    XtManageChild(highEdgeLabel);
    
    OneDHistHighBinW = XtVaCreateManagedWidget("highEdge1D",
    	    xmTextWidgetClass, OneDHistFormW,
            XmNcolumns, 10,
            XmNmaxLength, 20,
            XmNeditMode, XmSINGLE_LINE_EDIT,
    	    XmNbottomAttachment, XmATTACH_WIDGET,
    	    XmNbottomWidget,OneDHistCreateW,
    	    XmNbottomOffset,3,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftWidget,highEdgeLabel, 0);
    RemapDeleteKey(OneDHistHighBinW);
    XmTextSetString(OneDHistHighBinW, "1.");
    
    
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, 
     (s1=XmStringCreateSimple("Instance:"))); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftWidget,OneDHistHighBinW); ac++;
    XtSetArg(args[ac], XmNleftOffset,3); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomWidget,OneDHistCreateW); ac++;
    XtSetArg(args[ac], XmNbottomOffset,3); ac++;
    multLabel = 
       XmCreateLabelGadget(OneDHistFormW, "inst1Dlabel", args, ac);
    XmStringFree(s1);
    XtManageChild(multLabel);
    
    OneDHistMultW = XtVaCreateManagedWidget("inst1DTw",
    	    xmTextWidgetClass, OneDHistFormW,
            XmNcolumns, 10,
            XmNmaxLength, 20,
            XmNeditMode, XmSINGLE_LINE_EDIT,
    	    XmNbottomAttachment, XmATTACH_WIDGET,
    	    XmNbottomWidget,OneDHistCreateW,
    	    XmNbottomOffset,3,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftWidget,multLabel, 0);
    RemapDeleteKey(OneDHistMultW);
    XmTextSetString(OneDHistMultW, "All");
    
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, 
     (s1=XmStringCreateSimple("Title:"))); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftOffset,3); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomWidget,OneDHistNumBinsW); ac++;
    XtSetArg(args[ac], XmNbottomOffset,3); ac++;
    titleLabel = 
       XmCreateLabelGadget(OneDHistFormW, "title1Dlabel", args, ac);
    XmStringFree(s1);
    XtManageChild(titleLabel);
    
    OneDHistTitleW = XtVaCreateManagedWidget("title1DTw",
    	    xmTextWidgetClass, OneDHistFormW,
            XmNcolumns, 80,
            XmNmaxLength, 255,
            XmNeditMode, XmSINGLE_LINE_EDIT,
    	    XmNbottomAttachment, XmATTACH_WIDGET,
    	    XmNbottomWidget,OneDHistNumBinsW,
    	    XmNbottomOffset,3,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftWidget,titleLabel, 0);
    RemapDeleteKey(OneDHistTitleW);
    XmTextSetString(OneDHistTitleW, "Please enter a meaningful title here");
    /*
    ** Now the list of 1D histograms.
    */
    /* Create a form to hold the list and the top label */
    ac = 0;
    XtSetArg(args[ac], XmNmarginHeight, 0); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomWidget, OneDHistTitleW); ac++;
    XtSetArg(args[ac], XmNbottomOffset, 3); ac++;
    XtSetArg(args[ac], XmNleftOffset, 2); ac++;
    XtSetArg(args[ac], XmNtopOffset, 2); ac++;
    histForm = XmCreateForm(OneDHistFormW, "hist1DForm", args, ac);
    XtManageChild(histForm);

    ac = 0;
    XtSetArg(args[ac], XmNlabelString, 
     (s1=XmStringCreateSimple(
     "One Dimensional Histogram Listing"))); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftOffset,3); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
    histLabel = 
       XmCreateLabelGadget(histForm, "HIST1Dlabel", args, ac);
    XmStringFree(s1);
    XtManageChild(histLabel);
    
    ac = 0;
    XtSetArg(args[ac], XmNitems, (st1=StringTable(1,
     "No histogram defined                        "))); ac++;
    XtSetArg(args[ac], XmNitemCount, 1); ac++;
    XtSetArg(args[ac], XmNscrollBarDisplayPolicy, XmAS_NEEDED); ac++;
    XtSetArg(args[ac], XmNvisibleItemCount, 10); ac++;
    XtSetArg(args[ac], XmNselectionPolicy, XmBROWSE_SELECT); ac++;
    XtSetArg(args[ac], XmNlistSizePolicy, XmCONSTANT); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNrightAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNtopWidget, histLabel); ac++;
    XtSetArg(args[ac], XmNtopOffset, 0); ac++;
    XtSetArg(args[ac], XmNleftOffset, 3); ac++;
    XtSetArg(args[ac], XmNbottomOffset, 4); ac++;
    OneDHistHistoListW = XmCreateScrolledList(histForm, "hist1DList", args,ac);
    FreeStringTable(st1);
    XtAddCallback(OneDHistHistoListW, XmNbrowseSelectionCallback,
                  (XtCallbackProc)  oneDHistListCB, NULL);
    XtManageChild(OneDHistHistoListW);
    OneDHistSizeOfList = 10;
    OneDHistNumberListed = 0;
    OneDHistListedIDs = (int *) malloc(sizeof(int) * OneDHistSizeOfList);
    /*
    ** Now the list of Ntuple Variables. Also a few Label to refer to the 
    ** Ntuple category/title listed on the Main panel.
    ** 
    */
    /* Create a form to hold the list and the top label */
    ac = 0;
    XtSetArg(args[ac], XmNmarginHeight, 0); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftWidget, histForm); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomWidget, OneDHistTitleW); ac++;
    XtSetArg(args[ac], XmNbottomOffset, 3); ac++;
    XtSetArg(args[ac], XmNleftOffset, 6); ac++;
    XtSetArg(args[ac], XmNtopOffset, 2); ac++;
    ntupleForm = XmCreateForm(OneDHistFormW, "ntu1DForm", args, ac);
    XtManageChild(ntupleForm);

    ac = 0;
    XtSetArg(args[ac], XmNlabelString, 
     (s1=XmStringCreateSimple("Selected NTuple Synopsis"))); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftOffset,3); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
    ntupleLabel = 
       XmCreateLabelGadget(ntupleForm, "NTU1DDlabel", args, ac);
    XmStringFree(s1);
    XtManageChild(ntupleLabel);
    
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, 
     (s1=XmStringCreateSimple("Category:"))); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftOffset, 3); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopOffset, 1); ac++;
    XtSetArg(args[ac], XmNtopWidget,ntupleLabel ); ac++;
    ntupleCatLabel = 
       XmCreateLabelGadget(ntupleForm, "NTU1Dlabel", args, ac);
    XmStringFree(s1);
    XtManageChild(ntupleCatLabel);
    
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, 
     (s1=XmStringCreateSimple("None----------------------"))); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftWidget,ntupleCatLabel ); ac++;
    XtSetArg(args[ac], XmNleftOffset,3); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopOffset, 1); ac++;
    XtSetArg(args[ac], XmNtopWidget,ntupleLabel ); ac++;
    OneDHistNtupleCategoryLabelW = 
       XmCreateLabelGadget(ntupleForm, "NTU1DCatlabel", args, ac);
    XmStringFree(s1);
    XtManageChild(OneDHistNtupleCategoryLabelW);
    
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, 
     (s1=XmStringCreateSimple("Title:"))); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftOffset,3); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopWidget, ntupleCatLabel); ac++;
    XtSetArg(args[ac], XmNtopOffset, 1); ac++;
    ntupleTitleLabel = 
       XmCreateLabelGadget(ntupleForm, "NTU1Dlabel", args, ac);
    XmStringFree(s1);
    XtManageChild(ntupleTitleLabel);
    
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, 
     (s1=XmStringCreateSimple("None----------------------"))); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftWidget,ntupleCatLabel ); ac++;
    XtSetArg(args[ac], XmNleftOffset,3); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopWidget,ntupleCatLabel ); ac++;
    XtSetArg(args[ac], XmNtopOffset, 1); ac++;
    OneDHistNtupleTitleLabelW = 
       XmCreateLabelGadget(ntupleForm, "NTU1DTitlabel", args, ac);
    XmStringFree(s1);
    XtManageChild(OneDHistNtupleTitleLabelW);
    
    ac = 0;
    XtSetArg(args[ac], XmNitems, (st1=StringTable(1, " "))); ac++;
    XtSetArg(args[ac], XmNitemCount, 1); ac++;
    XtSetArg(args[ac], XmNscrollBarDisplayPolicy, XmAS_NEEDED); ac++;
    XtSetArg(args[ac], XmNvisibleItemCount, 10); ac++;
    XtSetArg(args[ac], XmNselectionPolicy, XmBROWSE_SELECT); ac++;
    XtSetArg(args[ac], XmNlistSizePolicy, XmCONSTANT); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNrightAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNtopWidget, OneDHistNtupleTitleLabelW); ac++;
    XtSetArg(args[ac], XmNtopOffset, 2); ac++;
    XtSetArg(args[ac], XmNleftOffset, 3); ac++;
    OneDHistVariableListW = XmCreateScrolledList(ntupleForm, "hist1DList", args,ac);
    FreeStringTable(st1);
    XtManageChild(OneDHistVariableListW);
    FirstOneDHistogram = True;
    XtManageChild(OneDHistFormW);
    
    	    
}
static void createOneDHistCB(Widget w, nTuBrowserInfo * nTuBrDummy,
                               caddr_t call_data) 
{
    createOneDHistActual(True);
}
static void createOneDHistActual(int newHisto)
{
   int countHisto, countNTu, countVar, ivar, ivarH, nBins, idh, uid, im, jn;
   int i, index, k, *ivk, ll, lastFixed;
   float from, to;
   double dfrom, dto;
   XmString *selectedItems;
   int *posListHisto, *posListNTu, *posListVar;
   char *str, category[24], x_label[20]; 
   nTuDDL *ddl;
   descrGenNtuple *dNTu;
   nTuBrowserInfo *nTuBr;
   varGenNtuple *var;
   Arg args[2];
   nTuBroHs1D *nTuH1;
   char *cDat;
   long pp; 


    if (HistoIsBrowserNtuInit == NULL) {
       hs_initialize("Mcfio Ntuple Browser");
       hs_histoscope(1);
       HistoIsBrowserNtuInit = 1;
       XtSetSensitive(McfioHsResetBtnW, True);
       hs_update();
       setTimer(500);
    }
    nTuBr = CurrentNTupleBrowserSelected;
    ddl = mcf_GetNTuByPtrID(CurrentNTupleBrowserSelected->id);
    if (ddl->descrNtu == NULL) dNTu = ddl->reference->descrNtu;
	    else dNTu = ddl->descrNtu;
    
    if (!XmListGetSelectedPos(OneDHistVariableListW, &posListVar, &countVar)) {
	DialogF(DF_WARN, OneDHistShellW, 1, "Please select a Variable",
		"Acknowledged");
	return;
    }
    ivarH = (*posListVar) - 1; ivar = ivarH -1;
    /*
    ** Get how many time we have to histogram the variable
    */
    if (ivar < dNTu->firstIndexed) im = -1;
    else {
     str = XmTextGetString(OneDHistMultW);
     if (GetIntText(OneDHistMultW, &im) == TEXT_READ_OK) {
        if (im > dNTu->maxMultiplicity) {
             DialogF(DF_WARN, OneDHistShellW, 1, 
 "Incorrect instance, above maximum Multiplicty.","Acknowledged");
             return;
        } else im--;
      } else {
        if ((strcmp(str,"All") == 0) || (strcmp(str,"ALL") == 0) ||
            (strcmp(str,"all") == 0)) im = -1;
        else if ((strcmp(str,"First") == 0) || (strcmp(str,"FIRST") == 0) ||
            (strcmp(str,"first") == 0)) im = 0;
        else if ((strcmp(str,"Second") == 0) || (strcmp(str,"SECOND") == 0) ||
            (strcmp(str,"second") == 0)) im = 1;
        else if ((strcmp(str,"Third") == 0) || (strcmp(str,"THIRD") == 0) ||
            (strcmp(str,"third") == 0)) im = 2;
        else {
             DialogF(DF_WARN, OneDHistShellW, 1, 
 "Incorrect instance, please use a number","Acknowledged");
             return;
        }
      }
    }     
/*
** We now read out the widget and define the histogram
*/
     str = XmTextGetString(OneDHistTitleW);
     if (GetIntTextWarn(OneDHistNumBinsW, &nBins,
                        "Number of Bins", TRUE) != TEXT_READ_OK) return;
     if (GetFloatTextWarn(OneDHistLowBinW, &dfrom,
                        "Low Bin Edge", TRUE) != TEXT_READ_OK) return;
     if (GetFloatTextWarn(OneDHistHighBinW, &dto,
                        "High Bin Edge", TRUE) != TEXT_READ_OK) return;
      from = (float) dfrom; to = (float) dto;
      /*
      ** At the beginning, the HistoList has a dummy item
      */
      if (FirstOneDHistogram) uid = 1;
      else if (newHisto) {
        XtSetArg (args[0], XmNitemCount, &uid);
        XtGetValues(OneDHistHistoListW, args, 1);
        uid++;
      } else uid = CurrentHistoUID; 
      if (ivarH == 0) 
          strcpy(x_label,"Multiplicity");
      else {
         var = dNTu->variables[ivar];
         if(strlen(var->name) < 17) strcpy(x_label,var->name);
         else {
            strncpy(x_label,var->name,16);
            strcpy(&x_label[16],"...");
         }
     }    
     idh = hs_create_1d_hist(uid, str,  "OneDHist", x_label, "Yield", 
                             nBins, from, to);
     XtFree(str);                        
     /*
     ** Store the idh, we need it for filling. Find the first free place on
     ** the list.
     */
    if (nTuBr->nHistoItems == nTuBr->sizeOfLists)
           mcfioC_ExtendBrowserAnalysis(nTuBr);
     nTuH1 = (nTuBroHs1D *) malloc(sizeof(nTuBroHs1D));
     index=0; 
     while (nTuBr->hsItemList[index] != NULL)  index++;
     nTuBr->hsItemList[index] =  (nTuBroHsGeneral *) nTuH1;
     nTuBr->nHistoItems++;
     nTuBr->nHisto1D++;
     nTuH1->id = idh;
     nTuH1->type = HS_1D_HISTOGRAM;
     nTuH1->varNumber = ivarH;
     nTuH1->subBlock = im;
     nTuH1->varIndices = NULL;
     if(dNTu->firstIndexed == -1) lastFixed = dNTu->numVariables;
           else lastFixed = dNTu->firstIndexed;
     if (ivarH != 0) { 
        var = dNTu->variables[ivar];
       if (var->numDim >= 1) {
          nTuH1->varIndices= (int *) malloc(sizeof(int) * var->numDim);
          ivk = nTuH1->varIndices;
          getVarIndexDialog(ivk, var->numDim, var->dimensions);
       } 
     }  
     if (nTuBr->data == NULL) mcfioC_createBrowserData(nTuBr);
     cDat = (char *) nTuBr->data;
     if (ivarH == 0) {
        cDat += dNTu->multOffset;
        nTuH1->lDat = (long *) cDat;
     } else {
          if (ivar < lastFixed) { 
            pp = ((long) nTuBr->data) +  dNTu->variables[ivar]->offset;
            nTuH1->lDat = (long *) pp;
          } else  nTuH1->lDat = NULL; 
               /* A specific subVariable (leaf), we'll compute the 
          		data pointer at filling time */
     }     		
     XtSetSensitive(OneDHistModifyW, True);
     XtSetSensitive(OneDHistDeleteW, True); 
     redisplay1DHistoList(); 
     FirstOneDHistogram = False; 
     hs_update();
}
static void modifyOneDHistCB(Widget w, nTuBrowserInfo * nTuBr,
                               caddr_t call_data)
{
    int *posList, count;
    if (!XmListGetSelectedPos(OneDHistHistoListW, &posList, &count)) {
	DialogF(DF_WARN, OneDHistShellW, 1, 
	"Please select a Histogram", "Acknowledged");
	return;
    }
     deleteOneDHistCB(w, NULL, NULL);
     createOneDHistActual(False);
}

static void deleteOneDHistCB(Widget w, nTuBrowserInfo *nTuBrDummy,
                               caddr_t call_data)
{
    int i, idPos, *posList, count, id, uid, jn;
    nTuBrowserInfo *nTuBr;
    nTuBroHs1D *nTuH1;
     
                                  
    if (!XmListGetSelectedPos(OneDHistHistoListW, &posList, &count)) {
	DialogF(DF_WARN, OneDHistShellW, 1, "Please select a Histogram",
		"Acknowledged");
	return;
    }
    id = OneDHistListedIDs[((*posList) -1)];
    CurrentHistoUID = hs_uid(id);
    nTuBr = oneDHistNTupleSource(id);
    if (nTuBr == NULL) {
         DialogF(DF_ERR, OneDHistShellW, 1,
                 "Internal Error in deleteOneDHist\nPlease report",
		"Acknowledged");
         return;
     }
    hs_delete(id);
    hs_update();
    for (i=0; i<nTuBr->sizeOfLists; i++) { 
       if (nTuBr->hsItemList[i] == NULL) continue;
        if (id == nTuBr->hsItemList[i]->id ) {
             nTuH1 = (nTuBroHs1D *) nTuBr->hsItemList[i];
             if (nTuH1->varIndices != NULL) free(nTuH1->varIndices);
             free(nTuH1);
             nTuBr->hsItemList[i] = NULL;       
             break;
         }
    }
    nTuBr->nHistoItems--;
    nTuBr->nHisto1D--;
    redisplay1DHistoList();
}
     
static void dismissOneDHistCB(Widget w, nTuBrowserInfo * nTuBr,
                               caddr_t call_data)
{
    XtUnmanageChild(OneDHistFormW);
}

static void redisplay1DHistoList()
{
    int i, id, nItems, l, uid, k, nh, jn, *il;
    nTuBrowserInfo *nTuBr;
    XmString *stringTable;
    
    for(jn=0, nh=0; jn< NumOfNTuples; jn++)  {
       nTuBr = NTupleBrowserList[jn];
       for (i=0; i<nTuBr->sizeOfLists; i++) {
         if (nTuBr->hsItemList[i] == NULL) continue;
          id = nTuBr->hsItemList[i]->id;
          if ( (id != -1) && (hs_type(id) == HS_1D_HISTOGRAM)) nh++;
       }    
    }
    OneDHistNumberListed = nh;
    if (nh == 0) {
       stringTable = (XmString *)XtMalloc((1) * sizeof(XmString));
       stringTable[1] = (XmString)0;
       setListItems(OneDHistHistoListW, stringTable, 0);
      /* free the string table */
       FreeStringTable(stringTable);
       return;
    } 
    if (nh > OneDHistSizeOfList) {
       il =  (int *) malloc(sizeof(int) * 2 * nh);
       memcpy((void *) il, (void *) OneDHistListedIDs, (sizeof(int) * nh));
       free(OneDHistListedIDs); OneDHistListedIDs = il;
       OneDHistSizeOfList = 2*nh;
    } 
    stringTable = (XmString *)XtMalloc((nh+1) * sizeof(XmString));
    for(jn=0, nh=0; jn< NumOfNTuples; jn++)  {
       nTuBr = NTupleBrowserList[jn];
       for (i=0; i<nTuBr->sizeOfLists; i++) {
         if (nTuBr->hsItemList[i] == NULL) continue;
         id = nTuBr->hsItemList[i]->id;
         if (nTuBr->hsItemList[i]->type == HS_1D_HISTOGRAM) {
             l = hs_title(id, BrowseUtil2htitle);
             stringTable[nh] = XmStringCreateSimple (BrowseUtil2htitle);
             OneDHistListedIDs[nh] = id;
             nh++; 
          }
        }
    }
    stringTable[nh] =  (XmString ) NULL;
    setListItems(OneDHistHistoListW, stringTable, nh);
    FreeStringTable(stringTable);
    return;
    
}                  
static void oneDHistListCB(Widget w, nTuBrowserInfo *nTuBrDummy,
                               caddr_t call_data)
{
    int i, idPos, *posList, count, id, l, jn, uid, ivar;
    float aa, bb;                               
    nTuBrowserInfo *nTuBr;
    nTuBroHs1D *nTuH1;

    XmListGetSelectedPos(OneDHistHistoListW, &posList, &count);
    if (count < 1) return;
    id = OneDHistListedIDs[((*posList) -1)];
    uid = hs_uid(id);
    /*
    ** Select the variable associated with this histogram.
    */
    nTuBr = oneDHistNTupleSource(id);
    if (nTuBr != CurrentNTupleBrowserSelected ) 
                    if (!mcfioC_SetSpecificNTupleBr(nTuBr)) return;
    for(i=0; i<CurrentNTupleBrowserSelected->sizeOfLists; i++) {
       nTuH1 = (nTuBroHs1D *) CurrentNTupleBrowserSelected->hsItemList[i];
       if (nTuH1 == NULL) continue;
       if (nTuH1->id == id) break;
    } 
    if (nTuH1 == NULL) {
         DialogF(DF_ERR, OneDHistShellW, 1,
                 "Internal Error in oneDHistListCB\nPlease report",
		"Acknowledged");
         return;
    }     
    ivar =  nTuH1->varNumber;
    XmListSelectPos(OneDHistVariableListW, (ivar+1), False);
    l = hs_title(id, BrowseUtil2htitle);
    XmTextSetString(OneDHistTitleW, BrowseUtil2htitle);
    l = hs_1d_hist_num_bins(id);
    SetIntText(OneDHistNumBinsW, l);
    hs_1d_hist_range(id, &aa, &bb);
    SetFloatText(OneDHistLowBinW, aa);
    SetFloatText(OneDHistHighBinW, bb);
    /*
    ** We now have to related this histogram to the associated Ntuple 
    */
    for (i=0; i<nTuBr->sizeOfLists; i++) {
       if (nTuBr->hsItemList[i] == NULL) continue;
           if (nTuBr->hsItemList[i]->id == id) {
               nTuH1 = (nTuBroHs1D *) nTuBr->hsItemList[i];
               if (nTuH1->subBlock == -1)  
                   XmTextSetString(OneDHistMultW, "All");
                else 
                   SetIntText(OneDHistMultW, nTuH1->subBlock);
            }
            break;
    }
    XtSetSensitive(OneDHistModifyW, True);
    XtSetSensitive(OneDHistDeleteW, True);
}
#endif                                
static nTuBrowserInfo *oneDHistNTupleSource(int hs_id) {

    nTuBrowserInfo *nTuBr;
    int i, jn;
    
    for(jn=0; jn< NumOfNTuples; jn++) { 
       nTuBr = NTupleBrowserList[jn];
       for (i=0; i<nTuBr->sizeOfLists; i++) 
         if ((nTuBr->hsItemList[i] != NULL) 
             && (nTuBr->hsItemList[i]->id == hs_id)) return nTuBr;
    }
    return NULL;
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
static void   getVarIndexDialog(int *loc, int nDim, int *maxVal)
{
    int i, valt, ok;
    char *endPtr, *t1;
	
    if (nDim == 1) {
        ok = -1;
        while (ok == -1) { 
        DialogF(DF_PROMPT,OneDHistShellW, 0,
           "The variable you wish to histogram is a Vector\n\
  Please fill in the requested index", BrowseUtil2htitle);
        mcfioC_RemoveWhiteSpace(BrowseUtil2htitle);
        valt = strtol(BrowseUtil2htitle, &endPtr, 10);
	if ((strlen(BrowseUtil2htitle) == 0) || (*endPtr != '\0'))
	   DialogF(DF_ERR,OneDHistShellW, 1,
	    "Please Try again, we need an integer number! ","Acknowledged");
	else {
	  if ((valt < 1) || (valt > maxVal[0]))  
	   DialogF(DF_ERR,OneDHistShellW, 1,
          "Wrong Index Value\nWe need a number between 1 and %d",
          "Acknowledged",maxVal[0]);
          else {
              ok = valt;
              loc[0] = valt;
          }
       }
     }
   } else {
tryagain:  
        DialogF(DF_PROMPT,OneDHistShellW, 0,
           "The variable you wish to histogram is a Matrix of dimension %d\n\
 Please fill in the requested indices\n\
 The format is 2,5....\n\Index numbering convention is assumed to be C",
  BrowseUtil2htitle, nDim);
        mcfioC_RemoveWhiteSpace(BrowseUtil2htitle);
        for (i=0, t1=BrowseUtil2htitle; i<nDim; i++) {
           valt = strtol(t1, &endPtr, 10);
	   if (strlen(BrowseUtil2htitle) == 0) {
	      DialogF(DF_ERR,OneDHistShellW, 1,
	       "Please Try again, we need integer numbers! ","Acknowledged");
	       goto tryagain;
	   }  else {
	     if ((valt < 1) || (valt > maxVal[i])) { 
	      DialogF(DF_ERR,OneDHistShellW, 1,
              "Wrong Index Value\nWe need a number between 1 and %d",
              "Acknowledged",maxVal[i]);
               goto tryagain;
              } else {
                loc[i] = valt;
                if (i != (nDim-1)) {
                   if (*endPtr == '\0') {
	               DialogF(DF_ERR,OneDHistShellW, 1,
	  "Please Try again, we need more integer numbers! ","Acknowledged");
	               goto tryagain;
	           }
	       t1 = endPtr; t1++;
              }
          }
       }
     }
   }    
}
          
          
/*
** setTimer - Set the timer
*/
static void setTimer(int timeInMillis)
{
    if (TimeOutID == 0) {
	gettimeofday(&TimeSetTimeOut, NULL);
	TimeOutTime = timeInMillis;
	TimeOutID =
	  XtAppAddTimeOut(XtWidgetToApplicationContext(McfioMainPanelW),
    	    		    timeInMillis > 0 ? timeInMillis : 1, 
    	    		    (XtTimerCallbackProc)timerUpCB, NULL);
    }
    else {
    	cancelTimer();
    	setTimer(timeInMillis);
    }
}

/*
** cancelTimer - Cancel the timer and reset timer variables.
*/
static void cancelTimer(void)
{
    if (TimeOutID != 0) {
    	XtRemoveTimeOut(TimeOutID);
    	TimeOutTime = 0;
    	TimeOutID = 0;
    	TimeSetTimeOut.tv_sec = 0;
    	TimeSetTimeOut.tv_usec = 0;
    }
}

/*
** timerUpCB - Timer is up.  Reset timer variables and see if there are updates
**	     to request
*/
static void timerUpCB(void)
{
    TimeOutTime = 0;
    TimeOutID = 0;
    TimeSetTimeOut.tv_sec = 0;
    TimeSetTimeOut.tv_usec = 0;

#ifdef HISTO
    hs_update();
#endif    
    setTimer(500);
}

