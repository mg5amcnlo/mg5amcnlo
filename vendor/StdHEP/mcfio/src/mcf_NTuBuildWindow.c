/*******************************************************************************
*									       *
* mcf_NTuBuildWindow.c -- Program for mcfast generalized nTuple Builder GUI    *
*									       *
*	P. Lebrun, September 1995.					       *
*									       *
*******************************************************************************/
/*
** Create, Close and related Callbacks to handle a Generalized NTuple 
** description.
*/
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <Xm/Xm.h>
#include <Xm/MainW.h>
#include <Xm/Form.h>
#include <Xm/List.h>
#include <Xm/LabelG.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/PushBG.h>
#include <Xm/ToggleB.h>
#include <Xm/Text.h>
#include <Xm/Scale.h>
#include <Xm/DialogS.h>
#include <Xm/PanedW.h>
#include <Xm/PanedWP.h>
#include <Xm/Separator.h>
#include <Xm/SeparatoG.h>
#include <X11/Intrinsic.h>
#include <X11/IntrinsicP.h>
#include <X11/StringDefs.h>
#include "mcf_nTupleDescript.h"
#include "mcf_nTupleBuild.h"
#include "mcf_BrowseUtil1.h"
#include "mcf_NTuBuildWindow.h"
#include "mcf_NTuBldMenu.h"
#include "DialogF.h"
#include "misc.h"
#include "getfiles.h"
#include "mcf_BrowseMainPanel.h"
#include "mcf_BrowseUtil2.h"

#define MAX_FIELD_DIMENSIONS 30

nTuBuildWindow *NTuBuildWindowList = NULL;
extern char *VarTypesNamesF77[N_VAR_TYPES];
extern char *VarTypesNamesC[N_VAR_TYPES];
extern NumOfNTuples;

/* Storage for pseudo-clipboard for cutting and pasting parameters */
static varGenNtuple VariableClipboard;
static varGenNtuple VariableForUndo;

/* Flag for the modifiedCB, not sure this is really necessary */
static int UpdatingFields = False;

static char *DimensionsNames[MAX_VAR_DIMENSIONS];

static int nTuBldWindowsOpen();
static void changeCB(Widget w, nTuBuildWindow *window, caddr_t call_data);
static void undoCB(Widget w, nTuBuildWindow *window, caddr_t call_data);
static void dismissCB(Widget w, nTuBuildWindow *window, caddr_t call_data);
static void cutCB(Widget w, nTuBuildWindow *window, caddr_t call_data);
static void copyCB(Widget w, nTuBuildWindow *window, caddr_t call_data);
static void pasteCB(Widget w, nTuBuildWindow *window, caddr_t call_data);
static void clearCB(Widget w, nTuBuildWindow *window, caddr_t call_data);
static void insertCB(Widget w, nTuBuildWindow *window, caddr_t call_data);
static void modifiedCB(Widget w, nTuBuildWindow *window, caddr_t call_data);
static void modifiedNameCB(Widget w, nTuBuildWindow *window,
                             caddr_t call_data);
static void nullCB(Widget w, nTuBuildWindow *window, caddr_t call_data);
static void listSelectionCB(Widget w, nTuBuildWindow *window,
	XmListCallbackStruct *call_data);
static void saveVarForUndo(nTuBuildWindow *window);
static int selectedListPosition(Widget listW);
static void typeMenuCB(Widget w, nTuBuildWindow *window, caddr_t call_data);
static void dimMenuCB(Widget w, nTuBuildWindow *window, caddr_t call_data);
static void indexedMenuCB(Widget w, nTuBuildWindow *window, caddr_t call_data);
static void createDimDialog(nTuBuildWindow *window);
static void fillDimensions(nTuBuildWindow *window);
static void dismissDimCB(Widget w, nTuBuildWindow *window, caddr_t call_data);
static void updateDimPanel(nTuBuildWindow *window);
static void getFilledDimensions(nTuBuildWindow *window);
static void removeWhiteSpace(char *string);

nTuBuildWindow *CreateNTuBuildWindow(Display *display, char *title,
						 int readOnly) 
{
    nTuBuildWindow *window;
    Widget main, menuBar, form, pane, formVar, clipForm;
    int topAttach;
    Widget listLbl, nameLbl, descriptLbl, titleLbl, versionLbl;
    Widget fixLbl, descriptVarLbl, multLbl, nameIndexLbl ;
    Widget dismissBtn, sep, sep2, typeMenu, button, dimMenu, indexedMenu;
    int p;
    XmString s1;
    int i, j, ac;
    Arg args[20];
    
    
    /* Create a window data structure and initialize it */
    window = (nTuBuildWindow *)XtMalloc(sizeof(nTuBuildWindow));
    if (!nTuBldWindowsOpen()) window->id = 1;
       else window->id = NTuBuildWindowList->id + 1;
    window->isSaved = False;
    window->hasChanged = False;
    window->dbinFileName = NULL;
    window->isReadOnly = readOnly;
    window->next = NTuBuildWindowList;
    window->selectedIndex = 0;
    NTuBuildWindowList = window;
    window->dimDialogW = NULL;
    window->descrNtu = (descrGenNtuple *) malloc(sizeof(descrGenNtuple));
    /* Initialization of the Ntuple description, at least the stuff we need */
    window->langEnv = F77_LANGUAGE;
    window->descrNtu->orgStyle = PARALLEL_ARRAY_NTU;
    window->descrNtu->address = NULL;
              /* All of these are not used in this context, but safer */
    window->descrNtu->multOffset  = 0;
    window->descrNtu->multXDROffset  = 0;         
    window->descrNtu->fenceOffset = 0;
    window->descrNtu->fenceXDROffset = 0; 
    window->descrNtu->subOffset = NULL;
    window->descrNtu->subXDROffset = NULL;
    window->descrNtu->numVariables = 0;
    window->descrNtu->numAvailable = NUM_START_VARIABLES;
    window->nameIndexBlank = True;
    window->multiplicityBlank = True;
    window->descrNtu->maxMultiplicity = 0;
    window->descrNtu->firstIndexed = 0; 
    window->titleBlank = True;
    window->descrNtu->title = NULL;
    window->descrNtu->description = NULL;
    strcpy(window->descrNtu->version,"1.00");
    /*
    ** Define a variable to be saved to implement undo feature
    */
    window->undoIndex = -1;
    window->undoVariable = 
       (varGenNtuple *) malloc(sizeof(varGenNtuple));
    window->undoVariable->nameBlank = True;
    window->undoVariable->name = NULL;
    window->undoVariable->description = NULL;
    window->undoVariable->type = INTEGER_NTU;
    window->undoVariable->isFixedSize = False;
    window->undoVariable->numDim = 0;
    for (j=0; j<MAX_VAR_DIMENSIONS; j++) 
            window->undoVariable->dimensions[j] = -1;
    /*
    ** The real variables now
    */
    window->descrNtu->varOrdering = NULL;
    window->descrNtu->variables =
       (varGenNtuple **) malloc(sizeof(varGenNtuple *) * NUM_START_VARIABLES);
    for (i=0; i<NUM_START_VARIABLES; i++) {
       window->descrNtu->variables[i] =   
       (varGenNtuple *) malloc(sizeof(varGenNtuple));
       window->descrNtu->variables[i]->nameBlank = True;
       window->descrNtu->variables[i]->name = NULL;
       window->descrNtu->variables[i]->description = NULL;
       window->descrNtu->variables[i]->type = INTEGER_NTU;
       window->descrNtu->variables[i]->isFixedSize = False;
       window->descrNtu->variables[i]->numDim = 0;
       window->descrNtu->variables[i]->offset = 0;
       window->descrNtu->variables[i]->offsetXDR = 0;
    }
    window->selectedIndex = 0;
    if (window->id == 1) CopyVarGenNtuple(window->descrNtu->variables[0],
                                    &VariableClipboard);
    /* Start butilding the panel... */	
    /* Create a toplevel shell to hold the window */
    window->shell = XtVaAppCreateShell (APP_NAME, APP_CLASS,
	    applicationShellWidgetClass, display,
	    XmNtitle, title,
	    XmNiconName, title,
	    XmNallowShellResize, False, 0);
	    
    /* Create a main window widget and the menu bar */
    main = XtVaCreateManagedWidget("mainWin", xmMainWindowWidgetClass,
    	    window->shell, NULL);
    AddMotifCloseCallback(XtParent(main), (XtCallbackProc)dismissCB, window);
 
    menuBar = CreateNTuBldMenuBar(main, window);
    
    /* Create a paned window so users can see long expressions */
    pane = XtVaCreateManagedWidget("panedW", xmPanedWindowWidgetClass, main,
    	    XmNmarginWidth, 0, XmNmarginHeight, 0, 0);
    
    /* Create a form widget for the title/description part of the window */
    form = XtVaCreateManagedWidget("ntuBldForm", xmFormWidgetClass, pane, 0);
    XtVaSetValues(form, XmNshadowThickness, 0, 0);
    
    /* Create the label for the title, and the title text  */
    titleLbl = XtVaCreateManagedWidget("titleLbl", xmLabelGadgetClass, form,
    	    XmNlabelString, s1=XmStringCreateSimple("Structure or Common Name"),
    	    XmNtopAttachment, XmATTACH_FORM,
    	    XmNtopOffset, 2,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 1, 0);
    XmStringFree(s1);
    
    window->titleW  = XtVaCreateManagedWidget("titleW",
    	    xmTextWidgetClass, form,
            XmNcolumns, 30,
            XmNmaxLength, 40,
            XmNeditMode, XmSINGLE_LINE_EDIT,
    	    XmNtopAttachment, XmATTACH_FORM,
    	    XmNtopOffset, 2,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftWidget,titleLbl,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 76, 0);
    RemapDeleteKey(window->titleW);
    XtAddCallback(window->titleW, XmNvalueChangedCallback,
    	    (XtCallbackProc)modifiedNameCB, (void *)window);
    if (readOnly) XtVaSetValues(window->titleW, XmNeditable, False, 0); 
    
    XtAddCallback(window->titleW, XmNmodifyVerifyCallback,
    	    (XtCallbackProc)modifiedCB, (void *)window);

    versionLbl = XtVaCreateManagedWidget("versionLbl",
            xmLabelGadgetClass, form,
    	    XmNlabelString, s1=XmStringCreateSimple("version"),
    	    XmNtopAttachment, XmATTACH_FORM,
    	    XmNtopOffset, 2,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 80, 0);
    XmStringFree(s1);
    
    window->versionW  = XtVaCreateManagedWidget("version",
    	    xmTextWidgetClass, form,
            XmNcolumns, 8,
            XmNmaxLength, 7,
            XmNeditMode, XmSINGLE_LINE_EDIT,
    	    XmNtopAttachment, XmATTACH_FORM,
    	    XmNtopOffset, 2,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftWidget, versionLbl,
    	    XmNrightAttachment, XmATTACH_FORM, 0);
    RemapDeleteKey(window->versionW);
    XmTextSetString(window->versionW, "1.00");
    if (readOnly) XtVaSetValues(window->versionW, XmNeditable, False, 0); 

    nameIndexLbl = XtVaCreateManagedWidget("nameIndexLbl", 
            xmLabelGadgetClass, form,
    	    XmNlabelString,
    	    s1=XmStringCreateSimple("Name of the master index"),
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopWidget, window->titleW,
    	    XmNtopOffset, 2,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 1, 0);
    XmStringFree(s1);
    
    window->nameIndexW  = XtVaCreateManagedWidget("nameIndexW",
    	    xmTextWidgetClass, form,
    	    XmNcolumns, 20,
    	    XmNrows, 1,
            XmNeditMode, XmSINGLE_LINE_EDIT,
    	    XmNmaxLength, 31,
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopWidget, window->titleW,
    	    XmNtopOffset, 2,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftWidget,nameIndexLbl, 0);
    RemapDeleteKey(window->nameIndexW);
    XtAddCallback(window->nameIndexW, XmNvalueChangedCallback,
    	    (XtCallbackProc)modifiedNameCB, (void *)window);
    XtAddCallback(window->nameIndexW, XmNmodifyVerifyCallback,
    	    (XtCallbackProc)modifiedCB, (void *)window);
    if (readOnly) XtVaSetValues(window->nameIndexW, XmNeditable, False, 0); 
    
    multLbl = XtVaCreateManagedWidget("multLbl", xmLabelGadgetClass, form,
    	    XmNlabelString,
    	    s1=XmStringCreateSimple("Maximum value "),
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopWidget, window->titleW,
    	    XmNtopOffset, 2,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftOffset, 5,
    	    XmNleftWidget, window->nameIndexW, 0);
    XmStringFree(s1);
    
    window->multiplicityW  = XtVaCreateManagedWidget("multW",
    	    xmTextWidgetClass, form,
    	    XmNcolumns, 6,
            XmNeditMode, XmSINGLE_LINE_EDIT,
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopWidget, window->titleW,
    	    XmNtopOffset, 2,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftWidget,multLbl, 0);
    RemapDeleteKey(window->multiplicityW);
    XtAddCallback(window->multiplicityW, XmNvalueChangedCallback,
    	    (XtCallbackProc)modifiedNameCB, (void *)window); 
    
    XtAddCallback(window->multiplicityW, XmNmodifyVerifyCallback,
    	    (XtCallbackProc)modifiedCB, (void *)window);
    if (readOnly) XtVaSetValues(window->multiplicityW, XmNeditable, False, 0); 
    	    
    /* Create the label for the description, 
       and the description scrolled text  */
    descriptLbl = XtVaCreateManagedWidget("descriptLbl",
            xmLabelGadgetClass, form,
    	    XmNlabelString, s1=XmStringCreateSimple("Description"),
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopOffset, 5 ,
    	    XmNtopWidget, window->multiplicityW,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 45, 0);
    XmStringFree(s1);
    
    ac = 0;
    XtSetArg(args[ac], XmNrows, 4); ac++;
    XtSetArg(args[ac], XmNcolumns, 80); ac++;
    XtSetArg(args[ac], XmNeditMode, XmMULTI_LINE_EDIT); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopWidget, descriptLbl); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
    window->descriptW = XmCreateScrolledText(form,
                                          "DescriptionText", args, ac);
    XtManageChild(window->descriptW);
    RemapDeleteKey(window->descriptW);
    /*
    ** Now, using the other part of the pane, create a new form to hold 
    ** the list of parameters..
    */
    formVar = XtVaCreateManagedWidget("ntuBldVForm", 
                                       xmFormWidgetClass, pane, 0);
    XtVaSetValues(formVar, XmNshadowThickness, 0, 0);
    
    /* Create the label for the columns of the scrolling list of parameters */
    listLbl = XtVaCreateManagedWidget("listLbl", xmLabelGadgetClass, formVar,
    	    XmNlabelString, s1=XmStringCreateSimple(
  " V#   Name                Type             Dimensions & Index       "),
  /*01234567890123456789012345678901234567890123456789012345678901234567890 */
    	    XmNtopAttachment, XmATTACH_FORM,
    	    XmNtopOffset, 2,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 17, 0);
    XmStringFree(s1);
    
    /* Create the OK, Apply, and Dismiss buttons, and separator */
    if (readOnly == False) {
       window->changeBtn = XtVaCreateManagedWidget("changeBtn",
    	    xmPushButtonGadgetClass, formVar,
    	    XmNlabelString, s1=XmStringCreateSimple("Change"),
    	    XmNbottomAttachment, XmATTACH_FORM,
    	    XmNbottomOffset, 4,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 10,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 30, 0);
       XmStringFree(s1);
       XtAddCallback(window->changeBtn, XmNactivateCallback,
    	    (XtCallbackProc)changeCB, (void *)window);
       XtVaSetValues(formVar, XmNdefaultButton, window->changeBtn, 0);
       XtSetSensitive(window->changeBtn, False);
       window->undoBtn = XtVaCreateManagedWidget("undoBtn",
    	    xmPushButtonGadgetClass, formVar,
    	    XmNlabelString, s1=XmStringCreateSimple("Undo"),
    	    XmNbottomAttachment, XmATTACH_FORM,
    	    XmNbottomOffset, 4,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 40,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 60, 0);
       XmStringFree(s1);
       XtAddCallback(window->undoBtn, XmNactivateCallback,
    	    (XtCallbackProc)undoCB, (void *)window);
       XtSetSensitive(window->undoBtn, False);
    } else {
         window->changeBtn = NULL;
         window->undoBtn = NULL;
    }     
    dismissBtn = XtVaCreateManagedWidget("dismissBtn",
    	    xmPushButtonGadgetClass, formVar,
    	    XmNlabelString, s1=XmStringCreateSimple("Dismiss"),
    	    XmNbottomAttachment, XmATTACH_FORM,
    	    XmNbottomOffset, 4,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 70,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 90, 0);
    XmStringFree(s1);
    XtAddCallback(dismissBtn, XmNactivateCallback,
    	    (XtCallbackProc)dismissCB, (void *)window);
    sep = XtVaCreateManagedWidget("sep", xmSeparatorGadgetClass, formVar,
    	    XmNbottomAttachment, XmATTACH_WIDGET,
    	    XmNbottomWidget, dismissBtn,
    	    XmNbottomOffset, 4,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 0,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 100, 0);
    /*
    **   Create the descrption label and text widget, for the selected
    **   Variable.
    */
    window->descriptVarW  = XtVaCreateManagedWidget("CommentW",
    	    xmTextWidgetClass, formVar,
            XmNeditMode, XmMULTI_LINE_EDIT,
    	    XmNcolumns, 70,
    	    XmNrows, 1,
    	    XmNmaxLength, 71,
            XmNeditMode, XmSINGLE_LINE_EDIT,
    	    XmNbottomAttachment, XmATTACH_WIDGET,
    	    XmNbottomOffset, 1,
    	    XmNbottomWidget, sep,
    	    XmNleftAttachment, XmATTACH_FORM,
    	    XmNleftOffset, 2, 0);
    RemapDeleteKey(window->descriptVarW);
    XtAddCallback(window->descriptVarW, XmNmodifyVerifyCallback,
    	    (XtCallbackProc)modifiedCB, (void *)window);
    if (readOnly) XtVaSetValues(window->descriptVarW, XmNeditable, False, 0); 
    
    descriptVarLbl = XtVaCreateManagedWidget("descriptVarLbl",
            xmLabelGadgetClass, formVar,
    	    XmNlabelString, s1=XmStringCreateSimple("Comment"),
    	    XmNbottomAttachment, XmATTACH_WIDGET,
    	    XmNbottomWidget, window->descriptVarW,
    	    XmNleftAttachment, XmATTACH_FORM,
    	    XmNleftOffset, 20, 0);
    XmStringFree(s1);
    
   /*
   ** Next is a complicated row for the Name field, ,
   ** and the type and dimension option menus, and wether this is a 
   ** fixed size Variable or not...
   */ 
    
    nameLbl = XtVaCreateManagedWidget("nameLbl",
            xmLabelGadgetClass, formVar,
    	    XmNlabelString, s1=XmStringCreateSimple("Name"),
    	    XmNbottomAttachment, XmATTACH_WIDGET,
    	    XmNbottomOffset, 5,
    	    XmNbottomWidget, descriptVarLbl,
    	    XmNleftAttachment, XmATTACH_FORM,
    	    XmNleftOffset, 1, 0); 
    XmStringFree(s1);
    
    window->nameW  = XtVaCreateManagedWidget("CommentW",
    	    xmTextWidgetClass, formVar,
    	    XmNcolumns, 20,
    	    XmNmaxLength, 31,
            XmNeditMode, XmSINGLE_LINE_EDIT,
    	    XmNbottomAttachment, XmATTACH_WIDGET,
    	    XmNbottomOffset, 1,
    	    XmNbottomWidget, descriptVarLbl,
    	    XmNleftAttachment, XmATTACH_WIDGET,
    	    XmNleftWidget, nameLbl,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 32, 0);
    RemapDeleteKey(window->nameW);
    XtAddCallback(window->nameW, XmNmodifyVerifyCallback,
    	    (XtCallbackProc)modifiedCB, (void *)window);
    	    
    XtAddCallback(window->nameW, XmNvalueChangedCallback,
    	    (XtCallbackProc)modifiedNameCB, (void *)window);
    if (readOnly) XtVaSetValues(window->nameW, XmNeditable, False, 0); 
    ac = 0;
    typeMenu = XmCreatePulldownMenu(formVar, "varType", args, ac);

    ac = 0;
    XtSetArg(args[ac], XmNspacing, 0); ac++;
    XtSetArg(args[ac], XmNmarginWidth, 0); ac++;
    XtSetArg(args[ac], XmNresizeWidth, False); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomOffset, 1); ac++;
    XtSetArg(args[ac], XmNbottomWidget, descriptVarLbl); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_POSITION); ac++;
    XtSetArg(args[ac], XmNleftPosition, 35); ac++;
    XtSetArg(args[ac], XmNsubMenuId, typeMenu); ac++;
    window->typeMenu = XmCreateOptionMenu(formVar, "varTypeMenu", args, ac);
    XtManageChild(window->typeMenu);
 
    for (i=0; i<N_VAR_TYPES; i++) {
    	window->typeBtns[i] = XtVaCreateManagedWidget("typeBtn", 
    	        xmPushButtonWidgetClass, typeMenu,
    		XmNlabelString, s1=XmStringCreateSimple(VarTypesNamesF77[i]),
    		XmNuserData, i, 0);
	XmStringFree(s1);
    	XtAddCallback(window->typeBtns[i], 
    	              XmNactivateCallback, (XtCallbackProc)typeMenuCB,
    		      (void *)window);
    }
    
    XtVaSetValues(window->typeMenu, XmNmenuHistory, 
                            window->typeBtns[INTEGER_NTU] , 0);
    if (readOnly) XtSetSensitive(window->typeMenu, False);                        
    ac = 0;
    dimMenu = XmCreatePulldownMenu(formVar, "Dimensions", args, ac);

    ac = 0;
    XtSetArg(args[ac], XmNspacing, 0); ac++;
    XtSetArg(args[ac], XmNmarginWidth, 0); ac++;
    XtSetArg(args[ac], XmNresizeWidth, False); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomOffset, 1); ac++;
    XtSetArg(args[ac], XmNbottomWidget, descriptVarLbl); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftOffset, 3); ac++;
    XtSetArg(args[ac], XmNleftWidget, window->typeMenu); ac++;
    XtSetArg(args[ac], XmNsubMenuId, dimMenu); ac++;
    window->dimNumMenu = XmCreateOptionMenu(formVar, "dimNumMenu", args, ac);
    XtManageChild(window->dimNumMenu);
 
    if (window->id == 1) {
        DimensionsNames[0]= "Scalar    ";
        DimensionsNames[1]= "Array     ";
        DimensionsNames[2]= "2-D Array ";
        DimensionsNames[3]= "3-D Array ";
    }    
    for (i=0; i<MAX_VAR_DIMENSIONS; i++) {
    	window->dimBtns[i] = XtVaCreateManagedWidget("dimBtn", 
    	        xmPushButtonWidgetClass, dimMenu,
    		XmNlabelString, s1=XmStringCreateSimple(DimensionsNames[i]),
    		XmNuserData, i, 0);
	XmStringFree(s1);
    	XtAddCallback(window->dimBtns[i],
    	              XmNactivateCallback, (XtCallbackProc)dimMenuCB,
    		      (void *)window);
    }
	
    if (readOnly) XtSetSensitive(window->dimNumMenu, False);                        
    
    
    ac = 0;
    indexedMenu = XmCreatePulldownMenu(formVar, "Indexing", args, ac);

    ac = 0;
    XtSetArg(args[ac], XmNspacing, 0); ac++;
    XtSetArg(args[ac], XmNmarginWidth, 0); ac++;
    XtSetArg(args[ac], XmNresizeWidth, False); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomOffset, 1); ac++;
    XtSetArg(args[ac], XmNbottomWidget, descriptVarLbl); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftOffset, 3); ac++;
    XtSetArg(args[ac], XmNleftWidget, window->dimNumMenu); ac++;
    XtSetArg(args[ac], XmNsubMenuId, indexedMenu); ac++;
    window->indexingMenu = XmCreateOptionMenu(formVar,
                                          "IndexingMenu", args, ac);
    XtManageChild(window->indexingMenu);
 
    window->indexingBtns[0] = XtVaCreateManagedWidget("indexingBtn", 
    	        xmPushButtonWidgetClass, indexedMenu,
    		XmNlabelString, s1=XmStringCreateSimple("Indexed"),
    		XmNuserData, 0, 0);
    XmStringFree(s1);
    XtAddCallback(window->indexingBtns[0], 
                  XmNactivateCallback, (XtCallbackProc)indexedMenuCB,
    		      (void *)window);
	
    window->indexingBtns[1] = XtVaCreateManagedWidget("indexingBtn", 
    	        xmPushButtonWidgetClass, indexedMenu,
    		XmNlabelString, s1=XmStringCreateSimple("One instance"),
    		XmNuserData, 1, 0);
    XmStringFree(s1);
    XtAddCallback(window->indexingBtns[1], 
                  XmNactivateCallback, (XtCallbackProc)indexedMenuCB,
    		      (void *)window);
    
    if (readOnly) XtSetSensitive(window->indexingMenu, False);                        
    sep2 = XtVaCreateManagedWidget("sep2", xmSeparatorGadgetClass, formVar,
    	    XmNbottomAttachment, XmATTACH_WIDGET,
    	    XmNbottomWidget, window->indexingMenu,
    	    XmNbottomOffset, 4,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 0,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 100, 0);
    /*
    ** Now, we have to create the scrolling list
    */
    /* Create the scrolling list to hold the parameter settings */
    ac = 0;
    XtSetArg(args[ac], XmNscrollBarDisplayPolicy, XmAS_NEEDED); ac++;
    XtSetArg(args[ac], XmNautomaticSelection, True); ac++;
    XtSetArg(args[ac], XmNcolumns, 69); ac++;
    XtSetArg(args[ac], XmNvisibleItemCount, 9); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopWidget, listLbl); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_POSITION); ac++;
    XtSetArg(args[ac], XmNleftPosition, 17); ac++;
    XtSetArg(args[ac], XmNrightAttachment, XmATTACH_POSITION); ac++;
    XtSetArg(args[ac], XmNrightPosition, 98); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomWidget, sep2); ac++;
    XtSetArg(args[ac], XmNbottomOffset, 1); ac++;
    window->listW = XmCreateScrolledList(formVar, "list", args, ac);
    XtManageChild(window->listW);
    XtAddCallback(window->listW, XmNbrowseSelectionCallback,
    	    (XtCallbackProc)listSelectionCB, (void *)window);
    
    /* Create the cut, copy, paste, and clear buttons, in their own form,
       for better keyboard traversal, and to avoid the default button
       appearance */
    clipForm = XtVaCreateManagedWidget("clipForm", xmFormWidgetClass, formVar,
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopWidget, listLbl,
    	    XmNbottomAttachment, XmATTACH_WIDGET,
    	    XmNbottomWidget, sep2,
    	    XmNbottomOffset, 1,
    	    XmNleftAttachment, XmATTACH_FORM,
    	    XmNleftOffset, 1,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 15, 0);
    if (readOnly) {
         window->copyBtn = XtVaCreateManagedWidget("copyBtn",
    	    xmPushButtonGadgetClass, clipForm,
    	    XmNlabelString, s1=XmStringCreateLtoR(
    	    "Copy \n Variable \n to \n Clipboard",
    	                             XmSTRING_DEFAULT_CHARSET),
    	    XmNtopAttachment, XmATTACH_FORM,
    	    XmNtopOffset, 5,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 5,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 95, 0);
        XmStringFree(s1);
        XtAddCallback(window->copyBtn, XmNactivateCallback,
    	    (XtCallbackProc)copyCB, (void *)window);
    	window->cutBtn = NULL;
    	window->pasteBtn = NULL;
    	window->clearBtn = NULL; 
    	window->insertBtn = NULL; 
    }  else {
       window->cutBtn = XtVaCreateManagedWidget("cutBtn",
    	    xmPushButtonGadgetClass, clipForm,
    	    XmNlabelString, s1=XmStringCreateSimple("Cut"),
    	    XmNtopAttachment, XmATTACH_FORM,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 5,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 95, 0);
       XmStringFree(s1);
       XtAddCallback(window->cutBtn, XmNactivateCallback,
    	    (XtCallbackProc)cutCB, (void *)window);
       window->copyBtn = XtVaCreateManagedWidget("copyBtn",
    	    xmPushButtonGadgetClass, clipForm,
    	    XmNlabelString, s1=XmStringCreateSimple("Copy"),
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopWidget, window->cutBtn,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 5,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 95, 0);
       XmStringFree(s1);
       XtAddCallback(window->copyBtn, XmNactivateCallback,
    	    (XtCallbackProc)copyCB, (void *)window);
       window->pasteBtn = XtVaCreateManagedWidget("pasteBtn",
    	    xmPushButtonGadgetClass, clipForm,
    	    XmNlabelString, s1=XmStringCreateSimple("Paste"),
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopWidget, window->copyBtn,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 5,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 95, 0);
       XmStringFree(s1);
       XtAddCallback(window->pasteBtn, XmNactivateCallback,
    	    (XtCallbackProc)pasteCB, (void *)window);
       window->clearBtn = XtVaCreateManagedWidget("clearBtn",
    	    xmPushButtonGadgetClass, clipForm,
    	    XmNlabelString, s1=XmStringCreateSimple("Clear"),
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopWidget, window->pasteBtn,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 5,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 95, 0);
       XmStringFree(s1);
       XtAddCallback(window->clearBtn, XmNactivateCallback,
    	    (XtCallbackProc)clearCB, (void *)window);
    	    
       window->insertBtn = XtVaCreateManagedWidget("insertBtn",
    	    xmPushButtonGadgetClass, clipForm,
    	    XmNlabelString, s1=XmStringCreateSimple("Insert"),
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopWidget, window->clearBtn,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 5,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 95, 0);
       XmStringFree(s1);
       XtAddCallback(window->insertBtn, XmNactivateCallback,
    	    (XtCallbackProc)insertCB, (void *)window);
    /* Set the order of keyboard traversal for the panel */
    
       XmAddTabGroup(window->titleW);
       XmAddTabGroup(window->nameIndexW);
       XmAddTabGroup(window->multiplicityW);
       XmAddTabGroup(window->descriptW);
       XmAddTabGroup(window->versionW);
    
       XmAddTabGroup(window->listW);
       XmAddTabGroup(window->nameW);
       XmAddTabGroup(window->descriptVarW);
       XmAddTabGroup(window->typeMenu);
       XmAddTabGroup(window->dimNumMenu);
       XmAddTabGroup(window->indexingMenu);
       XmAddTabGroup(window->changeBtn);
       XmAddTabGroup(window->undoBtn);
       XmAddTabGroup(dismissBtn);
       XmAddTabGroup(clipForm);

    }	    
    /* Display it all */
    XtRealizeWidget(window->shell);
    
    /* Fill in the list and the fields and select parameter # 1 */
    UpdateVariableList(window, 1);
    return window;
}


int  CloseNTuBuildWindow(nTuBuildWindow *window)
{
    int i, resp; 
    nTuBuildWindow *temp;

/*
** If use in conjunction with the Browser, skip the verification and 
** and the destruction of the descriptor itself, as we might need it 
** again.. 
*/ 
    if (window->isReadOnly != True) { 
       if ((window->isSaved != True) && 
           (window->hasChanged == True)) {
    	   resp = DialogF(DF_QUES, window->shell, 3,
    		"These definitions have not been saved\nDo you wish to save it?",
    		"Save", "No Save", "Cancel" );
           if (resp == 3) return GFN_CANCEL;
          if (resp == 1) SaveFileNTuBuildWindow(window);
       }
       /* If this is the last window open, just quit */
        if (NTuBuildWindowList==window && window->next==NULL ) exit(0);
    
       /* Remove the Ntuple Description, and various file name */
       DestroyGenNtuple(window->descrNtu);
       if (window->dbinFileName != NULL) free(window->dbinFileName); 
    } else {
    /*
    ** Look for which the Ntuple affected, set the pointer to NUll 
    */
       for (i=0; i<NumOfNTuples; i++) 
           if (NTupleBrowserList[i]->templateW == window)  
                 NTupleBrowserList[i]->templateW = NULL;
    }             
    /* Destroy the widgets */
    XtDestroyWidget(window->shell);
    
    /* Remove the window from the window list */
    if (NTuBuildWindowList == window)
	NTuBuildWindowList = window->next;
    else {
	for (temp = NTuBuildWindowList; temp != NULL; temp = temp->next) {
	    if (temp->next == window) {
		temp->next = window->next;
		break;
	    }
	}
    }
    XtFree((char *) window);
    return 0;
}
    
void  CloneNTuBuildWindow(nTuBuildWindow *fromWindow,
                          nTuBuildWindow *toWindow)
{
    char *text;
    varGenNtuple *fromVariable, *toVariable;
    descrGenNtuple *fromDNTu, *toDNTu;
    int i;
    text = XmTextGetString(fromWindow->titleW);
    if (text[0] != '\0') XmTextSetString(toWindow->titleW,text);
    XtFree(text);
    text = XmTextGetString(fromWindow->descriptW);
    if (text[0] != '\0') XmTextSetString(toWindow->descriptW, text);
    XtFree(text);
    text = XmTextGetString(fromWindow->versionW);
    if (text[0] != '\0') XmTextSetString(toWindow->versionW, text);
    XtFree(text);
    text = XmTextGetString(fromWindow->multiplicityW);
    if (text[0] != '\0') XmTextSetString(toWindow->multiplicityW, text);
    XtFree(text);
    text = XmTextGetString(fromWindow->nameIndexW);
    if (text[0] != '\0') XmTextSetString(toWindow->nameIndexW, text);
    XtFree(text);
    
    fromDNTu = fromWindow->descrNtu;
    toDNTu = toWindow->descrNtu;
    /*
    ** make room for variable in new window, if needed
    */
    while (fromDNTu->numAvailable > toDNTu->numAvailable ) {
       ExtendVariableList(toWindow);
       toDNTu = toWindow->descrNtu;
    }
    for (i=0; i< fromDNTu->numAvailable; i++) 
        CopyVarGenNtuple(fromDNTu->variables[i], toDNTu->variables[i]);
    UpdateVariableList(toWindow, 1);
    UpdateDialogVFields(toWindow);
}


static int nTuBldWindowsOpen()
{
    return NTuBuildWindowList != NULL;
}
void UpdateVariableList(nTuBuildWindow *window, int newPos)
{
    XmString *stringTable;
    char varString[72], maxMult[10], *tc;
    varGenNtuple *variable;
    descrGenNtuple *dNTu;
    int i, j, listPos, ll, pl;
    
    /* Save the currently selected list position because
       it is lost when the list contents are reset */
    listPos = newPos==0 ? selectedListPosition(window->listW) : newPos;

    dNTu = window->descrNtu;
    
    stringTable = (XmString *) XtMalloc(sizeof(XmString) * dNTu->numAvailable);

    /* Compose the entries for the variable list */
    if (window->multiplicityBlank)  sprintf(maxMult,"?");
       else {
         if (window->langEnv == F77_LANGUAGE)
             sprintf(maxMult, "%d", dNTu->maxMultiplicity);
         else  sprintf(maxMult, "%d", (dNTu->maxMultiplicity - 1)); 
    }        
    for (i=0; i<dNTu->numAvailable; i++) {
        variable=dNTu->variables[i]; 
    	sprintf(varString, "V%d", i+1);
    	for (j=strlen(varString); j<69; j++) varString[j] = ' ';
    	if (!variable->nameBlank) {
    	    ll = strlen(variable->name);
    	    if (ll <= 20)  
    	        strncpy(&varString[5], variable->name, ll);
    	    else {
    	        strncpy(&varString[5], variable->name, 16);
    	        for (j=22; j<25; j++) varString[j] = '.';
    	    }
    	    if (window->langEnv == F77_LANGUAGE) { 
    	        strncpy(&varString[26], VarTypesNamesF77[variable->type],
    	                       strlen(VarTypesNamesF77[variable->type]));
    	        if (variable->numDim == 0) {
    	            if (variable->isFixedSize)  
    	                sprintf(&varString[43],"Scalar");
    	            else 
    	                sprintf(&varString[43],"(i=1...%s)", maxMult);
    	        } else {
    	            varString[43] = '(';
    	            tc = &varString[44];
    	            for (j=0; j<variable->numDim; j++, tc+=pl) { 
    	                if (j == variable->numDim-1)
    	                    sprintf(tc,"%d%n",variable->dimensions[j],&pl);
    	                else 
    	                    sprintf(tc,"%d,%n",variable->dimensions[j],&pl);
    	            }         
    	            if (variable->isFixedSize) sprintf(tc,")");
    	            else sprintf(tc,",i=1...%s)", maxMult);
    	        }      	                             
    	    } else { 
    	        strncpy(&varString[26], VarTypesNamesC[variable->type],
    	                        strlen(VarTypesNamesC[variable->type]));
    	        if (variable->numDim == 0) {
    	            if (variable->isFixedSize)  
    	                sprintf(&varString[43],"Scalar");
    	            else 
    	                sprintf(&varString[43],"[i=0...%s]", maxMult);
    	        } else {
    	            tc = &varString[43];
    	            if (variable->isFixedSize == False) { 
    	                sprintf(tc,"[i=0...%s]%n", maxMult, &pl);  
    	                tc +=pl;
    	            }    
    	            for (j=variable->numDim-1; j>=0 ; j--, tc+=pl) 
    	                    sprintf(tc,"[%d]%n",variable->dimensions[j],&pl); 
    	        }      	                             
    	    }                            
    	}             
   	else varString[65] = '\0';
    	stringTable[i] = XmStringCreateSimple(varString);
    }
    
    /* Add the entries to the list */
    XtVaSetValues(window->listW, XmNitems, stringTable,
    	    XmNitemCount, dNTu->numAvailable, 0);
    for (i=0; i<dNTu->numAvailable; i++)
    	XmStringFree(stringTable[i]);

    XtFree((char *) stringTable);

    /* Select or reselect the proper entry in the list */
    XmListSelectPos(window->listW, listPos, newPos==0);
}

void UpdateDialogVFields(nTuBuildWindow *window)
{
    int i;
    int listPos = selectedListPosition(window->listW);
    descrGenNtuple *dNTu = window->descrNtu;
    varGenNtuple *variable = dNTu->variables[listPos-1];
    
    /* Set each field to match the parameter data structure */
    
    if (variable->nameBlank) {
        XmTextSetString(window->nameW, "\0");
        XmTextSetString(window->descriptVarW, "\0");
        return;
    }
    UpdatingFields = True;
    XmTextSetString(window->nameW, variable->name);
    if (variable->description != NULL) 
       XmTextSetString(window->descriptVarW, variable->description);
       else XmTextSetString(window->descriptVarW, "\0");
    XtVaSetValues(window->typeMenu, XmNmenuHistory,
                  window->typeBtns[variable->type],0);
    XtVaSetValues(window->dimNumMenu, XmNmenuHistory,
                  window->dimBtns[variable->numDim],0);
    if (variable->isFixedSize) i = 1;
        else i=0;
    XtVaSetValues(window->indexingMenu, XmNmenuHistory,
                  window->indexingBtns[i], 0);
    UpdatingFields = False;
    
}

static void changeCB(Widget w, nTuBuildWindow *window, caddr_t call_data)
{
    char *text;
    int listPos = selectedListPosition(window->listW);
    descrGenNtuple *dNTu = window->descrNtu;
    varGenNtuple *variable = dNTu->variables[listPos-1];
    int i;
    Widget wM;
    
    text = XmTextGetString(window->nameW);
    if (text[0] == '\0') { 
    	variable->nameBlank = True;
    	return;
    }
    else {
    	variable->nameBlank = False;
        if (variable->name != NULL) free(variable->name);
        variable->name = 
          (char *) malloc(sizeof(char) * (strlen(text)+1));
        strcpy(variable->name, text);
    }    
    XtFree(text);
    
    text = XmTextGetString(window->descriptVarW);
    if (variable->description != NULL) free(variable->description);
    if (text[0] == '\0')  
        variable->description = NULL;
      else {
        variable->description = 
           (char *) malloc(sizeof(char) * (strlen(text)+1));
        strcpy(variable->description, text);
    }    
    XtFree(text);
    XtVaGetValues(window->typeMenu, XmNmenuHistory, &wM , 0);
    XtVaGetValues(wM, XmNuserData, &i,0);
    variable->type = i;
    
    XtVaGetValues(window->dimNumMenu, XmNmenuHistory, &wM , 0);
    XtVaGetValues(wM, XmNuserData, &i,0);
    variable->numDim = i;
    /*
    ** Get the dimensions from the dimension dialog box
    */
    if (variable->numDim > 0){
        getFilledDimensions(window);
        if (variable->dimensions[0] <=0 ) fillDimensions(window);
        for (i=0; i<variable->numDim; i++) {
              if (variable->dimensions[i] <= 0) 
              DialogF(DF_ERR, window->shell,1,
            " Wrong or missing dimension related to arrary index number %d !",
               "Acknowledged", i);
           }    
    }                 
    XtVaGetValues(window->indexingMenu, XmNmenuHistory, &wM , 0);
    XtVaGetValues(wM, XmNuserData, &i,0);
    variable->isFixedSize = i;
    
    /* Save the old parameter values for undo */
    saveVarForUndo(window);
    
    UpdateVariableList(window, listPos);

    XtSetSensitive(window->saveItem, True);
    XtSetSensitive(window->saveAsItem, True);
    /* User will prefer the traversal on the list */
    XmProcessTraversal(window->listW, XmTRAVERSE_CURRENT);
    /*
    ** One should not be able to generate .in, .h, template files until 
    ** this information is saved.. 
    */
    XtSetSensitive(window->generateF77, False);
    XtSetSensitive(window->generateC, False);
    XtSetSensitive(window->generateDbin, False);
    window->hasChanged = True;
}

static void undoCB(Widget w, nTuBuildWindow *window, caddr_t call_data)
{
    int i, listPos = selectedListPosition(window->listW);
    descrGenNtuple *dNTu = window->descrNtu;
    varGenNtuple *varTmp = dNTu->variables[listPos-1];
    
    /* Swap the undo information with the last variable changed */
    CopyVarGenNtuple(window->undoVariable, varTmp);
        
    /* Make the list reflect the new values (in a wasteful but easy way) */
    UpdateVariableList(window, listPos);
    
    UpdateDialogVFields(window);
    window->isSaved = False;
}

static void dismissCB(Widget w, nTuBuildWindow *window, caddr_t call_data)
{
    CloseNTuBuildWindow(window); 
}

static void cutCB(Widget w, nTuBuildWindow *window, caddr_t call_data)
{
    copyCB(w, window, call_data);
    clearCB(w, window, call_data);
}

static void copyCB(Widget w, nTuBuildWindow *window, caddr_t call_data)
{
    descrGenNtuple *dNTu = window->descrNtu;
    int i;
    int listPos = selectedListPosition(window->listW);
    varGenNtuple *varTmp = dNTu->variables[listPos-1];

    /* Put this information in the Variable-clipboard */
    window->selectedIndex = listPos-1;
    CopyVarGenNtuple(varTmp, &VariableClipboard);

    /* User will prefer the traversal on the list, not on this button again */
    XmProcessTraversal(window->listW, XmTRAVERSE_CURRENT);
}

static void pasteCB(Widget w, nTuBuildWindow *window, caddr_t call_data)
{
    descrGenNtuple *dNTu = window->descrNtu;
    int listPos = selectedListPosition(window->listW);
    varGenNtuple *varTmp = dNTu->variables[listPos-1];
    
    window->selectedIndex = listPos-1;
    /* Save the old parameter values for undo */
    saveVarForUndo(window);

    /* Transfer the Variable-clipboard contents to the selected variable */
    CopyVarGenNtuple(&VariableClipboard, varTmp);
    
    /* Make the list reflect the new values (in a wasteful but easy way) */
    UpdateVariableList(window, 0);

    /* User will prefer the traversal on the list, not on this button again */
    XmProcessTraversal(window->listW, XmTRAVERSE_CURRENT);
}

static void clearCB(Widget w, nTuBuildWindow *window, caddr_t call_data)
{
    descrGenNtuple *dNTu = window->descrNtu;
    int listPos = selectedListPosition(window->listW);
    varGenNtuple *varTmp = dNTu->variables[listPos-1];
    
    /* Save the old variable definitions for undo */
    saveVarForUndo(window);
    
    /* Clear the fields in the parameter data struct */
    varTmp->nameBlank = True;
    if (varTmp->name != NULL) {
           free(varTmp->name); 
           varTmp->name = NULL;
    }
    
    if (varTmp->description != NULL) {
           free(varTmp->description); 
           varTmp->description = NULL;
    }
    /* Make the list reflect the new values (in a wasteful but easy way) */
    UpdateVariableList(window, 0);

    /* User will prefer the traversal on the list, not on this button again */
    XmProcessTraversal(window->listW, XmTRAVERSE_CURRENT);
}

static void insertCB(Widget w, nTuBuildWindow *window, caddr_t call_data)
{
    descrGenNtuple *dNTu = window->descrNtu;
    varGenNtuple *varEnd, *varB, *varA;
    int i, j, listPos = selectedListPosition(window->listW);
    
     varEnd = dNTu->variables[dNTu->numAvailable-1];
     if (varEnd->nameBlank != True) ExtendVariableList(window);
     for (i=dNTu->numAvailable-1; i>(listPos-1); i--) {
         varB =  dNTu->variables[i];
         varA = dNTu->variables[i-1];
         if ((varA->nameBlank != True) || (varB->nameBlank != True)) 
             CopyVarGenNtuple(varA, varB);
     }
     clearCB(window->clearBtn, window, call_data);
     
    /* Make the list reflect the new values (in a wasteful but easy way) */
    UpdateVariableList(window, 0);
             
           
}
static void modifiedCB(Widget w,  nTuBuildWindow *window, 
                                               caddr_t call_data)
{
       if ((!UpdatingFields) && (window->changeBtn != NULL)) 
    	   XtSetSensitive(window->changeBtn, True);
    window->isSaved = False;   
}

static void modifiedNameCB(Widget w,  nTuBuildWindow *window, 
                                               caddr_t call_data)
{
    descrGenNtuple *dNTu = window->descrNtu;
    int value;
    char *text, text2[32], *endPtr;
    varGenNtuple *varTmp;
    
    text = XmTextGetString(w);
    if (text[0] == '\0') {
        XtFree(text);
        return;
    }          
    if(w == window->titleW) {
       window->titleBlank = True;
       strncpy(text2, text, 31);
       if (strlen(text) >= 31 ) text2[32] = '\0';
       if (!VerifyVarGenName(text2)) 
           DialogF(DF_ERR, window->shell, 1, 
                  "For the COMMON name or typedef name,\n\
,You must enter an alpha numeric string \n\
with no special character, (for the first 31 char.) \n\
, the first char is not a number \n", "Acknowledged");
       else window->titleBlank = False;
    } else if (w == window->nameIndexW) {
       if (!VerifyVarGenName(text)) {
           window->nameIndexBlank = True; 
           DialogF(DF_ERR, window->shell, 1, 
          "For the name above,You must enter an alpha numeric string \n\
with no special character, first char is not a number \n", "Acknowledged");
 
           } else window->nameIndexBlank = False;
    } else if(w ==  window->multiplicityW) {
       window->multiplicityBlank = True;
       removeWhiteSpace(text);        /* Remove blanks and tabs */
       value = strtol(text, &endPtr, 10);
       if (*endPtr != '\0')			/* Whole string not parsed */
           DialogF(DF_ERR, window->shell, 1, 
           "You must enter an integer number", "Acknowledged");
       else {
           window->multiplicityBlank = False;
           window->descrNtu->maxMultiplicity = value;
           if (value == 0) { 
                XtVaSetValues(window->indexingMenu, XmNmenuHistory,
                               window->indexingBtns[1], 0);
                XtSetSensitive(window->indexingMenu, FALSE);
           } else {
                XtVaSetValues(window->indexingMenu, XmNmenuHistory,
                               window->indexingBtns[0], 0);
                XtSetSensitive(window->indexingMenu, TRUE);
                               
           }
       }    
    } else if (w == window->nameW) {
       varTmp = dNTu->variables[window->selectedIndex];
       if (!VerifyVarGenName(text)) {
           varTmp->nameBlank = True; 
           DialogF(DF_ERR, window->shell, 1, 
           "For this name, You must enter an alpha numeric string \n\
with no special character, first char is not a number \n", "Acknowledged");
 
           }
    } else if (w == window->titleW) {
       if (!VerifyVarGenName(text)) {
           DialogF(DF_ERR, window->shell, 1, 
           "For this name, You must enter an alpha numeric string \n\
with no special character, first char is not a number \n", "Acknowledged");
 
           }
    } else 
      printf("Internal error from modifiedNameCB, please report");	
    XtFree(text);	
    window->isSaved = False;   
}

static void listSelectionCB(Widget w, nTuBuildWindow *window,
	XmListCallbackStruct *call_data)
{
    descrGenNtuple *dNTu = window->descrNtu;
    
    int index = call_data->item_position - 1;
    window->selectedIndex = index;
    UpdateDialogVFields(window);
    if (window->changeBtn != NULL) XtSetSensitive(window->changeBtn, True);
    if (index == (dNTu->numAvailable -1)) ExtendVariableList(window); 
}

static void saveVarForUndo(nTuBuildWindow *window)
{    
    char *string, *tc, *tc2;
    int i;
    descrGenNtuple *dNTu = window->descrNtu;
    int listPos = selectedListPosition(window->listW);
    varGenNtuple *varTmp = dNTu->variables[listPos-1];
    
    
    /* Save the old definitions */
    CopyVarGenNtuple(varTmp, window->undoVariable);
    
    /* Un-dim the undo button */
    if (window->undoBtn != NULL) XtSetSensitive(window->undoBtn, True);
}
/*
** Get the position of the selection in the filter list widget
*/
static int selectedListPosition(Widget listW)
{
    int listPos;
    int *posList = NULL, posCount = 0;

    if (!XmListGetSelectedPos(listW, &posList, &posCount)) {
	fprintf(stderr, "NTuBuild: Internal error (nothing selected)\n");
    	return 0;
    }
    listPos = *posList;
    XtFree((char *)posList);
    return listPos;
}
static void typeMenuCB(Widget w, nTuBuildWindow *window, caddr_t call_data)
{
    int ival;
    descrGenNtuple *dNTu = window->descrNtu;
    varGenNtuple *varTmp = dNTu->variables[window->selectedIndex];
    
    XtVaGetValues(w, XmNuserData, &ival,0);
    varTmp->type = ival;
}
    
    	
static void dimMenuCB(Widget w, nTuBuildWindow *window, caddr_t call_data)
{
    int ival, oldDim;
    descrGenNtuple *dNTu = window->descrNtu;
    varGenNtuple *varTmp = dNTu->variables[window->selectedIndex];
    
    oldDim = varTmp->numDim;
    XtVaGetValues(w, XmNuserData, &ival,0);
    varTmp->numDim = ival;
    if (ival != 0)  
       fillDimensions(window);
}
static void indexedMenuCB(Widget w, nTuBuildWindow *window, caddr_t call_data)
{
    int ival;
    descrGenNtuple *dNTu = window->descrNtu;
    varGenNtuple *varTmp = dNTu->variables[window->selectedIndex];
    
    XtVaGetValues(w, XmNuserData, &ival,0);
    if (ival == 0) 
        varTmp->isFixedSize = False;
    else 
        varTmp->isFixedSize = True;
}
static void fillDimensions(nTuBuildWindow *window)
{  
    char *title, *comment, *text;
    descrGenNtuple *dNTu = window->descrNtu;
    varGenNtuple *varTmp = dNTu->variables[window->selectedIndex];

    if (window->dimDialogW == NULL) createDimDialog(window);    
    else {
       XtManageChild(window->dimDialogForm);
       XMapRaised(XtDisplay(window->dimDialogShell),
                  XtWindow(window->dimDialogShell));
    }
    text = XmTextGetString(window->nameW);
    if (text[0] == '\0') { 
       title = (char *) malloc(sizeof(char) * 80);
       strcpy(title, "Dimension(s) for unNamed Variable");
    } else {
       title = (char *) malloc(sizeof(char) * (strlen(text) + 20));
       sprintf(title,"Dimension(s) for %s", text);
    }   
    XtVaSetValues(window->dimDialogShell, XmNtitle, title, 0);
    free(title);
    XtFree(text);
    updateDimPanel(window);
}
                 
static void createDimDialog(nTuBuildWindow *window)
{
    descrGenNtuple *dNTu = window->descrNtu;
    varGenNtuple *varTmp = dNTu->variables[window->selectedIndex];
    
    Widget form, commentLbl;
    Widget dismissBtn, sep;
    int ll;
    XmString s1;
    int ac;
    char *comment;
    Arg args[20];
    
    /* Create the top level form & dialog shell */
    ac = 0;
    XtSetArg(args[ac], XmNautoUnmanage, False); ac++;
    form = XmCreateFormDialog(window->shell, "form", args, ac);
    window->dimDialogShell = XtParent(form);
    window->dimDialogForm = form;
    XtManageChild(form);
    AddMotifCloseCallback(XtParent(form), 
                         (XtCallbackProc)dismissDimCB, window);
    
    /* Create the dismiss button at the bottom */
    dismissBtn = XtVaCreateManagedWidget("dismissBtn",
    	    xmPushButtonGadgetClass, form,
    	    XmNlabelString, s1=XmStringCreateSimple("O.K. & Dismiss"),
    	    XmNbottomAttachment, XmATTACH_FORM,
    	    XmNbottomOffset, 4,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 20,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 80, 0);
    XmStringFree(s1);
    XtAddCallback(dismissBtn, XmNactivateCallback,
    	    (XtCallbackProc)dismissDimCB, (void *)window);
    XtVaSetValues(form, XmNdefaultButton, dismissBtn, 0);
    sep = XtVaCreateManagedWidget("sep", xmSeparatorGadgetClass, form,
    	    XmNbottomAttachment, XmATTACH_WIDGET,
    	    XmNbottomWidget, dismissBtn,
    	    XmNbottomOffset, 4,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 0,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNrightPosition, 100, 0);
     /* Create the multiline comment label */
#if AIX
    window->dimDialogComLbl = XtVaCreateManagedWidget("comDimLbl",
            xmLabelGadgetClass, form,
    	    XmNlabelString, s1=XmStringCreateLtoR(
"Help for \n dimensions \n        settings...    ", XmSTRING_DEFAULT_CHARSET),
    	    XmNtopAttachment, XmATTACH_FORM,
    	    XmNleftAttachment, XmATTACH_FORM, 0);
#else
    window->dimDialogComLbl = XtVaCreateManagedWidget("comDimLbl",
            xmLabelGadgetClass, form,
    	    XmNlabelString, s1=XmStringCreateLtoR(
"Help for \n dimensions \n        settings...    ", XmFONTLIST_DEFAULT_TAG),
    	    XmNtopAttachment, XmATTACH_FORM,
    	    XmNleftAttachment, XmATTACH_FORM, 0);
#endif
    XmStringFree(s1);
    /* Create the Text field underneath this comment */
    window->dimDialogW = XtVaCreateManagedWidget("dimText", 
            xmTextWidgetClass, form,
    	    XmNmaxLength, MAX_FIELD_DIMENSIONS,
    	    XmNbottomAttachment, XmATTACH_WIDGET,
    	    XmNbottomWidget, sep,
    	    XmNtopAttachment, XmATTACH_WIDGET,
    	    XmNtopWidget, window->dimDialogComLbl,
    	    XmNtopOffset, 2,
    	    XmNleftAttachment, XmATTACH_FORM, 
    	    XmNleftOffset, 10, 0);
    	        
    XmAddTabGroup(window->dimDialogW);
    XmAddTabGroup(dismissBtn);
    /* Display it all */
    XtRealizeWidget(window->dimDialogShell);
    
}
             
static void updateDimPanel(nTuBuildWindow *window)
{    
    /* Updated the comment, to guide the user to get correct syntax */
    
    char *comment, *tc;
    XmString s1;
    int i, ll, pl;
    descrGenNtuple *dNTu = window->descrNtu;
    varGenNtuple *varTmp = dNTu->variables[window->selectedIndex];
    char values[MAX_FIELD_DIMENSIONS+1];

    if (window->dimDialogW == NULL) return;
    if (varTmp->numDim < 1) return;    
    
    ll = varTmp->numDim * 80 + 100;
    if (!varTmp->isFixedSize) ll += 100;
    comment = (char *) malloc(sizeof(char) * ll); tc = comment;
    if (varTmp->isFixedSize) 
      sprintf (tc,
       "This variable is not indexed, e.g., fixed size\n%n",&pl);  
    else       sprintf (tc,
       "This variable is indexed, Do not include that dimension \n\
(as that dimension is identical for all indexed Variable)\n%n",&pl);
    tc += pl;
    switch(varTmp->numDim) {
        case 1:
           sprintf(tc,"Array: type one integer number");
           break;
        case 2:
           if (window->langEnv == F77_LANGUAGE)
             sprintf(tc,
"Matrix: type 2 integer numbers, separated by commas\n\
 (e.g. 4,5 (FORTRAN order convention!)) ");
           else 
             sprintf(tc,
"Matrix: type 2 integer numbers, separated by commas\n\
 (e.g. 5,4 (C order convention!)) ");
           
           break;
        default:
           if (window->langEnv == F77_LANGUAGE)
              sprintf(tc,
"Matrix: type %d integer numbers, separated by commas\n\
 (e.g. 4,5,... (FORTRAN ordering convention!)) ", varTmp->numDim);
           else 
              sprintf(tc,
"Matrix: type %d integer numbers, separated by commas\n\
 (e.g. 5,4,... (C ordering convention!)) ", varTmp->numDim);
           break;
     }
#if AIX
    s1 = XmStringCreateLtoR(comment, XmSTRING_DEFAULT_CHARSET);
#else
    s1 = XmStringCreateLtoR(comment, XmFONTLIST_DEFAULT_TAG);
#endif
    free(comment);       
    XtVaSetValues(window->dimDialogComLbl, XmNlabelString, s1, 0);
    XmStringFree(s1);
    /*
    ** Update now the text field 
    */
    tc = values;
    sprintf (tc, " ");
    if (varTmp->numDim > 0) {
      if (window->langEnv == F77_LANGUAGE) 
          for (i=0; i<varTmp->numDim; i++) {
              if (i == varTmp->numDim-1)
              sprintf(tc,"%d%n",varTmp->dimensions[i], &pl);
              else sprintf(tc,"%d,%n",varTmp->dimensions[i], &pl);  
              tc += pl;
          } else 
          for (i=varTmp->numDim-1; i>=0; i--) {
              if (i == 0)
              sprintf(tc,"%d%n",varTmp->dimensions[i], &pl);
              else sprintf(tc,"%d,%n",varTmp->dimensions[i], &pl);  
              tc += pl;
          }
    }     
    XmTextSetString(window->dimDialogW, values); 
}

static void dismissDimCB(Widget w, nTuBuildWindow *window, caddr_t call_data)
{
    getFilledDimensions(window);
    XtUnmanageChild(window->dimDialogForm);
}

static void nullCB(Widget w, nTuBuildWindow *window, caddr_t call_data)
{
    printf("NullCB\n");
}
static void getFilledDimensions(nTuBuildWindow *window)
{
    char *string, *tc, *tc2;
    int i;
    descrGenNtuple *dNTu = window->descrNtu;
    varGenNtuple *varTmp = dNTu->variables[window->selectedIndex];
    
    if (window->dimDialogW == NULL) return;    
    if (varTmp->numDim < 1) return;    
    string = XmTextGetString(window->dimDialogW);
    removeWhiteSpace(string);			/* Remove blanks and tabs */
    tc = string;
    if (window->langEnv == F77_LANGUAGE) 
        for (i=0; i<varTmp->numDim; i++) {
            if (strlen(tc) < 1) varTmp->dimensions[i] = -1;
            else {
                varTmp->dimensions[i] = strtol(tc, &tc2, 10);
                tc = tc2+1;
            }   
        } 
    else 
        for (i=i<varTmp->numDim-1; i>=0; i--) {
            if (strlen(tc) < 1) varTmp->dimensions[i] = -1;
            else {
                varTmp->dimensions[i] = strtol(tc, &tc2, 10);
                tc = tc2+1;
            }   
        } 
        
     XtFree(string);
}
/*
** Remove the white space (blanks and tabs) from a string
*/
static void removeWhiteSpace(char *string)
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


