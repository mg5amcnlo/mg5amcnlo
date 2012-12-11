/*******************************************************************************
*									       *
* settings.c -- Misc. dialogs for setting display parameters		       *
*									       *
* Copyright (c) 1991 Universities Research Association, Inc.		       *
* All rights reserved.							       *
* 									       *
* This material resulted from work developed under a Government Contract and   *
* is subject to the following license:  The Government retains a paid-up,      *
* nonexclusive, irrevocable worldwide license to reproduce, prepare derivative *
* works, perform publicly and display publicly by or for the Government,       *
* including the right to distribute to other Government contractors.  Neither  *
* the United States nor the United States Department of Energy, nor any of     *
* their employees, makes any warrenty, express or implied, or assumes any      *
* legal liability or responsibility for the accuracy, completeness, or         *
* usefulness of any information, apparatus, product, or process disclosed, or  *
* represents that its use would not infringe privately owned rights.           *
*                                        				       *
* Fermilab Nirvana GUI Library						       *
* August 10, 1991							       *
*									       *
* Written by Mark Edel							       *
*									       *
*******************************************************************************/
static char SCCSID[] = "@(#)settings.c	1.1	4/6/92";
#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/Scale.h>
#include <Xm/SelectioB.h>
#include <Xm/Text.h>
#include "spin/Spin.h"
#include "phase.h"
#include "space.h"
#include "phaseP.h"
 
static void createVectorLengthPanel(PhaseWindow *window);
static void createButtonRotationPanel(StdHepWindow *window);
static void resetBtnRotText(StdHepWindow *window, Widget selBox);
static void sliderCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void btnRotOkCB(Widget w, StdHepWindow *window, caddr_t call_data);
static void btnRotCancelCB(Widget w, StdHepWindow *window, caddr_t call_data);

void ShowVectorLengthPanel(PhaseWindow *window)
{
    if (window->vectorLengthPanel == NULL)
    	createVectorLengthPanel(window);
    else if (!XtIsManaged(window->vectorLengthPanel))
    	XtManageChild(window->vectorLengthPanel);
    else
    	XRaiseWindow(XtDisplay(window->vectorLengthPanel),
    		     XtWindow(XtParent(window->vectorLengthPanel)));
}    

void ShowButtonRotationPanel(StdHepWindow *window)
{
    if (window->btnRotationPanel == NULL)
    	createButtonRotationPanel(window);
    else if (!XtIsManaged(window->btnRotationPanel))
    	XtManageChild(window->btnRotationPanel);
    else
    	XRaiseWindow(XtDisplay(window->btnRotationPanel),
    		     XtWindow(XtParent(window->btnRotationPanel)));
}    

static void createVectorLengthPanel(PhaseWindow *window)
{
    Arg args[50];
    int ac;
    Widget selBox, slider;
    XmString s1, s2;

    ac = 0;
    XtSetArg(args[ac], XmNselectionLabelString, 
    	     (s1=MKSTRING("Percent of vector visible toward center"))); ac++;
    XtSetArg(args[ac], XmNokLabelString, 
    	     (s2=MKSTRING("Dismiss"))); ac++;
    selBox = XmCreatePromptDialog(window->shell, "vectorLengthBox", args, ac);
    XmStringFree(s1);
    XmStringFree(s2);
    XtUnmanageChild(XmSelectionBoxGetChild(selBox, XmDIALOG_TEXT));
    XtUnmanageChild(XmSelectionBoxGetChild(selBox, XmDIALOG_HELP_BUTTON));
    XtUnmanageChild(XmSelectionBoxGetChild(selBox, XmDIALOG_CANCEL_BUTTON));
    SET_ONE_RSRC(XtParent(selBox), XmNtitle, "Vector Length");

    ac = 0;
    XtSetArg(args[ac], XmNshowValue, True); ac++;
    XtSetArg(args[ac], XmNorientation, XmHORIZONTAL); ac++;
    XtSetArg(args[ac], XmNvalue, (int)(window->vectorLength * 100.)); ac++;
    slider = XmCreateScale(selBox, "vectorLength", args, ac);
    XtAddCallback(slider, XmNdragCallback,
                   (XtCallbackProc) sliderCB, window);
    XtAddCallback(slider, XmNvalueChangedCallback, 
                   (XtCallbackProc) sliderCB, window);
    XtManageChild(slider);
    
    XtManageChild(selBox);
    
    window->vectorLengthPanel = selBox;
}
    
static void createButtonRotationPanel(StdHepWindow *window)
{
    Arg args[50];
    int ac;
    Widget selBox;
    XmString s1;
    char incrText[20];

    ac = 0;
    XtSetArg(args[ac], XmNselectionLabelString, (s1=MKSTRING("\
Number of degrees to rotate when\n\
buttons or arrow keys are pressed"))); ac++;
    selBox = XmCreatePromptDialog(window->shell, "rotationIncrBox", args, ac);
    XtAddCallback(selBox, XmNapplyCallback,
                  (XtCallbackProc)  btnRotOkCB, window);
    XtAddCallback(selBox, XmNokCallback,
                  (XtCallbackProc)  btnRotOkCB, window);
    XtAddCallback(selBox, XmNcancelCallback,
                  (XtCallbackProc) btnRotCancelCB, window);
    XmStringFree(s1);
    XtUnmanageChild(XmSelectionBoxGetChild(selBox, XmDIALOG_HELP_BUTTON));
    XtManageChild(XmSelectionBoxGetChild(selBox, XmDIALOG_APPLY_BUTTON));
    SET_ONE_RSRC(XtParent(selBox), XmNtitle, "Button Rotation");
    resetBtnRotText(window, selBox);    
    XtManageChild(selBox);
    
    window->btnRotationPanel = selBox;
}

static void resetBtnRotText(StdHepWindow *window, Widget selBox)
{
    char resetText[20];

    sprintf(resetText, "%g", window->buttonRotateDegrees);
    XmTextSetString(XmSelectionBoxGetChild(selBox, XmDIALOG_TEXT), resetText);
}

static void sliderCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    int value;
    
    XmScaleGetValue(w, &value);
    if (window->vectorLength != ((double)value)/100.) {
	window->vectorLength = ((double)value)/100.;
	DrawEvent(window, False);
    }
}

static void btnRotOkCB(Widget w, StdHepWindow *window, caddr_t call_data)
{
    char *string;
    double value = -1.;
    int nRead;
    
    string = XmTextGetString(XmSelectionBoxGetChild(w, XmDIALOG_TEXT));
    nRead = sscanf(string, "%lf", &value);
    XtFree(string);
    if (nRead != 1 || value <= 0. || value > 100.) {
    	XBell(XtDisplay(w), 100);
    	resetBtnRotText(window, w);
    } else {
    	window->buttonRotateDegrees = (int)value;
    }
}

static void btnRotCancelCB(Widget w, StdHepWindow *window, caddr_t call_data)
{
    resetBtnRotText(window, w);
}
