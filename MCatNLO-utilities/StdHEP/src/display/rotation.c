/*******************************************************************************
*									       *
* rotation.c -- Read and set Euler rotation angles for event display	       *
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
* Written by Mark Edel & Paul Lebrun					       *
*									       *
*******************************************************************************/
static char SCCSID[] = "@(#)rotation.c	1.1	4/6/92";

#include <stdio.h>
#include <math.h>
#include <Xm/Xm.h>
#include <Xm/SelectioB.h>
#include <Xm/Text.h>
#include <Xm/Form.h>
#include <Xm/LabelG.h>
#include "spin/Spin.h"
#include "phase.h"
#include "space.h"
#include "phaseP.h"

enum angleNames {ALPHA, BETA, GAMMA};

#define PI M_PI
#define EPSILON 1.0E-4

#define RADIANS(x)  (M_PI * 2.0 * (x) / 360.0)
#define DEGREES(x)  ((x) / (M_PI * 2.0) * 360.0)

static void anglesToMatrix(double alpha, double beta, double gamma,
			   double m[3][3]);
static void matrixToAngles(double m[3][3], double *alpha,
			   double *beta, double *gamma);
static void createAbsRotationPanel(StdHepWindow *window);
static void absRotOkCB(Widget w, StdHepWindow *window, caddr_t call_data);
static void absRotCancelCB(Widget w, StdHepWindow *window, caddr_t call_data);

void ShowAbsRotationPanel(StdHepWindow *window)
{
    if (window->absRotationPanel == NULL)
    	createAbsRotationPanel(window);
    else if (!XtIsManaged(window->absRotationPanel))
    	XtManageChild(window->absRotationPanel);
    else
    	XRaiseWindow(XtDisplay(window->absRotationPanel),
    		     XtWindow(XtParent(window->absRotationPanel)));
}    

void UpdateRotationPanel(StdHepWindow *window)
{
    double alpha, beta, gamma, m[3][3];
    char rotString[100];
    XmString cRotString;
    
    if (window->absRotationPanel == NULL)
    	return;
    if (!XtIsManaged(window->absRotationPanel))
    	return;
    	
    SpinGetTransform(window->spin, m);
    matrixToAngles(m, &alpha, &beta, &gamma);
    sprintf(rotString, "Alpha %d, Beta %d, Gamma %d\n",
    	    (int)alpha, (int)beta, (int)gamma);
    cRotString = MKSTRING(rotString);
    SET_ONE_RSRC(window->absRotationLabel, XmNlabelString, cRotString);
    XmStringFree(cRotString);
}

static void anglesToMatrix(double alpha, double beta, double gamma,
			   double m[3][3])
{
    double cosAlpha, sinAlpha, cosBeta, sinBeta, cosGamma, sinGamma;
    
    sinAlpha = sin(RADIANS(alpha));
    sinBeta = sin(RADIANS(beta));
    sinGamma = sin(RADIANS(gamma));
    cosAlpha = cos(RADIANS(alpha));
    cosBeta = cos(RADIANS(beta));
    cosGamma = cos(RADIANS(gamma));
    
    m[0][0] = cosGamma*cosBeta*cosAlpha - sinGamma*sinAlpha;
    m[0][1] = 0. - sinGamma*cosBeta*cosAlpha - cosGamma*sinAlpha;
    m[0][2] = sinBeta*cosAlpha;
    m[1][0] = cosGamma*cosBeta*sinAlpha + sinGamma*cosAlpha;
    m[1][1] = 0. - sinGamma*cosBeta*sinAlpha + cosGamma*cosAlpha;
    m[1][2] = sinBeta*sinAlpha;
    m[2][0] = 0. - cosGamma*sinBeta;
    m[2][1] = sinGamma*sinBeta;
    m[2][2] = cosBeta;
}

/*
** Calculate current Euler rotation of a 3x3 matrix as three angles,
** alpha, beta, and gamma.  This routine solves the matrix
** equation 4.80 in "The Mathematical Methods of Physics" for
** alpha beta and gamma.
*/
static void matrixToAngles(double m[3][3], double *alpha,
			   double *beta, double *gamma)
{
    double cosAlpha, sinAlpha, cosBeta, sinBeta, cosGamma, sinGamma;
    double alphaResult, betaResult, gammaResult, betaResults[2];
    double temp, testMatrix[3][3];
    int i, j, k, verified;
    
    /* try solutions for both beta = (+/-)acos(m[2][2]) */
    cosBeta = m[2][2];
    betaResults[0] = acos(cosBeta);
    betaResults[1] = 0. - betaResults[0];
    for (k=0; k<=1; k++) {
    	betaResult = betaResults[k];
    	sinBeta = sin(betaResult);
    	if (fabs(sinBeta) < EPSILON) {
    	    /* beta angle close to or at zero */
    	    *beta = 0.;
    	    *gamma = 0.;
    	    /* compute alpha */
    	    sinAlpha = m[1][0];
    	    cosAlpha = m[0][0];
    	    temp = acos(cosAlpha);
    	    if (fabs(sin(temp) - sinAlpha) < EPSILON)
    	    	*alpha = DEGREES(temp);
    	    else if (fabs(sinAlpha + sin(temp)) < EPSILON)
    	    	*alpha = DEGREES(0.-temp);
    	    else
    	    	printf("Internal Error, input not an Euler matrix");
    	    if (*alpha < 0) *alpha += 360;
    	    return;
    	}
    	/* compute gamma */
    	sinGamma = m[2][1] / sinBeta;
    	cosGamma = 0.-m[2][0] / sinBeta;
    	temp = acos(cosGamma);
    	if (fabs(sin(temp) - sinGamma) < EPSILON)
    	    gammaResult = temp;
    	else if (fabs(sin(temp) + sinGamma) < EPSILON)
    	    gammaResult = 0.-temp;
    	else
    	    gammaResult = EPSILON;

    	/* compute alpha */
    	sinAlpha = m[1][2] / sinBeta;
    	cosAlpha = m[0][2] / sinBeta;
    	temp = acos(cosAlpha);
    	if (fabs(sin(temp) - sinAlpha) < EPSILON)
    	    alphaResult = temp;
    	else if (fabs(sin(temp) + sinAlpha) < EPSILON)
    	    alphaResult = 0.-temp;
    	else
    	    alphaResult = EPSILON;

    	/* Verify and select solution */
    	anglesToMatrix(DEGREES(alphaResult), DEGREES(betaResult), 
    	    	       DEGREES(gammaResult), testMatrix);
    	verified = True;
    	for (i=0; i<=1; i++) {
    	    for (j=0; j<=1; j++) {
    	    	if (fabs(testMatrix[i][j] - m[i][j]) > EPSILON)
    	    	    verified = False;
    	    }
    	}
    	if (verified) {
    	    if (alphaResult < 0) alphaResult += 2*PI;
    	    if (betaResult < 0) betaResult += 2*PI;
    	    if (gammaResult < 0) gammaResult += 2*PI;
    	    *alpha = DEGREES(alphaResult);
    	    *beta = DEGREES(betaResult);
    	    *gamma = DEGREES(gammaResult);
    	    return;
    	}
    }
    printf("Internal error, not an Euler matrix\n");
}

static void createAbsRotationPanel(StdHepWindow *window)
{
    Arg args[50];
    int ac;
    Widget form, selBox, alphaText, betaText, gammaText, currentValuesLabel;
    Widget topLabel, setToLabel, alphaLabel, betaLabel, gammaLabel;
    XmString s1;

    ac = 0;
    selBox = XmCreatePromptDialog(window->shell, "rotationBox", args, ac);
    XtAddCallback(selBox, XmNapplyCallback,
                 (XtCallbackProc)  absRotOkCB, window);
    XtAddCallback(selBox, XmNokCallback,
                 (XtCallbackProc)  absRotOkCB, window);
    XtAddCallback(selBox, XmNcancelCallback,
                 (XtCallbackProc)  absRotCancelCB, window);
    XtUnmanageChild(XmSelectionBoxGetChild(selBox, XmDIALOG_TEXT));
    XtUnmanageChild(XmSelectionBoxGetChild(selBox, XmDIALOG_SELECTION_LABEL));
    XtUnmanageChild(XmSelectionBoxGetChild(selBox, XmDIALOG_HELP_BUTTON));
    XtManageChild(XmSelectionBoxGetChild(selBox, XmDIALOG_APPLY_BUTTON));
    SET_ONE_RSRC(XtParent(selBox), XmNtitle, "Rotation Angles");
    
    ac = 0;
    form = XmCreateForm(selBox, "form", args, ac);

    ac = 0;
    XtSetArg(args[ac], XmNlabelString,
    		(s1=MKSTRING("Current (Euler) angles of rotation:"))); ac++;
    topLabel = XmCreateLabelGadget(form, "topLabel", args, ac);
    XmStringFree(s1);
    XtManageChild(topLabel);

    ac = 0;
    XtSetArg(args[ac], XmNalignment, XmALIGNMENT_CENTER); ac++;
    XtSetArg(args[ac], XmNlabelString, (s1=MKSTRING("alpha= beta= gamma="))); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNrightAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNtopWidget, topLabel); ac++;
    currentValuesLabel = XmCreateLabelGadget(form, "currentValuesLabel",
    					    args, ac);
    XmStringFree(s1);
    XtManageChild(currentValuesLabel);
 
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, (s1=MKSTRING("Set to:"))); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopWidget, currentValuesLabel); ac++;
    setToLabel = XmCreateLabelGadget(form, "setToLabel", args, ac);
    XmStringFree(s1);
    XtManageChild(setToLabel);
 
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, (s1=MKSTRING("Alpha"))); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNtopWidget, setToLabel); ac++;
    alphaLabel = XmCreateLabelGadget(form, "alphaLabel", args, ac);
    XmStringFree(s1);
    XtManageChild(alphaLabel);
 
    ac = 0;
    XtSetArg(args[ac], XmNcolumns, 3); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopWidget, setToLabel); ac++;
    XtSetArg(args[ac], XmNleftWidget, alphaLabel); ac++;
    alphaText = XmCreateText(form, "alphaText", args, ac);
    XtManageChild(alphaText);
 
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, (s1=MKSTRING("Beta"))); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNtopWidget, setToLabel); ac++;
    XtSetArg(args[ac], XmNleftWidget, alphaText); ac++;
    betaLabel = XmCreateLabelGadget(form, "betaLabel", args, ac);
    XmStringFree(s1);
    XtManageChild(betaLabel);

    ac = 0;
    XtSetArg(args[ac], XmNcolumns, 3); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopWidget, setToLabel); ac++;
    XtSetArg(args[ac], XmNleftWidget, betaLabel); ac++;
    betaText = XmCreateText(form, "betaText", args, ac);
    XtManageChild(betaText);
 
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, (s1=MKSTRING("Gamma"))); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNtopWidget, setToLabel); ac++;
    XtSetArg(args[ac], XmNleftWidget, betaText); ac++;
    gammaLabel = XmCreateLabelGadget(form, "gammaLabel", args, ac);
    XmStringFree(s1);
    XtManageChild(gammaLabel);

    ac = 0;
    XtSetArg(args[ac], XmNcolumns, 3); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopWidget, setToLabel); ac++;
    XtSetArg(args[ac], XmNleftWidget, gammaLabel); ac++;
    gammaText = XmCreateText(form, "gammaText", args, ac);
    XtManageChild(gammaText);

    XtManageChild(form);
    XtManageChild(selBox);
    window->absRotationPanel = selBox;
    window->absRotationLabel = currentValuesLabel;
    window->absRotFields[ALPHA] = alphaText;
    window->absRotFields[BETA] = betaText;
    window->absRotFields[GAMMA] = gammaText;
}

static void absRotOkCB(Widget w, StdHepWindow *window, caddr_t call_data)
{
    double angles[3], m[3][3], value;
    char *string;
    int i, nRead;
    
    for (i=ALPHA; i<=GAMMA; i++) {
	string = XmTextGetString(window->absRotFields[i]);
	nRead = sscanf(string, "%lf", &value);
	XtFree(string);
	if (nRead != 1 || value < 0. || value > 360.) {
    	    XBell(XtDisplay(w), 100);
    	    return;
	} else {
    	    angles[i] = (int)value;
	}
    }
    anglesToMatrix(angles[ALPHA], angles[BETA], angles[GAMMA], m);
    SpinSetTransform(window->spin, m);
    	
}

static void absRotCancelCB(Widget w, StdHepWindow *window, caddr_t call_data)
{
    int i;
    
    for (i=ALPHA; i<=GAMMA; i++)
	XmTextSetString(window->absRotFields[i], "");
}
