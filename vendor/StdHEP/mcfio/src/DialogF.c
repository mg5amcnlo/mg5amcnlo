/*******************************************************************************
*									       *
* DialogF -- modal dialog printf routine				       *
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
* their employees, makes any warranty, express or implied, or assumes any      *
* legal liability or responsibility for the accuracy, completeness, or         *
* usefulness of any information, apparatus, product, or process disclosed, or  *
* represents that its use would not infringe privately owned rights.           *
*                                        				       *
* Fermilab Nirvana GUI Library						       *
* April 26, 1991							       *
*									       *
* Written by Joy Kyriakopulos						       *
*									       *
*******************************************************************************/
static char SCCSID[] = "@(#)DialogF.c	1.8	8/19/94";
#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MessageB.h>
#include <Xm/DialogS.h>
#include <Xm/SelectioB.h>
#include <X11/StringDefs.h>
#include <X11/Intrinsic.h>
#include <stdarg.h>
#include "misc.h"
#include "DialogF.h"
#define NUM_DIALOGS_SUPPORTED 6
#define NUM_BUTTONS_SUPPORTED 3		/* except prompt dialog */
#define NUM_BUTTONS_MAXPROMPT 4

struct dfcallbackstruct {
    unsigned button;		/* button pressed by user		     */
    Boolean done_with_dialog;	/* set by callbacks; dialog can be destroyed */
    unsigned apply_up;		/* will = 1 when apply button managed	     */
};

static void apply_callback (Widget w, struct dfcallbackstruct *client_data,
	caddr_t call_data);
static void help_callback (Widget w, struct dfcallbackstruct *client_data,
	caddr_t call_data);
static void cancel_callback (Widget w, struct dfcallbackstruct *client_data,
	caddr_t call_data);
static void ok_callback (Widget w, struct dfcallbackstruct *client_data,
	caddr_t call_data);
static void focusCB(Widget w, Widget dialog, caddr_t call_data);

/******************************************************************************
 * DialogF ()								      *
 *									      *
 *	function to put up modal versions of the XmCreate<type>Dialog	      *
 *	functions (where <type> is Error, Information, Prompt, Question,      *
 *	Message, or Warning).  The purpose of this routine is to allow a      *
 *	printf-style dialog box to be invoked in-line without giving control  *
 *	back to the main loop.  The message string can contain vsprintf       *
 *	specifications.  DialogF displays the dialog in application-modal     *
 *	style, blocking the application and keeping the modal dialog as the   *
 *	top window until the user responds.  DialogF accepts a variable       *
 *	number of arguments, so the calling routine needs to #include         *
 *	<stdarg.h>.		     					      *
 *									      *
 *    Arguments:							      *
 *									      *
 *	unsigned   dialog_type	- dialog type (e.g. DF_ERR for error dialog)  *
 *				  (refer to DialogF.h for dialog type values) *
 *	Widget	   parent	- parent widget ID			      *
 *	unsigned   n		- # of buttons to display		      *
 *				  if = 0 use defaults in XmCreate<type>Dialog *
 *				  value in range 0 to NUM_BUTTONS_SUPPORTED   *
 *				  (for prompt dialogs: NUM_BUTTONS_MAXPROMPT) *
 *	char *     msgstr	- message string (may contain conversion      *
 *				  specifications for vsprintf)		      *
 *	char *	   input_string - if dialog type = DF_PROMPT, then:	      *
 *				  a character string array in which to put    *
 *				  the string input by the user.  Do NOT       *
 *				  include an input_string argument for other  *
 *				  dialog types.				      *
 *	char *	   but_lbl	- button label(s) for buttons requested       *
 *				  (if n > 0, one but_lbl argument per button) *
 *	<anytype>  <args>	- arguments for vsprintf (if any)	      *
 *									      *
 *    Returns:    - button selected by user (i.e. 1, 2, or 3. or 4 for prompt)*
 *									      *
 *    Examples:								      *
 *									      *
 *	but_pressed = DialogF (DF_QUES, toplevel, 3, "Direction?", "up",      *
 *				 "down", "other");			      *
 *	but_pressed = DialogF (DF_ERR, toplevel, 1, "You can't do that!",     *
 *				 "Acknowledged");			      *
 *	but_pressed = DialogF (DF_PROMPT, toplevel, 0, "New %s", 	      *
 *				new_sub_category, categories[i]);  	      *
*/

unsigned DialogF (unsigned dialog_type, Widget parent, unsigned n,
		char *msgstr, ...)		/* variable # arguments */
{
    va_list var;

    Widget dialog, dialog_shell;
    unsigned dialog_num, prompt;
    XmString but_lbl_xms[NUM_BUTTONS_MAXPROMPT];
    XmString msgstr_xms, input_string_xms, titstr_xms;
    char msgstr_vsp[DF_MAX_MSG_LENGTH+1];
    char *but_lbl, *input_string, *input_string_ptr;
    int argcount, num_but_lbls = 0, i;
    Arg args[256];
    
    struct dfcallbackstruct df;

    static int dialog_types[] = {		/* Supported dialog types */
		XmDIALOG_ERROR,
		XmDIALOG_INFORMATION,
		XmDIALOG_MESSAGE,
		XmDIALOG_QUESTION,
		XmDIALOG_WARNING,
		XmDIALOG_PROMPT
    };
    static char *dialog_name[] = {		/* Corresponding dialog names */
		"Error",
		"Information",
		"Message",
		"Question",
		"Warning",
		"Prompt"
    };
    static char *button_name[] = {		/* Motif button names */
		XmNokLabelString,
		XmNapplyLabelString,		/* button #2, if managed */
		XmNcancelLabelString,
		XmNhelpLabelString
    };

						/* Validate input parameters */
    if ((dialog_type > NUM_DIALOGS_SUPPORTED) || (dialog_type <= 0)) {
	printf ("\nError calling DialogF - Unsupported dialog type\n");
	return (0);
    }
    dialog_num = dialog_type - 1;
    prompt = (dialog_type == DF_PROMPT);
    if  (!prompt && (n > NUM_BUTTONS_SUPPORTED) ||
	  prompt && (n > NUM_BUTTONS_MAXPROMPT)) {
	printf ("\nError calling DialogF - Too many buttons specified\n");
	return (0);
    }

    df.done_with_dialog = False;
    va_start (var, msgstr);

    if (prompt) {		      /* Get where to put dialog input string */
	input_string = va_arg(var, char*);
    }
    if (n == NUM_BUTTONS_MAXPROMPT)
	df.apply_up = 1;		/* Apply button will be managed */
    else
    	df.apply_up = 0;		/* Apply button will not be managed */

    for (argcount = 0; argcount < n; ++argcount) {   /* Set up button labels */
	but_lbl = va_arg(var, char *);
	but_lbl_xms[num_but_lbls] = XmStringCreateLtoR (but_lbl, 
		XmSTRING_DEFAULT_CHARSET);
	XtSetArg (args[argcount], button_name[ df.apply_up ? argcount : 
		((argcount == 0) ? argcount : argcount+1) ], 
		but_lbl_xms[num_but_lbls++]); 
    }

    /* Get & translate msg string
    */
    vsprintf (msgstr_vsp, msgstr, var);
    msgstr_xms = XmStringCreateLtoR (msgstr_vsp, XmSTRING_DEFAULT_CHARSET);
    titstr_xms = XmStringCreateLtoR (" ", XmSTRING_DEFAULT_CHARSET);

    if (prompt) {				/* Prompt dialog */
	XtSetArg (args[argcount], XmNselectionLabelString, msgstr_xms);
		argcount++;
	XtSetArg (args[argcount], XmNdialogTitle, titstr_xms);
		argcount++;
#ifdef MOTIF10
	XtSetArg (args[argcount], XmNdialogStyle, XmDIALOG_APPLICATION_MODAL);
		argcount ++;
#else
	XtSetArg (args[argcount], XmNdialogStyle, XmDIALOG_FULL_APPLICATION_MODAL);
		argcount ++;
#endif

	dialog = XmCreatePromptDialog (parent, dialog_name[dialog_num], args,
			argcount);
	XtAddCallback (dialog, XmNokCallback, (XtCallbackProc)ok_callback,
		(char *)&df);
	XtAddCallback (dialog, XmNcancelCallback,
		(XtCallbackProc)cancel_callback, (char *)&df);
	XtAddCallback (dialog, XmNhelpCallback, (XtCallbackProc)help_callback,
		(char *)&df);
	XtAddCallback (dialog, XmNapplyCallback, (XtCallbackProc)apply_callback,
		(char *)&df);
	RemapDeleteKey(XmSelectionBoxGetChild(dialog, XmDIALOG_TEXT));

	/* Text area in prompt dialog should get focus, not ok button
	   since user enters text first.  To fix this, we need to turn
	   off the default button for the dialog, until after keyboard
	   focus has been established */
	XtVaSetValues(dialog, XmNdefaultButton, NULL, 0);
    	XtAddCallback(XmSelectionBoxGetChild(dialog, XmDIALOG_TEXT),
    		XmNfocusCallback, (XtCallbackProc)focusCB, (char *)dialog);

	/* Limit the length of the text that can be entered in text field */
	XtVaSetValues(XmSelectionBoxGetChild(dialog, XmDIALOG_TEXT),
		XmNmaxLength, DF_MAX_PROMPT_LENGTH-1, 0);
	
	switch (n) {		/* number of buttons requested */
	case 0: case 3:
	    break;		/* or default of 3 buttons */
	case 1:
	    XtUnmanageChild (XmSelectionBoxGetChild (dialog,
		XmDIALOG_CANCEL_BUTTON) );
	case 2:
	    XtUnmanageChild (XmSelectionBoxGetChild (dialog,
		XmDIALOG_HELP_BUTTON) );
	    break;
	case 4:
	    XtManageChild (XmSelectionBoxGetChild (dialog,
		XmDIALOG_APPLY_BUTTON) );
	    df.apply_up = 1;		/* apply button managed */
	default:
	    break;
	}				/* end switch */

	va_end(var);

	ManageDialogCenteredOnPointer(dialog);/* Pop it up, wait for response */
	
	/* Get the focus to the input string.  There is some timing problem
	   within Motif that requires this to be called several times */
	for (i=0; i<20; i++)
	    XmProcessTraversal(XmSelectionBoxGetChild(dialog, XmDIALOG_TEXT),
		    XmTRAVERSE_CURRENT);
	
	while (!df.done_with_dialog)
	    XtAppProcessEvent (XtWidgetToApplicationContext(dialog), XtIMAll);
 
	argcount = 0;			/* Pass back string user entered */
	XtSetArg (args[argcount], XmNtextString, &input_string_xms); argcount++;
	XtGetValues (dialog, args, argcount);
	XmStringGetLtoR (input_string_xms, XmSTRING_DEFAULT_CHARSET,
		&input_string_ptr);
	strcpy (input_string, input_string_ptr);  /* This step is necessary */
	XmStringFree(input_string_xms );
        XtFree(input_string_ptr);
	XtDestroyWidget(dialog);
    }						  /* End prompt dialog path */

    else {				/* MessageBox dialogs */
	XtSetArg (args[argcount], XmNmessageString, msgstr_xms); argcount++;

	XtSetArg (args[argcount], XmNdialogType, dialog_types[dialog_num]);
		argcount ++;
	XtSetArg (args[argcount], XmNdialogTitle, titstr_xms);
		argcount++;
#ifdef MOTIF10
	XtSetArg (args[argcount], XmNdialogStyle, XmDIALOG_APPLICATION_MODAL);
		argcount ++;
#else
	XtSetArg (args[argcount], XmNdialogStyle, XmDIALOG_FULL_APPLICATION_MODAL);
		argcount ++;
#endif
	dialog_shell = XmCreateDialogShell (parent, dialog_name[dialog_num],
			0, 0);
	dialog = XmCreateMessageBox (dialog_shell, "msg box", args, argcount);
	XtAddCallback (dialog, XmNokCallback, (XtCallbackProc)ok_callback,
		(char *)&df);
	XtAddCallback (dialog, XmNcancelCallback,
		(XtCallbackProc)cancel_callback, (char *)&df);
	XtAddCallback (dialog, XmNhelpCallback, (XtCallbackProc)help_callback,
		(char *)&df);

	va_end(var);

				/* Make extraneous buttons disappear */
	switch (n) {		/* n = number of buttons requested */
	case 0: case 3:
	    break;		/* default (0) gets you 3 buttons */
	case 1:
	    XtUnmanageChild (XmMessageBoxGetChild (dialog,
			XmDIALOG_CANCEL_BUTTON) );
	case 2:
	    XtUnmanageChild (XmMessageBoxGetChild (dialog,
			XmDIALOG_HELP_BUTTON) );
	    break;
	default:
	    break;
	}

	ManageDialogCenteredOnPointer(dialog);/* Pop it up, wait for response*/
	
	while (!df.done_with_dialog)
	    XtAppProcessEvent (XtWidgetToApplicationContext(dialog), XtIMAll);
	XtDestroyWidget(dialog_shell);
    }

    XmStringFree(msgstr_xms);
    XmStringFree(titstr_xms);
    for (i = 0; i < num_but_lbls; ++i)
	XmStringFree(but_lbl_xms[i]);

    df.apply_up = 0;			/* default is apply button unmanaged */

    return (df.button);
}

static void ok_callback (Widget w, struct dfcallbackstruct *client_data,
	caddr_t call_data)
{
    client_data->done_with_dialog = True;
    client_data->button = 1;		/* Return Button number pressed */
}

static void cancel_callback (Widget w, struct dfcallbackstruct *client_data,
	caddr_t call_data)
{
    client_data->done_with_dialog = True;
    client_data->button = 2 + client_data->apply_up; /* =3 if apply button managed */
}

static void help_callback (Widget w, struct dfcallbackstruct *client_data,
	caddr_t call_data)
{
    client_data->done_with_dialog = True;
    client_data->button = 3 + client_data->apply_up; /* =4 if apply button managed */
}

static void apply_callback (Widget w, struct dfcallbackstruct *client_data,
	caddr_t call_data)
{
    client_data->done_with_dialog = True;
    client_data->button = 2;		/* Motif puts between OK and cancel */
}

/*
** callback for returning default button status to the ok button once we're
** sure the text area in the prompt dialog has input focus.
*/
static void focusCB(Widget w, Widget dialog, caddr_t call_data)
{
    XtVaSetValues(dialog, XmNdefaultButton,
    	    XmSelectionBoxGetChild(dialog, XmDIALOG_OK_BUTTON), 0);
}
