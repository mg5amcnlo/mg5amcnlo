/*******************************************************************************
*									       *
* misc.c -- Miscelaneous Motif convenience functions			       *
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
* July 28, 1992								       *
*									       *
* Written by Mark Edel							       *
*									       *
*******************************************************************************/
static char SCCSID[] = "@(#)misc.c	1.39	2/1/96";
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#ifdef VMS
#include <types.h>
#include <unixio.h>
#include <file.h>
#endif /*VMS*/
#include <X11/Intrinsic.h>
#include <X11/Xatom.h>
#include <Xm/Xm.h>
#include <Xm/Label.h>
#include <Xm/LabelG.h>
#include <Xm/ToggleB.h>
#include <Xm/PushB.h>
#include <Xm/Separator.h>
#include <Xm/RowColumn.h>
#include <Xm/CascadeB.h>
#include <Xm/AtomMgr.h>
#include <Xm/Protocols.h>
#include <Xm/Text.h>
#include "DialogF.h"
#include "misc.h"

/* math.h on Sun mysteriously excludes strtod and other functions when
   POSIX compliance is turned on */
extern double strtod();

/* flags to enable/disable delete key remapping and pointer centered dialogs */
static int RemapDeleteEnabled = True;
static int PointerCenteredDialogsEnabled = False;

/* bitmap and mask for waiting (wrist-watch) cursor */
#define watch_x_hot 7
#define watch_y_hot 7
#define watch_width 16
#define watch_height 16
static unsigned char watch_bits[] = {
   0xe0, 0x07, 0xe0, 0x07, 0xe0, 0x07, 0xe0, 0x07, 0x10, 0x08, 0x08, 0x11,
   0x04, 0x21, 0x04, 0x21, 0xe4, 0x21, 0x04, 0x20, 0x08, 0x10, 0x10, 0x08,
   0xe0, 0x07, 0xe0, 0x07, 0xe0, 0x07, 0xe0, 0x07
};
#define watch_mask_width 16
#define watch_mask_height 16
static unsigned char watch_mask_bits[] = {
   0xf0, 0x0f, 0xf0, 0x0f, 0xf0, 0x0f, 0xf0, 0x0f, 0xf8, 0x1f, 0xfc, 0x3f,
   0xfe, 0x7f, 0xfe, 0x7f, 0xfe, 0x7f, 0xfe, 0x7f, 0xfc, 0x3f, 0xf8, 0x1f,
   0xf0, 0x0f, 0xf0, 0x0f, 0xf0, 0x0f, 0xf0, 0x0f
};

static void addMnemonicGrabs(Widget addTo, Widget w);
static void mnemonicCB(Widget w, XtPointer callData, XKeyEvent *event);
static void findAndActivateMnemonic(Widget w, unsigned int keycode);
static void removeWhiteSpace(char *string);
static void warnHandlerCB(String message);
static void passwdCB(Widget w, char * passTxt, XmTextVerifyCallbackStruct
	*txtVerStr);

/*
** Set up closeCB to be called when the user selects close from the
** window menu.  The close menu item usually activates f.kill which
** sends a WM_DELETE_WINDOW protocol request for the window.
*/
void AddMotifCloseCallback(Widget shell, XtCallbackProc closeCB, void *arg)
{
    static Atom wmpAtom, dwAtom = 0;
    Display *display = XtDisplay(shell);

    /* deactivate the built in delete response of killing the application */
    XtVaSetValues(shell, XmNdeleteResponse, XmDO_NOTHING, 0);

    /* add a delete window protocol callback instead */
    if (dwAtom == 0) {
    	wmpAtom = XmInternAtom(display, "WM_PROTOCOLS", TRUE);
    	dwAtom = XmInternAtom(display, "WM_DELETE_WINDOW", TRUE);
    }
    XmAddProtocolCallback(shell, wmpAtom, dwAtom, closeCB, arg);
}

/*
** Motif still generates spurrious passive grab warnings on both IBM and SGI
** This routine suppresses them
*/
void SuppressPassiveGrabWarnings(void)
{
    XtSetWarningHandler(warnHandlerCB);
}

/*
** This routine kludges around the problem of backspace not being mapped
** correctly when Motif is used between a server with a delete key in
** the traditional typewriter backspace position and a client that
** expects a backspace key in that position.  Though there are three
** distinct levels of key re-mapping in effect when a user is running
** a Motif application, none of these is really appropriate or effective
** for eliminating the delete v.s. backspace problem.  Our solution is,
** sadly, to eliminate the forward delete functionality of the delete key
** in favor of backwards delete for both keys.  So as not to prevent the
** user or the application from applying other translation table re-mapping,
** we apply re-map the key as a post-processing step, applied after widget
** creation.  As a result, the re-mapping necessarily becomes embedded
** throughout an application (wherever text widgets are created), and
** within library routines, including the Nirvana utility library.  To
** make this remapping optional, the SetDeleteRemap function provides a
** way for an application to turn this functionality on and off.  It is
** recommended that applications that use this routine provide an
** application resource called remapDeleteKey so savvy users can get
** their forward delete functionality back.
*/
void RemapDeleteKey(Widget w)
{
    static XtTranslations table = NULL;
    static char *translations =
    	"~Shift~Ctrl~Meta~Alt<Key>osfDelete: delete-previous-character()\n";

    if (RemapDeleteEnabled) {
    	if (table == NULL)
    	    table = XtParseTranslationTable(translations);
    	XtOverrideTranslations(w, table);
    }
}

void SetDeleteRemap(int state)
{
    RemapDeleteEnabled = state;
}

/*
** ManageDialogCenteredOnPointer is used in place of XtManageChild for
** popping up a dialog to enable the dialog to be centered under the
** mouse pointer.  Whether it pops up the dialog centered under the pointer
** or in its default position centered over the parent widget, depends on
** the value set in the SetPointerCenteredDialogs call.
*/ 
void ManageDialogCenteredOnPointer(Widget dialogChild)
{
    Widget shell = XtParent(dialogChild);
    Window root, child;
    unsigned int mask;
    unsigned int width, height, borderWidth, depth;
    int x, y, winX, winY, maxX, maxY;
    Boolean mappedWhenManaged;

    /* If this feature is not enabled, just manage the dialog */
    if (!PointerCenteredDialogsEnabled) {
    	XtManageChild(dialogChild);
    	return;
    }
    
    /* Temporarily set value of XmNmappedWhenManaged
       to stop the dialog from popping up right away */
    XtVaGetValues(shell, XmNmappedWhenManaged, &mappedWhenManaged, 0);
    XtVaSetValues(shell, XmNmappedWhenManaged, False, 0);
    
    /* Manage the dialog */
    XtManageChild(dialogChild);

    /* Get the pointer position (x, y) */
    XQueryPointer(XtDisplay(shell), XtWindow(shell), &root, &child,
	    &x, &y, &winX, &winY, &mask);

    /* Translate the pointer position (x, y) into a position for the new
       window that will place the pointer at its center */
    XGetGeometry(XtDisplay(shell), XtWindow(shell), &root, &winX, &winY,
    	    &width, &height, &borderWidth, &depth);
    width += 2 * borderWidth;
    height += 2 * borderWidth;
    x -= width/2;
    y -= height/2;

    /* Ensure that the dialog remains on screen */
    maxX = XtScreen(shell)->width - width;
    maxY = XtScreen(shell)->height - height;
    if (x < 0) x = 0;
    if (x > maxX) x = maxX;
    if (y < 0) y = 0;
    if (y > maxY) y = maxY;

    /* Set desired window position in the DialogShell */
    XtVaSetValues(shell, XmNx, x, XmNy, y, NULL);
    
    /* Map the widget */
    XtMapWidget(shell);
    
    /* Restore the value of XmNmappedWhenManaged */
    XtVaSetValues(shell, XmNmappedWhenManaged, mappedWhenManaged, 0);
}

/*
** Cause dialogs created by libNUtil.a routines (such as DialogF and
** GetNewFilename), and dialogs which use ManageDialogCenteredOnPointer
** to pop up over the pointer (state = True), or pop up in their default
** positions (state = False)
*/
void SetPointerCenteredDialogs(int state)
{
    PointerCenteredDialogsEnabled = state;
}

/*
** Add a handler for mnemonics in a dialog (Motif currently only handles
** mnemonics in menus) following the example of M.S. Windows.  To add
** mnemonics to a dialog, set the XmNmnemonic resource, as you would in
** a menu, on push buttons or toggle buttons, and call this function
** when the dialog is fully constructed.  Mnemonics added or changed
** after this call will not be noticed.  To add a mnemonic to a text field
** or list, set the XmNmnemonic resource on the appropriate label and set
** the XmNuserData resource of the label to the widget to get the focus
** when the mnemonic is typed.
*/
void AddDialogMnemonicHandler(Widget dialog)
{
    XtAddEventHandler(dialog, KeyPressMask, False,
    	    (XtEventHandler)mnemonicCB, (XtPointer)0);
    addMnemonicGrabs(dialog, dialog);
}

/*
** Removes the event handler and key-grabs added by AddDialogMnemonicHandler
*/
void RemoveDialogMnemonicHandler(Widget dialog)
{
    XtUngrabKey(dialog, AnyKey, Mod1Mask);
    XtRemoveEventHandler(dialog, KeyPressMask, False,
    	    (XtEventHandler)mnemonicCB, (XtPointer)0);
}

/*
** PopDownBugPatch
**
** Under some circumstances, popping down a dialog and its parent in
** rapid succession causes a crash.  This routine delays and
** processs events until receiving a ReparentNotify event.
** (I have no idea why a ReparentNotify event occurs at all, but it does
** mark the point where it is safe to destroy or pop down the parent, and
** it might have something to do with the bug.)  There is a failsafe in
** the form of a ~1.5 second timeout in case no ReparentNotify arrives.
** Use this sparingly, only when real crashes are observed, and periodically
** check to make sure that it is still necessary.
*/
void PopDownBugPatch(Widget w)
{
    time_t stopTime;

    stopTime = time(NULL) + 1;
    while (time(NULL) <= stopTime) {
    	XEvent event;
    	XtAppContext context = XtWidgetToApplicationContext(w);
    	XtAppPeekEvent(context, &event);
    	if (event.xany.type == ReparentNotify)
    	    return;
    	XtAppProcessEvent(context, XtIMAll);
    }
}

/*
** Convert a compound string to a C style null terminated string.
** Returned string must be freed by the caller.
*/
char *GetXmStringText(XmString fromString)
{
    XmStringContext context;
    char *text, *toPtr, *toString, *fromPtr;
    XmStringCharSet charset;
    XmStringDirection direction;
    Boolean separator;
    
    /* Malloc a buffer large enough to hold the string.  XmStringLength
       should always be slightly longer than necessary, but won't be
       shorter than the equivalent null-terminated string */ 
    toString = XtMalloc(XmStringLength(fromString));
    
    /* loop over all of the segments in the string, copying each segment
       into the output string and converting separators into newlines */
    XmStringInitContext(&context, fromString);
    toPtr = toString;
    while (XmStringGetNextSegment(context, &text,
    	    &charset, &direction, &separator)) {
    	for (fromPtr=text; *fromPtr!='\0'; fromPtr++)
    	    *toPtr++ = *fromPtr;
    	if (separator)
    	    *toPtr++ = '\n';
    }
    
    /* terminate the string, free the context, and return the string */
    *toPtr++ = '\0';
    XmStringFreeContext(context);
    return toString;
}

/*
** Get the XFontStruct that corresponds to the default (first) font in
** a Motif font list.  Since Motif stores this, it saves us from storing
** it or querying it from the X server.
*/
XFontStruct *GetDefaultFontStruct(XmFontList font)
{
    XFontStruct *fs;
    XmFontContext context;
    XmStringCharSet charset;

    XmFontListInitFontContext(&context, font);
    XmFontListGetNextFont(context, &charset, &fs);
    XmFontListFreeFontContext(context);
    XtFree(charset);
    return fs;
}
   
/*
** Create a string table suitable for passing to XmList widgets
*/
XmString* StringTable(int count, ... )
{
    va_list ap;
    XmString *array;
    int i;
    char *str;

    va_start(ap, count);
    array = (XmString*)XtMalloc((count+1) * sizeof(XmString));
    for(i = 0;  i < count; i++ ) {
    	str = va_arg(ap, char *);
	array[i] = XmStringCreateSimple(str);
    }
    array[i] = (XmString)0;
    va_end(ap);
    return(array);
}

void FreeStringTable(XmString *table)
{
    int i;

    for(i = 0; table[i] != 0; i++)
	XmStringFree(table[i]);
    XtFree((char *)table);
}

/*
** Simulate a button press.  The purpose of this routine is show users what
** is happening when they take an action with a non-obvious side effect,
** such as when a user double clicks on a list item.  The argument is an
** XmPushButton widget to "press"
*/ 
void SimulateButtonPress(Widget widget)
{
    XKeyPressedEvent keyEvent;
    
    memset((char *)&keyEvent, 0, sizeof(XKeyPressedEvent));
    keyEvent.type = KeyPress;
    keyEvent.serial = 1;
    keyEvent.send_event = True;
    keyEvent.display = XtDisplay(widget);
    keyEvent.window = XtWindow(widget);
    XtCallActionProc(widget, "ArmAndActivate", (XEvent *)&keyEvent, NULL, 0);
}

/*
** Add an item to an already established pull-down or pop-up menu, including
** mnemonics, accelerators and callbacks.
*/
Widget AddMenuItem(Widget parent, char *name, char *label,
			  char mnemonic, char *acc, char *accText,
			  XtCallbackProc callback, void *cbArg)
{
    Widget button;
    XmString st1, st2;
    
    button = XtVaCreateManagedWidget(name, xmPushButtonWidgetClass, parent, 
    	XmNlabelString, st1=XmStringCreateSimple(label),
    	XmNmnemonic, mnemonic,
    	XmNacceleratorText, st2=XmStringCreateSimple(accText),
    	XmNaccelerator, acc, NULL);
    XtAddCallback(button, XmNactivateCallback, callback, cbArg);
    XmStringFree(st1);
    XmStringFree(st2);
    return button;
}

/*
** Add a toggle button item to an already established pull-down or pop-up
** menu, including mnemonics, accelerators and callbacks.
*/
Widget AddMenuToggle(Widget parent, char *name, char *label,
		 	    char mnemonic, char *acc, char *accText,
		  	    XtCallbackProc callback, void *cbArg, int set)
{
    Widget button;
    XmString st1, st2;
    
    button = XtVaCreateManagedWidget(name, xmToggleButtonWidgetClass, parent, 
    	XmNlabelString, st1=XmStringCreateSimple(label),
    	XmNmnemonic, mnemonic,
    	XmNacceleratorText, st2=XmStringCreateSimple(accText),
    	XmNaccelerator, acc,
    	XmNset, set, NULL);
    XtAddCallback(button, XmNvalueChangedCallback, callback, cbArg);
    XmStringFree(st1);
    XmStringFree(st2);
    return button;
}

/*
** Add a separator line to a menu
*/
Widget AddMenuSeparator(Widget parent, char *name)
{
    Widget button;
    
    button = XmCreateSeparator(parent, name, NULL, 0);
    XtManageChild(button);
    return button;
}

/*
** Add a sub-menu to an established pull-down or pop-up menu, including
** mnemonics, accelerators and callbacks.  Returns the menu pane of the
** new sub menu.
*/
Widget AddSubMenu(Widget parent, char *name, char *label, char mnemonic)
{
    Widget menu;
    XmString st1;
    
    menu = XmCreatePulldownMenu(parent, name, NULL, 0);
    XtVaCreateManagedWidget(name, xmCascadeButtonWidgetClass, parent, 
    	XmNlabelString, st1=XmStringCreateSimple(label),
    	XmNmnemonic, mnemonic,
    	XmNsubMenuId, menu, 0);
    XmStringFree(st1);
    return menu;
}

/*
** SetIntLabel, SetFloatLabel, SetIntText, SetFloatText
**
** Set the text of a motif label or text widget to show an integer or
** floating number.
*/
void SetIntLabel(Widget label, int value)
{
    char labelString[20];
    XmString s1;
    
    sprintf(labelString, "%d", value);
    s1=XmStringCreateSimple(labelString);
    XtVaSetValues(label, XmNlabelString, s1, 0);
    XmStringFree(s1);
}
void SetFloatLabel(Widget label, double value)
{
    char labelString[20];
    XmString s1;
    
    sprintf(labelString, "%g", value);
    s1=XmStringCreateSimple(labelString);
    XtVaSetValues(label, XmNlabelString, s1, 0);
    XmStringFree(s1);
}
void SetIntText(Widget text, int value)
{
    char labelString[20];
    
    sprintf(labelString, "%d", value);
    XmTextSetString(text, labelString);
}
void SetFloatText(Widget text, double value)
{
    char labelString[20];
    
    sprintf(labelString, "%g", value);
    XmTextSetString(text, labelString);
}

/*
** GetIntText, GetFloatText, GetIntTextWarn, GetFloatTextWarn
**
** Get the text of a motif text widget as an integer or floating point number.
** The functions will return TEXT_READ_OK of the value was read correctly.
** If not, they will return either TEXT_IS_BLANK, or TEXT_NOT_NUMBER.  The
** GetIntTextWarn and GetFloatTextWarn will display a dialog warning the
** user that the value could not be read.  The argument fieldName is used
** in the dialog to help the user identify where the problem is.  Set
** warnBlank to true if a blank field is also considered an error.
*/
int GetFloatText(Widget text, double *value)
{
    char *strValue, *endPtr;
    int retVal;

    strValue = XmTextGetString(text);	/* Get Value */
    removeWhiteSpace(strValue);		/* Remove blanks and tabs */
    *value = strtod(strValue, &endPtr);	/* Convert string to double */
    if (strlen(strValue) == 0)		/* String is empty */
	retVal = TEXT_IS_BLANK;
    else if (*endPtr != '\0')		/* Whole string not parsed */
    	retVal = TEXT_NOT_NUMBER;
    else
	retVal = TEXT_READ_OK;
    XtFree(strValue);
    return retVal;
}
int GetIntText(Widget text, int *value)
{
    char *strValue, *endPtr;
    int retVal;

    strValue = XmTextGetString(text);		/* Get Value */
    removeWhiteSpace(strValue);			/* Remove blanks and tabs */
    *value = strtol(strValue, &endPtr, 10);	/* Convert string to long */
    if (strlen(strValue) == 0)			/* String is empty */
	retVal = TEXT_IS_BLANK;
    else if (*endPtr != '\0')			/* Whole string not parsed */
    	retVal = TEXT_NOT_NUMBER;
    else
	retVal = TEXT_READ_OK;
    XtFree(strValue);
    return retVal;
}
int GetFloatTextWarn(Widget text, double *value, char *fieldName, int warnBlank)
{
    int result;
    char *valueStr;
    
    result = GetFloatText(text, value);
    if (result == TEXT_READ_OK || (result == TEXT_IS_BLANK && !warnBlank))
    	return result;
    valueStr = XmTextGetString(text);
    if (result == TEXT_IS_BLANK)
    	DialogF (DF_ERR, text, 1, "Please supply %s value",
    		 "Acknowledged", fieldName);
    else /* TEXT_NOT_NUMBER */
    	DialogF (DF_ERR, text, 1,
    		 "Can't read %s value: \"%s\"",
    		 "Acknowledged", fieldName, valueStr);
    XtFree(valueStr);
    return result;
}
int GetIntTextWarn(Widget text, int *value, char *fieldName, int warnBlank)
{
    int result;
    char *valueStr;
    
    result = GetIntText(text, value);
    if (result == TEXT_READ_OK || (result == TEXT_IS_BLANK && !warnBlank))
    	return result;
    valueStr = XmTextGetString(text);
    if (result == TEXT_IS_BLANK)
    	DialogF (DF_ERR, text, 1, "Please supply a value for %s",
    		 "Acknowledged", fieldName);
    else /* TEXT_NOT_NUMBER */
    	DialogF (DF_ERR, text, 1,
    		 "Can't read integer value \"%s\" in %s",
    		 "Acknowledged", valueStr, fieldName);
    XtFree(valueStr);
    return result;
}

/*
 * PasswordText - routine to add a callback to any text widget so that all
 * 		  text typed by the user is echoed with asterisks, allowing
 *		  a password to be typed in without being seen.
 *
 * parameters: w       - text widget to add the callback to
 *             passTxt - pointer to a string created by caller of this routine.
 *		         **NOTE** The length of this string should be one 
 *			 greater than the maximum specified by XmNmaxLength.
 *			 This string is set to empty just before the callback
 *			 is added.
 */

void PasswordText(Widget w, char *passTxt)
{
    passTxt[0] = '\0';
    XtAddCallback(w, XmNmodifyVerifyCallback, (XtCallbackProc)passwdCB,passTxt);
}

/*
** BeginWait/EndWait
**
** Display/Remove a watch cursor over topCursorWidget and its descendents
*/
void BeginWait(Widget topCursorWidget)
{
    Display *display = XtDisplay(topCursorWidget);
    Pixmap pixmap;
    Pixmap maskPixmap;
    XColor xcolors[2];
    static Cursor  waitCursor = 0;
    
    /* if the watch cursor hasn't been created yet, create it */
    if (!waitCursor) {
	pixmap = XCreateBitmapFromData(display, DefaultRootWindow(display),
		(char *)watch_bits, watch_width, watch_height);

	maskPixmap = XCreateBitmapFromData(display, DefaultRootWindow(display),
		(char *)watch_mask_bits, watch_width, watch_height);

	xcolors[0].pixel = BlackPixelOfScreen(DefaultScreenOfDisplay(display));
	xcolors[1].pixel = WhitePixelOfScreen(DefaultScreenOfDisplay(display));

	XQueryColors(display, DefaultColormapOfScreen(
		DefaultScreenOfDisplay(display)), xcolors, 2);
	waitCursor = XCreatePixmapCursor(display, pixmap, maskPixmap,
		&xcolors[0], &xcolors[1], watch_x_hot, watch_y_hot);
	XFreePixmap(display, pixmap);
	XFreePixmap(display, maskPixmap);
    }

    /* display the cursor */
    XDefineCursor(display, XtWindow(topCursorWidget), waitCursor);
}

void EndWait(Widget topCursorWidget)
{
    XUndefineCursor(XtDisplay(topCursorWidget), XtWindow(topCursorWidget));
}

/*									     */
/* passwdCB: callback routine added by PasswordText routine.  This routine   */
/* 	     echoes each character typed as an asterisk (*) and a few other  */
/* 	     necessary things so that the password typed in is not visible   */
/*									     */

static void passwdCB(Widget w, char * passTxt, XmTextVerifyCallbackStruct
	*txtVerStr)

/* XmTextVerifyCallbackStruct:						      */
/*   int reason;	should be XmCR_MODIFYING_TEXT_VALUE 		      */
/*   XEvent  *event;	points to XEvent that triggered the callback	      */
/*   Boolean doit;	indicates whether action should be performed; setting */
/*			this to false negates the action		      */
/*   long currInsert, 	current position of insert cursor		      */
/*	  newInsert;	position user attempts to position the insert cursor  */
/*   long startPos,	starting position of the text to modify		      */
/*	  endPos;	ending position of the text to modify		      */
/*   XmTextBlock text;							      */

/* XmTextBlock (used to pass text around): 		    		      */
/*   char *ptr;                   points to text to be inserted 	      */
/*   int length;		  Number of bytes (length)		      */
/*   XmTextFormat format;         Representations format		      */

/* XmTextFormat: either FMT8BIT or FMT16BIT */

{
    int numCharsTyped, i, j, pos;

    /* ensure text format is 8-bit characters */
    if (txtVerStr->text->format != FMT8BIT)
	return;

    /* verify assumptions */
/*    if (txtVerStr->endPos < txtVerStr->startPos)
	fprintf(stderr, "User password callback error: endPos < startPos\n");
    if (strlen(passTxt) == 0 && txtVerStr->endPos != 0)
	fprintf(stderr, "User password callback error: no txt, but end != 0\n");

    printf("passTxt = %s, startPos = %d, endPos = %d, txtBlkAddr = %d\n",
	 passTxt, txtVerStr->startPos, txtVerStr->endPos, txtVerStr->text);
    if (txtVerStr->text != NULL && txtVerStr->text->ptr != NULL)
	printf("       string typed = %s, length = %d\n", txtVerStr->text->ptr,
		txtVerStr->text->length);
*/
    /* If necessary, expand/compress passTxt and insert any new text */
    if (txtVerStr->text != NULL && txtVerStr->text->ptr != NULL)
	numCharsTyped = txtVerStr->text->length;
    else
	numCharsTyped = 0;
    /* numCharsTyped = # chars to insert (that user typed) */
    /* j = # chars to expand (+) or compress (-) the password string */
    j = numCharsTyped - (txtVerStr->endPos - txtVerStr->startPos);
    if (j > 0) 				/* expand case: start at ending null  */
	for (pos = strlen(passTxt) + 1; pos >= txtVerStr->endPos; --pos)
	    passTxt[pos+j] = passTxt[pos];
    if (j < 0)				/* compress case */
	for (pos = txtVerStr->startPos + numCharsTyped; 
			     pos <= strlen(passTxt)+1; ++pos)
	    passTxt[pos] = passTxt[pos-j];
    /* then copy text to be inserted into passTxt */
    for (pos = txtVerStr->startPos, i = 0; i < numCharsTyped; ++i) {
	passTxt[pos+i] = *(txtVerStr->text->ptr + i);
	/* Replace text typed by user with asterisks (*) */
	*(txtVerStr->text->ptr + i) = '*';
    }
/*    printf("  Password string now = %s\n", passTxt);  */
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

static void warnHandlerCB(String message)
{
    if (strstr(message, "XtRemoveGrab"))
    	return;
    if (strstr(message, "Attempt to remove non-existant passive grab"))
    	return;
    fprintf(stderr, message);
}

/*
** Part of dialog mnemonic processing.  Search the widget tree under w
** for widgets with mnemonics.  When found, add a passive grab to the
** dialog widget for the mnemonic character, thus directing mnemonic
** events to the dialog widget.
*/
static void addMnemonicGrabs(Widget dialog, Widget w)
{
    char mneString[2];
    WidgetList children;
    int numChildren, i, isMenu;
    KeySym mnemonic = '\0';
    unsigned char rowColType;
    
    if (XtIsComposite(w)) {
	if (XtClass(w) == xmRowColumnWidgetClass) {
	    XtVaGetValues(w, XmNrowColumnType, &rowColType, 0);
	    isMenu = rowColType != XmWORK_AREA;
	} else
	    isMenu = False;
	if (!isMenu) {
	    XtVaGetValues(w, XmNchildren, &children, XmNnumChildren,
		    &numChildren, 0);
	    for (i=0; i<numChildren; i++)
    		addMnemonicGrabs(dialog, children[i]);
    	}
    } else {
	XtVaGetValues(w, XmNmnemonic, &mnemonic, 0);
	if (mnemonic != '\0') {
	    mneString[0] = mnemonic; mneString[1] = '\0';
	    XtGrabKey(dialog, XKeysymToKeycode(XtDisplay(dialog),
	    	    XStringToKeysym(mneString)), Mod1Mask,
	    	    True, GrabModeAsync, GrabModeAsync);
	}
    }
}

/*
** Callback routine for dialog mnemonic processing.
*/
static void mnemonicCB(Widget w, XtPointer callData, XKeyEvent *event)
{
    findAndActivateMnemonic(w, event->keycode);
}

/*
** Look for a widget in the widget tree w, with a mnemonic matching
** keycode.  When one is found, simulate a button press on that widget
** and give it the keyboard focus.  If the mnemonic is on a label,
** look in the userData field of the label to see if it points to
** another widget, and give that the focus.  This routine is just
** sufficient for NEdit, no doubt it will need to be extended for
** mnemonics on widgets other than just buttons and text fields.
*/
static void findAndActivateMnemonic(Widget w, unsigned int keycode)
{
    WidgetList children;
    int numChildren, i, isMenu;
    KeySym mnemonic = '\0';
    char mneString[2];
    Widget userData;
    unsigned char rowColType;
    
    if (XtIsComposite(w)) {
	if (XtClass(w) == xmRowColumnWidgetClass) {
	    XtVaGetValues(w, XmNrowColumnType, &rowColType, 0);
	    isMenu = rowColType != XmWORK_AREA;
	} else
	    isMenu = False;
	if (!isMenu) {
	    XtVaGetValues(w, XmNchildren, &children, XmNnumChildren,
		    &numChildren, 0);
	    for (i=0; i<numChildren; i++)
    		findAndActivateMnemonic(children[i], keycode);
    	}
    } else {
	XtVaGetValues(w, XmNmnemonic, &mnemonic, 0);
	if (mnemonic != '\0') {
	    mneString[0] = mnemonic; mneString[1] = '\0';
	    if (XKeysymToKeycode(XtDisplay(XtParent(w)),
	    	    XStringToKeysym(mneString)) == keycode) {
	    	if (XtClass(w) == xmLabelWidgetClass ||
	    		XtClass(w) == xmLabelGadgetClass) {
	    	    XtVaGetValues(w, XmNuserData, &userData, 0);
	    	    if (userData!=NULL && XtIsWidget(userData))
	    	    	XmProcessTraversal(userData, XmTRAVERSE_CURRENT);
	    	} else {
	    	    XmProcessTraversal(w, XmTRAVERSE_CURRENT);
	    	    SimulateButtonPress(w);
	    	}
	    }
	}
    }
}
