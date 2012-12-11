/*******************************************************************************
*									       *
* help.c -- Nirvana help display					       *
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
* November 13, 1991							       *
*									       *
* Written by Mark Edel							       *
*									       *
*******************************************************************************/
static char SCCSID[] = "@(#)help.c	1.6	8/19/94";
#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/Form.h>
#include <Xm/Text.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/CascadeB.h>
#include <Xm/ScrolledW.h>
#include <Xm/ScrollBar.h>
#include <Xm/Label.h>
#include "misc.h"
#include "help.h"

#define MAX_HELP_BM_WIDTH 640	/* Max width of a help bitmap before
				   using a horizontal scroll bar */
#define MAX_HELP_BM_HEIGHT 400	/* Max height of a help bitmap before
				   using a vertical scroll bar */
#define SCROLL_BAR_WIDTH 31	/* Amount of padding to add to set the width
				   of a scrolled window including a scroll bar
				   from the width of its contents */
#define SCROLL_FRAME_WIDTH 12	/* Amount of padding to add to set the width
				   of a scrolled window with no scroll bar
				   from the width of its contents */

static void dismissCB(Widget w, caddr_t clientData, caddr_t callData);
static void bitmapDismissCB(Widget w, Pixmap pmToDestroy, caddr_t callData);
static void helpMenuCB(Widget w, helpMenuInfo *info, caddr_t callData);

/*
** CreateHelpPulldownMenu
**
** Creates a working help menu. The caller should supply a parent menu bar
** or menu pane, and a null terminated array of pointers to menuInfo data
** structures.  If menuBar is a menu bar, the help menu will appear as a
** top level pulldown menu on the right of the menu bar.  If it is a menu
** pane, the help menu will appear as a sub-menu at the last item position.
** CreateHelpPulldownMenu creates the Help cascade button as well as the
** the menu pane and its contents.  The button has the name "Help", and
** the mnemonic "H" by default.  The data pointed to by menuInfo is not
** copied, so the caller must not deallocate it after passing it to
** CreateHelpPulldownMenu (we expect this to be used mostly with statically
** allocated help text).  menuInfo is an array of pointers to structures
** rather than the structures themselves, so programs which have several
** help menus can share items among them.
**
** Parameters
**
**	menuBar		Menu bar or menu pane in which to insert the help menu
**	menuInfo	Pointer to a null terminated array of pointers
**			to helpMenuInfo data structures.  This data is not
**			copied and must remain in place after the call
**			to  CreateHelpPulldownMenu
**
** Return Value
**
**	Returns the Help cascade button.  To get the menu pane of the new
**	menu, read the value of XmNsubMenuId from the returned widget.
*/
Widget CreateHelpPulldownMenu(Widget menuBar, helpMenuInfo **menuInfo)
{
    Widget menu, cascade;
    char widgetName[20];
    helpMenuInfo **info;
    XmString s1;
    int i;

    /* create the menu pane and fill in its items from the menuInfo array */
    menu = XmCreatePulldownMenu(menuBar, "help", NULL, 0);
    for (i=1, info=menuInfo; *info!=NULL; i++, info++) {
    	sprintf(widgetName, "help%d", i);
    	AddMenuItem(menu, widgetName, (**info).topic, (**info).mnemonic,
    		"", "", (XtCallbackProc)helpMenuCB, *info);
    	if ((**info).sepAfter) {
    	    sprintf(widgetName, "sep%d", i);
    	    AddMenuSeparator(menu, widgetName);
    	}
    }

    /* create the cascade button with name "Help" and mnemonic 'H' */
    cascade = XtVaCreateManagedWidget("help", xmCascadeButtonWidgetClass,
    	    menuBar, XmNlabelString, s1=XmStringCreateSimple("Help"),
	    XmNmnemonic, 'H', XmNsubMenuId, menu, 0);
    XmStringFree(s1);
    
    /* set the menu up as the designated help menu (puts it on the right
       of a menu bar or bottom of a submenu) */
    XtVaSetValues(menuBar, XmNmenuHelpWidget, cascade, 0);

    return cascade;
}

/*
** CreateHelpDialog
**
** Display a simple scrolled text dialog to present help information.
** Closing the dialog destroys it.
**
** Parameters
**
**	parent		Parent widget for dialog
**	title		Window title for the dialog
**	text		Null terminated string containing the text to present
**
** Return Value
**
**	Returns the top non-shell widget of the new dialog.
*/
Widget CreateHelpDialog(Widget parent, char *title, char *text)
{
    Widget form, button, textW;
    XmString st1;
    Arg al[20];
    int ac;

    form = XmCreateFormDialog(parent, "helpForm", NULL, 0);
    XtVaSetValues(form, XmNshadowThickness, 0, 0);

    ac = 0;
    XtSetArg(al[ac], XmNlabelString, st1=XmStringCreateSimple("Dismiss")); ac++;
    XtSetArg(al[ac], XmNbottomAttachment, XmATTACH_FORM);  ac++;
    XtSetArg(al[ac], XmNtopAttachment, XmATTACH_NONE);  ac++;
    XtSetArg(al[ac], XmNleftAttachment, XmATTACH_POSITION);  ac++;
    XtSetArg(al[ac], XmNrightAttachment, XmATTACH_POSITION);  ac++;
    XtSetArg(al[ac], XmNleftPosition, 35);  ac++;
    XtSetArg(al[ac], XmNrightPosition, 65);  ac++;
    button = XmCreatePushButton(form, "dismiss", al, ac);
    XtAddCallback(button, XmNactivateCallback, (XtCallbackProc)dismissCB, NULL);
    XmStringFree(st1);
    XtManageChild(button);
    XtVaSetValues(form, XmNdefaultButton, button, 0);
    
    ac = 0;
    XtSetArg(al[ac], XmNrows, 15);  ac++;
    XtSetArg(al[ac], XmNcolumns, 60);  ac++;
    XtSetArg(al[ac], XmNresizeHeight, False);  ac++;
    XtSetArg(al[ac], XmNtraversalOn, False); ac++;
    XtSetArg(al[ac], XmNwordWrap, True);  ac++;
    XtSetArg(al[ac], XmNscrollHorizontal, False);  ac++;
    XtSetArg(al[ac], XmNspacing, 0);  ac++;
    XtSetArg(al[ac], XmNeditMode, XmMULTI_LINE_EDIT);  ac++;
    XtSetArg(al[ac], XmNeditable, False);  ac++;
    XtSetArg(al[ac], XmNvalue, text);  ac++;
    XtSetArg(al[ac], XmNtopAttachment, XmATTACH_FORM);  ac++;
    XtSetArg(al[ac], XmNleftAttachment, XmATTACH_FORM);  ac++;
    XtSetArg(al[ac], XmNbottomAttachment, XmATTACH_WIDGET);  ac++;
    XtSetArg(al[ac], XmNrightAttachment, XmATTACH_FORM);  ac++;
    XtSetArg(al[ac], XmNbottomWidget, button);  ac++;
    textW = XmCreateScrolledText(form, "helpText", al, ac);
    XtManageChild(textW);
    
    XtVaSetValues(XtParent(form), XmNtitle, title, 0);
    ManageDialogCenteredOnPointer(form);
    
    return form;
}

/*
** CreateBitmapHelpDialog
**
** Display a simple scrolled bitmap dialog to present help information.
** Closing the dialog destroys it.
**
** Parameters
**
**	parent		Parent widget for dialog
**	title		Window title for the dialog
**	bits		Null terminated string containing the text to present
**	width		Width of the bitmap in pixels
**	height 		height ot the bitmap in pixels
**
** Return Value
**
**	Returns the top non-shell widget of the new dialog.
*/
Widget CreateBitmapHelpDialog(Widget parent, char *title, char *bits,
	int width, int height)
{
    Widget form, button, scrollW;
    XmString st1;
    Pixmap labelPixmap;
    Pixel fg, bg;
    int swWidth, swHeight, s;

    /* create a form dialog to hold everything */
    form = XmCreateFormDialog(parent, "helpForm", NULL, 0);
    XtVaSetValues(form, XmNshadowThickness, 0, 0);
    
    /* decide on a width and height for the scrolled window */
    swWidth = (width > MAX_HELP_BM_WIDTH ? MAX_HELP_BM_WIDTH : width) +
    	  (height > MAX_HELP_BM_HEIGHT ? SCROLL_BAR_WIDTH : SCROLL_FRAME_WIDTH);
    swHeight = (height > MAX_HELP_BM_HEIGHT ? MAX_HELP_BM_HEIGHT : height) +
	    (width > MAX_HELP_BM_WIDTH ? SCROLL_BAR_WIDTH : SCROLL_FRAME_WIDTH);

    /* create a pixmap to display from the bit data, the pixmap must be
       of the same depth as the window it will be drawn to */
    XtVaGetValues(form, XmNforeground, &fg, XmNbackground, &bg, 0);
    labelPixmap = XCreatePixmapFromBitmapData(XtDisplay(parent),
    	    RootWindowOfScreen(XtScreen(parent)), bits, width, height,
    	    fg, bg, DefaultDepthOfScreen(XtScreen(parent)));

    /* create the widgets in the window: a dismiss button, and a scrolled
       window containing a pixmap-displaying label */
    button = XtVaCreateManagedWidget("dismiss", xmPushButtonWidgetClass, form,
    	    XmNlabelString, st1=XmStringCreateSimple("Dismiss"),
    	    XmNbottomAttachment, XmATTACH_FORM,
    	    XmNtopAttachment, XmATTACH_NONE,
    	    XmNleftAttachment, XmATTACH_POSITION,
    	    XmNrightAttachment, XmATTACH_POSITION,
    	    XmNleftPosition, 35,
    	    XmNrightPosition, 65, 0);
    XtAddCallback(button, XmNactivateCallback, (XtCallbackProc)bitmapDismissCB,
    	    (char *)labelPixmap);
    XmStringFree(st1);
    XtVaSetValues(form, XmNdefaultButton, button, 0);
    scrollW = XtVaCreateManagedWidget("scroll", xmScrolledWindowWidgetClass,
    	    form,
    	    XmNwidth, swWidth,
    	    XmNheight, swHeight,
    	    XmNspacing, 0,
    	    XmNscrollingPolicy, XmAUTOMATIC,
    	    XmNtopAttachment, XmATTACH_FORM,
    	    XmNleftAttachment, XmATTACH_FORM,
    	    XmNbottomAttachment, XmATTACH_WIDGET,
    	    XmNrightAttachment, XmATTACH_FORM,
   	    XmNbottomWidget, button, 0);
    XtVaCreateManagedWidget("bitmap", xmLabelWidgetClass,  scrollW,
    	    XmNlabelType, XmPIXMAP,
    	    XmNlabelPixmap, labelPixmap, 0);
    
    /* set the title of the window */
    XtVaSetValues(XtParent(form), XmNtitle, title, 0);
    
    /* pop up and return wigit id of the help dialog just created */
    ManageDialogCenteredOnPointer(form);
    return form;
}

static void dismissCB(Widget w, caddr_t clientData, caddr_t callData)
{
    XtDestroyWidget(XtParent(w));
}

static void bitmapDismissCB(Widget w, Pixmap pmToDestroy, caddr_t callData)
{
    XFreePixmap(XtDisplay(w), pmToDestroy);
    XtDestroyWidget(XtParent(w));
}

static void helpMenuCB(Widget w, helpMenuInfo *info, caddr_t callData)
{
    if (info->type == HELP_TEXT)
    	CreateHelpDialog(w, info->topic, info->text);
    else /* HELP_BITMAP */
    	CreateBitmapHelpDialog(w, info->topic, info->text, info->width,
    		info->height);
}
