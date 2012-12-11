/*******************************************************************************
*									       *
* help.h -- Nirvana help display					       *
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
/* SCCS ID: help.h 1.3 9/20/93 */
enum helpTypes {HELP_TEXT, HELP_BITMAP};

typedef struct {
    char *topic;	/* name of subject, used for menu and window title */
    char mnemonic;	/* character from topic string to use as mnemonic */
    char sepAfter;	/* flag, if true put a separator after item in menu */
    enum helpTypes type;/* HELP_TEXT or HELP_BITMAP */
    char *text;		/* pointer to the help text or bitmap data to display
    			   in the help dialog */
    int width;		/* for HELP_BITMAP, width of the bitmap in pixels */
    int height;		/* "" height of the bitmap in pixels */
} helpMenuInfo;

Widget CreateHelpPulldownMenu(Widget menuBar, helpMenuInfo **menuInfo);
Widget CreateHelpDialog(Widget parent, char *title, char *text);
Widget CreateBitmapHelpDialog(Widget parent, char *title, char *bits,
	int width, int height);
