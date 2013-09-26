/*******************************************************************************
*									       *
* misc.h -- Miscelaneous Motif convenience functions			       *
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
/* SCCS ID: misc.h 1.17 1/11/96 */
#define TEXT_READ_OK 0
#define TEXT_IS_BLANK 1
#define TEXT_NOT_NUMBER 2

void AddMotifCloseCallback(Widget shell, XtCallbackProc closeCB, void *arg);
void SuppressPassiveGrabWarnings(void);
void PopDownBugPatch(Widget w);
void RemapDeleteKey(Widget w);
void SetDeleteRemap(int state);
void ManageDialogCenteredOnPointer(Widget dialogChild);
void SetPointerCenteredDialogs(int state);
void AddDialogMnemonicHandler(Widget dialog);
void RemoveDialogMnemonicHandler(Widget dialog);
char *GetXmStringText(XmString fromString);
XFontStruct *GetDefaultFontStruct(XmFontList font);
XmString* StringTable(int count, ...);
void FreeStringTable(XmString *table);
void SimulateButtonPress(Widget widget);
Widget AddMenuItem(Widget parent, char *name, char *label, char mnemonic,
	char *acc, char *accText, XtCallbackProc callback, void *cbArg);
Widget AddMenuToggle(Widget parent, char *name, char *label, char mnemonic,
	char *acc, char *accText, XtCallbackProc callback, void *cbArg,int set);
Widget AddMenuSeparator(Widget parent, char *name);
Widget AddSubMenu(Widget parent, char *name, char *label, char mnemonic);
void SetIntLabel(Widget label, int value);
void SetFloatLabel(Widget label, double value);
void SetIntText(Widget text, int value);
void SetFloatText(Widget text, double value);
int GetFloatText(Widget text, double *value);
int GetIntText(Widget text, int *value);
int GetFloatTextWarn(Widget text, double *value, char *fieldName,int warnBlank);
int GetIntTextWarn(Widget text, int *value, char *fieldName, int warnBlank);
void BeginWait(Widget topCursorWidget);
void EndWait(Widget topCursorWidget);
void PasswordText(Widget w, char *passTxt);
