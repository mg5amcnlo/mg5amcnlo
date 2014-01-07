/*******************************************************************************
*									       *
* mcf_NTuBuildWindow.h --    						       *
*									       *
*	P. Lebrun, September 1995.					       *
*									       *
*******************************************************************************/
nTuBuildWindow *CreateNTuBuildWindow(Display *display, char *title,
                                       int ReadOnly); 
int CloseNTuBuildWindow(nTuBuildWindow *window); 
void UpdateVariableList(nTuBuildWindow *window, int newPos);
void UpdateDialogVFields(nTuBuildWindow *window);
void CloneNTuBuildWindow(nTuBuildWindow *fromWindow, nTuBuildWindow *toWindow); 
