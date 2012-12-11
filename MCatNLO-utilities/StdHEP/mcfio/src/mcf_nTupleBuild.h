/*******************************************************************************
*									       *
* mcf_nTupleBuild.h -- Include file for mcfast generalized nTuple Builder GUI  *
*									       *
*	P. Lebrun, September 1995.					       *
*									       *
*******************************************************************************/
#define APP_NAME "mcf_ntuBuild" /* application name for loading X resources */
#define APP_CLASS "MCF_NtuBuild"

enum langTypes {F77_LANGUAGE, C_LANGUAGE};

typedef struct nTuBuildWindowRec {
    struct nTuBuildWindowRec *next;
    int id;
    int isSaved;
    int hasChanged;
    char *dbinFileName;
    int isReadOnly;
    descrGenNtuple *descrNtu;
    int undoIndex;
    int selectedIndex;
    char titleBlank, multiplicityBlank, nameIndexBlank;
    int langEnv;
    varGenNtuple *undoVariable;
    				/* Main panel widgets */
    Widget shell;
    Widget form;
    Widget titleW, descriptW, versionW, multiplicityW, nameIndexW; 
    Widget nameW, descriptVarW;
    Widget cutBtn, copyBtn, pasteBtn, clearBtn, insertBtn;
    Widget changeBtn, undoBtn;
    Widget listW;
    Widget typeMenu;
    Widget typeBtns[N_VAR_TYPES];
    Widget dimNumMenu;
    Widget dimBtns[4];
    Widget indexingMenu;
    Widget indexingBtns[2];
    Widget dimDialogW, dimDialogShell, dimDialogForm, dimDialogComLbl ;
    
    				/* Menu type Widgets */
    Widget saveItem;
    Widget saveAsItem; 
    Widget generateF77;
    Widget generateC;
    Widget generateDbin;
    Widget closeItem;
} nTuBuildWindow;

    
