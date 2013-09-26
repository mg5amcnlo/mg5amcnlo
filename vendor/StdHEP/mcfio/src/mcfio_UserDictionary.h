/*
** A small container to hold a set of user block declaration
**
* Written by Paul Lebrun, Aug 2001
*/

typedef struct _aUserBlockDecl {
	int blkNum;
	char *title;
} aUserBlockDecl;

typedef struct _allMCFIO_UserBlockDecl {
	int num;
	int numPreAlloc;
	aUserBlockDecl **decls;
}allMCFIO_UserBlockDecl ;

extern allMCFIO_UserBlockDecl *AllMCFIO_UserBlockDecl;

/*
** Internally used in mcfio. Return NULL if not on the list, 
** otherwise return the point to the relevant title block.
*/
char *mcfioC_UserBlockDescript(int blkNum);
	
