/*******************************************************************************
*									       *
* fileUtils.c -- File utilities for Nirvana applications		       *
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
* Modified by:	DMR - Ported to VMS (1st stage for Histo-Scope)		       *
*									       *
*******************************************************************************/
static char SCCSID[] = "@(#)fileUtils.c	1.4     8/29/94";
#include <string.h>
#ifdef VAXC
#define NULL (void *) 0
#endif /*VAXC*/
#ifdef VMS
#include "vmsparam.h"
#else
#include <sys/param.h>
#endif /*VMS*/
#include "fileUtils.h"

#define TRUE 1
#define FALSE 0

static int normalizePathname(char *pathname);
static int compressPathname(char *pathname);
static char *nextSlash(char *ptr);
static char *prevSlash(char *ptr);
static int compareThruSlash(char *string1, char *string2);
static void copyThruSlash(char **toString, char **fromString);

/*
** Decompose a Unix file name into a file name and a path
*/
int ParseFilename(char *fullname, char *filename, char *pathname)
{
    int fullLen = strlen(fullname);
    int i, pathLen, fileLen;
	    
#ifdef VMS
    /* find the last ] or : */
    for (i=fullLen-1; i>=0; i--) {
    	if (fullname[i] == ']' || fullname[i] == ':')
	    break;
    }
#else  /* UNIX */
    /* find the last slash */
    for (i=fullLen-1; i>=0; i--) {
    	if (fullname[i] == '/')
	    break;
    }
#endif

    /* move chars before / (or ] or :) into pathname,& after into filename */
    pathLen = i + 1;
    fileLen = fullLen - pathLen;
    strncpy(pathname, fullname, pathLen);
    pathname[pathLen] = 0;
    strncpy(filename, &fullname[pathLen], fileLen);
    filename[fileLen] = 0;
#ifdef VMS
    return TRUE;
#else     /* UNIX specific... Modify at a later date for VMS */
    return normalizePathname(pathname);
}

static int normalizePathname(char *pathname)
{
    char oldPathname[MAXPATHLEN], wd[MAXPATHLEN];

    /* if this is a relative pathname, prepend current directory */
    if (pathname[0] != '/') {
        /* make a copy of pathname to work from */
	strcpy(oldPathname, pathname);
	/* get the working directory */
	getcwd(wd, MAXPATHLEN);
	/* prepend it to the path */
	strcpy(pathname, wd);
	strcat(pathname, "/");
	strcat(pathname, oldPathname);
    }
    /* compress out .. and . */
    return compressPathname(pathname);
}


static int compressPathname(char *pathname)
{
    char *inPtr, *outPtr;

    /* compress out . and .. */
    inPtr = &pathname[1];		/* start after initial / */
    outPtr = &pathname[1];
    while (TRUE) {
	/* if the next component is "../", remove previous component */
	if (compareThruSlash(inPtr, "../")) {
	    /* error if already at beginning of string */
	    if (outPtr == &pathname[1])
	        return FALSE;
	    /* back up outPtr to remove last path name component */
	    outPtr = prevSlash(outPtr);
	    inPtr = nextSlash(inPtr);
	} else if (compareThruSlash(inPtr, "./")) {
	    /* don't copy the component if it's the redundant "./" */
	    inPtr = nextSlash(inPtr);
	} else {
	    /* copy the component to outPtr */
	    copyThruSlash(&outPtr, &inPtr);
	}
	if (inPtr == NULL) {
	    return TRUE;
	}
    }
}

static char *nextSlash(char *ptr)
{
    for(; *ptr!='/'; ptr++) {
    	if (*ptr == '\0')
	    return NULL;
    }
    return ptr + 1;
}

static char *prevSlash(char *ptr)
{
    for(ptr -= 2; *ptr!='/'; ptr--);
    return ptr + 1;
}

static int compareThruSlash(char *string1, char *string2)
{
    while (TRUE) {
    	if (*string1 != *string2)
	    return FALSE;
	if (*string1 =='\0' || *string1=='/')
	    return TRUE;
	string1++;
	string2++;
    }
}

static void copyThruSlash(char **toString, char **fromString)
{
    char *to = *toString;
    char *from = *fromString;
    
    while (TRUE) {
        *to = *from;
        if (*from =='\0') {
            *fromString = NULL;
            return;
        }
	if (*from=='/') {
	    *toString = to + 1;
	    *fromString = from + 1;
	    return;
	}
	from++;
	to++;
    }
#endif /* UNIX */
}
