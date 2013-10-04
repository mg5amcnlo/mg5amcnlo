/*******************************************************************************
*									       *
* stringUtils.c -- Functions and macros for Motif compound strings	       *
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
* July 11, 1991								       *
*									       *
* Written by Mark Edel							       *
*									       *
*******************************************************************************/
static char SCCSID[] = "@(#)stringUtils.c	1.3	9/24/92";
#include <Xm/Xm.h>
#include "stringUtils.h"

#define TOGGLE_CHAR '~'		/* escape character */
#define NORMAL_CHAR 'n'		/* set normal font */
#define GREEK_CHAR 'g'		/* set greek font */
#define SUPER_CHAR 'u'		/* set superscript font */
#define SUB_CHAR 'd'		/* set subscript font */

enum stringType {NORMAL, GREEK, SUPER, SUB};
#define NUM_TYPES 4

/* character set names */
static char TypeNames[NUM_TYPES][8] = {"normal", "greek", "super", "sub"};
static char FontChars[NUM_TYPES] =
	{NORMAL_CHAR, GREEK_CHAR, SUPER_CHAR, SUB_CHAR};

/* Function Prototypes */
XmString extendString(XmString cString, char *string, char *charset);
static int charsetNum(char *charset);
static int expandEscapeChars(char *fromString, char *toString, int maxLength);

/*
** MultiFontString
**
** Create a string containing several different fonts.  The input string can
** contain escape sequences which switch the font to one of five named
** Motif character sets.  To use this function, you need to have specified
** a fontList which contains fonts labeled with specific character
** set names.  The character sets used by this function are: normal,
** greek, superscript and subscript.  For example, your app-defaults or
** .Xdefaults file might contain a line similar to:
**
**   program*fontList:Rom14=normal, symb14=greek, sup=super, sub=sub
**
** The escape sequences that activate these fonts begin with the tilde (~)
** character.  ~n activates the normal character set, ~g activates the greek
** character set, ~u activates the superscript character set, ~d activates the
** subscript character set.  ~~ becomes a single ~ in the output string.  The
** caller is responsible for calling XmStringFree on the returned result when
** it is no longer needed.
**
** Arguments
**
**	string		A C style (null terminated) string.  The string can
**			contain escape sequences (~~, ~n, ~g, ~u, ~d) described
**			above to build a string with several fonts
**
** Return Value
**
**	A Motif style compound string.  The caller is responsible for freeing
**	this string to release the memory allocated to it.
*/
XmString MultiFontString(char *string)
{
    XmString cString;
    char *segment, *segPtr, *stringPtr;
    int segType;
    
    segment = XtMalloc(strlen(string)+1);
    segPtr = segment;
    segType = NORMAL;
    cString = XmStringCreateLtoR("" , TypeNames[NORMAL]);
    for(stringPtr=string; *stringPtr!='\0'; stringPtr++) {
    	if (*stringPtr == TOGGLE_CHAR) {
    	    stringPtr++;
    	    switch (*stringPtr) {
    	      case '\0':
    	        /* string ended in the middle of escape sequence */
    	        stringPtr--;
    	        break;
    	      case TOGGLE_CHAR:
    	    	/* double toggle character inserts toggle character */
    	    	*segPtr++ = *stringPtr;
    	    	break;
    	      case NORMAL_CHAR:
    	      case GREEK_CHAR:
    	      case SUPER_CHAR:
    	      case SUB_CHAR:
    	    	/* toggle to other character set after creating segment */
		if (segPtr > segment) {
		    *segPtr = '\0';
		    cString = extendString(cString, segment,TypeNames[segType]);
     	    	}
     	    	segPtr = segment;
    	    	if (*stringPtr == NORMAL_CHAR)
    	    	    segType = NORMAL;
    	    	if (*stringPtr == GREEK_CHAR)
    	    	    segType = GREEK;
    	    	if (*stringPtr == SUPER_CHAR)
    	    	    segType = SUPER;
    	    	if (*stringPtr == SUB_CHAR)
    	    	    segType = SUB;
    	    	break;
    	      default:
    	        /* unrecognized escape sequence, just copy character */
    	      	*segPtr++ = *stringPtr;
    	    }
    	} else {
    	    /* not the toggle character, just add to current segment */
    	    *segPtr++ = *stringPtr;
    	}
    }
    /* create the last segment */
    if (segPtr > segment) {
    	*segPtr = '\0';
	cString = extendString(cString, segment, TypeNames[segType]);
    }
    XtFree(segment);
    return cString;	    
}

/* extend an existing compuund string with a C string and character set */
XmString extendString(XmString cString, char *string, char *charset)
{
    XmString cTemp, cSegment;
    
    cSegment = XmStringCreateLtoR(string, charset);
    cTemp = XmStringConcat(cString, cSegment);
    XmStringFree(cString);
    XmStringFree(cSegment);
    return cTemp;
}

/*
** Convert a compound string back to a C style null terminated string
** restoring the font switching escape sequences from MultiFontString
*/
void GetMFString(XmString fromString, char *toString, int maxLength)
{
    XmStringContext context;
    char *text, *toPtr = toString;
    XmStringCharSet charset;
    XmStringDirection direction;
    Boolean separator;
    int newCharset, curCharset = NORMAL;
    
    /* loop over all of the segments in the string */
    XmStringInitContext(&context, fromString);
    while (XmStringGetNextSegment(context, &text,
    		&charset, &direction, &separator)) {
    	
    	/* if character set changes, include toggle escape sequence */
    	newCharset = charsetNum(charset);
    	if (curCharset != newCharset) {
    	    *toPtr++ = TOGGLE_CHAR;
    	    if (toPtr-toString >= maxLength - 1) break;
    	    *toPtr++ = FontChars[newCharset];
    	    if (toPtr-toString >= maxLength - 1) break;
    	    curCharset = newCharset;
    	}
    	
    	/* copy the new segment into the output string expanding escape chars */
    	toPtr += expandEscapeChars(text, toPtr, maxLength-(toPtr-toString)-1);
    	if (toPtr-toString >= maxLength - 1) break;
    	
    	/* if next segment is a separator, add a newline */
    	if (separator) {
    	    *toPtr++ = '\n';
    	    if (toPtr-toString >= maxLength - 1) break;
    	}
    }
    
    /* terminate the string, free the context, and return the length */
    *toPtr++ = '\0';
    XmStringFreeContext(context);
}

/*
** determine if a named character set is available in a font list
*/
Boolean CharsetAvailable(char *charset, XmFontList fontList)
{
#ifndef MOTIF10
    XmFontContext context;
    XFontStruct *font;
    char *fontCharset;
    
    /* Loop through all of the fonts in fontList, matching their character
       set names with the argument charset.  Return true if a match is found */
    if (!XmFontListInitFontContext(&context, fontList))
        return False;
    while (XmFontListGetNextFont(context, &fontCharset, &font)) {
    	if (!strcmp(charset, fontCharset)) {
    	    XtFree(fontCharset);
    	    XmFontListFreeFontContext(context);
    	    return True;
    	}
    	XtFree(fontCharset);
    }
    XmFontListFreeFontContext(context);
    return False;
#else
    return True;
#endif
}

/*
** return the index number that matches a charset string.  If there is
** no match, return the NORMAL charset index.
*/
static int charsetNum(char *charset)
{
    int i;
    
    for (i=0; i<NUM_TYPES; i++) {
	if (!strcmp(charset, TypeNames[i]))
	    return i;
    }
    /* name was not recognized */
    return NORMAL;
}

/*
** Expand the escape character into a double escape sequence where
** it is found in fromString, copy result into toString up to a maximum
** of maxLength characters.
*/
static int expandEscapeChars(char *fromString, char *toString, int maxLength)
{
    char *fromPtr = fromString, *toPtr = toString;
    
    while (*fromPtr!='\0' && toPtr-toString<maxLength) {
    	if (*fromPtr == TOGGLE_CHAR) {
    	    *toPtr++ = TOGGLE_CHAR;
    	    if (toPtr - toString == maxLength)
    	    	break;
    	}
    	*toPtr++ = *fromPtr++;
    }
    return toPtr - toString;
}  
