/************************************************************************
 *									*
 * DialogF.h -- interface for modal error dialog routine		*
 *									*
 * Copyright (c) 1991 Universities Research Association, Inc.           *
 * All rights reserved.                                                 *
 *                                                                      *
 * Fermilab Nirvana Project	                                   	*
 * April 26, 1991                                                       *
 *									*
 * Written by Joy Kyriakopulos						*
 *									*
 ************************************************************************/
/* SCCS ID: DialogF.h 1.4 12/16/93 */

/*
 *  To use DialogF, #include <stdarg.h>
 */
 
#define DF_ERR 1			/* Error Dialog       */
#define DF_INF 2			/* Information Dialog */
#define DF_MSG 3			/* Message Dialog     */
#define DF_QUES 4			/* Question Dialog    */
#define DF_WARN 5			/* Warning Dialog     */
#define DF_PROMPT 6			/* Prompt Dialog      */

#define DF_MAX_MSG_LENGTH 2047		/* longest message length supported */
#define DF_MAX_PROMPT_LENGTH 255	/* longest prompt string supported */

unsigned DialogF(unsigned, Widget, unsigned, char*, ...);
				/* variable # arguments */
