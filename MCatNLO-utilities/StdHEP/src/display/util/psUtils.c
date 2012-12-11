/*******************************************************************************
*	 								       *
* psUtils.c -- PostScript file output routines			               *
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
* April 16, 1992							       *
*									       *
* Written by Arnulfo Zepeda-Navratil				               *
*            Centro de Investigacion y Estudio Avanzados ( CINVESTAV )         *
*            Mexico                                                            *
*									       *
* With some portions from psFiles.c by Sanza T. Kazadi, Fermilab	       *
*                                                                              *
* June 1994 : Upgrade to Encapsulated Postscript, by P. Lebrun                 *
*             Include also a routine to draw rectangles                        *
*									       *
*******************************************************************************/
static char SCCSID[] = "@(#)psUtils.c	1.11	7/17/95";

#include <stdio.h>
#include <time.h>
#include <X11/Xlib.h>
#include <Xm/Xm.h>
#include "psUtils.h"

static void echPS(char *str, int siz, int posx, int posy, int anchor,
	char *fontname, double red, double green, double blue);
static void setXGCLineParams(Display *display, GC gc);
static void getXParms(Display *display, GC gc,
	unsigned short *red_ptr, unsigned short *green_ptr, 
	unsigned short *blue_ptr, double *lineWidth);
static XFontStruct *getFontStruct(XmFontList font);

/* Parameters of the current open file  */
static PSWidth  = 0;
static PSHeight = 0; 
static FILE *PSFile;

/* Maximum color component value in X Windows */
#define COLOR_FACTOR 65535.0

/* Coordinate conversion if necessary is done here.  This is really dumb, and
   should be eliminated by properly scaling the PS coordinate system in the
   first place.  It's left here because fixing it will probably break nfit.
   (fixing it would also allow us to better integrate the 3D widgets which
   already use the more rational scaling method) */
#define x_X2PS(x)     (x)
#define y_X2PS(y)     (PSHeight?PSHeight-y:(y))

/*
** Open a text file for one page of PostScript output, append text if
** already created.  On error, return a NULL pointer.  This routine
** also sets up the coordinate system and clipping for later output routines.
*/
FILE *OpenPS(char fname[], int width, int height)
{
    time_t tt;
    PSFile = fopen(fname,"w");
    if (PSFile == NULL)
    	return NULL;
    fprintf(PSFile, "%%!PS-Adobe-3.0 EPSF-3.0\n");
    fprintf(PSFile, "%%%%BoundingBox: %d %d %d %d \n", PAGE_MARGIN,
    	    PAGE_MARGIN, width*72/75 + PAGE_MARGIN, height*72/75 + PAGE_MARGIN);
    fprintf(PSFile, "%%%%Title: %s \n", fname);
    time(&tt);
    fprintf(PSFile, "%%%%CreationDate: %s \n", ctime(&tt));
    
    /* Set up the page margin.  This is essential since most PostScript
       printers can't print all of the way to the edge of the page.  For
       encapsulated PostScript, we aren't allowed to use initclip to find
       out the real boundaries of the page, so this guess must suffice */
    fprintf(PSFile, "%d %d translate\n", PAGE_MARGIN, PAGE_MARGIN);
    
    /* Scale coordinates to 75 dpi rather than 72 because most X screens
       are closer to 75, and because being an even multiple of 300, simple
       line thicknesses like 1.0 and .5 will be more uniform */
    fprintf(PSFile, "72 75 div dup scale\n");

    /* Save the graphics state so that further scaling and transformations
       can be undone if necessary (as in PSSetWindowPagePosition) */
    fprintf(PSFile, "gsave\n");
    
    /* Define PostScript procedures for drawing dots and line segments:
       d draws a dot using x and y coordinates from the stack, l draws
       a line between arguments x1 y1 x2 y2 from the stack, rect draws
       a rectangle, fillrect fills a rectangle */
    fprintf(PSFile, "%% Draw a dot. Args are x, y\n\
/d {\n\
currentlinewidth 2 div sub moveto\n\
0 currentlinewidth rlineto stroke\n\
} def\n\
%% Draw a line.  Args are x1, y1, x2, y2\n\
/l {\n\
moveto lineto stroke\n\
} def\n\
%% Draw a rectangle.  Args are width, height, bottom, left\n\
/rect {\n\
gsave\n\
translate\n\
matrix currentmatrix\n\
3 1 roll\n\
scale\n\
newpath\n\
0 0 moveto\n\
0 1 lineto\n\
1 1 lineto\n\
1 0 lineto\n\
closepath\n\
setmatrix\n\
stroke\n\
grestore\n\
} def\n\
%% Draw a filled rectangle.  Args are width, height, bottom, left\n\
/fillrect {\n\
gsave\n\
translate\n\
scale\n\
newpath\n\
0 0 moveto\n\
0 1 lineto\n\
1 1 lineto\n\
1 0 lineto\n\
closepath\n\
fill\n\
grestore\n\
} def\n\
%% Draw an arc.  Args are: x, y, radius, angle1, angle2\n\
/xarc {\n\
newpath\n\
arc\n\
stroke\n\
} def\n\
%% Draw a filled arc.  Args are: x, y, radius, angle1, angle2\n\
/fillarc {\n\
newpath\n\
arc\n\
fill\n\
} def\n");
    
    /* Clip to the width and height of the window on the screen */
    fprintf(PSFile, "0 0 moveto\n");
    fprintf(PSFile, "%d 0 lineto\n", width);
    fprintf(PSFile, "%d %d lineto\n", width, height);
    fprintf(PSFile, "0 %d lineto\n", height);
    fprintf(PSFile, "closepath clip\n");
    fprintf(PSFile, "newpath\n");

    /* save the width and height */
    PSWidth = width;
    PSHeight = height;

    return PSFile;
}

/*
** End writing to a PostScript file
*/
void EndPS(void)
{
    fprintf(PSFile, "showpage grestore\n");
    fclose(PSFile);
}

/*
** Get the file currently being written using the PSxxx routines below
*/
FILE *PSGetFile(void)
{
    return PSFile;
}

/*
** Change the file to be written when using the PSxxx routines below (Note:
** you still need to use OpenPS to create the necessary preamble definitions
** for most of the PSxxx drawing routines to work properly).  "fp" must
** represent a file which is already open for writing.
*/
void PSSetFile(FILE *fp)
{
    PSFile = fp;
}

/*
** Set the rectangle on the PostScript page which will be equivalent to
** the X window for drawing with the PSxxx commands in this module.
** The coordinate system (set up by OpenPS) is 75dpi with 0,0 at
** PAGE_MARGIN, PAGE_MARGIN.
*/
void PSSetWindowPagePosition(int left, int bottom, int width, int height)
{
    /* Undo cliping from previous window page position */
    fprintf(PSFile, "grestore\n");
    fprintf(PSFile, "gsave\n");
    
    /* Clip to the width and height of the window on the screen */
    fprintf(PSFile, "%d %d translate\n", left, bottom);
    fprintf(PSFile, "0 0 moveto\n");
    fprintf(PSFile, "%d 0 lineto\n", width);
    fprintf(PSFile, "%d %d lineto\n", width, height);
    fprintf(PSFile, "0 %d lineto\n", height);
    fprintf(PSFile, "closepath clip\n");
    fprintf(PSFile, "newpath\n");
    
    /* Reset the origin
    fprintf(PSFile, "%d %d translate\n", bottom, left);

    /* Save the width and height.  PSHeight is used for flipping vertical
       axis coordinates (in X (0,0) is at the top of the page) */
    PSWidth = width;
    PSHeight = height;
}

/*
** Draw a continuous connected line from an array of X XPoint structures
** to the PostScript file
*/
void PSDrawLines(Display *display, Drawable w, GC gc,
	XPoint *points, int nPoints, int mode)
{
    int i;
    XPoint *point;

    if (mode != CoordModeOrigin) {
    	fprintf(stderr, "PSDrawLines only does CoordModeOrigin (so far)\n");
    	return;
    }
    
    /* set line drawing parameters from contents of X graphics context */
    setXGCLineParams(display, gc);

    /* generate PostScript calls to the line drawing procedure l defined 
       in OpenPS above to draw each of the segments in the array */
    fprintf(PSFile, "%d %d moveto\n", x_X2PS(points->x), y_X2PS(points->y));
    for(i=1, point= &points[1]; i<nPoints; i++, point++) 
        fprintf(PSFile, "%d %d lineto\n", x_X2PS(point->x), y_X2PS(point->y));
    fprintf(PSFile, "stroke\n");
}

/*
** Draw colored line segments from an array of X XSegment structures
** to the PostScript file
*/
void PSDrawSegments(Display *display, Drawable w, GC gc,
	XSegment *segment, int nSegments)
{
    int j;

    /* set line drawing parameters from contents of X graphics context */
    setXGCLineParams(display, gc);

    /* generate PostScript calls to the line drawing procedure l defined 
       in OpenPS above to draw each of the segments in the array */
    for(j=0; j<nSegments; j++,segment++) 
        fprintf(PSFile, "%d %d %d %d l\n",
        	x_X2PS(segment->x1), y_X2PS(segment->y1),
                x_X2PS(segment->x2), y_X2PS(segment->y2));
}

void PSDrawLine(Display *display, Drawable w, GC gc, int x1, int y1,
	int x2, int y2)
{
    XSegment seg;
    
    seg.x1 = x1; seg.x2 = x2; seg.y1 = y1; seg.y2 = y2;
    PSDrawSegments(display, w, gc, &seg, 1);
}

/*
** Draw colored line segments from a floating point equivalent of the
** XSegment data structure to the PostScript File.  The coordinate system
** is still assumed to be set up to be equivalent to X coordinate system,
** the floating point values just allow lines to be positioned at a greater
** precision within the 72 dpi grid of the screen coordinate system.
*/
void PSFloatDrawSegments(Display *display, Drawable w, GC gc, 
	FloatSegment *segment, int nSegments)
{
    int j;

    /* set line drawing parameters from contents of X graphics context */
    setXGCLineParams(display, gc);

    /* generate PostScript calls to the line drawing procedure l defined 
       in OpenPS above to draw each of the segments in the array */
    for(j=0; j<nSegments; j++,segment++) 
        fprintf(PSFile, "%g %g %g %g l\n",
        	x_X2PS(segment->x1), y_X2PS(segment->y1),
                x_X2PS(segment->x2), y_X2PS(segment->y2));
}

/*
** Draw a continuous connected line from a floating point equivalent of the
** XPoint data structure to the PostScript File.  The coordinate system
** is still assumed to be set up to be equivalent to X coordinate system,
** the floating point values just allow lines to be positioned at a greater
** precision within the 72 dpi grid of the screen coordinate system.
*/
void PSFloatDrawLines(Display *display, Drawable w, GC gc,
	FloatPoint *points, int nPoints)
{
    int i;
    FloatPoint *point;

    /* set line drawing parameters from contents of X graphics context */
    setXGCLineParams(display, gc);

    /* generate PostScript calls to the line drawing procedure l defined 
       in OpenPS above to draw each of the segments in the array */
    fprintf(PSFile, "%g %g moveto\n", x_X2PS(points->x), y_X2PS(points->y));
    for(i=1, point= &points[1]; i<nPoints; i++, point++) 
        fprintf(PSFile, "%g %g lineto\n", x_X2PS(point->x), y_X2PS(point->y));
    fprintf(PSFile, "stroke\n");
}

/*
** Draw colored points from an X XPoint structure to the PostScript file
*/
void PSDrawPoints(Display *display, Drawable w, GC gc,
                  XPoint *point, int npoints, int mode)
{
    int j;

    /* Set line drawing parameters from contents of X graphics context */
    setXGCLineParams(display, gc);

    /* generate PostScript calls to the line drawing procedure l defined 
       in OpenPS above to draw each of the segments in the array */
    for(j=0; j<npoints; j++,point++)
    	fprintf(PSFile, "%d %d d\n", x_X2PS(point->x), y_X2PS(point->y));
}

/*
** Display a colored Motif compound string in Times Roman at a point size
** one point smaller than the default font of the fontlist specified.  
** Parameters are the same as XmStringDraw, except minus layout direction
** and clipping.  The routine can't yet handle multiline strings,
** clipping, or right-to-left character sets.
*/
void PSDrawXmString(Display *display, Drawable w, XmFontList font, XmString msg,
		    GC gc, int x, int y, int width, int alignment)
{
    char *ansiMsg;
    XFontStruct *fs = getFontStruct(font);
    int adjX, adjY, anchor;
    
    /* Calculate revised coordinates and anchor mode for drawing the string,
       using the parameters in the form required by XmStringDraw */
    adjY = y + fs->ascent;	/* change y to top left corner of string */
    if (alignment == XmALIGNMENT_BEGINNING) {
    	adjX = x;
    	anchor = PS_LEFT;
    } else if (alignment == XmALIGNMENT_CENTER) {
    	adjX = x + width/2;	
    	anchor = PS_CENTER;
    } else /* XmALIGNMENT_END */ {
        adjX = x + width;
        anchor = PS_RIGHT;
    }
    
    /* Convert the string to a C style stle string and call PSDrawString */
    XmStringGetLtoR(msg, XmSTRING_DEFAULT_CHARSET, &ansiMsg);
    PSDrawString(display, w, gc, fs, adjX, adjY, anchor, ansiMsg);
    XtFree(ansiMsg);
}

/*
** Display a colored string in Times Roman at a point size
** one point smaller than the default font specified in fs
*/
void PSDrawString(Display *display, Drawable w, GC gc, XFontStruct *fs,
                   int x, int y, int anchor, char *msg)
{
    double lw;
    unsigned short red, green, blue;

    getXParms(display, gc, &red, &green, &blue, &lw);

    echPS(msg, fs->ascent + fs->descent - 2, x_X2PS(x), y_X2PS(y),
    	  anchor, "Times-Roman",     
    	  (float)red   / COLOR_FACTOR,
          (float)green / COLOR_FACTOR,
          (float)blue  / COLOR_FACTOR);
}

/*
** Draw rectangles from an array of X XRectangle structures
*/
void PSDrawRectangles(Display *display, Drawable w, GC gc,
                  XRectangle *rects, int nRects)
{
    int j;
    XRectangle *rect;

    /* Set line drawing parameters from contents of X graphics context */
    setXGCLineParams(display, gc);

    for(j=0, rect=rects; j<nRects; j++, rect++)
        fprintf(PSFile,"%d %d %d %d rect\n", rect->width, rect->height,
        	x_X2PS(rect->x), y_X2PS(rect->y) - rect->height);
}

/*
** Draw filled rectangles from an array of X XRectangle structures
*/
void PSFillRectangles(Display *display, Drawable w, GC gc,
	XRectangle *rects, int nRects)
{
    int j;
    XRectangle *rect;

    /* Set line drawing parameters from contents of X graphics context */
    setXGCLineParams(display, gc);

    for(j=0, rect=rects; j<nRects; j++, rect++)
        fprintf(PSFile,"%d %d %d %d fillrect\n", rect->width, rect->height,
        	x_X2PS(rect->x), y_X2PS(rect->y) - rect->height);
}

/*
** Draw arcs from an array of X XArc structures
*/
void PSDrawArcs(Display *display, Drawable w, GC gc,
	XArc *arcs, int nArcs)
{
    int j;
    XArc *arc;

    /* Set line drawing parameters from contents of X graphics context */
    setXGCLineParams(display, gc);

    for(j=0, arc=arcs; j<nArcs; j++, arc++) {
    	if (arc->width != arc->height) {
    	    fprintf(stderr, "PSDrawArcs doesn't yet support eliptical arcs\n");
    	    return;
    	}
        fprintf(PSFile,"%d %d %d %d %d xarc\n", x_X2PS(arc->x + arc->width/2),
        	y_X2PS(arc->y - arc->height/2), arc->width/2, arc->angle1/64,
        	arc->angle2/64);
    }
}

/*
** Draw filled arcs from an array of X XArc structures
*/
void PSFillArcs(Display *display, Drawable w, GC gc,
	XArc *arcs, int nArcs)
{
    int j;
    XArc *arc;

    /* Set line drawing parameters from contents of X graphics context */
    setXGCLineParams(display, gc);

    for(j=0, arc=arcs; j<nArcs; j++, arc++) {
    	if (arc->width != arc->height) {
    	    fprintf(stderr, "PSDrawArcs doesn't yet support eliptical arcs\n");
    	    return;
    	}
        fprintf(PSFile,"%d %d %d %d %d fillarc\n",
        	x_X2PS(arc->x + arc->width/2), y_X2PS(arc->y - arc->height/2),
        	arc->width/2, arc->angle1/64, arc->angle2/64);
    }
}

/*
** draw an image
*/
void PSDrawImage(display, w, gc, scanLength, scanLines, bitsPerSample,
                 x, y, bitmap)
       Display *display;
       Drawable w;
       GC gc;
       int scanLength, scanLines, bitsPerSample, x, y;
       unsigned char *bitmap;
{
    int i, disp_x, disp_y, letter;

    /* set image drawing parameters from the contents of X graphics context */
    setXGCLineParams(display, gc);

    /* translate the origin to drawing position */
    disp_x = x;
    disp_y = y ;
    /* draw image */
    fprintf(PSFile, "%d %d translate\n", x_X2PS(disp_x), y_X2PS(disp_y));
    fprintf(PSFile, "%d %d %d [%d 0 0 %d 0 0] {<", scanLength, 
                     scanLines, bitsPerSample, scanLength, scanLines);
    for (i = 0; i < scanLength; i++) {
      letter = (int)(bitmap[i] & 0xf0);

      fprintf(PSFile, "%02x", bitmap[i]);
    }
    fprintf(PSFile, ">} image\n");

    /* restore origin */
    fprintf(PSFile, "%d %d translate\n", -x_X2PS(disp_x), -y_X2PS(disp_y));
}
   

/*
** Draw dashed (or solid) line segments (dashed lines can't be drawn by
** PSDrawSegments because X does not provide access to the complete dash
** information in the gc
*/
void PSDrawDashedSegments(Display *display, Drawable w, GC gc,
	XSegment *segments, int nSegments, char *dashList, int dashOffset)
{
    int i, len;
    XGCValues valuesRet;

    /* if dashed lines are turned off in the GC, draw solid */
    XGetGCValues(display, gc, GCLineStyle, &valuesRet);
    if (valuesRet.line_style == LineSolid) {
    	PSDrawSegments(display, w, gc, segments, nSegments);
    	return;
    }
  
    /* transform and set dash list */
    len = strlen(dashList);
    if (len != 0) {
	fprintf(PSFile, "["); 
	for (i = 0; i < len - 1; i++) 
	      fprintf(PSFile, "%3d ", (int)dashList[i]);
	fprintf(PSFile, "%3d] %d setdash\n", (int)(dashList[len-1]), 
        	dashOffset);  
    }

    /* draw the segments */
    PSDrawSegments(display, w, gc, segments, nSegments);

    /* set the line style back to solid line */
    fprintf(PSFile, "[] 0 setdash\n");
}

/*
** Draw continuous dashed lines from the floating point equivalent of the
** XPoint data structure
*/
void PSFloatDrawDashedLines(Display *display, Drawable w, GC gc,
	FloatPoint *points, int nPoints, char *dashList, int dashOffset)
{
    int i, len;
    XGCValues valuesRet;

    /* if dashed lines are turned off in the GC, draw solid */
    XGetGCValues(display, gc, GCLineStyle, &valuesRet);
    if (valuesRet.line_style == LineSolid) {
    	PSFloatDrawLines(display, w, gc, points, nPoints);
    	return;
    }

    /* transform and set dash list */
    len = strlen(dashList);
    if (len != 0) {
	fprintf(PSFile, "["); 
	for (i = 0; i < len - 1; i++) 
	      fprintf(PSFile, "%3d ", (int)dashList[i]);
	fprintf(PSFile, "%3d] %d setdash\n", (int)(dashList[len-1]), 
        	dashOffset);  
    }

    /* draw the lines */
    PSFloatDrawLines(display, w, gc, points, nPoints);

    /* set the line style back to solid line */
    fprintf(PSFile, "[] 0 setdash\n");
}
    
/*
** Draw a dashed line (dashed lines can't be drawn by PSDrawLine because
** X does not give access to the complete dash information in the gc
*/
void PSDrawDashedLine(Display *display, Drawable w, GC gc, int x1, int y1,
	int x2, int y2, char *dashList, int dashOffset)
{
    XSegment seg;
    
    seg.x1 = x1; seg.x2 = x2; seg.y1 = y1; seg.y2 = y2;
    PSDrawDashedSegments(display, w, gc, &seg, 1, dashList, dashOffset);
}

/*
** Change the current clip rectangle
*/
void PSSetClipRectangle(int x1, int y1, int x2, int y2)
{
    fprintf(PSFile, "newpath\n");
    fprintf(PSFile, "%d %d moveto\n", x_X2PS(x1), y_X2PS(y1));
    fprintf(PSFile, "%d %d lineto\n", x_X2PS(x2), y_X2PS(y1));
    fprintf(PSFile, "%d %d lineto\n", x_X2PS(x2), y_X2PS(y2));
    fprintf(PSFile, "%d %d lineto\n", x_X2PS(x1), y_X2PS(y2));
    fprintf(PSFile, " %d %d lineto\n", x_X2PS(x1), y_X2PS(y1));
    fprintf(PSFile, "closepath clip newpath\n");
}

/* 
** Output colored text
*/
static void echPS(char *str, int siz, int posx, int posy, int anchor,
		  char *fontname, double red, double green, double blue)
{   
    fprintf(PSFile, "%.2f %.2f %.2f setrgbcolor ", red, green, blue);
    fprintf(PSFile, "/%s findfont %04d scalefont setfont\n", fontname, siz);
    if (anchor == PS_LEFT)
    	fprintf(PSFile, "%d %d moveto\n", posx, posy);
    else if (anchor == PS_CENTER)
    	fprintf(PSFile, "(%s) stringwidth pop 2 div neg %d add %d moveto\n",
    		str, posx, posy);
    else if (anchor == PS_RIGHT)
    	fprintf(PSFile, "(%s) stringwidth pop neg %d add %d moveto\n",
    		str, posx, posy);
    else {
    	fprintf(stderr, "Internal error: bad anchor value in echPS\n");
    	return;
    }	
    fprintf(PSFile, "(%s) show\n", str);
}

/*
** Set a subset of GC parameters relating to line appearance for subsequent
** drawing operations
*/
static void setXGCLineParams(Display *display, GC gc)
{
    double lineWidth;
    unsigned short red, green, blue;

    /* Set line drawing parameters from contents of X graphics context */
    getXParms(display, gc, &red, &green, &blue, &lineWidth);
    fprintf(PSFile, "%.2f %.2f %.2f setrgbcolor ", (float)red/COLOR_FACTOR,
    	    (float)green/COLOR_FACTOR, (float)blue/COLOR_FACTOR);
    fprintf(PSFile, "%.2f setlinewidth\n", lineWidth);
}

/* 
** Obtain X Window related drawing parameters
** and massage them for PostScript printers
*/
static void getXParms(Display *display, GC gc, unsigned short *red_ptr,
	unsigned short *green_ptr, unsigned short *blue_ptr, double *lineWidth)
{ 
    XGCValues valuesRet;
    XColor ret_color;

    XGetGCValues(display, gc, GCForeground | GCLineWidth, &valuesRet);
    /*XGetWindowAttributes(display, w, window_attributes);*/
    ret_color.pixel = valuesRet.foreground;
    ret_color.flags = DoRed | DoGreen | DoBlue ;
    XQueryColor(display, /*window_attributes.colormap*/   /* Get color rgb */
    	    DefaultColormap(display,0), &ret_color);
    *red_ptr   = ret_color.red;
    *green_ptr = ret_color.green;
    *blue_ptr  = ret_color.blue;
    *lineWidth = valuesRet.line_width == 0 ? 0.5 : valuesRet.line_width;
}

/*
** Get the XFontStruct that corresponds to the default (first) font in
** a Motif font list.  Since Motif stores this, it saves us from storing
** it or querying it from the X server.
*/
static XFontStruct *getFontStruct(XmFontList font)
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
