*
* $Id: qqvrtx.inc,v 1.1 2005/09/29 19:27:22 garren Exp $
*
* $Log: qqvrtx.inc,v $
* Revision 1.1  2005/09/29 19:27:22  garren
* 5.04.02
*
* Revision 1.1 1994/10/07 23:57:40 zfiles
* New include files for QQ.
*
*
*CMZ : 1.02/61 04/10/94 17.22.38 by Paul Avery
*CMZ : 1.03/70 15/10/93 08.44.26 by Paul Avery
*CMZ : 08/10/93 16.00.06 by Paul Avery
*>> Author :
* 14/10/96 Lynn Garren: Change XVTX, TVTX, RVTX from real to double precision.
*
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
* QQVRTX.INC
*
* QQ vertex list
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C Contains a list of vertices. When a particle decays, is absorbed
C or converts to e+e- a new vertex is created. Various pointers
C relate the vertex list to the full particle list in QQTRAK.INC.
C
C VPROD(IT) is the vertex at which particle IT was produced while
C IVDECA(IT) is the vertex at which particle IT either decayed, was
C absorbed or converted, or 0 if nothing happened. Conversely,
C ITRKIN(IV) is the incoming track to vertex IV or 0 if IV = 1 (the
C production vertex has no incoming track). ITRKOU(IV) is the first
C outgoing track from vertex IV or 0 if the track is absorbed.
C >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>-
C
* NVRTX # vertices
* XVTX X,Y,Z of vertex in meters
* TVTX T of vertex in meters
* RVTX SQRT(X**2 + Y**2)
* ITRKIN Pos. in track list of incoming trk
* NTRKOU # outgoing tracks
* ITRKOU Position in track list of 1st outgoing track
* IVKODE (1=decay, 2=absorbed, 3=photon conv.)
      INTEGER NVRTX
      COMMON/QQVRT1/
     * NVRTX
      DOUBLE PRECISION XVTX, TVTX, RVTX
      INTEGER ITRKIN, NTRKOU, ITRKOU, IVKODE
      COMMON/QQVRT2/
     * XVTX(MCVRTX,3), TVTX(MCVRTX), RVTX(MCVRTX),
     * ITRKIN(MCVRTX), NTRKOU(MCVRTX), ITRKOU(MCVRTX),
     * IVKODE(MCVRTX)
C
