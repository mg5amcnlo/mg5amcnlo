*
* $Id: qqprop.inc,v 1.1 2005/09/29 19:27:22 garren Exp $
*
* $Log: qqprop.inc,v $
* Revision 1.1  2005/09/29 19:27:22  garren
* 5.04.02
*
* Revision 1.1 1994/10/07 23:57:39 zfiles
* New include files for QQ.
*
*
*CMZ : 1.02/61 03/10/94 17.34.16 by Paul Avery
*CMZ : 1.03/70 11/10/93 09.28.52 by Paul Avery
*CMZ : 08/10/93 13.00.47 by Paul Avery
*>> Author :
* 17/10/96 Lynn Garren: Add double precision conditionals.
*
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
* QQPROP.INC
*
* Particle properties
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*
C
* NPMNQQ Minimum particle # defined in list
* NPMXQQ Maximum particle # defined in list
*
* IDPDGQ(IP) PDG particle code
* IDMC(IP) Stable particle code (not really used in QQ)
* LPARTY(IP) Parity of particle
* CPARTY(IP) C parity of particle
* IMIXPP(IP) IMIXPP(IP) = particle it mixes into (0 if no mixing)
* ICPMIX(IP) ICPMIX(IP) = particle it mixes into (0 if no CP violation)
* HIDEQQ(IP) TRUE ==> hidden particle, never appears in decay history.
* When produced, it is decayed immediately.
      INTEGER NPMNQQ, NPMXQQ
      INTEGER IDPDGQ, IDMC, LPARTY, CPARTY, IMIXPP, ICPMIX
      LOGICAL HIDEQQ
      COMMON/QQPAR1/
     * NPMNQQ, NPMXQQ,
     * IDPDGQ(-20:MCNUM),
     * IDMC(-20:MCNUM),
     * LPARTY(-20:MCNUM), CPARTY(-20:MCNUM),
     * IMIXPP(-20:MCNUM), ICPMIX(-20:MCNUM),
     * HIDEQQ(-20:MCNUM)
* AMASS(IP) Particle mass
* CHARGE(IP) Charge
* CTAU(IP) Lifetime in meters
* SPIN(IP) spin (0, 0.5, 1.0, 1.5, 2.0, ...)
* RWIDTH(IP) widths
* RMASMN(IP) Minimum mass (for RWIDTH > 0.)
* RMASMX(IP) Maximum mass (for RWIDTH > 0.)
* RMIXPP(IP) RMIXPP(IP) = dM/Gamma = mixing strength
* RCPMIX(IP) RCPMIX(IP) = CP violation parameter
      DOUBLE PRECISION AMASS, CTAU, RWIDTH, RMASMN, RMASMX
      REAL CHARGE, SPIN
      DOUBLE PRECISION RMIXPP, RCPMIX
      COMMON/QQPAR2/
     * AMASS(-20:MCNUM), CTAU(-20:MCNUM),
     * RWIDTH(-20:MCNUM), RMASMN(-20:MCNUM), RMASMX(-20:MCNUM),
     * RMIXPP(-20:MCNUM), RCPMIX(-20:MCNUM),
     * CHARGE(-20:MCNUM),
     * SPIN(-20:MCNUM)
* INVMC List of particle codes for stable particle (Inverse of IDMC)
      INTEGER INVMC
      COMMON/QQPAR3/
     * INVMC(0:MCSTBL)
* NPOLQQ Number of polarized decays
* IPOLQQ List of daughter particles
      INTEGER NPOLQQ, IPOLQQ
      COMMON/QQPAR4/
     * NPOLQQ, IPOLQQ(5,MPOLQQ)
* QNAME qqbar names (37th is generic qqbar)
* PNAME Particle names
      CHARACTER QNAME*10, PNAME*10
      COMMON/QQPAR5/
     * QNAME(37), PNAME(-20:MCNUM)
      CHARACTER*10 PNAMEA(MCNUM1)
      EQUIVALENCE (PNAME(-20), PNAMEA)
*
