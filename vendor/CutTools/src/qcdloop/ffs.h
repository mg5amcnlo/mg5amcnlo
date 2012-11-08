	integer memory
	parameter(memory=12)
	logical lwrite,ltest,l4also,ldc3c4,lmem,lwarn,ldot,onshel,lsmug,
     +		lnasty
	integer nwidth,nschem,idot
	DOUBLE PRECISION xloss,precx,precc,xalogm,xclogm,xalog2,xclog2,
     +		reqprc,x0,x05,x1,x2,x4,pi,pi6,pi12,xlg2,bf(20),
     +		xninv(30),xn2inv(30),xinfac(30),
     +		fpij2(3,3),fpij3(6,6),fpij4(10,10),fpij5(15,15),
     +		fpij6(21,21),fdel2,fdel3,fdel4s,fdel4,fdl3i(5),
     +		fdl3ij(6,6),fdl4i(6)
	COMPLEX c0,c05,c1,c2,c4,c2ipi,cipi2,
     +		cfpij2(3,3),cfpij3(6,6),cfpij4(10,10),cfpij5(15,15),
     +		cfpij6(21,21),cmipj(3,3),c2sisj(4,4),cfdl4s,ca1
	integer nevent,ner,id,idsub,inx(4,4),isgn(4,4),isgn34,isgnal,
     +		iold(13,12),isgrot(10,12),irota3,irota4,irota5,irota6
	integer idum93(2)
	parameter(x0 = 0.d0,x1 = 1.d0,x05 = .5d0,x2 = 2.d0,x4 = 4.d0,
     +		c0 = (0.E0,0.E0),c05 = (.5D0,0.E0),c1 = (1.E0,0.E0),
     +		c2 = (2.E0,0.E0),c4 = (4.E0,0.E0))
	parameter(
     +		c2ipi = (0.E+0,6.28318530717958647692528676655896D+0),
     +		cipi2 = (0.E+0,9.869604401089358618834490999876D+0),
     +		pi  = 3.14159265358979323846264338327948D+0,
     +		pi6 = 1.644934066848226436472415166646D+0,
     +		pi12 = .822467033424113218236207583323D+0,
     +		xlg2 = .6931471805599453094172321214581D+0)
	common /ffsign/isgn34,isgnal
	common /ffprec/ xloss,precx,precc,xalogm,xclogm,xalog2,xclog2,
     +		reqprc
	common /ffflag/ lwrite,ltest,l4also,ldc3c4,lmem,lwarn,ldot,
     +		nevent,ner,id,idsub,nwidth,nschem,onshel,idot
	common /ffcnst/ bf,xninv,xn2inv,xinfac,inx,isgn,iold,isgrot
	common /ffrota/ irota3,irota4,irota5,irota6
	common /ffdot/ fpij2,fpij3,fpij4,fpij5,fpij6
	common /ffdel/ fdel2,fdel3,fdel4s,fdel4,fdl3i,fdl3ij,fdl4i
	common /ffcdot/ cfpij2,cfpij3,cfpij4,cfpij5,cfpij6
	common /ffcdel/ cfdl4s
	common /ffsmug/ lsmug,lnasty,idum93,cmipj,c2sisj,ca1
