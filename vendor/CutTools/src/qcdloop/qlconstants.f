      double precision pi,pisq,pisqo6
      parameter(pi=3.14159265358979d0,pisq=pi*pi,pisqo6=pisq/6d0)

      double precision zip,half,one,two,three,four,eight
      parameter(zip=0d0,half=0.5d0,one=1d0,two=2d0)
      parameter(three=3d0,four=4d0,eight=8d0)

      double complex im,impi,czip,chalf,cone,ctwo,c2ipi
      parameter(im=(0d0,1d0),impi=(0d0,3.14159265358979d0),
     . czip=(0d0,0d0),chalf=(0.5d0,0d0),cone=(1d0,0d0),ctwo=(2d0,0d0),
     . c2ipi=2d0*impi)

      DOUBLE COMPLEX nan
      parameter (nan = (1D123, 1D123))


