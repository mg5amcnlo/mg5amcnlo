      implicit none
      integer i,j
      real*8 mcmass(21)


      double precision stlow1, stlow2,stupp1,stupp2, md
      double precision min_py_sudakov
      integer id, itype,iseed
      double precision temp1,temp2,tmp1PDF,tmp2PDF


      double precision fx(-7:7),x,xmu,nnfx(-6:7)

c      call NNPDFDriver('NNPDF23nlo_as_0119_qed_mem0.grid')
      call NNPDFDriver('NNPDF23_lo_as_0130_qed_mem0.grid')

c
      do i=1,21
        mcmass(i)=0.d0
      enddo
      include 'MCmasses_PYTHIA8.inc'

      stupp2=200d0
      stupp1=20d0
      md=2000d0
      id=1
      itype=1
      iseed=59822
      min_py_sudakov=0.0000001d0
      x=1d0
      stlow1=stupp1
      stlow2=stupp1
c
      call pythia_init(mcmass)
c$$$      call dire_init(mcmass)
      do j=1,10
         x=x/2d0
c$$$c$$$      iseed=59822
c$$$      call pythia_get_no_emission_prob(temp1, stupp, stlow, md, id,
c$$$     $     itype, iseed, min_py_sudakov)
c$$$      call pythia_get_no_emission_prob_x(temp1, stupp1, stlow1, md, id,
c$$$     $     itype, iseed, min_py_sudakov,x)
      temp1=1d0
      call pythia_get_no_emission_prob_x(temp2, stupp2, stlow2, md, id,
     $     itype, iseed, min_py_sudakov,x)
c$$$      call dire_get_no_emission_prob(temp1, stupp,
c$$$     #     stlow, md, id, itype, iseed, min_py_sudakov)
c$$$      iseed=59822
c$$$         call dire_get_no_emission_prob_x(temp1, stupp, stlow1, md, id,
c$$$     $        itype, iseed, min_py_sudakov,x)
c$$$         call dire_get_no_emission_prob_x(temp2, stupp2, stlow2, md, id,
c$$$     $        itype, iseed, min_py_sudakov,x)


! PDFs
c$$$      xmu=stlow1
      xmu=stupp2
      do i=-7,7
         fx(i)=0d0
      enddo
      call NNevolvePDF(x,xmu,nnfx)
      do i=-5,5
         fx(i)=nnfx(i)/x
      enddo
      fx(7)=nnfx(7)/x

      tmp1PDF=fx(mod(id,21))


c$$$      xmu=stlow2
      xmu=stupp1
      do i=-7,7
         fx(i)=0d0
      enddo
      call NNevolvePDF(x,xmu,nnfx)
      do i=-5,5
         fx(i)=nnfx(i)/x
      enddo
      fx(7)=nnfx(7)/x
      
      tmp2PDF=fx(mod(id,21))

      write (*,*) 'id=', id, 'x=', x, 'Pi=',
     $   temp2/temp1,tmp2PDF/tmp1PDF,
     $  'Pi/PDFratio=', (temp2/temp1)/(tmp2PDF/tmp1PDF),
     $   temp2,temp1,(temp2/temp1)/(tmp2PDF/tmp1PDF)**2

      enddo

      return
      end
