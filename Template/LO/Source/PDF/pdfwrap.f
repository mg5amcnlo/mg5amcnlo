      subroutine pdfwrap
      implicit none
C
C     INCLUDE
C
      include 'pdf.inc'
      include '../alfas.inc'

c-------------------
c     START THE CODE
c-------------------      

      nloop=2 ! NLO running unless set otherwise

C
c  MRST2002
c  1     NLO   0.1197    0.00949 
c  2     NNLO  0.1154    0.00685
C  
      if     (pdlabel .eq. 'mrs02nl') then
      asmz=0.1197d0 
      elseif     (pdlabel .eq. 'mrs02nn') then
      asmz=0.1154d0
C
c  MRST2001
c  1     alf119  central gluon, a_s       323      0.119    0.00927  
c  2     alf117  lower a_s                290      0.117    0.00953  
c  3     alf121  higher a_s               362      0.121    0.00889  
c  4     j121    better fit to jet data   353      0.121    0.00826  
C
      elseif     (pdlabel .eq. 'mrs0119') then
      asmz=0.119d0
      elseif     (pdlabel .eq. 'mrs0117') then
      asmz=0.117d0
      elseif     (pdlabel .eq. 'mrs0121') then
      asmz=0.121d0
      elseif     (pdlabel .eq. 'mrs01_j') then
      asmz=0.121d0
C
C MRS99
C  1     COR01  central gluon, a_s    300      0.1175   0.00537  C
C  2     COR02  higher gluon          300      0.1175   0.00497  C
C  3     COR03  lower gluon           300      0.1175   0.00398  C
C  4     COR04  lower a_s             229      0.1125   0.00585  C
C  5     COR05  higher a_s            383      0.1225   0.00384  C
C  6     COR06  quarks up             303.3    0.1178   0.00497  C
C  7     COR07  quarks down           290.3    0.1171   0.00593  C
C  8     COR08  strange up            300      0.1175   0.00524  C
C  9     COR09  strange down          300      0.1175   0.00524  C
C  10    C0R10  charm up              300      0.1175   0.00525  C
C  11    COR11  charm down            300      0.1175   0.00524  C
C  12    COR12  larger d/u            300      0.1175   0.00515  C
C
      elseif     (pdlabel .eq. 'mrs99_1') then
      asmz=0.1175d0
      elseif (pdlabel .eq. 'mrs99_2') then
      asmz=0.1175d0
      elseif (pdlabel .eq. 'mrs99_3') then
      asmz=0.1175d0
      elseif (pdlabel .eq. 'mrs99_4') then
      asmz=0.1125d0
      elseif (pdlabel .eq. 'mrs99_5') then
      asmz=0.1225d0
      elseif (pdlabel .eq. 'mrs99_6') then
      asmz=0.1178d0
      elseif (pdlabel .eq. 'mrs99_7') then
      asmz=0.1171d0
      elseif (pdlabel .eq. 'mrs99_8') then
      asmz=0.1175d0
      elseif (pdlabel .eq. 'mrs99_9') then
      asmz=0.1175d0
      elseif (pdlabel .eq. 'mrs9910') then
      asmz=0.1175d0
      elseif (pdlabel .eq. 'mrs9911') then
      asmz=0.1175d0
      elseif (pdlabel .eq. 'mrs9912') then
      asmz=0.1175d0
C
C MRS98
C    ft08a  central gluon, a_s  300      0.1175   0.00561  
C    ft09a  higher gluon        300      0.1175   0.00510  
C    ft11a  lower gluon         300      0.1175   0.00408  
C    ft24a  lower a_s           229      0.1125   0.00586  
C    ft23a  higher a_s          383      0.1225   0.00410  
C
      elseif (pdlabel .eq. 'mrs98z1') then
      asmz=0.1175d0
      elseif (pdlabel .eq. 'mrs98z2') then
      asmz=0.1175d0
      elseif (pdlabel .eq. 'mrs98z3') then
      asmz=0.1175d0
      elseif (pdlabel .eq. 'mrs98z4') then
      asmz=0.1125d0
      elseif (pdlabel .eq. 'mrs98z5') then
      asmz=0.1225d0
      elseif (pdlabel .eq. 'mrs98ht') then
c-- real value
      asmz=0.1170d0
c-- modified - DEBUG
      asmz=0.1175d0
      write(6,*) 'alpha_s(MZ) for mrs98ht has been modified from'
      write(6,*) 'the inherent 0.1170 to a new value of 0.1175'    
C
C  MRS98LO
C    lo05a  central gluon, a_s  174      0.1250   0.01518  
C    lo09a  higher gluon        174      0.1250   0.01616  
C    lo10a  lower gluon         174      0.1250   0.01533  
C    lo01a  lower a_s           136      0.1200   0.01652  
C    lo07a  higher a_s          216      0.1300   0.01522  
C
      elseif (pdlabel .eq. 'mrs98l1') then
      asmz=0.125d0
      nloop=1
      elseif (pdlabel .eq. 'mrs98l2') then
      asmz=0.125d0
      nloop=1
      elseif (pdlabel .eq. 'mrs98l3') then
      asmz=0.125d0
      nloop=1
      elseif (pdlabel .eq. 'mrs98l4') then
      asmz=0.120d0
      nloop=1
      elseif (pdlabel .eq. 'mrs98l5') then
      asmz=0.130d0
      nloop=1
C
C CTEQ4
C   1      CTEQ4M   Standard MSbar scheme   0.116        1.6      cteq4m.tbl
C   2      CTEQ4D   Standard DIS scheme     0.116        1.6      cteq4d.tbl
C   3      CTEQ4L   Leading Order           0.116        1.6      cteq4l.tbl
C   4      CTEQ4A1  Alpha_s series          0.110        1.6      cteq4a1.tbl
C   5      CTEQ4A2  Alpha_s series          0.113        1.6      cteq4a2.tbl
C   6      CTEQ4A3  same as CTEQ4M          0.116        1.6      cteq4m.tbl
C   7      CTEQ4A4  Alpha_s series          0.119        1.6      cteq4a4.tbl
C   8      CTEQ4A5  Alpha_s series          0.122        1.6      cteq4a5.tbl
C   9      CTEQ4HJ  High Jet                0.116        1.6      cteq4hj.tbl
C   10     CTEQ4LQ  Low Q0                  0.114        0.7      cteq4lq.tbl
C
      elseif (pdlabel .eq. 'cteq3_m') then
      asmz=0.112d0
      elseif (pdlabel .eq. 'cteq3_l') then
c---??????
      asmz=0.112d0
      nloop=1
      elseif (pdlabel .eq. 'cteq3_d') then
c---??????
      asmz=0.112d0
      elseif (pdlabel .eq. 'cteq4_m') then
      asmz=0.116d0
      elseif (pdlabel .eq. 'cteq4_d') then
      asmz=0.116d0
      elseif (pdlabel .eq. 'cteq4_l') then 
      asmz=0.132d0
      nloop=1
      elseif (pdlabel .eq. 'cteq4a1') then
      asmz=0.110d0
      elseif (pdlabel .eq. 'cteq4a2') then
      asmz=0.113d0
      elseif (pdlabel .eq. 'cteq4a3') then
      asmz=0.116d0
      elseif (pdlabel .eq. 'cteq4a4') then
      asmz=0.119d0
      elseif (pdlabel .eq. 'cteq4a5') then
      asmz=0.122d0
      elseif (pdlabel .eq. 'cteq4hj') then
      asmz=0.116d0
      elseif (pdlabel .eq. 'cteq4lq') then
      asmz=0.114d0
C
C ---------------------------------------------------------------------------
C  Iset   PDF        Description       Alpha_s(Mz)  Lam4  Lam5   Table_File
C ---------------------------------------------------------------------------
C   1    CTEQ5M   Standard MSbar scheme   0.118     326   226    cteq5m.tbl
C   2    CTEQ5D   Standard DIS scheme     0.118     326   226    cteq5d.tbl
C   3    CTEQ5L   Leading Order           0.127     192   146    cteq5l.tbl
C   4    CTEQ5HJ  Large-x gluon enhanced  0.118     326   226    cteq5hj.tbl
C   5    CTEQ5HQ  Heavy Quark             0.118     326   226    cteq5hq.tbl
C   6    CTEQ5F3  Nf=3 FixedFlavorNumber  0.106     (Lam3=395)   cteq5f3.tbl
C   7    CTEQ5F4  Nf=4 FixedFlavorNumber  0.112     309   XXX    cteq5f4.tbl
C         --------------------------------------------------------
C   8    CTEQ5M1  Improved CTEQ5M         0.118     326   226    cteq5m1.tbl
C   9    CTEQ5HQ1 Improved CTEQ5HQ        0.118     326   226    ctq5hq1.tbl
C ---------------------------------------------------------------------------
C
      elseif (pdlabel .eq. 'cteq5_m') then
      Call SetCtq5(1)
      asmz=0.118d0
      elseif (pdlabel .eq. 'cteq5_d') then
      Call SetCtq5(2)
      asmz=0.118d0
      elseif (pdlabel .eq. 'cteq5_l') then
      Call SetCtq5(3)
      asmz=0.127d0
      nloop=1
      elseif (pdlabel .eq. 'cteq5l1') then
      asmz=0.127d0
      nloop=1
      elseif (pdlabel .eq. 'cteq5hj') then
      Call SetCtq5(4)
      asmz=0.118d0
      elseif (pdlabel .eq. 'cteq5hq') then
      Call SetCtq5(5)
      asmz=0.118d0
      elseif (pdlabel .eq. 'cteq5f3') then
      Call SetCtq5(6)
      asmz=0.106d0
      elseif (pdlabel .eq. 'cteq5f4') then
      Call SetCtq5(7)
      asmz=0.112d0
      elseif (pdlabel .eq. 'cteq5m1') then
      Call SetCtq5(8)
      asmz=0.118d0
      elseif (pdlabel .eq. 'ctq5hq1') then
      Call SetCtq5(9)
      asmz=0.118d0
C
C   1    CTEQ6M   Standard MSbar scheme   0.118     326   226    cteq6m.tbl
C   2    CTEQ6D   Standard DIS scheme     0.118     326   226    cteq6d.tbl
C   3    CTEQ6L   Leading Order           0.118**   326** 226    cteq6l.tbl
C   4    CTEQ6L1  Leading Order           0.130**   215** 165    cteq6l1.tbl
C
C Note:CTEQ6L1 uses the LO running alpha_s 
C
      elseif (pdlabel .eq. 'cteq6_m') then
      asmz=0.118d0
      Call SetCtq6(1)
      elseif (pdlabel .eq. 'cteq6_d') then
      asmz=0.118d0
      Call SetCtq6(2)
      elseif (pdlabel .eq. 'cteq6_l') then
      asmz=0.118d0
      Call SetCtq6(3)
      elseif (pdlabel .eq. 'cteq6l1') then
      asmz=0.130d0
      nloop=1
      Call SetCtq6(4)


c---------------------------------------------------------------

C
C NNPDF2.3 sets
C   1      NNPDF2.3QED LO  QCD+QED  alphas(MZ) = 0.119       NNPDF23_lo_as_0119_qed_mem0.grid 
C   2      NNPDF2.3QED LO  QCD+QED  alphas(MZ) = 0.130       NNPDF23_lo_as_0130_qed_mem0.grid 
C   3      NNPDF2.3QED NLO  QCD+QED  alphas(MZ) = 0.119       NNPDF23_nlo_as_0130_qed_mc_mem0.grid  -- Positive Definite set
C
      elseif (pdlabel .eq. 'nn23lo') then
      call NNPDFDriver('NNPDF23_lo_as_0119_qed_mem0.grid')      
      call NNinitPDF(0)
      asmz=0.119d0

      elseif (pdlabel .eq. 'nn23lo1') then
      call NNPDFDriver('NNPDF23_lo_as_0130_qed_mem0.grid')      
      call NNinitPDF(0)
      asmz=0.130d0

      elseif (pdlabel .eq. 'nn23nlo') then
      call NNPDFDriver('NNPDF23nlo_as_0119_qed_mem0.grid')      
      call NNinitPDF(0)
      asmz=0.119d0

c---------------------------------------------------------------
c---------------------------------------------------------------


      else
          write(6,*) 'Unimplemented distribution= ',pdlabel
          write(6,*) 'Implemented are: ',
     .'mrs02nl,','mrs02nn,',
     .'mrs0119,','mrs0117,','mrs0121,','mrs01_j,',
     .'mrs99_1,','mrs99_2,','mrs99_3,','mrs99_4,','mrs99_5,','mrs99_6,',
     .'mrs99_7,','mrs99_8,','mrs99_9,','mrs9910,','mrs9911,','mrs9912,',
     .'mrs98z1,','mrs98z2,','mrs98z3,','mrs98z4,','mrs98z5,','mrs98ht,',
     .'mrs98l1,','mrs98l2,','mrs98l3,','mrs98l4,','mrs98l5,',
     .'cteq3_m,','cteq3_l,','cteq3_d,',
     .'cteq4_m,','cteq4_d,','cteq4_l,','cteq4a1,','cteq4a2,',
     .'cteq4a3,','cteq4a4,','cteq4a5,','cteq4hj,','cteq4lq,',
     .'cteq5_m,','cteq5_d,','cteq5_l,','cteq5hj,','cteq5hq,',
     .'cteq5f3,','cteq5f4,','cteq5m1,','ctq5hq1,','cteq5l1,',
     .'cteq6_m,','cteq6_d,','cteq6_l,','cteq6l1,',
     .'nn23lo,','nn23lo1,','nn23nlo,'
c
c     make madgraph to stop evaluating
      stop 1
c	   write(6,*) 'Setting it to default cteq6l1'
c       pdlabel='cteq6l1'
c       asmz=0.130d0
c       nloop=1
c       Call SetCtq6(4)
      endif      
      return
      end
 

