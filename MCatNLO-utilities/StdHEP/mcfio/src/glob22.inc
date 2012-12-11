C     ntuBuild
C     Creation Date : Sat Oct 28 14:17:14 1995
C     User Comments
C      
C    
C     Variable description
C    
C     triggerbit1: this is a trigger bit field  
C     something: this is junk variable  
C    
      integer nglob_max  
      Parameter (nglob_max = 12) 
      structure /glob_v_struct/ 
            Byte            v5_pad(2)
            Character      *30 missingetphi
            Integer*2       V6_new(80)
            Real            v5_new(5,6)
            Real            v7_now2 
            Real            missingetnomuon 
            Double Precision   v7_now 
            Double Precision   v7_now_last 
      end structure
C       
      structure /glob_struct/ 
          character*12 version
           integer nglob  
            Byte            eventnumber(8)
            Logical         triggerbit1 
            Integer         runnumber 
            Integer         No_way 
            Real            missinget 
            Double Precision   something 
            integer fence 
            record /glob_v_struct/ glob_var(nglob_max)
      end structure
C         
C         
       record /glob_struct/ glob_c 
C         
      COMMON/glob/glob_c 
C      End
