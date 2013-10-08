C     ntuBuild
C     Creation Date : Mon Oct 30 07:50:45 1995
C     User Comments
C      
C    
C     Variable description
C    
C     triggerbit1: this is a trigger bit field  
C     something: this is junk variable  
C    
      integer ngl23  
      integer ngl23_max  
      Parameter (ngl23_max = 500) 
      character*12 version_glo23 
      Logical         triggerbit1(2) 
      Integer          eventnumber
      Integer          runnumber
      Integer          No_way
      Real             missinget
      Double Precision    something
      Real            Amtrack(5,8,ngl23_max) 
      Real            v5_new(10,10,ngl23_max) 
      Real            v7_now2(ngl23_max)
      Real            missingetnomuon(ngl23_max)
      integer fence_glo23 
      COMMON/glo23/version_glo23, 
     & ngl23, 
     & triggerbit1, eventnumber, runnumber, No_way, missinget, 
     & something, Amtrack, v5_new, v7_now2, missingetnomuon, 
     & fence_glo23
C      End
