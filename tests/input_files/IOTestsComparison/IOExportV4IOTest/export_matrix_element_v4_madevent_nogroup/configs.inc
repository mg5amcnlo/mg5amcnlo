C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA (IFOREST(I,-1,1),I=1,2)/4,3/
      DATA (SPROP(I,-1,1),I=1,1)/21/
      DATA TPRID(-1,1)/0/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/1,3/
      DATA TPRID(-1,2)/2/
      DATA (SPROP(I,-1,2),I=1,1)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/-1,4/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA (IFOREST(I,-1,3),I=1,2)/1,4/
      DATA TPRID(-1,3)/2/
      DATA (SPROP(I,-1,3),I=1,1)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/-1,3/
C     Number of configs
      DATA MAPCONFIG(0)/3/
