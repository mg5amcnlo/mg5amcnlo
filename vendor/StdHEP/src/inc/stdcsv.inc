C -------------------------------------------------------------
C use this to pass information when parsing the csv mass width file
C
      character cline*132	! input line
      character ispin*3,gpar*1,jspin*4,ppar*1,cpar*1,antip*1
      character cchg*4,rank*1,status*1
      character name*17,quarklist*15
      common/stdline/cline,ispin,gpar,jspin,ppar,cpar,antip,cchg,
     &               rank,status,name,quarklist

      integer pdgid,chg
      double precision mval,mpos,mneg,wval,wpos,wneg
      common/stdmw/mval,mpos,mneg,wval,wpos,wneg,chg,pdgid

      save /stdline/,/stdmw/
C
C -------------------------------------------------------------
