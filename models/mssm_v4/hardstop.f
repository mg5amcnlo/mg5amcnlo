cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                                                                      c
c HARD_STOP                                                            c
c                                                                      c
c subroutine to stop the program                                       c
c (only used for serious problems, please report to author)            c
c                                                                      c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine HARD_STOP

      print*, " HARD_STOP: code stopped, fatal and serious error "
      stop

      end 
