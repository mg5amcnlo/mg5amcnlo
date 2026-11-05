     subroutine write_good_hel(stream_id)
        implicit none
        include 'maxamps.inc'
        integer stream_id
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB, MAXFLAVPERPROC)
        INTEGER NTRY(MAXFLAVPERPROC)
        common/BLOCK_GOODHEL/NTRY,GOODHEL
        write(stream_id,*) GOODHEL
        return
        end


        subroutine read_good_hel(stream_id)
        implicit none
        include 'genps.inc'
        include 'maxamps.inc'
        integer stream_id
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB, MAXFLAVPERPROC)
        INTEGER NTRY(MAXFLAVPERPROC)
        common/BLOCK_GOODHEL/NTRY,GOODHEL
        read(stream_id,*) GOODHEL
        NTRY(:) = MAXTRIES + 1
        return
        end

        subroutine init_good_hel()
        implicit none
        include 'maxamps.inc'
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB, MAXFLAVPERPROC)
        INTEGER NTRY(MAXFLAVPERPROC)
        INTEGER I

        GOODHEL(:,:) = .false.
        NTRY(:) = 0
        end
        
        integer function get_maxsproc()
        implicit none
        get_maxsproc = 1
        return 
        end
        
