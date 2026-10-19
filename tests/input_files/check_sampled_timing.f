      program check_sampled_timing
      use test_clock
      use mint_timing_fixture
      implicit none
      include 'timing_variables.inc'
      double precision elapsed(timing_ncategories),last_clock,total
      common/timing_clocks/elapsed,last_clock
      integer(kind=8) points,samples,sampler_state
      common/timing_counts/points,samples,sampler_state
      integer i,block_samples,positions(100),expected(100),nseed,mode
      integer,allocatable::seed(:)
      real randoms(10000),value
      character*32 arg
      integer callback_points,sampled_calls,closing_calls
      logical callback_sampled
      common/test_callback_counts/callback_points,sampled_calls,
     $     closing_calls,callback_sampled
      double precision x(ndimmax),fun
      external fun
      call get_command_argument(1,arg)
      call timing_reset()
      if(timing_sampled)stop 1
      if(arg.eq.'sampling')then
         positions=0
         expected=0
         call random_seed(size=nseed)
         allocate(seed(nseed))
         seed=7429
         call random_seed(put=seed)
         call random_number(randoms)
         call random_seed(put=seed)
         block_samples=0
         do i=1,10000
            call timing_begin_point()
            call random_number(value)
            if(value.ne.randoms(i))stop 'Physics RNG changed'
            if(timing_sampled)then
               block_samples=block_samples+1
               positions((i-1)/100+1)=mod(i-1,100)+1
            endif
            call timing_enter(timing_born)
            call timing_leave(timing_born)
            call timing_end_point()
            if(timing_sampled)stop 'Timing leaked across points'
            if(mod(i,100).eq.0)then
               if(block_samples.ne.1)stop 'Not one sample per block'
               block_samples=0
            endif
         enddo
         if(points.ne.10000_8.or.samples.ne.100_8)stop 'Wrong counts'
         if(reads.ne.400)stop 'Unsampled points read clocks'
         if(count(positions.eq.positions(1)).gt.10)
     $        stop 'Sampling aliases a periodic point pattern'
         expected=positions
         call timing_reset()
         do i=1,10000
            call timing_begin_point()
            if(timing_sampled)
     $           positions((i-1)/100+1)=mod(i-1,100)+1
            call timing_end_point()
         enddo
         if(any(positions.ne.expected))stop 'Sampler not reproducible'
      elseif(arg.eq.'exclusive')then
         call select_point()
         reads=0
         clock=1d0
         call timing_enter(timing_phase_space)
         clock=3d0
         call timing_enter(timing_phase_space)
         clock=5d0
         call timing_enter(timing_born)
         clock=11d0
         call timing_leave(timing_born)
         clock=13d0
         call timing_leave(timing_phase_space)
         clock=17d0
         call timing_leave(timing_phase_space)
         clock=23d0
         call timing_end_point()
         if(elapsed(timing_born).ne.6d0)stop 'Born interval'
         if(elapsed(timing_phase_space).ne.10d0)stop 'Nested interval'
         if(elapsed(timing_weights).ne.7d0)stop 'Default interval'
         if(sum(elapsed).ne.23d0)stop 'Categories overlap'
         if(reads.ne.5)stop 'Repeated category reads clocks'
      elseif(arg.eq.'empty')then
         call timing_enter(-1)
         call timing_leave(-1)
         call timing_end_point()
         if(reads.ne.0)stop 'Idle timers read clocks'
         call timing_report(42.5d0)
      elseif(arg.eq.'estimate'.or.arg.eq.'clock_cost')then
         if(arg.eq.'clock_cost')read_cost=0.25d0
         call select_point()
         clock=clock+1d0
         call timing_enter(timing_born)
         clock=clock+2d0
         call timing_leave(timing_born)
         clock=clock+1d0
         call timing_end_point()
         call timing_report(42.5d0)
      elseif(arg.eq.'mismatch')then
         call select_point()
         call timing_enter(timing_born)
         call timing_leave(timing_real)
         stop 'Failed to reject mismatched timing category'
      elseif(arg.eq.'unclosed')then
         call select_point()
         call timing_enter(timing_born)
         call timing_end_point()
         stop 'Failed to reject unclosed timing category'
      elseif(arg.eq.'mint')then
         do mode=0,2
            call timing_reset()
            callback_points=0
            sampled_calls=0
            closing_calls=0
            callback_sampled=.false.
            fixed_order=mode.eq.2
            imode=mod(mode,2)
            ifold=1
            do i=1,1000
               x=0.25d0
               call compute_integrand(fun,x,1d0)
               if(timing_sampled)stop 'MINT left point active'
            enddo
            if(points.ne.1000_8.or.samples.ne.10_8)stop 'MINT counts'
            if(callback_points.ne.1000)stop 'Lost integrand callbacks'
            if(sampled_calls.ne.10)stop 'Wrong sampled callbacks'
            if(fixed_order)then
               if(closing_calls.ne.0)stop 'Unexpected FO callback'
            else
               if(closing_calls.ne.1000)stop 'Lost closing callbacks'
            endif
         enddo
      else
         stop 'Unknown timing check'
      endif
      write(*,*)'PASS sampled timing ',trim(arg)
      end

      subroutine select_point()
      use test_clock
      implicit none
      include 'timing_variables.inc'
      clock=0d0
      do
         call timing_begin_point()
         if(timing_sampled)return
         call timing_end_point()
      enddo
      end

      double precision function fun(x,vol,ifirst,f1)
      use test_clock
      implicit none
      include 'timing_variables.inc'
      integer ifirst,callback_points,sampled_calls,closing_calls
      logical callback_sampled
      common/test_callback_counts/callback_points,sampled_calls,
     $     closing_calls,callback_sampled
      double precision x(2),vol,f1(2)
      if(ifirst.eq.0)then
         callback_points=callback_points+1
         callback_sampled=timing_sampled
         if(timing_sampled)sampled_calls=sampled_calls+1
      elseif(ifirst.eq.2)then
         closing_calls=closing_calls+1
         if(timing_sampled.neqv.callback_sampled)
     $        stop 'Sampling changed within a MINT point'
      else
         stop 'Unexpected integrand callback'
      endif
      if(timing_sampled)call timing_enter(timing_born)
      clock=clock+2d0
      f1=x*vol
      fun=sum(f1)
      if(timing_sampled)call timing_leave(timing_born)
      end
