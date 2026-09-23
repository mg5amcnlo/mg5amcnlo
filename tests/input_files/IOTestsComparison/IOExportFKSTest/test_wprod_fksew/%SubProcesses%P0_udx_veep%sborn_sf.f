      subroutine sborn_sf(p,m,n,ans)
      use mc_born_types
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      double precision p(0:3,nexternal-1),ans
      type(BornRequest),save::request
      type(BornResult),save::result
      integer m,n
      logical need_color_links,need_charge_links
      common/c_need_links/need_color_links,need_charge_links
      double precision amp_split_soft(amp_split_size)
      common/to_amp_split_soft/amp_split_soft
      request%%colour=need_color_links
      request%%charge=need_charge_links
      request%%m=m
      request%%n=n
      call mc_born_local(p,request,result)
      ans=result%%correlation
      amp_split_soft=result%%soft
      amp_split_cnt=result%%split_counterterms
      end
