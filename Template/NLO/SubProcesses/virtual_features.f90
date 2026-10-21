! Kinematic inputs for native virtual approximations. No fit or point state.
module virtual_features
  implicit none
  private
  integer, parameter, public :: dp=kind(1d0), maxlegs=64, maxfeatures=2048
  public :: pooled_virtual_invariants,pooled_virtual_dipole_inputs
  public :: pooled_virtual_antenna_remainder,canonical_coordinates
  public :: distinct_masses,dot4,epsilon4,logratio
contains
  recursive subroutine canonical_coordinates(p,incoming,coordinates,valid)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    real(dp),intent(in) :: p(0:,:)
    logical,intent(in) :: incoming(:)
    real(dp),intent(out) :: coordinates(0:3,size(incoming))
    logical,intent(out) :: valid
    real(dp) :: cm(0:3,size(incoming)),total(0:3),beta(3),z(3),x(3),y(3),t(3),s,q,b2,g,dot,coef,norm
    integer :: leg,anchor
    valid=.false.;coordinates=0d0;total=0d0
    do leg=1,size(incoming)
       if(incoming(leg)) total=total+p(:,leg)
    enddo
    s=dot4(total,total)
    if(s<=0d0.or.total(0)<=0d0) return
    q=sqrt(s);beta=total(1:3)/total(0);b2=sum(beta**2)
    if(b2>=1d0) return
    g=total(0)/q
    do leg=1,size(incoming)
       dot=sum(p(1:3,leg)*beta);coef=g*g/(g+1d0)*dot-g*p(0,leg)
       cm(0,leg)=g*(p(0,leg)-dot);cm(1:3,leg)=p(1:3,leg)+coef*beta
    enddo
    do anchor=1,size(incoming)
       if(incoming(anchor).eqv.(count(incoming)==2)) exit
    enddo
    z=cm(1:3,anchor);norm=sqrt(sum(z*z))
    if(norm<=q*1d-14) return
    z=z/norm
    do leg=1,size(incoming)
       if(incoming(leg)) cycle
       t=cm(1:3,leg)-sum(cm(1:3,leg)*z)*z;norm=sqrt(sum(t*t))
       if(norm>q*1d-12) exit
    enddo
    if(leg>size(incoming)) return
    x=t/norm;y=cross3(z,x)
    do leg=1,size(incoming)
       if(incoming(leg)) cycle
       coordinates(:,leg)=[4d0*cm(0,leg)/q-1d0,2d0*sum(cm(1:3,leg)*x)/q, &
            2d0*sum(cm(1:3,leg)*y)/q,2d0*sum(cm(1:3,leg)*z)/q]
    enddo
    valid=all(ieee_is_finite(coordinates))
  end subroutine

  subroutine distinct_masses(mass,unique,nm)
    real(dp),intent(in) :: mass(:)
    real(dp),intent(out) :: unique(:)
    integer,intent(out) :: nm
    integer :: i,j
    real(dp) :: swap
    nm=0
    do i=1,size(mass)
       if(mass(i)<=0d0) cycle
       if(any(unique(1:nm)==mass(i))) cycle
       nm=nm+1
       unique(nm)=mass(i)
    enddo
    do i=1,nm-1
       do j=i+1,nm
          if(unique(i)<=unique(j)) cycle
          swap=unique(i);unique(i)=unique(j);unique(j)=swap
       enddo
    enddo
  end subroutine

  recursive real(dp) function real_dilog(x) result(value)
    ! Real Li_2 on the real axis. Transform to a convergent power series
    ! on [0,1/2], including the real analytic continuation across x=1.
    real(dp),intent(in) :: x
    real(dp),parameter :: pi2=9.869604401089358618834490999876d0
    real(dp) :: term,add
    integer :: k
    if(x>2d0) then
       value=pi2/3d0-0.5d0*log(x)**2-real_dilog(1d0/x)
    elseif(x>1d0) then
       value=pi2/6d0-log(x)*log(x-1d0)-real_dilog(1d0-x)
    elseif(x==1d0) then
       value=pi2/6d0
    elseif(x>0.5d0) then
       value=pi2/6d0-log(x)*log(1d0-x)-real_dilog(1d0-x)
    elseif(x<0d0) then
       value=-real_dilog(x/(x-1d0))-0.5d0*log(1d0-x)**2
    else
       value=0d0;term=x
       do k=1,100
          add=term/real(k,dp)**2
          value=value+add
          if(abs(add)<=2d-16) exit
          term=term*x
       enddo
    endif
  end function

  subroutine pooled_virtual_antenna_remainder(p,incoming,i,j,k,value,valid)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    real(dp),intent(in) :: p(0:,:)
    logical,intent(in) :: incoming(:)
    integer,intent(in) :: i,j,k
    real(dp),intent(out) :: value
    logical,intent(out) :: valid
    real(dp),parameter :: pi2=9.869604401089358618834490999876d0
    real(dp) :: total(0:3),triple(0:3),den,u,v,shat
    integer :: leg
    value=0d0;valid=.false.;total=0d0
    do leg=1,size(incoming)
       if(incoming(leg)) total=total+p(:,leg)
    enddo
    shat=dot4(total,total)
    triple=merge(-1d0,1d0,incoming(i))*p(:,i)+merge(-1d0,1d0,incoming(j))*p(:,j)+ &
         merge(-1d0,1d0,incoming(k))*p(:,k)
    den=dot4(triple,triple)
    if(shat<=0d0.or.abs(den)<=1d-12*shat) return
    u=2d0*dot4(p(:,i),p(:,j))*merge(1d0,-1d0,incoming(i).eqv.incoming(j))/den
    v=2d0*dot4(p(:,j),p(:,k))*merge(1d0,-1d0,incoming(j).eqv.incoming(k))/den
    if(u==0d0.or.v==0d0) return
    value=real_dilog(1d0-u)+real_dilog(1d0-v)+log(abs(u))*log(abs(v))-pi2/6d0
    if(u<0d0.and.v<0d0) value=value-pi2
    valid=ieee_is_finite(value)
    if(.not.valid) value=0d0
  end subroutine

  subroutine pooled_virtual_dipole_inputs(p,mass,incoming,qes2,correlation,x,valid)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    ! Colour-space one-loop logarithms and the massive cusp/Coulomb shapes.
    ! Correlations retain the Born provider's normalization and sign. Their
    ! coefficients are fitted, not assumed universal for finite remainders.
    real(dp),intent(in) :: p(0:,:),mass(:),qes2,correlation(:,:)
    logical,intent(in) :: incoming(:)
    real(dp),intent(out) :: x(4)
    logical,intent(out) :: valid
    real(dp),parameter :: pi=3.1415926535897932384626433832795d0
    real(dp) :: s,l,cusp,v,a(0:3),b(0:3),scale,dot,mm,delta,error,c
    integer :: i,j,n
    x=0d0;valid=.false.;n=size(mass)
    if(size(correlation,1)/=n.or.size(correlation,2)/=n.or.size(p,2)/=n.or.size(incoming)/=n) return
    if(.not.all(ieee_is_finite(correlation)).or.qes2<=0d0) return
    do i=1,n-1
       do j=i+1,n
          c=correlation(i,j)
          if(c==0d0) cycle
          s=2d0*dot4(p(:,i),p(:,j))
          if(s<=0d0.or..not.ieee_is_finite(s)) return
          l=logratio(s,qes2)
          cusp=l
          if(mass(i)>0d0) cusp=cusp-0.5d0*logratio(mass(i)**2,qes2)
          if(mass(j)>0d0) cusp=cusp-0.5d0*logratio(mass(j)**2,qes2)
          if(mass(i)>0d0.and.mass(j)>0d0) then
             scale=max(maxval(abs(p(:,i))),maxval(abs(p(:,j))),mass(i),mass(j))
             a=p(:,i)/scale;b=p(:,j)/scale
             dot=dot4(a,b);mm=(mass(i)/scale)*(mass(j)/scale)
             delta=dot-mm
             error=64d0*epsilon(1d0)*(abs(a(0)*b(0))+sum(abs(a(1:3)*b(1:3)))+mm)
             if(delta<=error.or.mm<=0d0) return
             v=sqrt((delta/dot)*((dot+mm)/dot))
             if(v<1d-4) then
                cusp=1d0+v*v/3d0+v**4/5d0
             else
                cusp=(log(dot)+log(1d0+v)-log(mm))/v
             endif
             x(3)=x(3)+c*cusp*l
             if(.not.incoming(i).and..not.incoming(j)) x(4)=x(4)+c*pi*pi/v
          endif
          x(1)=x(1)+c*cusp
          x(2)=x(2)+c*(l*l-merge(pi*pi,0d0,incoming(i).eqv.incoming(j)))
       enddo
    enddo
    valid=all(ieee_is_finite(x))
  end subroutine

  subroutine pooled_virtual_invariants(p,mass,incoming,pdg,mur2,cat,x,valid,qes2)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    real(dp),intent(in) :: p(0:,:),mass(:),mur2
    logical,intent(in) :: incoming(:)
    integer,intent(in) :: pdg(:)
    integer,intent(out) :: cat
    real(dp),allocatable,intent(out) :: x(:)
    logical,intent(out) :: valid
    real(dp),intent(in) :: qes2
    real(dp) :: total(0:3),shat,unique(maxlegs),a(0:3),b(0:3),c(0:3),d(0:3),den
    real(dp) :: scale,energy,spatial,dot,mm,delta,error
    integer :: i,j,k,n,nm,ia,ib,nmassive
    valid=.false.
    cat=9
    if(.not.ieee_is_finite(qes2).or.qes2<=0d0) return
    n=size(pdg)
    if(n<3.or.n>maxlegs.or.size(p,1)/=4.or.size(p,2)/=n.or.size(mass)/=n.or.size(incoming)/=n) return
    if(.not.all(ieee_is_finite(p)).or..not.all(ieee_is_finite(mass)).or..not.ieee_is_finite(mur2)) return
    if(any(mass<0d0).or.mur2<=0d0.or.count(incoming)<1.or.count(incoming)>2) return
    total=0d0
    do i=1,n
       if(incoming(i)) total=total+p(:,i)
    enddo
    shat=dot4(total,total)
    if(shat<=0d0) return
    call distinct_masses(mass,unique,nm)
    nmassive=count(mass>0d0.and..not.incoming)
    k=7+n*(n-1)/2+nm+(n-2)*(n-3)/2+nmassive*(nmassive-1)/2
    if(k>maxfeatures) return
    allocate(x(k))
    x(1)=logratio(shat,mur2)
    k=1
    do i=1,n-1
       do j=i+1,n
          k=k+1
          x(k)=logratio(abs(2d0*dot4(p(:,i),p(:,j))),shat)
       enddo
    enddo
    do i=1,nm
       k=k+1
       x(k)=logratio(shat,unique(i)**2)
    enddo
    x(k+1)=logratio(shat,qes2)
    k=k+1
    do i=1,n-1
       if(incoming(i).or.mass(i)<=0d0) cycle
       do j=i+1,n
          if(incoming(j).or.mass(j)<=0d0) cycle
          scale=max(maxval(abs(p(:,i))),maxval(abs(p(:,j))),mass(i),mass(j))
          a=p(:,i)/scale;b=p(:,j)/scale
          energy=a(0)*b(0);spatial=sum(a(1:3)*b(1:3))
          dot=energy-spatial;mm=(mass(i)/scale)*(mass(j)/scale)
          delta=dot-mm
          error=64d0*epsilon(1d0)*(abs(energy)+abs(spatial)+mm)
          if(delta<=error.or.dot<=0d0) return
          k=k+1
          x(k)=1d0/sqrt((delta/dot)*((dot+mm)/dot))
       enddo
    enddo
    x(k+1:k+4)=0d0
    k=k+4
    k=k+1;x(k)=log(shat)
    ia=1;ib=2
    if(count(incoming)==2) then
       do ia=1,n
          if(incoming(ia)) exit
       enddo
       do ib=ia+1,n
          if(incoming(ib)) exit
       enddo
    else
       do ia=1,n
          if(incoming(ia)) exit
       enddo
       do ib=1,n
          if(.not.incoming(ib)) exit
       enddo
    endif
    ! Dimensionless vectors avoid fourth powers of the physical scale.
    a=p(:,ia)/sqrt(shat);b=p(:,ib)/sqrt(shat)
    do i=1,n-1
       if(i==ia.or.i==ib) cycle
       c=p(:,i)/sqrt(shat)
       do j=i+1,n
          if(j==ia.or.j==ib) cycle
          d=p(:,j)/sqrt(shat)
          den=sqrt(max(0d0,16d0*dot4(a,c)*dot4(b,c)*dot4(a,d)*dot4(b,d)))
          k=k+1;x(k)=0d0
          if(den>1d-15) x(k)=max(-1d0,min(1d0,2d0*epsilon4(a,b,c,d)/den))
       enddo
    enddo
    valid=all(ieee_is_finite(x))
  end subroutine

  pure recursive function cross3(a,b) result(c)
    real(dp),intent(in) :: a(3),b(3)
    real(dp) :: c(3)
    c=[a(2)*b(3)-a(3)*b(2),a(3)*b(1)-a(1)*b(3),a(1)*b(2)-a(2)*b(1)]
  end function

  pure recursive real(dp) function epsilon4(a,b,c,d) result(value)
    real(dp),intent(in) :: a(0:3),b(0:3),c(0:3),d(0:3)
    real(dp) :: cd(3),bd(3),bc(3)
    cd=cross3(c(1:3),d(1:3))
    bd=cross3(b(1:3),d(1:3))
    bc=cross3(b(1:3),c(1:3))
    value=a(0)*sum(b(1:3)*cd)-b(0)*sum(a(1:3)*cd) &
         +c(0)*sum(a(1:3)*bd)-d(0)*sum(a(1:3)*bc)
  end function

  real(dp) function dot4(a,b) result(value)
    real(dp),intent(in) :: a(0:3),b(0:3)
    value=a(0)*b(0)-dot_product(a(1:3),b(1:3))
  end function

  real(dp) function logratio(a,b) result(value)
    real(dp),intent(in) :: a,b
    value=log(max(a/b,1d-15))
  end function

end module virtual_features
