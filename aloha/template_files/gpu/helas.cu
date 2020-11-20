  //--------------------------------------------------------------------------

  __device__
  inline const fptype& pIparIp4Ievt( const fptype* momenta1d, // input: momenta as AOSOA[npagM][npar][4][neppM]
                                     const int ipar,
                                     const int ip4,
                                     const int ievt )
  {
    //mapping for the various scheme AOS, OSA, ...

    using mgOnGpu::np4;
    using mgOnGpu::npar;
    const int neppM = mgOnGpu::neppM; // ASA layout: constant at compile-time
    fptype (*momenta)[npar][np4][neppM] = (fptype (*)[npar][np4][neppM]) momenta1d; // cast to multiD array pointer (AOSOA)
    const int ipagM = ievt/neppM; // #eventpage in this iteration
    const int ieppM = ievt%neppM; // #event in the current eventpage in this iteration
    //return allmomenta[ipagM*npar*np4*neppM + ipar*neppM*np4 + ip4*neppM + ieppM]; // AOSOA[ipagM][ipar][ip4][ieppM]
    return momenta[ipagM][ipar][ip4][ieppM];
  }

  //--------------------------------------------------------------------------

__device__ void ixxxxx(const fptype* allmomenta, fptype fmass, int nhel, int nsf,
                       cxtype fi[6],
#ifndef __CUDACC__
                 const int ievt,
#endif
                 const int ipar )          // input: particle# out of npar

) {
    mgDebug( 0, __FUNCTION__ );
#ifndef __CUDACC__
    // ** START LOOP ON IEVT **
    //for (int ievt = 0; ievt < nevt; ++ievt)
#endif
    {
#ifdef __CUDACC__
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread) in grid
      //printf( "ixzxxxM0: ievt=%d threadId=%d\n", ievt, threadIdx.x );
#endif

      const fptype& pvec0 = pIparIp4Ievt( allmomenta, ipar, 0, ievt );
      const fptype& pvec1 = pIparIp4Ievt( allmomenta, ipar, 1, ievt );
      const fptype& pvec2 = pIparIp4Ievt( allmomenta, ipar, 2, ievt );
      const fptype& pvec3 = pIparIp4Ievt( allmomenta, ipar, 3, ievt );

  cxtype chi[2];
  fptype sf[2], sfomega[2], omega[2], pp, pp3, sqp0p3, sqm[2];
  int ip, im, nh;

  fptype p[4] = {0, pvec0, pvec1, pvec2};
  p[0] = sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]+fmass*fmass);
  fi[0] = cxtype(-p[0] * nsf, -p[3] * nsf);
  fi[1] = cxtype(-p[1] * nsf, -p[2] * nsf);
  nh = nhel * nsf;
  if (fmass != 0.0) {
    pp = min(p[0], sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]));
    if (pp == 0.0) {
      sqm[0] = sqrt(std::abs(fmass));
      sqm[1] = (fmass < 0) ? -abs(sqm[0]) : abs(sqm[0]);
      ip = (1 + nh) / 2;
      im = (1 - nh) / 2;
      fi[2] = ip * sqm[ip];
      fi[3] = im * nsf * sqm[ip];
      fi[4] = ip * nsf * sqm[im];
      fi[5] = im * sqm[im];
    } else {
      sf[0] = (1 + nsf + (1 - nsf) * nh) * 0.5;
      sf[1] = (1 + nsf - (1 - nsf) * nh) * 0.5;
      omega[0] = sqrt(p[0] + pp);
      omega[1] = fmass / omega[0];
      ip = (1 + nh) / 2;
      im = (1 - nh) / 2;
      sfomega[0] = sf[0] * omega[ip];
      sfomega[1] = sf[1] * omega[im];
      pp3 = max(pp + p[3], 0.0);
      chi[0] = cxtype(sqrt(pp3 * 0.5 / pp), 0);
      if (pp3 == 0.0) {
        chi[1] = cxtype(-nh, 0);
      } else {
        chi[1] =
            cxtype(nh * p[1], p[2]) / sqrt(2.0 * pp * pp3);
      }
      fi[2] = sfomega[0] * chi[im];
      fi[3] = sfomega[0] * chi[ip];
      fi[4] = sfomega[1] * chi[im];
      fi[5] = sfomega[1] * chi[ip];
    }
  } else {
    if (p[1] == 0.0 and p[2] == 0.0 and p[3] < 0.0) {
      sqp0p3 = 0.0;
    } else {
      sqp0p3 = sqrt(max(p[0] + p[3], 0.0)) * nsf;
    }
    chi[0] = cxtype(sqp0p3, 0.0);
    if (sqp0p3 == 0.0) {
      chi[1] = cxtype(-nhel * sqrt(2.0 * p[0]), 0.0);
    } else {
      chi[1] = cxtype(nh * p[1], p[2]) / sqp0p3;
    }
    if (nh == 1) {
      fi[2] = cxtype(0.0, 0.0);
      fi[3] = cxtype(0.0, 0.0);
      fi[4] = chi[0];
      fi[5] = chi[1];
    } else {
      fi[2] = chi[1];
      fi[3] = chi[0];
      fi[4] = cxtype(0.0, 0.0);
      fi[5] = cxtype(0.0, 0.0);
    }
  }
    // ** END LOOP ON IEVT **
    mgDebug( 1, __FUNCTION__ );
  return;
}

__device__ void ipzxxx(const fptype pvec[3], int nhel, int nsf, cxtype fi[6])
{
  // ASSUMPTION FMASS == 0
  // PX = PY = 0
  // E = P3 (E>0)

  fi[0] = cxtype (-pvec[2] * nsf, -pvec[2] * nsf);
  fi[1] = cxtype (0.,0.);
  int nh = nhel * nsf;

  cxtype sqp0p3 = cxtype(sqrt(2.* pvec[2]) * nsf, 0.);

  fi[2]=fi[1];
  fi[3]=(nh== 1)*fi[1]   + (nh==-1)*sqp0p3;
  fi[4]=(nh== 1)*sqp0p3 + (nh==-1)*fi[1];
  fi[5]=fi[1];
}

__device__ void imzxxx(const fptype pvec[3], int nhel, int nsf, cxtype fi[6])
{
  // ASSUMPTION FMASS == 0
  // PX = PY = 0
  // E = -P3 (E>0)
  //printf("p3 %f", pvec[2]);
  fi[0] = cxtype (pvec[2] * nsf, -pvec[2] * nsf);
  fi[1] = cxtype (0., 0.);
  int nh = nhel * nsf;
  cxtype  chi = cxtype (-nhel * sqrt(-2.0 * pvec[2]), 0.0);

  fi[2]=(nh== 1)*fi[1]   + (nh==-1)*chi;
  fi[3]=fi[1];
  fi[4]=fi[1];
  fi[5]=(nh== 1)*chi   + (nh==-1)*fi[1];
}

__device__ void ixzxxx(const fptype pvec[3],  int nhel, int nsf, cxtype fi[6])
{
  // ASSUMPTIONS: FMASS == 0
  // Px and Py are not zero

  //cxtype chi[2];
  //fptype sf[2], sfomega[2], omega[2], pp, pp3, sqp0p3, sqm[2];
  //int ip, im, nh;
  float p[4] = {0, (float) pvec[0], (float) pvec[1], (float) pvec[2]};
  p[0] = sqrtf(p[3] * p[3] + p[1] * p[1] + p[2] * p[2]);

  fi[0] = cxtype (-p[0] * nsf, -pvec[2] * nsf);
  fi[1] = cxtype (-pvec[0] * nsf, -pvec[1] * nsf);
  int nh = nhel * nsf;

  float sqp0p3 = sqrtf(p[0] + p[3]) * nsf;
  cxtype chi0 = cxtype (sqp0p3, 0.0);
  cxtype chi1 = cxtype (nh * p[1]/sqp0p3, p[2]/sqp0p3);
  cxtype CZERO = cxtype(0.,0.);

  fi[2]=(nh== 1)*CZERO   + (nh==-1)*chi1;
  fi[3]=(nh== 1)*CZERO   + (nh==-1)*chi0;
  fi[4]=(nh== 1)*chi0    + (nh==-1)*CZERO;
  fi[5]=(nh== 1)*chi1    + (nh==-1)*CZERO;
  return;
}


/*
__device__ void txxxxx(fptype pvec[3], fptype tmass, int nhel, int nst,
                       cxtype tc[18]) {
  cxtype ft[6][4], ep[4], em[4], e0[4];
  fptype pt, pt2, pp, pzpt, emp, sqh, sqs;
  int i, j;
  
  fptype p[4] = {0, pvec[0], pvec[1], pvec[2]};
  p[0] = sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]+tmass*tmass);
  sqh = sqrt(0.5);
  sqs = sqrt(0.5 / 3);

  pt2 = p[1] * p[1] + p[2] * p[2];
  pp = min(p[0], sqrt(pt2 + p[3] * p[3]));
  pt = min(pp, sqrt(pt2));

  ft[4][0] = cxtype(p[0] * nst, p[3] * nst);
  ft[5][0] = cxtype(p[1] * nst, p[2] * nst);

  // construct eps+
  if (nhel >= 0) {
    if (pp == 0) {
      ep[0] = cxtype(0, 0);
      ep[1] = cxtype(-sqh, 0);
      ep[2] = cxtype(0, nst * sqh);
      ep[3] = cxtype(0, 0);
    } else {
      ep[0] = cxtype(0, 0);
      ep[3] = cxtype(pt / pp * sqh, 0);

      if (pt != 0) {
        pzpt = p[3] / (pp * pt) * sqh;
        ep[1] = cxtype(-p[1] * pzpt, -nst * p[2] / pt * sqh);
        ep[2] = cxtype(-p[2] * pzpt, nst * p[1] / pt * sqh);
      } else {
        ep[1] = cxtype(-sqh, 0);
        ep[2] =
            cxtype(0, nst * (p[3] < 0) ? -abs(sqh) : abs(sqh));
      }
    }
  }

  // construct eps-
  if (nhel <= 0) {
    if (pp == 0) {
      em[0] = cxtype(0, 0);
      em[1] = cxtype(sqh, 0);
      em[2] = cxtype(0, nst * sqh);
      em[3] = cxtype(0, 0);
    } else {
      em[0] = cxtype(0, 0);
      em[3] = cxtype(-pt / pp * sqh, 0);

      if (pt != 0) {
        pzpt = -p[3] / (pp * pt) * sqh;
        em[1] = cxtype(-p[1] * pzpt, -nst * p[2] / pt * sqh);
        em[2] = cxtype(-p[2] * pzpt, nst * p[1] / pt * sqh);
      } else {
        em[1] = cxtype(sqh, 0);
        em[2] =
            cxtype(0, nst * (p[3] < 0) ? -abs(sqh) : abs(sqh));
      }
    }
  }

  // construct eps0
  if (std::labs(nhel) <= 1) {
    if (pp == 0) {
      e0[0] = cxtype(0, 0);
      e0[1] = cxtype(0, 0);
      e0[2] = cxtype(0, 0);
      e0[3] = cxtype(1, 0);
    } else {
      emp = p[0] / (tmass * pp);
      e0[0] = cxtype(pp / tmass, 0);
      e0[3] = cxtype(p[3] * emp, 0);

      if (pt != 0) {
        e0[1] = cxtype(p[1] * emp, 0);
        e0[2] = cxtype(p[2] * emp, 0);
      } else {
        e0[1] = cxtype(0, 0);
        e0[2] = cxtype(0, 0);
      }
    }
  }

  if (nhel == 2) {
    for (j = 0; j < 4; j++) {
      for (i = 0; i < 4; i++)
        ft[i][j] = ep[i] * ep[j];
    }
  } else if (nhel == -2) {
    for (j = 0; j < 4; j++) {
      for (i = 0; i < 4; i++)
        ft[i][j] = em[i] * em[j];
    }
  } else if (tmass == 0) {
    for (j = 0; j < 4; j++) {
      for (i = 0; i < 4; i++)
        ft[i][j] = 0;
    }
  } else if (tmass != 0) {
    if (nhel == 1) {
      for (j = 0; j < 4; j++) {
        for (i = 0; i < 4; i++)
          ft[i][j] = sqh * (ep[i] * e0[j] + e0[i] * ep[j]);
      }
    } else if (nhel == 0) {
      for (j = 0; j < 4; j++) {
        for (i = 0; i < 4; i++)
          ft[i][j] =
              sqs * (ep[i] * em[j] + em[i] * ep[j] + 2.0 * e0[i] * e0[j]);
      }
    } else if (nhel == -1) {
      for (j = 0; j < 4; j++) {
        for (i = 0; i < 4; i++)
          ft[i][j] = sqh * (em[i] * e0[j] + e0[i] * em[j]);
      }
    } else {
      // sr fixme // std::cerr << "Invalid helicity in txxxxx.\n";
      // sr fixme // std::exit(1);
    }
  }

  tc[0] = ft[4][0];
  tc[1] = ft[5][0];

  for (j = 0; j < 4; j++) {
    for (i = 0; i < 4; i++)
      tc[j * 4 + i + 2] = ft[j][i];
  }
}

*/

__device__ void vxxxxx(const fptype pvec[3], fptype vmass, int nhel, int nsv,
                       cxtype vc[6]) {
  fptype hel, hel0, pt, pt2, pp, pzpt, emp, sqh;
  int nsvahl;

  fptype p[4] = {0, pvec[0], pvec[1], pvec[2]};
  p[0] = sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]+vmass*vmass);

  sqh = sqrt(0.5);
  hel = fptype(nhel);
  nsvahl = nsv * std::abs(hel);
  pt2 = (p[1] * p[1]) + (p[2] * p[2]);
  pp = min(p[0], sqrt(pt2 + (p[3] * p[3])));
  pt = min(pp, sqrt(pt2));
  vc[0] = cxtype(p[0] * nsv, p[3] * nsv);
  vc[1] = cxtype(p[1] * nsv, p[2] * nsv);
  if (vmass != 0.0) {
    hel0 = 1.0 - std::abs(hel);
    if (pp == 0.0) {
      vc[2] = cxtype(0.0, 0.0);
      vc[3] = cxtype(-hel * sqh, 0.0);
      vc[4] = cxtype(0.0, nsvahl * sqh);
      vc[5] = cxtype(hel0, 0.0);
    } else {
      emp = p[0] / (vmass * pp);
      vc[2] = cxtype(hel0 * pp / vmass, 0.0);
      vc[5] =
          cxtype(hel0 * p[3] * emp + hel * pt / pp * sqh, 0.0);
      if (pt != 0.0) {
        pzpt = p[3] / (pp * pt) * sqh * hel;
        vc[3] = cxtype(hel0 * p[1] * emp - p[1] * pzpt,
                                        -nsvahl * p[2] / pt * sqh);
        vc[4] = cxtype(hel0 * p[2] * emp - p[2] * pzpt,
                                        nsvahl * p[1] / pt * sqh);
      } else {
        vc[3] = cxtype(-hel * sqh, 0.0);
        vc[4] = cxtype(0.0, nsvahl * (p[3] < 0) ? -abs(sqh)
                                                                 : abs(sqh));
      }
    }
  } else {
    pp = p[0];
    pt = sqrt((p[1] * p[1]) + (p[2] * p[2]));
    vc[2] = cxtype(0.0, 0.0);
    vc[5] = cxtype(hel * pt / pp * sqh, 0.0);
    if (pt != 0.0) {
      pzpt = p[3] / (pp * pt) * sqh * hel;
      vc[3] = cxtype(-p[1] * pzpt, -nsv * p[2] / pt * sqh);
      vc[4] = cxtype(-p[2] * pzpt, nsv * p[1] / pt * sqh);
    } else {
      vc[3] = cxtype(-hel * sqh, 0.0);
      vc[4] =
          cxtype(0.0, nsv * (p[3] < 0) ? -abs(sqh) : abs(sqh));
    }
  }
  return;
}

__device__ void sxxxxx(const fptype pvec[3], fptype smass, int nss, cxtype sc[3]) {
  fptype p[4] = {0, pvec[0], pvec[1], pvec[2]};
  p[0] = sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]+smass*smass);
  sc[2] = cxtype(1.00, 0.00);
  sc[0] = cxtype(p[0] * nss, p[3] * nss);
  sc[1] = cxtype(p[1] * nss, p[2] * nss);
  return;
}

__device__ void oxxxxx(const fptype pvec[3], fptype fmass, int nhel, int nsf,
                       cxtype fo[6]) {
  cxtype chi[2];
  fptype sf[2], sfomeg[2], omega[2], pp, pp3, sqp0p3, sqm[2];
  int nh, ip, im;
  
  fptype p[4] = {0, pvec[0], pvec[1], pvec[2]};
  p[0] = sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]+fmass*fmass);

  fo[0] = cxtype(p[0] * nsf, p[3] * nsf);
  fo[1] = cxtype(p[1] * nsf, p[2] * nsf);
  nh = nhel * nsf;
  if (fmass != 0.000) {
    pp = min(p[0], sqrt((p[1] * p[1]) + (p[2] * p[2]) + (p[3] * p[3])));
    if (pp == 0.000) {
      sqm[0] = sqrt(std::abs(fmass));
      sqm[1] = (fmass < 0) ? -abs(sqm[0]) : abs(sqm[0]);
      ip = -((1 - nh) / 2) * nhel;
      im = (1 + nh) / 2 * nhel;
      fo[2] = im * sqm[std::abs(ip)];
      fo[3] = ip * nsf * sqm[std::abs(ip)];
      fo[4] = im * nsf * sqm[std::abs(im)];
      fo[5] = ip * sqm[std::abs(im)];
    } else {
      pp = min(p[0], sqrt((p[1] * p[1]) + (p[2] * p[2]) + (p[3] * p[3])));
      sf[0] = fptype(1 + nsf + (1 - nsf) * nh) * 0.5;
      sf[1] = fptype(1 + nsf - (1 - nsf) * nh) * 0.5;
      omega[0] = sqrt(p[0] + pp);
      omega[1] = fmass / omega[0];
      ip = (1 + nh) / 2;
      im = (1 - nh) / 2;
      sfomeg[0] = sf[0] * omega[ip];
      sfomeg[1] = sf[1] * omega[im];
      pp3 = max(pp + p[3], 0.00);
      chi[0] = cxtype(sqrt(pp3 * 0.5 / pp), 0.00);
      if (pp3 == 0.00) {
        chi[1] = cxtype(-nh, 0.00);
      } else {
        chi[1] =
            cxtype(nh * p[1], -p[2]) / sqrt(2.0 * pp * pp3);
      }
      fo[2] = sfomeg[1] * chi[im];
      fo[3] = sfomeg[1] * chi[ip];
      fo[4] = sfomeg[0] * chi[im];
      fo[5] = sfomeg[0] * chi[ip];
    }
  } else {
    if ((p[1] == 0.00) and (p[2] == 0.00) and (p[3] < 0.00)) {
      sqp0p3 = 0.00;
    } else {
      sqp0p3 = sqrt(max(p[0] + p[3], 0.00)) * nsf;
    }
    chi[0] = cxtype(sqp0p3, 0.00);
    if (sqp0p3 == 0.000) {
      chi[1] = cxtype(-nhel, 0.00) * sqrt(2.0 * p[0]);
    } else {
      chi[1] = cxtype(nh * p[1], -p[2]) / sqp0p3;
    }
    if (nh == 1) {
      fo[2] = chi[0];
      fo[3] = chi[1];
      fo[4] = cxtype(0.00, 0.00);
      fo[5] = cxtype(0.00, 0.00);
    } else {
      fo[2] = cxtype(0.00, 0.00);
      fo[3] = cxtype(0.00, 0.00);
      fo[4] = chi[1];
      fo[5] = chi[0];
    }
  }
  return;
}

__device__ void opzxxx(const fptype pvec[3], int nhel, int nsf, cxtype fo[6])
{
  // ASSUMPTIONS FMASS =0
  // PX = PY =0
  // E = PZ
  fo[0] = cxtype (pvec[2] * nsf, pvec[2] * nsf);
  fo[1] = cxtype (0., 0.);
  int nh = nhel * nsf;

  cxtype CSQP0P3 = cxtype (sqrt(2.* pvec[2]) * nsf, 0.00);

    fo[2]=(nh== 1)*CSQP0P3 + (nh==-1)*fo[1];
    fo[3]=fo[1];
    fo[4]=fo[1];
    fo[5]=(nh== 1)*fo[1]   + (nh==-1)*CSQP0P3;
}

__device__ void omzxxx(const fptype pvec[3], int nhel, int nsf, cxtype fo[6])
{
  // ASSUMPTIONS FMASS =0
  // PX = PY =0
  // E = -PZ (E>0)

  fo[0] = cxtype (-pvec[2] * nsf, pvec[2] * nsf);
  fo[1] = cxtype (0., 0.);
  int nh = nhel * nsf;
  cxtype chi = cxtype (-nhel, 0.00) * sqrt(-2.0 * pvec[2]);

  fo[2]=(nh== 1)*fo[1] + (nh==-1)*fo[1];
  fo[3]=(nh== 1)*chi + (nh==-1)*fo[1];;
  fo[4]=(nh== 1)*fo[1] + (nh==-1)*chi;
  fo[5]=(nh== 1)*fo[1] + (nh==-1)*chi;

  return;
}

__device__ void oxzxxx(const fptype pvec[3], int nhel, int nsf, cxtype fo[6])
{
  // ASSUMPTIONS FMASS =0
  // PT > 0

  float p[4] = {0, (float) pvec[0], (float) pvec[1], (float) pvec[2]};
  p[0] = sqrtf(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]);

  fo[0] = cxtype (p[0] * nsf, pvec[2] * nsf);
  fo[1] = cxtype (pvec[0] * nsf, pvec[1] * nsf);
  int nh = nhel * nsf;

  float sqp0p3 = sqrtf(p[0] + p[3]) * nsf;
  cxtype chi0 = cxtype (sqp0p3, 0.00);
  cxtype chi1 = cxtype (nh * p[1]/sqp0p3, -p[2]/sqp0p3);
  cxtype zero = cxtype (0.00, 0.00);

  fo[2]=(nh== 1)*chi0 + (nh==-1)*zero;
  fo[3]=(nh== 1)*chi1 + (nh==-1)*zero;
  fo[4]=(nh== 1)*zero + (nh==-1)*chi1;
  fo[5]=(nh== 1)*zero + (nh==-1)*chi0;

  return;
}