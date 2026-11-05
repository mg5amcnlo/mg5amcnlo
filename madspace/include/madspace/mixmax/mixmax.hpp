/*
 * mixmax.hpp
 *
 * C++ implementation of the MIXMAX random number generator.
 *
 *  Copyright (2008-2023) by Konstantin Savvidy.
 *
 *  Free to use, academic or commercial. Do not redistribute without permission.
 *
 *  G.K.Savvidy and N.G.Ter-Arutyunian,
 *  On the Monte Carlo simulation of physical systems,
 *  J.Comput.Phy 97, 566 (1991);
 *  Preprint EPI-865-16-86, Yerevan, Jan. 1986
 *
 *  K.Savvidy
 *  The MIXMAX random number generator
 *  Comp. Phy Commun. 196 (2015), pp 161-165
 *  http://dx.doi.org/10.1016/j.cpc.2015.06.003
 *
 *  K.Savvidy and G.Savvidy
 *  Spectrum and Entropy of C-system MIXMAX random number generator
 *  Chaos, Solitons & Fractals, Volume 91, (2016) pp. 33-38
 *  http://dx.doi.org/10.1016/j.chao2016.05.003
 *
 */

#ifndef __MIXMAX_H
#define __MIXMAX_H

#include <cstdint>

#if (defined(__CUDACC__) || defined (__CUDA_ARCH__) || defined(__HIPCC__))
#define _dev __host__ __device__
#else
#define _dev
#include <array>
#include <iostream>
#endif

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
#define _constmem __constant__
#else
#define _constmem
#endif

class alignas(128) mixmax_engine
{
static const int N = 17;
static constexpr int BITS=61;
static constexpr uint64_t M61=2305843009213693951ULL;
static constexpr uint64_t MERSBASE=M61;
static constexpr double INV_MERSBASE=(0.43368086899420177360298E-18);


    static constexpr long long int SPECIAL   = 0;
    static constexpr long long int SPECIALMUL= 36;
    // Note the potential for confusion...

private: // state
    uint64_t V[N] = {0}; // init with all zero's - not the same as seeding with seed=0!
    uint64_t sumtot = {0};
    int counter = N;

public:
using result_type = uint64_t;                                 // should it be double?
    static constexpr uint64_t min() {return 0;}
    static constexpr uint64_t max() {return M61;}

    _dev inline uint64_t get_next() ;
    _dev inline double flat();


    _dev inline mixmax_engine(); // Constructor, no seeds
    _dev inline mixmax_engine(uint32_t clusterID, uint32_t machineID, uint32_t runID, uint32_t  streamID );       // Constructor with four 32-bit seeds
    _dev inline mixmax_engine(uint64_t seedval){seed_uniquestream( 0, 0, (uint32_t)(seedval>>32), (uint32_t)seedval );}; // Constructor with one 64bit seed
    _dev inline void seed(uint64_t seedval){seed_uniquestream( 0, 0, (uint32_t)(seedval>>32), (uint32_t)seedval );} // seed with one 64-bit seed
    _dev inline void seed_uniquestream( uint32_t clusterID, uint32_t machineID, uint32_t runID, uint32_t  streamID );

    _dev inline mixmax_engine Branch();
    _dev inline void BranchInplace();


    _dev inline mixmax_engine& operator=(const mixmax_engine& other );
    _dev inline operator double() { return flat(); }
    _dev inline operator float() { return float( flat() ); }
    _dev inline operator unsigned int() { return static_cast<unsigned int>(get_next()); }

    _dev inline uint64_t operator()()
    {
        return get_next();
    }

    // --- substream seeding building blocks, used by the GPU seeder ---
    static constexpr int state_size = N;
    static constexpr uint64_t mod_prime = M61;
    // Y <- (sum_j coef[j] A^j) Y, i.e. apply the polynomial-in-A `coef` (state_size terms).
    _dev static inline void apply_poly(uint64_t* Y, const uint64_t* coef);
    // Y <- P_r(A) Y: apply the skip operator for seed bit r (row r of the skip matrix).
    // The operators for different r commute, so bits may be applied in any order/grouping.
    _dev static inline void apply_skip_bit(uint64_t* Y, int r);
    // field ops mod mod_prime: acc + a*b, a + b, and Y <- A*Y (returns new sum(Y))
    _dev static inline uint64_t mod_mul_add(uint64_t acc, uint64_t a, uint64_t b) {
        return fmodmulM61(acc, a, b);
    }
    _dev static inline uint64_t mod_add(uint64_t a, uint64_t b) { return modadd(a, b); }
    _dev static inline uint64_t advance_state(uint64_t* Y, uint64_t sum) {
        return iterate_raw_vec(Y, sum);
    }
    // raw N-vector access, for in-place skipping
    _dev uint64_t* state_data() { return V; }
    _dev const uint64_t* state_data() const { return V; }
    // recompute sumtot from V and arm the engine (counter = 1), as seed_uniquestream does
    _dev void finalize_skipped_state() {
        uint64_t s = 0;
        for (int i = 0; i < N; i++) { s = modadd(s, V[i]); }
        sumtot = s;
        counter = 1;
    }
    // vielbein with the two high seed words applied: `high0` at effective bit positions
    // [64,96), `high1` at [96,128). This part of the seed is the global run seed and does
    // not change within a run, so callers cache the result.
    _dev static void run_seed_prefix(uint64_t* out, uint32_t high0, uint32_t high1) {
        for (int i = 0; i < N; i++) { out[i] = 0; }
        out[0] = 1;
        for (int b = 0; b < 32; b++) { if ((high0 >> b) & 1u) { apply_skip_bit(out, 64 + b); } }
        for (int b = 0; b < 32; b++) { if ((high1 >> b) & 1u) { apply_skip_bit(out, 96 + b); } }
    }
    // seed from `prefix` (a state with some seed bits pre-applied), then apply the low
    // seed words: `low0` at effective bit positions [0,32), `low1` at [32,64). Equivalent
    // to seed_uniquestream when prefix carries exactly the bits [64,128).
    _dev void seed_from_state(const uint64_t* prefix, uint32_t low0, uint32_t low1) {
        for (int i = 0; i < N; i++) { V[i] = prefix[i]; }
        for (int b = 0; b < 32; b++) { if ((low0 >> b) & 1u) { apply_skip_bit(V, b); } }
        for (int b = 0; b < 32; b++) { if ((low1 >> b) & 1u) { apply_skip_bit(V, 32 + b); } }
        finalize_skipped_state();
    }

private:
    _dev static inline uint64_t MOD_MULSPEC(uint64_t k);
    _dev inline void seed_vielbein(); // seeds with the unit vector {1,0,0,...}
    _dev static inline uint64_t iterate_raw_vec(uint64_t* Y, uint64_t sumtotOld);
    _dev inline uint64_t apply_bigskip(uint32_t clusterID, uint32_t machineID, uint32_t runID, uint32_t  streamID );
    _dev static inline uint64_t modadd(uint64_t foo, uint64_t bar){return MOD_MERSENNE(foo+bar);};
    _dev static inline uint64_t fmodmulM61(uint64_t cum, uint64_t s, uint64_t a);
    _dev static inline uint64_t MOD_MERSENNE(uint64_t k) {return ((((k)) & MERSBASE) + (((k)) >> BITS) );}
    _dev static inline uint64_t MULWU(uint64_t k) {return (( (k)<<(SPECIALMUL) & M61) | ( (k) >> (BITS-SPECIALMUL))  );}
    void inline print_state();

public:

#if (!defined(__CUDACC__) && !defined (__CUDA_ARCH__))
  template <class CharT, class Traits>
  friend std::basic_ostream<CharT, Traits> &
  operator<<(std::basic_ostream<CharT, Traits> &ost, const mixmax_engine &me) {
    // save the state of RNG to stream
    int j;
    ost << "mixmax state, file version 1.0\n";
    ost << "N=" << N << "; V[N]={";
    for (j = 0; (j < (N - 1)); j++) {
      ost << (std::uint64_t)me.V[j] << ", ";
    }
    ost << me.V[N - 1];
    ost << "}; ";
    ost << "counter=" << (std::uint64_t)me.counter << "; ";
    ost << "sumtot=" << (std::uint64_t)me.sumtot << ";\n";
    ost.flush();
    return ost;
  }

  template <class CharT, class Traits>
  friend std::basic_istream<CharT, Traits> &
  operator>>(std::basic_istream<CharT, Traits> &in, mixmax_engine &me) {
    // will set std::ios::failbit and throw an exception if format is not right
    std::array<std::uint64_t, N> vec;
    std::uint64_t sum = 0, sumtmp = 0, counter = 0;
    in.ignore(150, '='); // eat chars up to N=
    CharT xxxchar;
    try {
      std::basic_string<CharT> line;
      if (std::getline(in, line)) {
        std::basic_istringstream<CharT> iss(line);
        std::basic_string<CharT> token;
        getline(iss, token, xxxchar = (';'));
        int i = std::stoi(token);
        if (i != N) {
          std::cerr
              << "ERROR: Wrong dimension of the MIXMAX RNG state on input, "
              << i << " vs " << N << "\n";
        }                    // else{std::cerr <<"Dim ok "<< i << "\n";}
        iss.ignore(15, '{'); // eat chars up to V[N]={
        for (int j = 0; j < N - 1; j++) {
          std::getline(iss, token, xxxchar = ',');
          if (!iss.fail()) {
            vec[j] = std::stoull(token);
            sum = me.modadd(sum, vec[j]);
          } // std::cerr <<vec[j]<<" + ";}
        }
        std::getline(iss, token, xxxchar = '}');
        if (!iss.fail()) {
          vec[N - 1] = std::stoull(token);
          sum = me.modadd(sum, vec[N - 1]);
        } else {
          std::cerr << "ERROR: last elem empty\n";
        }
        iss.ignore(10, '=');
        std::getline(iss, token, xxxchar = ';');
        counter = std::stoi(token);
        iss.ignore(10, '=');
        std::getline(iss, token, xxxchar = ';');
        sumtmp = std::stoull(token);
      } else {
        std::cerr << "ERROR: nothing to read\n";
      }
    } catch (const std::exception &e) {
      std::cerr << "ERROR: Exception caught in MIXMAX RNG while reading state: "
                << e.what() << "'\n"; // "\nfrom string='" << token <<
      in.setstate(std::ios::failbit);
      throw;
    }
    if (sum == sumtmp && counter > 0 && counter < N) {
      for (int i=0; i < N; i++){ me.V[i] = vec[i];};
      me.counter = counter;
      me.sumtot = sumtmp;
    } else {
      std::cerr << "ERROR: incorrect checksum or out of range counter while "
                   "reading  MIXMAX RNG state.\n"
                << counter << " " << sumtmp << " vs " << sum << "\n";
      in.setstate(std::ios::failbit);
    }
    return in;
  }
#endif // #indef __CUDA_ARCH__

};


namespace mixmax_detail {
_constmem static const uint64_t skipMat17[128][17] =
#include "mixmax_skip_N17.c"
;
}


_dev mixmax_engine  ::mixmax_engine()
// constructor, with no params, fast and seeds with a unit vector
{
    seed_vielbein();
}

_dev mixmax_engine  ::mixmax_engine(uint32_t clusterID, uint32_t machineID, uint32_t runID, uint32_t  streamID)
// constructor, no need to allocate, just seed
{
    seed_uniquestream( clusterID,  machineID,  runID,  streamID );
}


_dev uint64_t mixmax_engine  ::iterate_raw_vec(uint64_t* Y, uint64_t sumtotOld)
{
    // operates with a raw vector, uses known sum of elements of Y
    uint64_t  tempP, tempV;
    Y[0] = ( tempV = sumtotOld);
    uint64_t sumtot = Y[0], ovflow = 0; // will keep a running sum of all new elements
    tempP = 0;              // will keep a partial sum of all old elements
    for (int i=1; i<N; ++i){
            uint64_t tempPO = MULWU(tempP);
            tempP = modadd(tempP, Y[i]);
            tempV = MOD_MERSENNE(tempV+tempP+tempPO); // new Y[i] = old Y[i] + old partial * m
    Y[i] = tempV;
    sumtot += tempV; if (sumtot < tempV) {++ovflow;}
    }
    return MOD_MERSENNE(MOD_MERSENNE(sumtot) + (ovflow <<3 ));
}

_dev uint64_t mixmax_engine  ::get_next()
{
   int i = counter;

   if ((i<=(N-1)) )
   {
     ++counter;
     return V[i];
   }
   else
   {
     sumtot = iterate_raw_vec(V, sumtot);
     counter=2;
     return V[1];
   }
}

_dev double mixmax_engine  ::flat()             // Returns a random double with all 53 bits random, in the range (0,1]
{    /* cast to signed int trick suggested by Andrzej Gorlich     */
    int64_t Z=(int64_t)get_next();
    return INV_MERSBASE*static_cast<double>(Z);
}

_dev void mixmax_engine  ::seed_vielbein()
{
    for (int i=0; i < N; i++){
        V[i] = 0;
    }
    V[0] = 1;
    counter = N;  // set the counter to N if iteration should happen right away
    sumtot = 1;
}


_dev void mixmax_engine  ::seed_uniquestream( uint32_t clusterID, uint32_t machineID, uint32_t runID, uint32_t  streamID ){
    seed_vielbein();
    //print_state();
    sumtot = apply_bigskip(clusterID,  machineID,  runID,   streamID );
    //std::cerr << "seeding mixmax with: " << clusterID << ", " << machineID << ", " <<   runID <<  ", " <<  streamID << "\n";
    counter = 1;
}


_dev uint64_t mixmax_engine  ::apply_bigskip( uint32_t clusterID, uint32_t machineID, uint32_t runID, uint32_t  streamID ){
    /*
     makes a derived state vector, Vout, from the mother state vector Vin
     by skipping a large number of steps, determined by the given seeding ID's

     it is mathematically guaranteed that the substreams derived in this way from the SAME (!!!) Vin will not collide provided
     1) at least one bit of ID is different
     2) less than 10^100 numbers are drawn from the stream
     (this is good enough : a single CPU will not exceed this in the lifetime of the universe, 10^19 sec,
     even if it had a clock cycle of Planch time, 10^44 Hz )

     Caution: never apply this to a derived vector, just choose some mother vector Vin, for example the unit vector by seed_vielbein(X,0),
     and use it in all your runs, just change runID to get completely nonoverlapping streams of random numbers on a different day.

     clusterID and machineID are provided for the benefit of large organizations who wish to ensure that a simulation
     which is running in parallel on a large number of  clusters and machines will have non-colliding source of random number

     did i repeat it enough times? the non-collision guarantee is absolute, not probabilistic

     */


    uint32_t IDvec[4] = {streamID, runID, machineID, clusterID};
    uint64_t Y[N];

    for (int i=0; i<N; i++) { Y[i] = V[i]; }
    for (int IDindex=0; IDindex<4; IDindex++) { // go from lower order to higher order ID
        uint32_t id=IDvec[IDindex];
        int r = IDindex*32;
        while (id){
            if (id & 1) { apply_skip_bit(Y, r); }
            id = (id >> 1); r++; // bring up the r-th bit in the ID
        }
    }
    uint64_t insumtot=0;
    for (int i=0; i<N; i++){ V[i] = Y[i]; insumtot = modadd( insumtot, Y[i]); } ;  // returns sumtot, and copy the vector over to Vout
    return (insumtot) ;
}

// Y <- (sum_j coef[j] A^j) Y. Every operator in the algebra generated by A is such a
// polynomial (Cayley-Hamilton), so this applies both individual skip operators and any
// product of them, given in the same 17-coefficient basis.
_dev void mixmax_engine  ::apply_poly(uint64_t* Y, const uint64_t* coef){
    uint64_t cum[N];
    uint64_t insumtot=0;
    for (int i=0; i<N; i++){ cum[i] = 0; insumtot = modadd( insumtot, Y[i]); }
    for (int j=0; j<N; j++){                  // j is lag, enumerates terms of the poly
        for (int i=0; i<N; i++){
            cum[i] = fmodmulM61( cum[i], coef[j], Y[i] );
        }
        insumtot = iterate_raw_vec(Y, insumtot);
    }
    for (int i=0; i<N; i++){ Y[i] = cum[i]; }
}

// Y <- P_r(A) Y, the skip operator for seed bit r (row r of the skip matrix).
_dev void mixmax_engine  ::apply_skip_bit(uint64_t* Y, int r){
    apply_poly(Y, mixmax_detail::skipMat17[r]);
}



_dev inline uint64_t mixmax_engine  ::fmodmulM61(uint64_t cum, uint64_t s, uint64_t a)
{
    uint64_t o,ph,pl,ah,al;
    uint64_t MASK32=0xFFFFFFFFULL;
    o=(s)*a;
    ph = ((s)>>32);
    pl = (s) & MASK32;
    ah = a>>32;
    al = a & MASK32;
    o = (o & M61) + ((ph*ah)<<3) + ((ah*pl+al*ph + ((al*pl)>>32))>>29) ;
    o += cum;
    o = (o & M61) + ((o>>61));
    return o;
}

#if  !defined(__CUDA_ARCH__)
#include <cstdio>
 void mixmax_engine  ::print_state(){
    int j;
    fprintf(stdout, "mixmax state, file version 1.0\n" );
    fprintf(stdout, "N=%u; V[N]={", N );
    for (j=0; (j< (N-1) ); j++) {
        fprintf(stdout, "%llu, ", (unsigned long long)V[j] );
    }
    fprintf(stdout, "%llu", (unsigned long long)V[N-1] );
    fprintf(stdout, "}; " );
    fprintf(stdout, "counter=%u; ", counter );
    fprintf(stdout, "sumtot=%llu;\n", (unsigned long long)sumtot );
}
#endif

_dev mixmax_engine  mixmax_engine  ::Branch(){
    sumtot = iterate_raw_vec(V, sumtot); counter = N-1;
    mixmax_engine tmp=*this;
    tmp.BranchInplace();
    return tmp;
}

_dev mixmax_engine  & mixmax_engine  ::operator=(const mixmax_engine& other ){
    for (int i=0; i<N; i++) { V[i] = other.V[i];}
    sumtot = other.sumtot;
    counter = other.counter;
    return *this;
}

_dev void mixmax_engine  ::BranchInplace(){
    // Dont forget to iterate the mother, when branching the daughter, or else will have collisions!
    // a 64-bit LCG from Knuth line 26, is used to mangle a vector component
    constexpr uint64_t MULT64=6364136223846793005ULL;
    uint64_t tmp=V[1];
    V[1] *= MULT64; V[1] &= MERSBASE;
    sumtot = modadd( sumtot , V[1] - tmp + MERSBASE);
    sumtot = iterate_raw_vec(V, sumtot);
    counter = 1;
}

#endif      // __MIXMAX_H
