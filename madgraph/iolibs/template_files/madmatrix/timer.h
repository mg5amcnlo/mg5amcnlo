// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: S. Roiser (Feb 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: O. Mattelaer, S. Roiser, A. Valassi (2020-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MGONGPUTIMER_H
#define MGONGPUTIMER_H 1

#include <chrono>
#include <iostream>

namespace mgOnGpu
{

  /*
  high_resolution_clock
  steady_clock
  system_clock

  from https://www.modernescpp.com/index.php/the-three-clocks
  and https://codereview.stackexchange.com/questions/196245/extremely-simple-timer-class-in-c
  */

  template<typename T>
  class Timer
  {
  public:
    Timer()
      : m_StartTime( T::now() ) {}
    virtual ~Timer() {}
    void Start();
    float GetDuration();
    void Info();
  private:
    typedef typename T::time_point TTP;
    TTP m_StartTime;
  };

  template<typename T>
  void
  Timer<T>::Start()
  {
    m_StartTime = T::now();
  }

  template<typename T>
  float
  Timer<T>::GetDuration()
  {
    std::chrono::duration<float> duration = T::now() - m_StartTime;
    return duration.count();
  }

  template<typename T>
  void
  Timer<T>::Info()
  {
    typedef typename T::period TPER;
    typedef typename std::ratio_multiply<TPER, std::kilo> MilliSec;
    typedef typename std::ratio_multiply<TPER, std::mega> MicroSec;
    std::cout << std::boolalpha << std::endl;
    std::cout << "clock info: " << std::endl;
    std::cout << "  is steady: " << T::is_steady << std::endl;
    std::cout << "  precision: " << TPER::num << "/" << TPER::den << " second " << std::endl;
    std::cout << std::fixed;
    std::cout << "             " << static_cast<double>( MilliSec::num ) / MilliSec::den << " milliseconds " << std::endl;
    std::cout << "             " << static_cast<double>( MicroSec::num ) / MicroSec::den << " microseconds " << std::endl;
    std::cout << std::endl;
  }

}
#endif // MGONGPUTIMER_H
