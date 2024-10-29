#ifndef MGONGPUTIMERMAP_H
#define MGONGPUTIMERMAP_H 1

#include <cassert>
#include <map>
#include <string>
#include <fstream>

#include "nvtx.h"
#include "timer.h"
#define TIMERTYPE std::chrono::high_resolution_clock

namespace mgOnGpu
{
  class TimerMap
  {

  public:

    TimerMap() : m_timer(), m_active(""), m_partitionTimers(), m_partitionIds() {}
    virtual ~TimerMap() {}

    // Start the timer for a specific partition (key must be a non-empty string)
    // Stop the timer for the current partition if there is one active
    float start( const std::string& key )
    {
      assert( key != "" );
      // Close the previously active partition
      float last = stop();
      // Switch to a new partition
      m_timer.Start();
      m_active = key;
      if( m_partitionTimers.find(key) == m_partitionTimers.end() )
      {
        m_partitionIds[key] = m_partitionTimers.size();
        m_partitionTimers[key] = 0;
      }
      // Open a new Cuda NVTX range
      NVTX_PUSH( key.c_str(), m_partitionIds[key] );
      // Return last duration
      return last;
    }

    // Stop the timer for the current partition if there is one active
    float stop()
    {
      // Close the previously active partition
      float last = 0;
      if ( m_active != "" )
      {
        last = m_timer.GetDuration();
        m_partitionTimers[m_active] += last;
      }
      m_active = "";
      // Close the current Cuda NVTX range
      NVTX_POP();
      // Return last duration
      return last;
    }

    // Dump the overall results
    void dump(std::ostream& ostr = std::cout, bool json=false)
    {
      // Improve key formatting
      const std::string totalKey = "TOTAL      "; // "TOTAL (ANY)"?
      //const std::string totalBut2Key = "TOTAL (n-2)";
      const std::string total123Key = "TOTAL (123)";
      const std::string total23Key = "TOTAL  (23)";
      const std::string total1Key = "TOTAL   (1)";
      const std::string total2Key = "TOTAL   (2)";
      const std::string total3Key = "TOTAL   (3)";
      size_t maxsize = 0;
      for ( auto ip : m_partitionTimers )
        maxsize = std::max( maxsize, ip.first.size() );
      maxsize = std::max( maxsize, totalKey.size() );
      // Compute the overall total
      size_t ipart = 0;
      float total = 0;
      //float totalBut2 = 0;
      float total123 = 0;
      float total23 = 0;
      float total1 = 0;
      float total2 = 0;
      float total3 = 0;
      for ( auto ip : m_partitionTimers )
      {
        total += ip.second;
        //if ( ipart != 0 && ipart+1 != m_partitionTimers.size() ) totalBut2 += ip.second;
        if ( ip.first[0] == '1' || ip.first[0] == '2' || ip.first[0] == '3' ) total123 += ip.second;
        if ( ip.first[0] == '2' || ip.first[0] == '3' ) total23 += ip.second;
        if ( ip.first[0] == '1' ) total1 += ip.second;
        if ( ip.first[0] == '2' ) total2 += ip.second;
        if ( ip.first[0] == '3' ) total3 += ip.second;
        ipart++;
      }
      // Dump individual partition timers and the overall total
      if (json) {
        std::string s1 = "\"", s2 = "\" : \"", s3 = " sec\",";
        ostr << std::setprecision(6); // set precision (default=6): affects all floats
        ostr << std::fixed; // fixed format: affects all floats
        for ( auto ip : m_partitionTimers )
          ostr << s1 << ip.first << s2 << ip.second << s3 << std::endl;
        ostr << s1 << totalKey << s2 << total << s3 << std::endl
             << s1 << total123Key << s2 << total123 << s3 << std::endl
             << s1 << total23Key << s2 << total23 << s3 << std::endl
             << s1 << total3Key << s2 << total3 << " sec \"" << std::endl;
        ostr << std::defaultfloat; // default format: affects all floats
      }
      else {
        // NB: 'setw' affects only the next field (of any type)
        ostr << std::setprecision(6); // set precision (default=6): affects all floats
        ostr << std::fixed; // fixed format: affects all floats
        for ( auto ip : m_partitionTimers )
          ostr << std::setw(maxsize) << ip.first << " : "
               << std::setw(12) << ip.second << " sec" << std::endl;
        ostr << std::setw(maxsize) << totalKey << " : "
             << std::setw(12) << total << " sec" << std::endl
             << std::setw(maxsize) << total123Key << " : "
             << std::setw(12) << total123 << " sec" << std::endl
             << std::setw(maxsize) << total23Key << " : "
             << std::setw(12) << total23 << " sec" << std::endl
             << std::setw(maxsize) << total1Key << " : "
             << std::setw(12) << total1 << " sec" << std::endl
             << std::setw(maxsize) << total2Key << " : "
             << std::setw(12) << total2 << " sec" << std::endl
             << std::setw(maxsize) << total3Key << " : "
             << std::setw(12) << total3 << " sec" << std::endl;
        ostr << std::defaultfloat; // default format: affects all floats
      }
    }

  private:

    Timer<TIMERTYPE> m_timer;
    std::string m_active;
    std::map< std::string, float > m_partitionTimers;
    std::map< std::string, uint32_t > m_partitionIds;

  };

}

#endif // MGONGPUTIMERMAP_H
