// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: S. Hageboeck (Nov 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef COMMONRANDOMNUMBERS_H_
#define COMMONRANDOMNUMBERS_H_ 1

#include <future>
#include <random>
#include <thread>
#include <vector>

namespace CommonRandomNumbers
{

  /// Create `n` random numbers using simple c++ engine.
  template<typename T>
  std::vector<T> generate( std::size_t n, std::minstd_rand::result_type seed = 1337 )
  {
    std::vector<T> result;
    result.reserve( n );

    std::minstd_rand generator( seed );
    std::uniform_real_distribution<T> distribution( 0.0, 1.0 );

    for( std::size_t i = 0; i < n; ++i )
    {
      result.push_back( distribution( generator ) );
    }

    return result;
  }

  /// Create `nBlock` blocks of random numbers.
  /// Each block uses a generator that's seeded with `seed + blockIndex`, and blocks are generated in parallel.
  template<typename T>
  std::vector<std::vector<T>> generateParallel( std::size_t nPerBlock, std::size_t nBlock, std::minstd_rand::result_type seed = 1337 )
  {
    std::vector<std::vector<T>> results( nBlock );
    std::vector<std::thread> threads;
    const auto partPerThread = nBlock / std::thread::hardware_concurrency() + ( nBlock % std::thread::hardware_concurrency() != 0 );

    auto makeBlock = [nPerBlock, nBlock, seed, &results]( std::size_t partitionBegin, std::size_t partitionEnd )
    {
      for( std::size_t partition = partitionBegin; partition < partitionEnd && partition < nBlock; ++partition )
      {
        results[partition] = generate<T>( nPerBlock, seed + partition );
      }
    };

    for( unsigned int threadId = 0; threadId < std::thread::hardware_concurrency(); ++threadId )
    {
      threads.emplace_back( makeBlock, threadId * partPerThread, ( threadId + 1 ) * partPerThread );
    }

    for( auto& thread: threads )
    {
      thread.join();
    }

    return results;
  }

  /// Starts asynchronous generation of random numbers. This uses as many threads as cores, and generates blocks of random numbers.
  /// These become available at unspecified times, but the blocks 0, 1, 2, ... are generated first.
  /// Each block is seeded with seed + blockIndex to generate stable sequences.
  /// \param[in/out] promises Vector of promise objects storing blocks of random numbers.
  /// \param[in] nPerBlock Configures number of entries generated per block.
  /// \param[in] nBlock Configures the number of blocks generated.
  /// \param[in] nThread Optional concurrency.
  /// \param[in] seed Optional seed.
  template<typename T>
  void startGenerateAsync( std::vector<std::promise<std::vector<T>>>& promises, std::size_t nPerBlock, std::size_t nBlock, unsigned int nThread = std::thread::hardware_concurrency(), std::minstd_rand::result_type seed = 1337 )
  {
    promises.resize( nBlock );
    std::vector<std::thread> threads;

    auto makeBlocks = [=, &promises]( std::size_t threadID )
    {
      for( std::size_t partition = threadID; partition < nBlock; partition += nThread )
      {
        auto values = generate<T>( nPerBlock, seed + partition );
        promises[partition].set_value( std::move( values ) );
      }
    };

    for( unsigned int threadId = 0; threadId < nThread; ++threadId )
    {
      std::thread( makeBlocks, threadId ).detach();
    }
  }

}

#endif /* COMMONRANDOMNUMBERS_H_ */
