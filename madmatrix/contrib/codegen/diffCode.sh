#!/bin/bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Sep 2021) for the MG5aMC CUDACPP plugin.
# Further modified by: A. Valassi (2021-2024).
# Integrated with the MadGraph7 project in Feb 2026.

diff --no-dereference -x '*log.txt' -x 'nsight_logs' -x '*.o' -x '*.o.*' -x '*.a' -x '*.exe' -x 'lib' -x 'build.*' -x '.build.*' -x '*~' -x 'include' $* 
