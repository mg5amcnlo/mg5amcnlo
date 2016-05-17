#! /usr/bin/env python

import sys

fifo = open(sys.argv[1],'r')
while raw_input("Continue?") in ['y','Y','']:
    evt = [fifo.readline()]
    while not evt[-1].startswith('E'):
        evt.append(fifo.readline())
    print 'Read the following event:'
    print '-------------------------'
    print ''.join(evt)
