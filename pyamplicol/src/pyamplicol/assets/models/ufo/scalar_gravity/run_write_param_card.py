# type: ignore
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from six.moves import range

import sys
import os
import importlib

if '__main__' == __name__:
    dir_path = os.path.join(os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(dir_path, os.path.pardir))
    pw = importlib.import_module('%s.%s' % (
        os.path.basename(dir_path), 'write_param_card'))
    del sys.path[0]

    pw.ParamCardWriter('./param_card.dat', generic=True)
    print('write ./param_card.dat')
