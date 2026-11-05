Examples
========

Short, self-contained scripts showing how to use MadSpace from Python. Each example uses
only the pieces of the API it needs to make its point; see the :doc:`Python API <python-api>`
for the full reference.

Interfacing with MadSpace is done through NumPy, unless the example involves machine learning
or a GPU, in which case it uses PyTorch instead. The code shown on these pages is taken
verbatim from the ``.rst`` source and run as part of the test suite, so it is guaranteed to
work against the current MadSpace build.

.. toctree::
   :maxdepth: 1

   examples/simple-mappings
   examples/cuts
   examples/diagram-mapping
   examples/integration-order
   examples/matrix-element
   examples/gpu
   examples/pdf
   examples/integrator
   examples/integrator-madnis
   examples/flow-training
