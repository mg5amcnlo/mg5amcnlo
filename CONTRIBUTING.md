# MadGraph5_aMC@NLO
## Test Suite
- Unit tests:
> `./tests/test_manager.py [--verbose=1] [--logging=INFO]`
- Acceptance tests:
> `./madgraph5/tests/test_manager.py -p A`
- Parallel tests (warning - this will take a LOOONG time!):
> `./madgraph5/tests/test_manager.py -p P`

(Please check the files in madgraph5/tests/parallel_tests to choose running of individual tests)
Run individual tests by specifying the test name.

# Create release
> `python bin/create_release.py`
