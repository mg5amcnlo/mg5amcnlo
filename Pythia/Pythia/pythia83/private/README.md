# Overview

This directory contains a collection of scripts and directories for
internal Pythia collaboration use only. Most of these are used when
preparing a release.

* `checkXML.py` - checks if the XML documenation pages
  are well formed.

* `converXML.cc` - converts the XML into the HTML manual.

* `docker` - contains documentation and scripts used in building
  Docker images for testing and the Python interface. Further
  documentation is provided in the
  [Docker documentation](docker/README.md).
  
* `doxygen.cfg` - the Doxygen configuration for Pythia.

* `doxygen.sh` - build Doxygen for Pythia.

* `indexMains.py` - builds the examples search index
  (`htmldoc/Examples.js`), (`htmldoc/ExampleKeywords.html`), the
  example pages (`htmldoc/examples/`).

* `indexXML.py` - builds `htmldoc/Index.js`, the search index for the
  HTML manual.

* `release` - creates a tarball for the release; *use this with caution*.

* `test` - contains the regression testing suite. Further
  documentation is provided in the
  [testing documentation](tests/README.md).
  
* `trimfile` - checks that files are formatted following the Pythia
  conventions.

# Release

This is partially based on an original checklist from Torbjörn. The
checklist for a release can be separated into two stages. The first
stage ensures that the code is running as expected, while the second
stage updates documentation and formatting. Note that the first stage
is primarily superseded by CI/CD checks within Gitlab and is no longer
necessary. The second stage has been consolidated in the the script
`release.sh` and can be run as follows.
```bash
./release.sh <final three version digits, e.g. 303> 
```
Note that this generates a number of logs which **must** be inspected
before proceeding. Most notably, the `trimfile` output needs to be
checked.

## Code Checks

1. Generate the Python interface. Note, if any changes are made to
   `include` after this step, the interface needs to be generated
   again.
   ```bash
   cd plugins/python
   ./generate
   ```

2. Check the tests for any regressions from the last commit. See the
   [tests documetation](tests/README.md) for further details. Ensure
   the output is as expected.
   ```bash
   cd private/tests
   ./testgit --branch=master --ref=<reference> --first=<first> --last=<last>
   ```
   Running `./testgit --help` gives documentation on available options.

3. Check all the examples run. This can be done using the Docker
   testing image. Beginning in the top level of the Pythia release, do
   the following.
   ```bash
   # Start the Docker container (the cap flag allows use of gdb).
   docker run -i -t -v "$PWD:$PWD" -w $PWD -u `id -u` --cap-add=SYS_PTRACE --rm pythia8/dev:test bash --norc
   
   # Configure Pythia with all packages.
   export PATH=$PATH:/hep/fastjet3/bin:/hep/hepmc3/bin:/hep/lhapdf5/bin:/hep/rivet/bin
   ./configure --with-evtgen=/hep/evtgen --with-fastjet3 --with-hepmc2=/hep/hepmc2 --with-hepmc3 --with-lhapdf5 --with-lhapdf6=/hep/lhapdf6 --with-powheg-bin=/hep/powheg/ --with-rivet --with-root --with-gzip --with-python --with-mg5mes --with-openmp
   
   # Build Pythia.
   make -j6
   
   # Run the examples.
   cd examples
   ./runmains --threads=6
   ```
   Running `./runmains --help` gives documentation on available options.

## Documentation.

1. Update version number in `Pythia.h`, `Pythia.cc`, `Version.xml` and
   `UpdateHistory.xml`.
  
2. Update authors list in `AUTHORS`, `Pythia.cc` and `Frontpage.xml`.

3. Include the most current worksheet.

4. Update the date in `Version.xml` and `UpdateHistory.xml`.

5. Check the `XML` is formatted correctly by running python `checkXML.py`
   from within `private`, and correct any mismatch.
   ```bash
   cd private
   ./checkXML.py
   ```
   
6. Run `private/trimfile -d` for the directories `src`,
  `include/Pythia8`, `include/Pythia8Plugins`, `share/Pythia8/xmldoc`,
  and `examples`. Tabs can be fixed in Emacs by `ctrl-x h` followed by
  `cmnd-x untabify`. Ensure the correct date is replaced in
  `trimfile.`
  ```bash
  make private/trimfile
  ./private/trimfile -d src
  ./private/trimfile -d include/Pythia8
  ./private/trimfile -d include/Pythia8Plugins
  ./private/trimfile -d share/Pythia8/xmldoc
  ./private/trimfile -d examples
  ```
   
7. Run `indexXML.py` and `indexMains.py` in `private` to build the
   search index and example index.
   ```bash
   cd private
   ./indexXML.py
   ./indexMains.py
   ```

8. Run `convertXML` to convert to HTML.
   ```bash
   make private/convertXML
   ./private/convertXML
   ```
