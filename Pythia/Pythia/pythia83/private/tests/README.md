# Running Tests

This directory contains a simple test suite to be used by Pythia
developers. The intended use is as follows:

* When checking out a (new) branch to implement or modify something,
  first do (in this tests directory):
  ```bash
  make check
  ```
  which will fail as there are no output files to compare with. You can
  use the -j flag to make if you have a lot of cores. Then do
  ```
  make checkdir
  ```
  which will copy the output files to the cmp sub-directory.

* After you have finished your modifications but BEFORE you push your
  changes to gitlab, do,
  ```bash
  make check
  ```
  again. In the best of all worlds this time you will get a summary looking like
  ```
  Tried:    32
  Passed:   32
  Differed: 0
  Failed:   0
  ```
  in which case you can go ahead and push you changes, feeling
  reassured that you at least have not done something very stupid. If
  the number of `Failed` tests is larger than one, then you
  definitely need to fix your code before pushing.

  If there is a number larger than zero for `Differed`, then the
  situation is a bit more difficult. It could be that you just changed
  the random number sequence somewhere. But it could also mean that
  you by mistake have changed something that is giving unintended
  consequences.

* Not until you have fixed all the `Failed` and fixed or clearly
  understand why some tests `Differed` should you push your changes to
  gitlab. If there still are differences you should then clearly state this
  in the commit message.

# Comparison Tool

The `testgit` tool automates checking differences between `git`
commits. This is done by providing a reference commit via the
`--ref=<ref>` argument where `<ref>` is the commit hash. If the
reference commit is omitted, then the first commit of the requested
branch is used. The branch is specified with `--branch=<branch>` which
defaults to `master`.
```bash
./testgit --branch=master --ref=ee355a75
```
It is also possible to compare each commit to the previous with
`--ref=PREVIOUS`.

The script finds the linear history for the requested branch and
compares the test output for each commit to the reference commit. It
is possible to specify the first commit for comparison with the option
`--first=<first hash>` and the last commit for comparison with the
option `--last=<last hash>` where the arguments are the commit
hashes. By default the first commit for comparison is the first commit
in the history, and the last commit for comparison is the last commit
in the history.

The history may have interleaved commits from multiple branches,
e.g. the third commit might be based on the first commit, not the
second. It is possible to follow the lineage of a commit with the
`--parents` flag. Here, the lineage for the commit specified with
`--last=<last hash>` is traced backwards. The comparison order may be
reversed via the `--reverse` flag. Commits where all tests fail can be
skipped with the `--skip` flag.

The test output for each test is cached to `<out>/<index>.<hash>`
where `<out>` is the output directory, `<index>` is the index of that
commit in the linear history, and `<hash>` is the hash for that
commit. All commit hashes, with their corresponding index from the
linear history can be printed with the `--list` flag. Because running
each test suite is time consuming, the script caches results and skips
cached commits. Cached results are searched for in the following order
`<index>.<hash>`, `<hash>`, and `*.<hash>` in the `<out>`
directory. The output directory can be specified with the argument
`--outdir=<out>` which defaults to `commits`. Cached results are
ignored if the `--force` flag is used.

In many cases it is necessary to patch a specific commit so that it
compiles and/or so the tests can be run. Patches for each commit can
be provided in the `<dif>` directory specified by the argument
`--difdir=<dif>` which defaults to `patches`. Patches are searched for
in the same order as cached results. Following the linear history, the
last provided patch is applied to all subsequent commits until a new
patch supersedes the previous patch. This patching mechanism is
invariant under other options.

Patches can be prepared with the `--patch=<hash>` command. This
creates the directories `old` and `new` for the commit `<hash>`. The
`new` directory is using the patches from `--difdir` and may be
modified to create the new patch. Once changes have been made in
`new`, the option `--patch` without argument should be specified, and
will create the relevant patch in `--difdir`.
```bash
./testgit --patch=ee355a75
# Make the patch changes in "new".
./testgit --patch
```

Alternatively, a patch can be directly created between two commits
with `--patch=<hash 1>/<hash 2>`. Here, a patch is created for `<hash
2>` so that when applied to `<hash 2>` the result is identical to
`<hash 1>` as patched from `--difdir`. This patch contains all the
differences between `<hash 1>` and `<hash 2>` and can be modified to
determine which change causes the modified test behaviour.

The patch, build, test build, and test summary logs, including both
`stdout` and `stderr`, are stored in the cached output for each commit
as `patch.log`, `build.log`, `test.log`, and `testsum.log`,
respectively. When rerunning with a different with a different
reference, each `testsum.log` is updated to the new comparison. The
aggregated test summary for all commits is provided in the file
`<log>` which can be specified by `--log-<log>` and defaults to
`testsums.log`.

Three debug levels are available and can be specified by
`--debug=<level>` with the following `<level>` values.
0. Most output is suppressed except a summary of the branch name and
   commits processed.
1. All `stdout` from patching, building, and testing is now also
   printed to the screen.
2. Additionally, `stderr` is now printed to the screen.
The default debug value is `0`.

Below, a list of of the reference tests, and observed changes are
documented. All necessary patches for checking these reference tests
are available in the `patches` directory.

# 8.300 - 8.301

* ee355a75 (000) - first 8.300 commit and **starting reference**.

* 53f7bda5 (001) - bug fix in `src/SimpleTimeShower.cc` ported from
  8.234 changes a number of tests. This commit becomes the **new
  reference**.
  ```bash
  ./testgit --ref=ee355a75 --first=53f7bda5 --last=53f7bda5
  ```
  ```
  Tried:    33
  Passed:   12
  Differed: 21
  Failed:   0
  ```

* cd15a8ed (0003) - a number of bug fixes ported from 8.243 are
  included. This commit becomes the **new reference**.
  * `test009` - process `Top:ggm2ttbar` added to `src/ProcessContainer.cc`.
  * `test068` - kinematics for `Sigma2ggm2qqbar::sigmaKin()` changed in
    `src/SigmaEW.cc`.
  * `test101` - change in `src/MultiPartionInteractions.cc` leads to
    difference in number of MPI.
  ```bash
  ./testgit --ref=53f7bda5 --first=cd15a8ed --last=cd15a8ed
  ```
  ```
  Tried:    33
  Passed:   30
  Differed: 3
  Failed:   0
  ```
  
* fa4b4778 (0091) - the switch to a common two-body phase-space
  generator, `Random::phaseSpace2` changes the random number sequence
  and causes a number of differences in the tests. This commit becomes
  the **new reference**.
  ```bash
  ./testgit --ref=cd15a8ed --first=fa4b4778 --last=fa4b4778
  ```
  ```
  Tried:    33
  Passed:   12
  Differed: 21
  Failed:   0
  ```

* de4eb4a2 (0381) - last 8.301 commit. Here, there is a difference in
  `allowCalc` for `test007` where the lifetime is no longer calculated
  from the width for the `GeneralResonace` in the example.
  ```bash
  ./testgit --ref=fa4b4778 --first=de4eb4a2 --last=de4eb4a2
  ```
  ```
  Tried:    33
  Passed:   32
  Differed: 1
  Failed:   0
  ```

# 8.301 - 8.302

* de4eb4a2 (0381) - last 8.301 commit and **starting reference**.

* aa82119e (0410) - incorrect fix for SCA issues 18 and 19 creates a
  significant difference for commits along this branch history.

* 76dec759 (0434) - there is an expected difference with de4eb4a2 because of
  significant changes to the PDF handling mechanism in the `speedyPDF`
  branch. This commit becomes the **new reference**.
  ```bash
  ./testgit --ref=de4eb4a2 --first=76dec759 --last=76dec759
  ```
  ```
  Tried:    33
  Passed:   12
  Differed: 21
  ```

* 8d47afa6 (0583) - `tauCalc` overrides the `allowCalcWidth` flag
  causing a difference in `test007`. This is the intended behaviour,
  and so this commit becomes the **new reference**.
  ```bash
  ./testgit --ref=76dec759 --first=566483e1 --last=566483e1
  ```
  ```
  Tried:    33
  Passed:   32
  Differed: 1
  ```

# 8.302 - 8.303

* d764ab1c - reference commit with automatic testing. No differences wrt.
  previous commit, but includes script changes for testing, so
  this commit becomes the **new reference**. 

* 05ebfda6 - fix for "dangling" gluon in top decays changing color
  flows. All tests with top decays are modified: `test007`, `test009`,
  `test029`. This commit becomes the **new reference**.

* 51ee7db7 - added new `test202` for Vincia, this becomes the **new
  reference**.

* 2bd0c84b - photoproduction for hadron beams. This results in changes
  to the output of `test068`, a photoproduction example, so this
  commit becomes the **new reference**.

* 7b15ba7a - hadronic rescattering is introduced. This is by default
  off, but has introduced a number of changes in `ParticleData.xml`,
  `ParticleDecays.cc`, and `MiniStringFragmentation.cc` which cause
  differences in random number sequences for a number of tests. The
  first and second changes are needed for hadronic resonances; the
  third is needed for the small strings present in hadronic
  rescattering. This commit becomes the **new reference**.

* 52725ba9 - fixes out-of-bound access of `Q2RenSave` due to
  mixed-uses of the `iDS` variables, and protects against unphysical
  `x` values for diffractive systems with masses greater than the
  centre-of-mass energy. These changes modify the behaviour of the
  diffractive machinery, and change the output of `test068`. This
  commit becomes the **new reference**.

# 8.303 - 8.304

* 56b3a8b3 - several small changes have led to a change in the random
  number sequences. Particularly, a bug fix to how the Breit-Wigner
  shapes are introduced for narrow resonances causes differences. This
  commit becomes the **new reference**.

* 4f32268b - c and b quarks now can produce two- or three-body final
  states further above threshold than light quarks can. This could
  break the sequence of events generated, even if overall event
  properties hardly are affected. This commit becomes the **new
  reference**.

* d588ada5 - temporary check for Vincia code review. This commit
  becomes the **new reference**.

* d3e11a75 - `master` merged into Vincia branch with changes from
  4f32268b.

* 796ff96f - introduced `--with-mg5mes` configuration flag for Vincia
  tests, modifies output of `test200` and `test202`. 

* 9525c642 - New `test201`, first introduced in 20304071 provides
  further Vincia checks. This commit becomes the **new reference**.

* b0dc1399 - Further fixes in Vincia so `test201` completes
  succesfully, This commit becomes the **new reference**.

* 98cc0735 - Merging of `master` into `vincia-release8304` and picking
  up 248024b4. This fixes a Hadronic rescattering bug where the pi+ K+
  and pi+ Kbar0 elastic cross sections are set equal to the total
  ones. Only modifies results of `test110`. This commit becomes the
  **new reference**.

* 28bbbac1 - A number of Vincia changes have been made, which result
  differences for `test200`, `test201`, and `test202`. This commit
  becomes the **new reference**.

* eec27d6a - The `test201` was moved to `test203` to better align with
  the examples. Verbose output was turned off in all `test20X`
  examples. This commit becomes the **new reference**.

* 737ec363 - Fix for the event information in hard diffraction
  events. Differences in `test061` and `test068` are expected. This
  commit becomes the **new reference**.

* 774033ab - Included double quarkonia production in both second hard
  processes and MPI. This changes the random number sequence in MPI,
  which causes most tests to differ. This commit becomes the **new
  reference**.
  ```
  Tried:    37
  Passed:   12
  Differed: 25
  Failed:   0
  ```

* b496ea17 - A bug was fixed in `forceTimeShower` which changes the
  output of `test200`, `test202`, and `test203`. This commit becomes
  the **new reference**.

* 16405bf1 - A few bug fixes in Vincia have changed the behaviour of
  `test202` and `test203`. The unstable boost of the Dalitz decay has
  been stabilized which changes `test006`. This commit becomes the
  **new reference**.

* 07004b07 - Changed to using `log1p`, which causes small differences
  in `test029` due to `ColourReconnectionHooks` modifications. This
  commit becomes the **new reference**.

# 8.304 - 8.305

* 4d3f792a - Added new test for Dire. This commit becomes the **new
  reference**.

* 76830603 - Changes in Vincia initialization leads to different
  results in `test202` and `test203`. This commit becomes the **new
  reference**.
  
* db916285 - Bug fix for initial quark treatement in
  `src/VinciaCommon.cc` leads to a change in `test202` and
  `test203`. This commit becomes the **new reference**.
