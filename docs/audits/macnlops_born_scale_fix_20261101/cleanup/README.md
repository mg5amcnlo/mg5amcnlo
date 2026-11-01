# Cleanup verification

The earlier snapshot.json and source/ tree retain the implementation epoch used by the native campaign. They are unchanged. The later behavior-preserving cleanup has its own source hashes and proof in ../cleanup_validation.json.

The C++ S/H support gates now share one helper. The Fortran reference helper uses a scalar loop temporary and drops an unused include; native call and sum order are unchanged. All138 regression tests pass in one run, 140 complete native fixture records are byte-identical before/after, and all four real Wbb Fortran objects compile with zero warnings. The earlier19,812-event/eight-process campaign was not rerun for this refactor.

Large native inputs, binaries and fixture event files remain at /export/slow1/rikkert/macnlops_born_scale_fix_20261101/cleanup.
