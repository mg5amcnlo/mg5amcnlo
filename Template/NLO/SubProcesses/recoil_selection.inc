c Explicit recoil request, kept separate from each point's active frame.
c The mask contains recoilers only, in the current real leg labels.
      logical fks_recoil_requested,fks_requested_recoilers(nexternal)
      common /c_fks_recoil_request/ fks_recoil_requested,
     $     fks_requested_recoilers
