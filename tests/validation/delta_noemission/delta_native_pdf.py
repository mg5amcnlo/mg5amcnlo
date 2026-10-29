"""The generated MG5 built-in nn23nlo PDF, using its native Fortran driver.

The driver returns x*f, whereas pdg2pdf returns f; the factor x cancels in
all Delta PDF ratios (including flavour sums at a common Bjorken x).
"""
import ctypes as C
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np


class NativePDF:
    def __init__(self, process, work=None):
        process=Path(process).resolve()
        self._temp=tempfile.TemporaryDirectory(prefix='delta_native_pdf_') if work is None else None
        self.work=(Path(self._temp.name) if self._temp else Path(work)).resolve()
        self.work.mkdir(parents=True,exist_ok=True)
        source=process/'Source/PDF/NNPDFDriver.f'
        opener=process/'Source/PDF/opendata.f'
        grid=process/'lib/Pdfdata/NNPDF23nlo_as_0119_qed_mem0.grid'
        lib=self.work/'native_pdf.so'
        command=['gfortran','-O2','-std=legacy','-shared','-fPIC','-ffixed-line-length-none',
                 str(source),str(opener),'-o',str(lib)]
        subprocess.run(command,check=True,capture_output=True,text=True)
        self.lib=C.CDLL(str(lib))
        self.lib.nnpdfdriver_.argtypes=[C.c_char_p,C.c_size_t]
        self.lib.nninitpdf_.argtypes=[C.POINTER(C.c_int)]
        self.lib.nnevolvepdf_.argtypes=[C.POINTER(C.c_double)]*3
        link=self.work/'nn23.grid'
        if not link.exists():link.symlink_to(grid)
        old=os.getcwd()
        try:
            # Production OpenData uses a 100-character local buffer. A short
            # local filename also works when the process path is longer.
            os.chdir(self.work)
            self.lib.nnpdfdriver_(b'nn23.grid',9)
        finally:os.chdir(old)
        self.lib.nninitpdf_(C.byref(C.c_int(0)))
        self.provenance=dict(pdf='nn23nlo',convention='x*f; x cancels in Delta PDF ratios',
            command=command,sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [source,opener,grid]})

    @lru_cache(maxsize=8192)
    def vector(self,x,q):
        if not (0<x<=1 and q>=0):raise ValueError((x,q))
        if x==1:return tuple([0.]*14)
        values=np.zeros(14)
        xx=C.c_double(x);qq=C.c_double(q)
        self.lib.nnevolvepdf_(C.byref(xx),C.byref(qq),values.ctypes.data_as(C.POINTER(C.c_double)))
        return tuple(values)

    def __call__(self,pdg,x,q):
        id=0 if pdg==21 else 7 if pdg==22 else pdg
        if abs(id)>7:return 0.
        # Production fdist excludes top density even though the driver has it.
        if abs(id)==6:return 0.
        return self.vector(float(x),float(q))[id+6]

    def close(self):
        self.vector.cache_clear()
        if self._temp:self._temp.cleanup();self._temp=None
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
