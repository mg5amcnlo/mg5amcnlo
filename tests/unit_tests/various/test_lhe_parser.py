################################################################################
#
# Copyright (c) 2012 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Test the validity of the LHE parser"""

import unittest
import madgraph.various.lhe_parser as lhe_parser
import madgraph .various.misc as misc
import tempfile
import os
import shutil
pjoin = os.path.join
from madgraph import MG5DIR

class TESTLHEParser(unittest.TestCase):

    def setUp(self):
        
        debugging = False
        if debugging:
            self.path = pjoin(MG5DIR, "tmp_lhe_test")
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            os.mkdir(pjoin(MG5DIR, "tmp_test"))
        else:
            self.path = tempfile.mkdtemp(prefix='test_mg5')

    def tearDown(self):
        
        if self.path != pjoin(MG5DIR, "tmp_lhe_test"):
            shutil.rmtree(self.path)



    def test_parsing_lo_weight(self):
        """test that our parser can handle a large range of lo_weight format"""

        def parse_lo_weight_old(evt):
            """parsing for unittest onlyx"""

            
            start, stop = evt.tag.find('<mgrwt>'), evt.tag.find('</mgrwt>')
    
            if start != -1 != stop :
                text = evt.tag[start+8:stop]
    #<rscale>  3 0.29765919e+03</rscale>
    #<asrwt>0</asrwt>
    #<pdfrwt beam="1">  1       21 0.15134321e+00 0.29765919e+03</pdfrwt>
    #<pdfrwt beam="2">  1       21 0.38683649e-01 0.29765919e+03</pdfrwt>
    #<totfact> 0.17315115e+03</totfact>
                evt.loweight={}
                for line in text.split('\n'):
                    line = line.replace('<', ' <').replace("'",'"')
                    if 'rscale' in line:
                        _, nqcd, scale, _ = line.split()
                        evt.loweight['n_qcd'] = int(nqcd)
                        evt.loweight['ren_scale'] = float(scale)
                    elif '<pdfrwt beam="1"' in line:
                        args = line.split()
                        evt.loweight['n_pdfrw1'] = int(args[2])
                        npdf = evt.loweight['n_pdfrw1']
                        evt.loweight['pdf_pdg_code1'] = [int(i) for i in args[3:3+npdf]]
                        evt.loweight['pdf_x1'] = [float(i) for i in args[3+npdf:3+2*npdf]]
                        evt.loweight['pdf_q1'] = [float(i) for i in args[3+2*npdf:3+3*npdf]]
                    elif '<pdfrwt beam="2"' in line:
                        args = line.split()
                        evt.loweight['n_pdfrw2'] = int(args[2])
                        npdf = evt.loweight['n_pdfrw2']
                        evt.loweight['pdf_pdg_code2'] = [int(i) for i in args[3:3+npdf]]
                        evt.loweight['pdf_x2'] = [float(i) for i in args[3+npdf:3+2*npdf]]
                        evt.loweight['pdf_q2'] = [float(i) for i in args[3+2*npdf:3+3*npdf]]
                    elif '<asrwt>' in line:
                        args = line.replace('>','> ').split()
                        nalps = int(args[1])
                        evt.loweight['asrwt'] = [float(a) for a in args[2:2+nalps]] 
                        
                    elif 'totfact' in line:
                        args = line.replace('>','> ').split()
                        evt.loweight['tot_fact'] = float(args[1])
            else:
                return None
            return evt.loweight

        
        events=["""
<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
</event>
""","""
<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
<mgrwt>
<rscale>  2 0.12500000E+03</rscale>
<asrwt>0</asrwt>
<pdfrwt beam="1">  1        4 0.11319990E+00 0.12500000E+03</pdfrwt>
<pdfrwt beam="2">  1       -1 0.59528052E+00 0.12500000E+03</pdfrwt>
<totfact>-0.27352270E-03</totfact>
</mgrwt>
</event>""",
"""
<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
<mgrwt>
<rscale>  2 0.12500000E+03</rscale>
<asrwt> 1 0.11 </asrwt>
<pdfrwt beam='1'>  1        4 0.11319990E+00 0.12500000E+03</pdfrwt>
<pdfrwt beam=2>    2      1 -1  0.2 0.11e-02 0.59528052E+00 0.12500000E+03</pdfrwt>
<totfact> 115 </totfact>
</mgrwt>
</event>""",
"""<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
<mgrwt>
<rscale>2 0.12500000E+03</rscale>
<asrwt>1 0.11 </asrwt>
<pdfrwt beam='1'>  1        4 0.11319990E+00 0.12500000e+03 </pdfrwt>
<pdfrwt beam=2>    2      1 -1  0.2 0.11e-02 0.59528052E+00 0.12500000E+03 </pdfrwt>
<totfact> 115.001 </totfact>
</mgrwt>
</event>""",
"""<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
<mgrwt>
<rscale>2 0.12500000E+03</rscale>
<asrwt>1 0.11 </asrwt>
<pdfrwt beam='1'>  1        4 0.11319990E+00 0.12500000e+03 </pdfrwt>
<pdfrwt beam=2>    2      1 -1  0.2 0.11e-02 0.59528052E+00 0.12500000E+03 </pdfrwt>
<totfact> 115.001 </totfact>
</mgrwt>
</event>""",
"""<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
<mgrwt>
<rscale>  2 0.26956472E+02</rscale>
    <asrwt>  1 0.46373112E+02</asrwt>
    <pdfrwt beam="1">  1       21 0.13689253E-01 0.52142986E+01</pdfrwt>
    <pdfrwt beam="2">  1       21 0.29841683E-01 0.46373112E+02</pdfrwt>
    <totfact> 0.15951072E+03</totfact>
</mgrwt>
</event>
"""]
     
        solutions = [None, 
                  {'pdf_pdg_code1': [4], 'asrwt': [], 'pdf_pdg_code2': [-1], 'pdf_q1': [125.0], 'pdf_q2': [125.0], 'n_pdfrw1': 1, 'n_pdfrw2': 1, 'tot_fact': -0.0002735227, 'pdf_x2': [0.59528052], 'pdf_x1': [0.1131999], 'n_qcd': 2, 'ren_scale': 125.0},
                  {'pdf_pdg_code1': [4], 'asrwt': [0.11], 'pdf_pdg_code2': [1, -1], 'pdf_q1': [125.0], 'pdf_q2': [0.59528052, 125.0], 'ren_scale': 125.0, 'n_pdfrw1': 1, 'n_pdfrw2': 2, 'pdf_x2': [0.2, 0.0011], 'pdf_x1': [0.1131999], 'n_qcd': 2, 'tot_fact': 115.0},
                  {'pdf_pdg_code1': [4], 'asrwt': [0.11], 'pdf_pdg_code2': [1, -1], 'pdf_q1': [125.0], 'pdf_q2': [0.59528052, 125.0], 'ren_scale': 125.0, 'n_pdfrw1': 1, 'n_pdfrw2': 2, 'pdf_x2': [0.2, 0.0011], 'pdf_x1': [0.1131999], 'n_qcd': 2, 'tot_fact': 115.001},
                  {'pdf_pdg_code1': [4], 'asrwt': [0.11], 'pdf_pdg_code2': [1, -1], 'pdf_q1': [125.0], 'pdf_q2': [0.59528052, 125.0], 'ren_scale': 125.0, 'n_pdfrw1': 1, 'n_pdfrw2': 2, 'pdf_x2': [0.2, 0.0011], 'pdf_x1': [0.1131999], 'n_qcd': 2, 'tot_fact': 115.001},
                  {'pdf_pdg_code1': [21], 'asrwt': [46.373112], 'pdf_pdg_code2': [21], 'pdf_q1': [5.2142986], 'pdf_q2': [46.373112], 'ren_scale': 26.956472, 'n_pdfrw1': 1, 'n_pdfrw2': 1, 'pdf_x2': [0.029841683], 'pdf_x1': [0.013689253], 'n_qcd': 2, 'tot_fact': 159.51072},
                  None]
                  
     
        for i,evt in enumerate(events):
            evt1 = lhe_parser.Event(evt)
            evt2 = lhe_parser.Event(evt)
            lo = evt1.parse_lo_weight()
            try:
                lo2 = parse_lo_weight_old(evt2)
            except:
                pass
            else:
                if lo:
                    for key in lo2:
                        self.assertEqual(lo[key], lo2[key])
            self.assertEqual(lo, solutions[i])
         
     
     
        
        

    def test_read_write_lhe(self):
        """test that we can read/write an lhe event file"""
        
        input= """<LesHouchesEvents version="1.0">
<header>
DATA
</header>
<init>
     2212     2212  0.70000000000E+04  0.70000000000E+04 0 0 10042 10042 3  1
  0.16531958660E+02  0.18860728290E+00  0.17208000000E+00   0
</init>
<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
</event>
<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
# balbalblb
#bbbb3
</event>            
<event>
  7     66 +1.5024446e-03 3.15138740e+02 7.95774720e-02 9.66701260e-02
       21 -1    0    0  502  501 +0.0000000e+00 +0.0000000e+00 -6.4150959e+01 6.41553430e+01 7.49996552e-01 0.0000e+00 0.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 +8.1067989e+02 8.10679950e+02 3.11899968e-01 0.0000e+00 0.0000e+00
       24  2    1    2    0    0 -1.5294533e+02 +1.0783429e+01 +4.5796553e+02 4.89600040e+02 8.04190039e+01 0.0000e+00 0.0000e+00
        2  1    3    3  503    0 -1.5351296e+02 +1.2743130e+01 +4.6093709e+02 4.85995489e+02 0.00000000e+00 0.0000e+00 0.0000e+00
       -1  1    3    3    0  503 +5.6763429e-01 -1.9597014e+00 -2.9715566e+00 3.60455091e+00 0.00000000e+00 0.0000e+00 0.0000e+00
       23  1    1    2    0    0 +4.4740095e+01 +3.2658177e+01 +4.6168760e+01 1.16254200e+02 9.11880036e+01 0.0000e+00 0.0000e+00
        1  1    1    2  502    0 +1.0820523e+02 -4.3441605e+01 +2.4239464e+02 2.68981060e+02 3.22945297e-01 0.0000e+00 0.0000e+00
# 2  5  2  2  1 0.11659994e+03 0.11659994e+03 8  0  0 0.10000000e+01 0.88172677e+00 0.11416728e+01 0.00000000e+00 0.00000000e+00
  <rwgt>
    <wgt id='1001'> +9.1696000e+03 </wgt>
    <wgt id='1002'> +1.1264000e+04 </wgt>
    <wgt id='1003'> +6.9795000e+03 </wgt>
    <wgt id='1004'> +9.1513000e+03 </wgt>
    <wgt id='1005'> +1.1253000e+04 </wgt>
  </rwgt>
</event>
</LesHouchesEvents> 
        
        """
        
        open(pjoin(self.path,'event.lhe'),'w').write(input)
        
        input = lhe_parser.EventFile(pjoin(self.path,'event.lhe'))
        self.assertEqual(input.banner, """<LesHouchesEvents version="1.0">
<header>
DATA
</header>
<init>
     2212     2212  0.70000000000E+04  0.70000000000E+04 0 0 10042 10042 3  1
  0.16531958660E+02  0.18860728290E+00  0.17208000000E+00   0
</init>
""")
        
        nb_event = 0
        txt = ""
        for event in input:
            nb_event +=1
            new = lhe_parser.Event(text=str(event))
            for part1,part2 in zip(event, new):
                self.assertEqual(part1, part2)
            self.assertEqual(new, event, '%s \n !=\n %s' % (new, event))
            txt += str(event)
        self.assertEqual(nb_event, 3)
    
        target = """<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000000e+00 +0.0000000000e+00 +1.1943355000e+01 1.1943354600e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000000e+00 +0.0000000000e+00 -1.0679326000e+03 1.0679326200e+03 0.0000000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155000e+00 +4.2744556000e+01 -7.9238049000e+02 7.9761999700e+02 8.0419007300e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155000e+00 -4.2744556000e+01 -2.6360878000e+02 2.8225597900e+02 9.1188003500e+01 1.8975e-26 1.0000e+00
</event>
<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000000e+00 +0.0000000000e+00 +1.1943355000e+01 1.1943354600e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000000e+00 +0.0000000000e+00 -1.0679326000e+03 1.0679326200e+03 0.0000000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155000e+00 +4.2744556000e+01 -7.9238049000e+02 7.9761999700e+02 8.0419007300e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155000e+00 -4.2744556000e+01 -2.6360878000e+02 2.8225597900e+02 9.1188003500e+01 1.8975e-26 1.0000e+00
# balbalblb
#bbbb3
</event>
<event>
 7     66 +1.5024446e-03 3.15138740e+02 7.95774720e-02 9.66701260e-02
       21 -1    0    0  502  501 +0.0000000000e+00 +0.0000000000e+00 -6.4150959000e+01 6.4155343000e+01 7.4999655200e-01 0.0000e+00 0.0000e+00
        2 -1    0    0  501    0 +0.0000000000e+00 +0.0000000000e+00 +8.1067989000e+02 8.1067995000e+02 3.1189996800e-01 0.0000e+00 0.0000e+00
       24  2    1    2    0    0 -1.5294533000e+02 +1.0783429000e+01 +4.5796553000e+02 4.8960004000e+02 8.0419003900e+01 0.0000e+00 0.0000e+00
        2  1    3    3  503    0 -1.5351296000e+02 +1.2743130000e+01 +4.6093709000e+02 4.8599548900e+02 0.0000000000e+00 0.0000e+00 0.0000e+00
       -1  1    3    3    0  503 +5.6763429000e-01 -1.9597014000e+00 -2.9715566000e+00 3.6045509100e+00 0.0000000000e+00 0.0000e+00 0.0000e+00
       23  1    1    2    0    0 +4.4740095000e+01 +3.2658177000e+01 +4.6168760000e+01 1.1625420000e+02 9.1188003600e+01 0.0000e+00 0.0000e+00
        1  1    1    2  502    0 +1.0820523000e+02 -4.3441605000e+01 +2.4239464000e+02 2.6898106000e+02 3.2294529700e-01 0.0000e+00 0.0000e+00
# 2  5  2  2  1 0.11659994e+03 0.11659994e+03 8  0  0 0.10000000e+01 0.88172677e+00 0.11416728e+01 0.00000000e+00 0.00000000e+00
<rwgt>
<wgt id='1001'> +9.1696000e+03 </wgt>
<wgt id='1002'> +1.1264000e+04 </wgt>
<wgt id='1003'> +6.9795000e+03 </wgt>
<wgt id='1004'> +9.1513000e+03 </wgt>
<wgt id='1005'> +1.1253000e+04 </wgt>
</rwgt>
</event>
"""


        self.assertEqual(target.split('\n'), txt.split('\n'))


    def test_read_write_gzip(self):
        """ """ 
        
        input= """<LesHouchesEvents version="1.0">
<header>
DATA
</header>
<init>
     2212     2212  0.70000000000E+04  0.70000000000E+04 0 0 10042 10042 3  1
  0.16531958660E+02  0.18860728290E+00  0.17208000000E+00   0
</init>
<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
</event>
<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
# balbalblb
#bbbb3
</event>            
<event>
  7     66 +1.5024446e-03 3.15138740e+02 7.95774720e-02 9.66701260e-02
       21 -1    0    0  502  501 +0.0000000e+00 +0.0000000e+00 -6.4150959e+01 6.41553430e+01 7.49996552e-01 0.0000e+00 0.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 +8.1067989e+02 8.10679950e+02 3.11899968e-01 0.0000e+00 0.0000e+00
       24  2    1    2    0    0 -1.5294533e+02 +1.0783429e+01 +4.5796553e+02 4.89600040e+02 8.04190039e+01 0.0000e+00 0.0000e+00
        2  1    3    3  503    0 -1.5351296e+02 +1.2743130e+01 +4.6093709e+02 4.85995489e+02 0.00000000e+00 0.0000e+00 0.0000e+00
       -1  1    3    3    0  503 +5.6763429e-01 -1.9597014e+00 -2.9715566e+00 3.60455091e+00 0.00000000e+00 0.0000e+00 0.0000e+00
       23  1    1    2    0    0 +4.4740095e+01 +3.2658177e+01 +4.6168760e+01 1.16254200e+02 9.11880036e+01 0.0000e+00 0.0000e+00
        1  1    1    2  502    0 +1.0820523e+02 -4.3441605e+01 +2.4239464e+02 2.68981060e+02 3.22945297e-01 0.0000e+00 0.0000e+00
# 2  5  2  2  1 0.11659994e+03 0.11659994e+03 8  0  0 0.10000000e+01 0.88172677e+00 0.11416728e+01 0.00000000e+00 0.00000000e+00
  <rwgt>
    <wgt id='1001'> +9.1696000e+03 </wgt>
    <wgt id='1002'> +1.1264000e+04 </wgt>
    <wgt id='1003'> +6.9795000e+03 </wgt>
    <wgt id='1004'> +9.1513000e+03 </wgt>
    <wgt id='1005'> +1.1253000e+04 </wgt>
  </rwgt>
</event>
</LesHouchesEvents> 
        
        """
        
        open(pjoin(self.path, 'event.lhe'),'w').write(input)
        input_lhe = lhe_parser.EventFile(pjoin(self.path, 'event.lhe.gz'))
        output_lhe = lhe_parser.EventFile(pjoin(self.path, 'event2.lhe.gz'),'w')
        output_lhe.write(input_lhe.banner)
        for event in input_lhe:
            output_lhe.write(str(event))
        output_lhe.close()
        self.assertTrue(pjoin(self.path,'event2.lhe.gz'))
        text = open(pjoin(self.path, 'event2.lhe.gz')).read()
        self.assertFalse(text.startswith('<LesHouchesEvents version="1.0">'))
        misc.gunzip(pjoin(self.path,'event2.lhe.gz'))
        self.assertTrue(pjoin(self.path,'event2.lhe'))
        input_lhe = lhe_parser.EventFile(pjoin(self.path, 'event.lhe'))
        

