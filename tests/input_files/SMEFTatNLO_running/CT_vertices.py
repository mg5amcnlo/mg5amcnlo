# This file was automatically created by FeynRules 2.4.78
# Mathematica version: 12.0.0 for Mac OS X x86 (64-bit) (April 7, 2019)
# Date: Wed 1 Apr 2020 19:36:05


from object_library import all_vertices, all_CTvertices, Vertex, CTVertex
import particles as P
import CT_couplings as C
import lorentz as L


V_1 = CTVertex(name = 'V_1',
               type = 'R2',
               particles = [ P.g, P.g, P.G0, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.VVSS11, L.VVSS14, L.VVSS16, L.VVSS20 ],
               loop_particles = [ [ [P.g] ], [ [P.t] ] ],
               couplings = {(0,3,0):C.R2GC_569_1756,(0,1,1):[ C.R2GC_1296_466, C.R2GC_618_1801 ],(0,2,1):C.R2GC_629_1812,(0,0,1):C.R2GC_630_1813})

V_2 = CTVertex(name = 'V_2',
               type = 'R2',
               particles = [ P.g, P.g, P.G__minus__, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.VVSS11, L.VVSS13, L.VVSS14, L.VVSS17, L.VVSS20 ],
               loop_particles = [ [ [P.b] ], [ [P.b, P.t] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.g] ], [ [P.t] ] ],
               couplings = {(0,4,4):C.R2GC_569_1756,(0,1,0):C.R2GC_1324_541,(0,1,2):C.R2GC_1324_542,(0,1,3):C.R2GC_1324_543,(0,1,5):C.R2GC_1324_544,(0,2,5):C.R2GC_1497_967,(0,2,1):[ C.R2GC_1497_968, C.R2GC_618_1801 ],(0,3,1):C.R2GC_857_1939,(0,0,1):C.R2GC_630_1813})

V_3 = CTVertex(name = 'V_3',
               type = 'R2',
               particles = [ P.g, P.g, P.H, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.VVSS11, L.VVSS14, L.VVSS16, L.VVSS20 ],
               loop_particles = [ [ [P.g] ], [ [P.t] ] ],
               couplings = {(0,3,0):C.R2GC_569_1756,(0,1,1):[ C.R2GC_1297_467, C.R2GC_618_1801 ],(0,2,1):C.R2GC_629_1812,(0,0,1):C.R2GC_630_1813})

V_4 = CTVertex(name = 'V_4',
               type = 'R2',
               particles = [ P.g, P.g, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.VVS11, L.VVS14, L.VVS16 ],
               loop_particles = [ [ [P.g] ], [ [P.t] ] ],
               couplings = {(0,2,0):C.R2GC_570_1757,(0,0,1):[ C.R2GC_631_1814, C.R2GC_1298_468 ],(0,1,1):C.R2GC_595_1779})

V_5 = CTVertex(name = 'V_5',
               type = 'R2',
               particles = [ P.g, P.g, P.g ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVV8 ],
               loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.g] ], [ [P.t] ] ],
               couplings = {(0,0,0):C.R2GC_1905_1362,(0,0,1):C.R2GC_1905_1363,(0,0,2):C.R2GC_1681_1170})

V_6 = CTVertex(name = 'V_6',
               type = 'R2',
               particles = [ P.g, P.g, P.g, P.g ],
               color = [ 'd(-1,1,3)*d(-1,2,4)', 'd(-1,1,3)*f(-1,2,4)', 'd(-1,1,4)*d(-1,2,3)', 'd(-1,1,4)*f(-1,2,3)', 'd(-1,2,3)*f(-1,1,4)', 'd(-1,2,4)*f(-1,1,3)', 'f(-1,1,2)*f(-1,3,4)', 'f(-1,1,3)*f(-1,2,4)', 'f(-1,1,4)*f(-1,2,3)', 'Identity(1,2)*Identity(3,4)', 'Identity(1,3)*Identity(2,4)', 'Identity(1,4)*Identity(2,3)' ],
               lorentz = [ L.VVVV14, L.VVVV15, L.VVVV17, L.VVVV22 ],
               loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.g] ], [ [P.t] ] ],
               couplings = {(0,0,0):C.R2GC_1305_480,(0,0,1):C.R2GC_1305_481,(2,0,0):C.R2GC_1305_480,(2,0,1):C.R2GC_1305_481,(6,0,0):C.R2GC_1910_1370,(6,0,1):C.R2GC_1910_1371,(6,0,2):C.R2GC_1683_1172,(7,0,0):C.R2GC_1909_1368,(7,0,1):C.R2GC_1909_1369,(7,0,2):C.R2GC_1683_1172,(5,0,0):C.R2GC_1302_474,(5,0,1):C.R2GC_1302_475,(1,0,0):C.R2GC_1302_474,(1,0,1):C.R2GC_1302_475,(4,0,0):C.R2GC_1302_474,(4,0,1):C.R2GC_1302_475,(3,0,0):C.R2GC_1302_474,(3,0,1):C.R2GC_1302_475,(8,0,0):C.R2GC_1304_478,(8,0,1):C.R2GC_1304_479,(11,3,0):C.R2GC_1303_476,(11,3,1):C.R2GC_1303_477,(10,3,0):C.R2GC_1303_476,(10,3,1):C.R2GC_1303_477,(9,3,1):C.R2GC_568_1755,(0,1,0):C.R2GC_1305_480,(0,1,1):C.R2GC_1305_481,(2,1,0):C.R2GC_1305_480,(2,1,1):C.R2GC_1305_481,(6,1,0):C.R2GC_1908_1366,(6,1,1):C.R2GC_1908_1367,(6,1,2):C.R2GC_1682_1171,(8,1,0):C.R2GC_1909_1368,(8,1,1):C.R2GC_1909_1369,(8,1,2):C.R2GC_1683_1172,(5,1,0):C.R2GC_1302_474,(5,1,1):C.R2GC_1302_475,(1,1,0):C.R2GC_1302_474,(1,1,1):C.R2GC_1302_475,(7,1,0):C.R2GC_1304_478,(7,1,1):C.R2GC_1304_479,(4,1,0):C.R2GC_1302_474,(4,1,1):C.R2GC_1302_475,(3,1,0):C.R2GC_1302_474,(3,1,1):C.R2GC_1302_475,(0,2,0):C.R2GC_1305_480,(0,2,1):C.R2GC_1305_481,(2,2,0):C.R2GC_1305_480,(2,2,1):C.R2GC_1305_481,(7,2,0):C.R2GC_1907_1364,(7,2,1):C.R2GC_1907_1365,(7,2,2):C.R2GC_1682_1171,(8,2,0):C.R2GC_1907_1364,(8,2,1):C.R2GC_1907_1365,(8,2,2):C.R2GC_1682_1171,(5,2,0):C.R2GC_1302_474,(5,2,1):C.R2GC_1302_475,(1,2,0):C.R2GC_1302_474,(1,2,1):C.R2GC_1302_475,(4,2,0):C.R2GC_1302_474,(4,2,1):C.R2GC_1302_475,(3,2,0):C.R2GC_1302_474,(3,2,1):C.R2GC_1302_475})

V_7 = CTVertex(name = 'V_7',
               type = 'R2',
               particles = [ P.g, P.g, P.g, P.G0, P.G0 ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVVSS25 ],
               loop_particles = [ [ [P.g] ], [ [P.t] ] ],
               couplings = {(0,0,0):C.R2GC_1892_1326,(0,0,1):C.R2GC_1892_1327})

V_8 = CTVertex(name = 'V_8',
               type = 'R2',
               particles = [ P.g, P.g, P.g, P.G__minus__, P.G__plus__ ],
               color = [ 'd(1,2,3)', 'f(1,2,3)' ],
               lorentz = [ L.VVVSS20, L.VVVSS25, L.VVVSS26 ],
               loop_particles = [ [ [P.b] ], [ [P.b, P.t] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.g] ], [ [P.t] ] ],
               couplings = {(1,1,4):C.R2GC_1892_1326,(1,1,1):C.R2GC_1892_1327,(1,0,0):C.R2GC_1355_637,(1,0,2):C.R2GC_1355_638,(1,0,3):C.R2GC_1355_639,(1,0,5):C.R2GC_1355_640,(0,2,0):C.R2GC_1354_633,(0,2,2):C.R2GC_1354_634,(0,2,3):C.R2GC_1354_635,(0,2,5):C.R2GC_1354_636})

V_9 = CTVertex(name = 'V_9',
               type = 'R2',
               particles = [ P.g, P.g, P.g, P.H, P.H ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVVSS25 ],
               loop_particles = [ [ [P.g] ], [ [P.t] ] ],
               couplings = {(0,0,0):C.R2GC_1892_1326,(0,0,1):C.R2GC_1892_1327})

V_10 = CTVertex(name = 'V_10',
                type = 'R2',
                particles = [ P.g, P.g, P.g, P.H ],
                color = [ 'f(1,2,3)' ],
                lorentz = [ L.VVVS17 ],
                loop_particles = [ [ [P.g] ], [ [P.t] ] ],
                couplings = {(0,0,0):C.R2GC_1919_1374,(0,0,1):C.R2GC_1919_1375})

V_11 = CTVertex(name = 'V_11',
                type = 'R2',
                particles = [ P.g, P.g, P.g, P.g, P.H ],
                color = [ 'd(-1,1,3)*d(-1,2,4)', 'd(-1,1,3)*f(-1,2,4)', 'd(-1,1,4)*d(-1,2,3)', 'd(-1,1,4)*f(-1,2,3)', 'd(-1,2,3)*f(-1,1,4)', 'd(-1,2,4)*f(-1,1,3)', 'f(-1,1,2)*f(-1,3,4)', 'f(-1,1,3)*f(-1,2,4)', 'f(-1,1,4)*f(-1,2,3)', 'Identity(1,2)*Identity(3,4)', 'Identity(1,3)*Identity(2,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.VVVVS12, L.VVVVS13, L.VVVVS15, L.VVVVS20 ],
                loop_particles = [ [ [P.g] ], [ [P.t] ] ],
                couplings = {(0,0,0):C.R2GC_571_1758,(2,0,0):C.R2GC_571_1758,(6,0,0):C.R2GC_1924_1381,(6,0,1):C.R2GC_1923_1380,(7,0,0):C.R2GC_1923_1379,(7,0,1):C.R2GC_1923_1380,(5,0,0):C.R2GC_575_1762,(1,0,0):C.R2GC_575_1762,(4,0,0):C.R2GC_575_1762,(3,0,0):C.R2GC_575_1762,(8,0,0):C.R2GC_572_1759,(11,3,0):C.R2GC_574_1761,(10,3,0):C.R2GC_574_1761,(9,3,0):C.R2GC_573_1760,(0,1,0):C.R2GC_571_1758,(2,1,0):C.R2GC_571_1758,(6,1,0):C.R2GC_1922_1378,(6,1,1):C.R2GC_1921_1377,(8,1,0):C.R2GC_1923_1379,(8,1,1):C.R2GC_1923_1380,(5,1,0):C.R2GC_575_1762,(1,1,0):C.R2GC_575_1762,(7,1,0):C.R2GC_572_1759,(4,1,0):C.R2GC_575_1762,(3,1,0):C.R2GC_575_1762,(0,2,0):C.R2GC_571_1758,(2,2,0):C.R2GC_571_1758,(7,2,0):C.R2GC_1921_1376,(7,2,1):C.R2GC_1921_1377,(8,2,0):C.R2GC_1921_1376,(8,2,1):C.R2GC_1921_1377,(5,2,0):C.R2GC_575_1762,(1,2,0):C.R2GC_575_1762,(4,2,0):C.R2GC_575_1762,(3,2,0):C.R2GC_575_1762})

V_12 = CTVertex(name = 'V_12',
                type = 'R2',
                particles = [ P.b__tilde__, P.t, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS41, L.FFS42, L.FFS43, L.FFS45, L.FFS47, L.FFS48, L.FFS50 ],
                loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,0,2):C.R2GC_2085_1481,(0,0,3):C.R2GC_2425_1729,(0,0,1):[ C.R2GC_2425_1730, C.R2GC_2394_1688 ],(0,4,3):C.R2GC_889_1968,(0,5,1):C.R2GC_951_2002,(0,6,3):C.R2GC_871_1951,(0,1,0):C.R2GC_1931_1387,(0,3,2):C.R2GC_2079_1475,(0,3,1):C.R2GC_2418_1718,(0,2,2):C.R2GC_2063_1470})

V_13 = CTVertex(name = 'V_13',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS31, L.FFS32, L.FFS34, L.FFS36, L.FFS45, L.FFS55 ],
                loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                couplings = {(0,4,1):C.R2GC_1834_1270,(0,4,0):C.R2GC_1717_1199,(0,5,1):C.R2GC_1840_1272,(0,5,0):C.R2GC_1718_1200,(0,0,1):C.R2GC_1679_1169,(0,0,0):[ C.R2GC_2190_1550, C.R2GC_1715_1197 ],(0,1,1):C.R2GC_1832_1269,(0,2,0):C.R2GC_2187_1547,(0,3,0):C.R2GC_872_1952})

V_14 = CTVertex(name = 'V_14',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS33, L.FFS37, L.FFS38, L.FFS40 ],
                loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                couplings = {(0,1,1):C.R2GC_1842_1273,(0,1,0):[ C.R2GC_2258_1597, C.R2GC_1716_1198 ],(0,0,0):C.R2GC_2186_1546,(0,2,1):C.R2GC_1831_1268,(0,3,0):C.R2GC_873_1953})

V_15 = CTVertex(name = 'V_15',
                type = 'R2',
                particles = [ P.b__tilde__, P.t, P.G__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFSS27, L.FFSS29, L.FFSS30, L.FFSS31 ],
                loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,0,1):C.R2GC_2024_1436,(0,0,2):C.R2GC_2419_1719,(0,0,0):C.R2GC_2419_1720,(0,3,2):C.R2GC_907_1985,(0,1,0):C.R2GC_907_1985,(0,2,1):C.R2GC_1997_1410,(0,2,0):C.R2GC_2343_1654})

V_16 = CTVertex(name = 'V_16',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.G0, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFSS24, L.FFSS26, L.FFSS37, L.FFSS43 ],
                loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                couplings = {(0,1,1):C.R2GC_1821_1259,(0,1,0):C.R2GC_2227_1575,(0,2,0):C.R2GC_2221_1569,(0,3,0):C.R2GC_642_1824,(0,0,0):C.R2GC_2217_1565})

V_17 = CTVertex(name = 'V_17',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.G__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFSS25, L.FFSS26, L.FFSS30, L.FFSS37, L.FFSS38, L.FFSS41, L.FFSS42, L.FFSS43 ],
                loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,2,1):C.R2GC_1993_1406,(0,2,2):C.R2GC_1686_1175,(0,6,1):C.R2GC_2005_1418,(0,6,2):C.R2GC_1688_1177,(0,1,1):C.R2GC_2022_1434,(0,1,2):C.R2GC_2227_1575,(0,3,1):C.R2GC_2021_1433,(0,3,2):C.R2GC_2409_1703,(0,3,0):C.R2GC_2409_1704,(0,7,2):C.R2GC_642_1824,(0,0,1):C.R2GC_2001_1414,(0,5,0):C.R2GC_977_2023,(0,4,0):C.R2GC_2217_1565})

V_18 = CTVertex(name = 'V_18',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.G0, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFSS23, L.FFSS25, L.FFSS30, L.FFSS35, L.FFSS42 ],
                loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                couplings = {(0,2,1):C.R2GC_1811_1249,(0,2,0):C.R2GC_1687_1176,(0,4,1):C.R2GC_1817_1255,(0,4,0):C.R2GC_1689_1178,(0,0,1):C.R2GC_1820_1258,(0,0,0):C.R2GC_2229_1577,(0,3,1):C.R2GC_1819_1257,(0,3,0):C.R2GC_2223_1571,(0,1,1):C.R2GC_1812_1250,(0,1,0):C.R2GC_2218_1566})

V_19 = CTVertex(name = 'V_19',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.H, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFSS24, L.FFSS26, L.FFSS37, L.FFSS43 ],
                loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                couplings = {(0,1,1):C.R2GC_1822_1260,(0,1,0):C.R2GC_2228_1576,(0,2,0):C.R2GC_2222_1570,(0,3,0):C.R2GC_642_1824,(0,0,0):C.R2GC_2217_1565})

V_20 = CTVertex(name = 'V_20',
                type = 'R2',
                particles = [ P.t__tilde__, P.b, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS44, L.FFS45, L.FFS46, L.FFS52, L.FFS53, L.FFS54, L.FFS60 ],
                loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,3,2):C.R2GC_2084_1480,(0,3,3):C.R2GC_2424_1727,(0,3,1):[ C.R2GC_2424_1728, C.R2GC_2393_1687 ],(0,0,3):C.R2GC_888_1967,(0,2,1):C.R2GC_950_2001,(0,6,0):C.R2GC_650_1826,(0,4,3):C.R2GC_2133_1509,(0,1,2):C.R2GC_2079_1475,(0,1,1):C.R2GC_2418_1718,(0,5,2):C.R2GC_2064_1471})

V_21 = CTVertex(name = 'V_21',
                type = 'R2',
                particles = [ P.t__tilde__, P.b, P.G__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFSS30, L.FFSS33, L.FFSS34, L.FFSS36 ],
                loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,3,1):C.R2GC_2023_1435,(0,3,2):C.R2GC_2420_1721,(0,3,0):C.R2GC_2420_1722,(0,2,2):C.R2GC_905_1983,(0,1,0):C.R2GC_905_1983,(0,0,1):C.R2GC_1997_1410,(0,0,0):C.R2GC_2343_1654})

V_22 = CTVertex(name = 'V_22',
                type = 'R2',
                particles = [ P.vt__tilde__, P.ta__minus__, P.b__tilde__, P.t ],
                color = [ 'Identity(3,4)' ],
                lorentz = [ L.FFFF18 ],
                loop_particles = [ [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_2350_1659})

V_23 = CTVertex(name = 'V_23',
                type = 'R2',
                particles = [ P.ta__plus__, P.ta__minus__, P.b__tilde__, P.b ],
                color = [ 'Identity(3,4)' ],
                lorentz = [ L.FFFF18, L.FFFF24 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_1443_935,(0,1,0):C.R2GC_1692_1181})

V_24 = CTVertex(name = 'V_24',
                type = 'R2',
                particles = [ P.t__tilde__, P.b, P.ta__plus__, P.vt ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFFF18 ],
                loop_particles = [ [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_2350_1659})

V_25 = CTVertex(name = 'V_25',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.u__tilde__, P.u ],
                color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
                loop_particles = [ [ [P.g, P.t], [P.g, P.u] ], [ [P.g, P.t, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_2327_1638,(0,0,1):C.R2GC_2327_1639,(0,2,0):C.R2GC_2307_1616,(0,2,1):C.R2GC_2307_1617,(0,3,0):C.R2GC_2333_1646,(0,3,1):C.R2GC_2333_1647,(0,1,0):C.R2GC_2331_1642,(0,1,1):C.R2GC_2331_1643,(1,0,0):C.R2GC_2328_1640,(1,0,1):C.R2GC_2328_1641,(1,2,0):C.R2GC_2308_1618,(1,2,1):C.R2GC_2308_1619,(1,3,0):C.R2GC_2334_1648,(1,3,1):C.R2GC_2334_1649,(1,1,0):C.R2GC_2332_1644,(1,1,1):C.R2GC_2332_1645})

V_26 = CTVertex(name = 'V_26',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.u__tilde__, P.u ],
                color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.FFFF18, L.FFFF28 ],
                loop_particles = [ [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_2305_1612,(0,0,1):C.R2GC_2305_1613,(0,1,0):C.R2GC_2307_1616,(0,1,1):C.R2GC_2307_1617,(1,0,0):C.R2GC_2306_1614,(1,0,1):C.R2GC_2306_1615,(1,1,0):C.R2GC_2308_1618,(1,1,1):C.R2GC_2308_1619})

V_27 = CTVertex(name = 'V_27',
                type = 'R2',
                particles = [ P.t__tilde__, P.b, P.d__tilde__, P.u ],
                color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.FFFF18 ],
                loop_particles = [ [ [P.b, P.d, P.g], [P.g, P.t, P.u] ], [ [P.b, P.g, P.t], [P.d, P.g, P.u] ], [ [P.b, P.g, P.u], [P.d, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_2447_1742,(0,0,1):C.R2GC_2447_1743,(0,0,2):C.R2GC_2447_1741,(1,0,0):C.R2GC_2445_1739,(1,0,1):C.R2GC_2445_1740,(1,0,2):C.R2GC_2445_1738})

V_28 = CTVertex(name = 'V_28',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.t__tilde__, P.t ],
                color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
                loop_particles = [ [ [P.c, P.g], [P.g, P.t] ], [ [P.c, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_2327_1638,(0,0,1):C.R2GC_2327_1639,(0,1,0):C.R2GC_2307_1616,(0,1,1):C.R2GC_2307_1617,(0,2,0):C.R2GC_2331_1642,(0,2,1):C.R2GC_2331_1643,(0,3,0):C.R2GC_2333_1646,(0,3,1):C.R2GC_2333_1647,(1,0,0):C.R2GC_2328_1640,(1,0,1):C.R2GC_2328_1641,(1,1,0):C.R2GC_2308_1618,(1,1,1):C.R2GC_2308_1619,(1,2,0):C.R2GC_2332_1644,(1,2,1):C.R2GC_2332_1645,(1,3,0):C.R2GC_2334_1648,(1,3,1):C.R2GC_2334_1649})

V_29 = CTVertex(name = 'V_29',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.c__tilde__, P.c ],
                color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.FFFF18, L.FFFF28 ],
                loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g], [P.c, P.g] ] ],
                couplings = {(0,0,1):C.R2GC_2305_1612,(0,0,0):C.R2GC_2305_1613,(0,1,1):C.R2GC_2307_1616,(0,1,0):C.R2GC_2307_1617,(1,0,1):C.R2GC_2306_1614,(1,0,0):C.R2GC_2306_1615,(1,1,1):C.R2GC_2308_1618,(1,1,0):C.R2GC_2308_1619})

V_30 = CTVertex(name = 'V_30',
                type = 'R2',
                particles = [ P.t__tilde__, P.b, P.s__tilde__, P.c ],
                color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.FFFF18 ],
                loop_particles = [ [ [P.b, P.c, P.g], [P.g, P.s, P.t] ], [ [P.b, P.g, P.s], [P.c, P.g, P.t] ], [ [P.b, P.g, P.t], [P.c, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_2447_1741,(0,0,1):C.R2GC_2447_1742,(0,0,2):C.R2GC_2447_1743,(1,0,0):C.R2GC_2445_1738,(1,0,1):C.R2GC_2445_1739,(1,0,2):C.R2GC_2445_1740})

V_31 = CTVertex(name = 'V_31',
                type = 'R2',
                particles = [ P.u__tilde__, P.d, P.b__tilde__, P.t ],
                color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.FFFF18 ],
                loop_particles = [ [ [P.b, P.d, P.g], [P.g, P.t, P.u] ], [ [P.b, P.g, P.t], [P.d, P.g, P.u] ], [ [P.b, P.g, P.u], [P.d, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_2447_1742,(0,0,1):C.R2GC_2447_1743,(0,0,2):C.R2GC_2447_1741,(1,0,0):C.R2GC_2445_1739,(1,0,1):C.R2GC_2445_1740,(1,0,2):C.R2GC_2445_1738})

V_32 = CTVertex(name = 'V_32',
                type = 'R2',
                particles = [ P.d__tilde__, P.d, P.t__tilde__, P.t ],
                color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
                loop_particles = [ [ [P.d, P.g], [P.g, P.t] ], [ [P.d, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_2305_1612,(0,0,1):C.R2GC_2305_1613,(0,1,0):C.R2GC_2335_1650,(0,1,1):C.R2GC_2335_1651,(0,2,0):C.R2GC_2331_1642,(0,2,1):C.R2GC_2331_1643,(0,3,0):C.R2GC_2357_1671,(0,3,1):C.R2GC_2357_1672,(1,0,0):C.R2GC_2306_1614,(1,0,1):C.R2GC_2306_1615,(1,1,0):C.R2GC_2336_1652,(1,1,1):C.R2GC_2336_1653,(1,2,0):C.R2GC_2332_1644,(1,2,1):C.R2GC_2332_1645,(1,3,0):C.R2GC_2358_1673,(1,3,1):C.R2GC_2358_1674})

V_33 = CTVertex(name = 'V_33',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.d__tilde__, P.d ],
                color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.FFFF18, L.FFFF28 ],
                loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.g], [P.d, P.g] ] ],
                couplings = {(0,0,1):C.R2GC_2327_1638,(0,0,0):C.R2GC_2327_1639,(0,1,1):C.R2GC_2335_1650,(0,1,0):C.R2GC_2335_1651,(1,0,1):C.R2GC_2328_1640,(1,0,0):C.R2GC_2328_1641,(1,1,1):C.R2GC_2336_1652,(1,1,0):C.R2GC_2336_1653})

V_34 = CTVertex(name = 'V_34',
                type = 'R2',
                particles = [ P.c__tilde__, P.s, P.b__tilde__, P.t ],
                color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.FFFF18 ],
                loop_particles = [ [ [P.b, P.c, P.g], [P.g, P.s, P.t] ], [ [P.b, P.g, P.s], [P.c, P.g, P.t] ], [ [P.b, P.g, P.t], [P.c, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_2447_1741,(0,0,1):C.R2GC_2447_1742,(0,0,2):C.R2GC_2447_1743,(1,0,0):C.R2GC_2445_1738,(1,0,1):C.R2GC_2445_1739,(1,0,2):C.R2GC_2445_1740})

V_35 = CTVertex(name = 'V_35',
                type = 'R2',
                particles = [ P.s__tilde__, P.s, P.t__tilde__, P.t ],
                color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
                loop_particles = [ [ [P.g, P.s], [P.g, P.t] ], [ [P.g, P.s, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_2305_1612,(0,0,1):C.R2GC_2305_1613,(0,1,0):C.R2GC_2335_1650,(0,1,1):C.R2GC_2335_1651,(0,2,0):C.R2GC_2331_1642,(0,2,1):C.R2GC_2331_1643,(0,3,0):C.R2GC_2357_1671,(0,3,1):C.R2GC_2357_1672,(1,0,0):C.R2GC_2306_1614,(1,0,1):C.R2GC_2306_1615,(1,1,0):C.R2GC_2336_1652,(1,1,1):C.R2GC_2336_1653,(1,2,0):C.R2GC_2332_1644,(1,2,1):C.R2GC_2332_1645,(1,3,0):C.R2GC_2358_1673,(1,3,1):C.R2GC_2358_1674})

V_36 = CTVertex(name = 'V_36',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.s__tilde__, P.s ],
                color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.FFFF18, L.FFFF28 ],
                loop_particles = [ [ [P.b, P.g], [P.g, P.s] ], [ [P.b, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_2327_1638,(0,0,1):C.R2GC_2327_1639,(0,1,0):C.R2GC_2335_1650,(0,1,1):C.R2GC_2335_1651,(1,0,0):C.R2GC_2328_1640,(1,0,1):C.R2GC_2328_1641,(1,1,0):C.R2GC_2336_1652,(1,1,1):C.R2GC_2336_1653})

V_37 = CTVertex(name = 'V_37',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.t__tilde__, P.t ],
                color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.FFFF17, L.FFFF18, L.FFFF24, L.FFFF25, L.FFFF26, L.FFFF27, L.FFFF28, L.FFFF29 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(1,0,0):C.R2GC_1929_1385,(0,1,0):C.R2GC_1929_1385,(0,2,0):C.R2GC_2130_1507,(1,5,0):C.R2GC_2130_1507,(1,3,0):C.R2GC_2130_1507,(1,4,0):C.R2GC_2135_1510,(0,6,0):C.R2GC_2130_1507,(0,7,0):C.R2GC_2135_1510,(0,0,0):C.R2GC_1930_1386,(1,1,0):C.R2GC_1930_1386,(1,2,0):C.R2GC_2131_1508,(0,5,0):C.R2GC_2131_1508,(0,3,0):C.R2GC_2131_1508,(0,4,0):C.R2GC_2136_1511,(1,6,0):C.R2GC_2131_1508,(1,7,0):C.R2GC_2136_1511})

V_38 = CTVertex(name = 'V_38',
                type = 'R2',
                particles = [ P.t__tilde__, P.b, P.b__tilde__, P.t ],
                color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.FFFF17, L.FFFF18, L.FFFF25, L.FFFF26, L.FFFF27 ],
                loop_particles = [ [ [P.b, P.g], [P.g, P.t] ], [ [P.b, P.g, P.t] ] ],
                couplings = {(1,0,0):C.R2GC_2351_1660,(1,0,1):C.R2GC_2351_1661,(0,1,0):C.R2GC_2354_1665,(0,1,1):C.R2GC_2354_1666,(1,4,0):C.R2GC_2335_1650,(1,4,1):C.R2GC_2335_1651,(1,2,0):C.R2GC_2355_1667,(1,2,1):C.R2GC_2355_1668,(1,3,0):C.R2GC_2357_1671,(1,3,1):C.R2GC_2357_1672,(0,0,1):C.R2GC_2352_1662,(1,1,0):C.R2GC_2353_1663,(1,1,1):C.R2GC_2353_1664,(0,4,0):C.R2GC_2336_1652,(0,4,1):C.R2GC_2336_1653,(0,2,0):C.R2GC_2356_1669,(0,2,1):C.R2GC_2356_1670,(0,3,0):C.R2GC_2358_1673,(0,3,1):C.R2GC_2358_1674})

V_39 = CTVertex(name = 'V_39',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.b__tilde__, P.b ],
                color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.FFFF17, L.FFFF18, L.FFFF24, L.FFFF25, L.FFFF27, L.FFFF28 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(1,0,0):C.R2GC_1929_1385,(0,1,0):C.R2GC_1929_1385,(0,2,0):C.R2GC_1927_1383,(1,4,0):C.R2GC_1927_1383,(1,3,0):C.R2GC_1927_1383,(0,5,0):C.R2GC_1927_1383,(0,0,0):C.R2GC_1930_1386,(1,1,0):C.R2GC_1930_1386,(1,2,0):C.R2GC_1928_1384,(0,4,0):C.R2GC_1928_1384,(0,3,0):C.R2GC_1928_1384,(1,5,0):C.R2GC_1928_1384})

V_40 = CTVertex(name = 'V_40',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.ve__tilde__, P.ve ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFFF18, L.FFFF24 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_1441_933,(0,1,0):C.R2GC_1702_1188})

V_41 = CTVertex(name = 'V_41',
                type = 'R2',
                particles = [ P.ve__tilde__, P.e__minus__, P.b__tilde__, P.t ],
                color = [ 'Identity(3,4)' ],
                lorentz = [ L.FFFF18 ],
                loop_particles = [ [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_2348_1657})

V_42 = CTVertex(name = 'V_42',
                type = 'R2',
                particles = [ P.t__tilde__, P.b, P.e__plus__, P.ve ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFFF18 ],
                loop_particles = [ [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_2348_1657})

V_43 = CTVertex(name = 'V_43',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.e__plus__, P.e__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFFF18, L.FFFF28 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_1441_933,(0,1,0):C.R2GC_1690_1179})

V_44 = CTVertex(name = 'V_44',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.vm__tilde__, P.vm ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFFF18, L.FFFF24 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_1442_934,(0,1,0):C.R2GC_1703_1189})

V_45 = CTVertex(name = 'V_45',
                type = 'R2',
                particles = [ P.vm__tilde__, P.mu__minus__, P.b__tilde__, P.t ],
                color = [ 'Identity(3,4)' ],
                lorentz = [ L.FFFF18 ],
                loop_particles = [ [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_2349_1658})

V_46 = CTVertex(name = 'V_46',
                type = 'R2',
                particles = [ P.t__tilde__, P.b, P.mu__plus__, P.vm ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFFF18 ],
                loop_particles = [ [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_2349_1658})

V_47 = CTVertex(name = 'V_47',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.mu__plus__, P.mu__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFFF18, L.FFFF28 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_1442_934,(0,1,0):C.R2GC_1691_1180})

V_48 = CTVertex(name = 'V_48',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.vt__tilde__, P.vt ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFFF18, L.FFFF24 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_1443_935,(0,1,0):C.R2GC_1704_1190})

V_49 = CTVertex(name = 'V_49',
                type = 'R2',
                particles = [ P.e__plus__, P.e__minus__, P.t__tilde__, P.t ],
                color = [ 'Identity(3,4)' ],
                lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_1693_1182,(0,1,0):C.R2GC_1690_1179,(0,2,0):C.R2GC_1702_1188,(0,3,0):C.R2GC_1699_1185})

V_50 = CTVertex(name = 'V_50',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.ve__tilde__, P.ve ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFFF18 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_1693_1182})

V_51 = CTVertex(name = 'V_51',
                type = 'R2',
                particles = [ P.mu__plus__, P.mu__minus__, P.t__tilde__, P.t ],
                color = [ 'Identity(3,4)' ],
                lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_1695_1183,(0,1,0):C.R2GC_1691_1180,(0,2,0):C.R2GC_1703_1189,(0,3,0):C.R2GC_1700_1186})

V_52 = CTVertex(name = 'V_52',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.vm__tilde__, P.vm ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFFF18 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_1695_1183})

V_53 = CTVertex(name = 'V_53',
                type = 'R2',
                particles = [ P.ta__plus__, P.ta__minus__, P.t__tilde__, P.t ],
                color = [ 'Identity(3,4)' ],
                lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_1697_1184,(0,1,0):C.R2GC_1692_1181,(0,2,0):C.R2GC_1704_1190,(0,3,0):C.R2GC_1701_1187})

V_54 = CTVertex(name = 'V_54',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.vt__tilde__, P.vt ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFFF18 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_1697_1184})

V_55 = CTVertex(name = 'V_55',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV57, L.FFV70, L.FFV71, L.FFV82, L.FFV92 ],
                loop_particles = [ [ [P.b] ], [ [P.g, P.u] ], [ [P.t] ] ],
                couplings = {(0,1,2):C.R2GC_1154_257,(0,0,1):[ C.R2GC_1685_1174, C.R2GC_1466_958 ],(0,3,2):C.R2GC_1156_259,(0,2,0):C.R2GC_1011_15,(0,2,2):C.R2GC_1011_16,(0,4,0):C.R2GC_1012_17,(0,4,2):C.R2GC_1012_18})

V_56 = CTVertex(name = 'V_56',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV57, L.FFV70, L.FFV71, L.FFV82, L.FFV92 ],
                loop_particles = [ [ [P.b] ], [ [P.c, P.g] ], [ [P.t] ] ],
                couplings = {(0,1,2):C.R2GC_1154_257,(0,0,1):[ C.R2GC_1685_1174, C.R2GC_1466_958 ],(0,3,2):C.R2GC_1156_259,(0,2,0):C.R2GC_1011_15,(0,2,2):C.R2GC_1011_16,(0,4,0):C.R2GC_1012_17,(0,4,2):C.R2GC_1012_18})

V_57 = CTVertex(name = 'V_57',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV57, L.FFV61, L.FFV67, L.FFV68, L.FFV70, L.FFV71, L.FFV73, L.FFV75, L.FFV82, L.FFV87, L.FFV92, L.FFV94, L.FFV98 ],
                loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.g, P.t] ], [ [P.t] ] ],
                couplings = {(0,9,4):C.R2GC_1151_254,(0,4,4):C.R2GC_1150_253,(0,0,3):[ C.R2GC_1685_1174, C.R2GC_2247_1591 ],(0,2,4):C.R2GC_1143_246,(0,8,4):C.R2GC_598_1782,(0,6,0):C.R2GC_1855_1283,(0,7,4):C.R2GC_1007_10,(0,12,4):C.R2GC_577_1764,(0,5,0):C.R2GC_1307_485,(0,5,1):C.R2GC_1307_486,(0,5,2):C.R2GC_1307_487,(0,11,4):C.R2GC_576_1763,(0,10,0):C.R2GC_1308_488,(0,10,1):C.R2GC_1308_489,(0,10,2):C.R2GC_1308_490,(0,3,3):C.R2GC_910_1987,(0,1,3):C.R2GC_911_1988})

V_58 = CTVertex(name = 'V_58',
                type = 'R2',
                particles = [ P.d__tilde__, P.d, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV57, L.FFV70, L.FFV71, L.FFV82, L.FFV92 ],
                loop_particles = [ [ [P.b] ], [ [P.d, P.g] ], [ [P.t] ] ],
                couplings = {(0,1,2):C.R2GC_1155_258,(0,0,1):[ C.R2GC_639_1821, C.R2GC_1452_944 ],(0,3,2):C.R2GC_1153_256,(0,2,0):C.R2GC_1010_13,(0,2,2):C.R2GC_1010_14,(0,4,0):C.R2GC_1009_12,(0,4,2):C.R2GC_1008_11})

V_59 = CTVertex(name = 'V_59',
                type = 'R2',
                particles = [ P.s__tilde__, P.s, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV57, L.FFV70, L.FFV71, L.FFV82, L.FFV92 ],
                loop_particles = [ [ [P.b] ], [ [P.g, P.s] ], [ [P.t] ] ],
                couplings = {(0,1,2):C.R2GC_1155_258,(0,0,1):[ C.R2GC_639_1821, C.R2GC_1452_944 ],(0,3,2):C.R2GC_1153_256,(0,2,0):C.R2GC_1010_13,(0,2,2):C.R2GC_1010_14,(0,4,0):C.R2GC_1009_12,(0,4,2):C.R2GC_1008_11})

V_60 = CTVertex(name = 'V_60',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV57, L.FFV58, L.FFV70, L.FFV71, L.FFV73, L.FFV75, L.FFV82, L.FFV92 ],
                loop_particles = [ [ [P.b] ], [ [P.b, P.g] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                couplings = {(0,2,4):C.R2GC_1152_255,(0,0,1):[ C.R2GC_639_1821, C.R2GC_1452_944 ],(0,6,4):C.R2GC_1153_256,(0,5,0):C.R2GC_991_2033,(0,4,4):C.R2GC_1856_1284,(0,1,0):C.R2GC_1009_12,(0,3,2):C.R2GC_1306_482,(0,3,3):C.R2GC_1306_483,(0,3,4):C.R2GC_1306_484,(0,7,4):C.R2GC_1008_11})

V_61 = CTVertex(name = 'V_61',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV57, L.FFV70, L.FFV71, L.FFV82, L.FFV92 ],
                loop_particles = [ [ [P.b] ], [ [P.g, P.u] ], [ [P.t] ] ],
                couplings = {(0,0,1):C.R2GC_1926_1382,(0,1,2):C.R2GC_1160_263,(0,3,2):C.R2GC_1162_265,(0,2,0):C.R2GC_1029_42,(0,2,2):C.R2GC_1029_43,(0,4,0):C.R2GC_1030_44,(0,4,2):C.R2GC_1030_45})

V_62 = CTVertex(name = 'V_62',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV57, L.FFV70, L.FFV71, L.FFV82, L.FFV92 ],
                loop_particles = [ [ [P.b] ], [ [P.c, P.g] ], [ [P.t] ] ],
                couplings = {(0,0,1):C.R2GC_1926_1382,(0,1,2):C.R2GC_1160_263,(0,3,2):C.R2GC_1162_265,(0,2,0):C.R2GC_1029_42,(0,2,2):C.R2GC_1029_43,(0,4,0):C.R2GC_1030_44,(0,4,2):C.R2GC_1030_45})

V_63 = CTVertex(name = 'V_63',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV57, L.FFV64, L.FFV67, L.FFV70, L.FFV71, L.FFV74, L.FFV82, L.FFV87, L.FFV92, L.FFV94, L.FFV97 ],
                loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.g, P.t] ], [ [P.t] ] ],
                couplings = {(0,7,4):C.R2GC_599_1783,(0,0,3):[ C.R2GC_1926_1382, C.R2GC_1674_1167 ],(0,2,4):C.R2GC_1145_248,(0,3,4):C.R2GC_1157_260,(0,6,4):C.R2GC_600_1784,(0,5,0):C.R2GC_1024_36,(0,5,4):C.R2GC_1024_37,(0,10,4):C.R2GC_583_1769,(0,4,0):C.R2GC_1027_39,(0,4,1):C.R2GC_1316_515,(0,4,2):C.R2GC_1316_516,(0,9,4):C.R2GC_1315_514,(0,8,0):C.R2GC_1317_517,(0,8,1):C.R2GC_1317_518,(0,8,2):C.R2GC_1317_519,(0,1,3):C.R2GC_912_1989})

V_64 = CTVertex(name = 'V_64',
                type = 'R2',
                particles = [ P.d__tilde__, P.d, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV57, L.FFV70, L.FFV71, L.FFV82, L.FFV92 ],
                loop_particles = [ [ [P.b] ], [ [P.d, P.g] ], [ [P.t] ] ],
                couplings = {(0,0,1):C.R2GC_1926_1382,(0,1,2):C.R2GC_1161_264,(0,3,2):C.R2GC_1159_262,(0,2,0):C.R2GC_1028_40,(0,2,2):C.R2GC_1028_41,(0,4,0):C.R2GC_1027_39,(0,4,2):C.R2GC_1026_38})

V_65 = CTVertex(name = 'V_65',
                type = 'R2',
                particles = [ P.s__tilde__, P.s, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV57, L.FFV70, L.FFV71, L.FFV82, L.FFV92 ],
                loop_particles = [ [ [P.b] ], [ [P.g, P.s] ], [ [P.t] ] ],
                couplings = {(0,0,1):C.R2GC_1926_1382,(0,1,2):C.R2GC_1161_264,(0,3,2):C.R2GC_1159_262,(0,2,0):C.R2GC_1028_40,(0,2,2):C.R2GC_1028_41,(0,4,0):C.R2GC_1027_39,(0,4,2):C.R2GC_1026_38})

V_66 = CTVertex(name = 'V_66',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV57, L.FFV58, L.FFV70, L.FFV71, L.FFV74, L.FFV82, L.FFV92 ],
                loop_particles = [ [ [P.b] ], [ [P.b, P.g] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                couplings = {(0,0,1):C.R2GC_1926_1382,(0,2,4):C.R2GC_1158_261,(0,5,4):C.R2GC_1159_262,(0,4,0):C.R2GC_1024_37,(0,4,4):C.R2GC_1024_36,(0,1,0):C.R2GC_1027_39,(0,3,2):C.R2GC_1315_512,(0,3,3):C.R2GC_1315_513,(0,3,4):C.R2GC_1315_514,(0,6,4):C.R2GC_1026_38})

V_67 = CTVertex(name = 'V_67',
                type = 'R2',
                particles = [ P.d__tilde__, P.u, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV70, L.FFV71 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_1991_1404,(0,0,1):[ C.R2GC_2315_1626, C.R2GC_2326_1637 ],(0,1,0):C.R2GC_1600_1103})

V_68 = CTVertex(name = 'V_68',
                type = 'R2',
                particles = [ P.s__tilde__, P.c, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV70, L.FFV71 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_1991_1404,(0,0,1):[ C.R2GC_2315_1626, C.R2GC_2326_1637 ],(0,1,0):C.R2GC_1600_1103})

V_69 = CTVertex(name = 'V_69',
                type = 'R2',
                particles = [ P.b__tilde__, P.t, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV70, L.FFV71, L.FFV72, L.FFV74, L.FFV77, L.FFV79 ],
                loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                couplings = {(0,0,1):C.R2GC_1992_1405,(0,0,0):[ C.R2GC_2315_1626, C.R2GC_2434_1737 ],(0,5,0):C.R2GC_986_2028,(0,4,1):C.R2GC_1511_982,(0,3,1):C.R2GC_808_1906,(0,2,1):C.R2GC_810_1907,(0,1,2):C.R2GC_1600_1103})

V_70 = CTVertex(name = 'V_70',
                type = 'R2',
                particles = [ P.u__tilde__, P.d, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV70, L.FFV71 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_1991_1404,(0,0,1):[ C.R2GC_2315_1626, C.R2GC_2326_1637 ],(0,1,0):C.R2GC_1600_1103})

V_71 = CTVertex(name = 'V_71',
                type = 'R2',
                particles = [ P.c__tilde__, P.s, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV70, L.FFV71 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_1991_1404,(0,0,1):[ C.R2GC_2315_1626, C.R2GC_2326_1637 ],(0,1,0):C.R2GC_1600_1103})

V_72 = CTVertex(name = 'V_72',
                type = 'R2',
                particles = [ P.t__tilde__, P.b, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV106, L.FFV70, L.FFV71, L.FFV72, L.FFV74, L.FFV93 ],
                loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                couplings = {(0,1,1):C.R2GC_1992_1405,(0,1,0):[ C.R2GC_2315_1626, C.R2GC_2434_1737 ],(0,0,0):C.R2GC_987_2029,(0,4,1):C.R2GC_808_1906,(0,3,1):C.R2GC_810_1907,(0,5,1):C.R2GC_1510_981,(0,2,2):C.R2GC_1600_1103})

V_73 = CTVertex(name = 'V_73',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV57, L.FFV70, L.FFV71, L.FFV82, L.FFV87, L.FFV88, L.FFV92 ],
                loop_particles = [ [ [P.b] ], [ [P.g, P.u] ], [ [P.t] ] ],
                couplings = {(0,1,2):C.R2GC_1801_1245,(0,1,1):[ C.R2GC_1707_1193, C.R2GC_1465_957 ],(0,4,1):[ C.R2GC_1713_1195, C.R2GC_1467_959 ],(0,3,2):C.R2GC_1804_1248,(0,3,1):C.R2GC_700_1868,(0,0,1):C.R2GC_1724_1202,(0,5,1):C.R2GC_1725_1203,(0,2,0):C.R2GC_1168_273,(0,2,2):C.R2GC_1168_274,(0,6,0):C.R2GC_1169_275,(0,6,2):C.R2GC_1169_276})

V_74 = CTVertex(name = 'V_74',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV57, L.FFV70, L.FFV71, L.FFV82, L.FFV87, L.FFV88, L.FFV92 ],
                loop_particles = [ [ [P.b] ], [ [P.c, P.g] ], [ [P.t] ] ],
                couplings = {(0,1,2):C.R2GC_1801_1245,(0,1,1):[ C.R2GC_1707_1193, C.R2GC_1465_957 ],(0,4,1):[ C.R2GC_1713_1195, C.R2GC_1467_959 ],(0,3,2):C.R2GC_1804_1248,(0,3,1):C.R2GC_700_1868,(0,0,1):C.R2GC_1724_1202,(0,5,1):C.R2GC_1725_1203,(0,2,0):C.R2GC_1168_273,(0,2,2):C.R2GC_1168_274,(0,6,0):C.R2GC_1169_275,(0,6,2):C.R2GC_1169_276})

V_75 = CTVertex(name = 'V_75',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV103, L.FFV104, L.FFV107, L.FFV57, L.FFV61, L.FFV67, L.FFV68, L.FFV70, L.FFV71, L.FFV73, L.FFV75, L.FFV82, L.FFV84, L.FFV87, L.FFV88, L.FFV92, L.FFV94, L.FFV98 ],
                loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.g, P.t] ], [ [P.t] ] ],
                couplings = {(0,7,4):C.R2GC_1797_1242,(0,7,3):[ C.R2GC_1707_1193, C.R2GC_1726_1204 ],(0,13,4):C.R2GC_1227_381,(0,13,3):[ C.R2GC_1713_1195, C.R2GC_2256_1595 ],(0,11,4):C.R2GC_1803_1247,(0,11,3):C.R2GC_1727_1205,(0,3,3):C.R2GC_1724_1202,(0,5,4):C.R2GC_612_1795,(0,14,3):C.R2GC_1725_1203,(0,0,4):C.R2GC_1163_266,(0,1,4):C.R2GC_611_1794,(0,9,0):C.R2GC_1915_1372,(0,10,4):C.R2GC_1164_267,(0,17,4):C.R2GC_602_1786,(0,8,0):C.R2GC_1376_702,(0,8,1):C.R2GC_1376_703,(0,8,2):C.R2GC_1376_704,(0,16,4):C.R2GC_601_1785,(0,15,0):C.R2GC_1377_705,(0,15,1):C.R2GC_1377_706,(0,15,2):C.R2GC_1377_707,(0,15,4):C.R2GC_1377_708,(0,6,3):C.R2GC_917_1994,(0,12,3):C.R2GC_916_1993,(0,2,3):C.R2GC_914_1991,(0,4,3):C.R2GC_918_1995})

V_76 = CTVertex(name = 'V_76',
                type = 'R2',
                particles = [ P.d__tilde__, P.d, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV57, L.FFV70, L.FFV71, L.FFV82, L.FFV87, L.FFV91, L.FFV92 ],
                loop_particles = [ [ [P.b] ], [ [P.d, P.g] ], [ [P.t] ] ],
                couplings = {(0,1,2):C.R2GC_1802_1246,(0,1,1):[ C.R2GC_658_1834, C.R2GC_1479_965 ],(0,4,1):[ C.R2GC_663_1839, C.R2GC_1453_945 ],(0,3,2):C.R2GC_1800_1244,(0,3,1):C.R2GC_673_1849,(0,0,1):C.R2GC_672_1848,(0,5,1):C.R2GC_674_1850,(0,2,0):C.R2GC_1167_271,(0,2,2):C.R2GC_1167_272,(0,6,0):C.R2GC_1166_270,(0,6,2):C.R2GC_1165_269})

V_77 = CTVertex(name = 'V_77',
                type = 'R2',
                particles = [ P.s__tilde__, P.s, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV57, L.FFV70, L.FFV71, L.FFV82, L.FFV87, L.FFV91, L.FFV92 ],
                loop_particles = [ [ [P.b] ], [ [P.g, P.s] ], [ [P.t] ] ],
                couplings = {(0,1,2):C.R2GC_1802_1246,(0,1,1):[ C.R2GC_658_1834, C.R2GC_1479_965 ],(0,4,1):[ C.R2GC_663_1839, C.R2GC_1453_945 ],(0,3,2):C.R2GC_1800_1244,(0,3,1):C.R2GC_673_1849,(0,0,1):C.R2GC_672_1848,(0,5,1):C.R2GC_674_1850,(0,2,0):C.R2GC_1167_271,(0,2,2):C.R2GC_1167_272,(0,6,0):C.R2GC_1166_270,(0,6,2):C.R2GC_1165_269})

V_78 = CTVertex(name = 'V_78',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV57, L.FFV70, L.FFV71, L.FFV73, L.FFV75, L.FFV82, L.FFV87, L.FFV91, L.FFV92, L.FFV94 ],
                loop_particles = [ [ [P.b] ], [ [P.b, P.g] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                couplings = {(0,1,4):C.R2GC_1799_1243,(0,1,1):[ C.R2GC_658_1834, C.R2GC_1451_943 ],(0,6,1):[ C.R2GC_663_1839, C.R2GC_1453_945 ],(0,5,4):C.R2GC_1800_1244,(0,5,1):C.R2GC_673_1849,(0,0,1):C.R2GC_672_1848,(0,7,1):C.R2GC_674_1850,(0,4,0):C.R2GC_997_2037,(0,3,4):C.R2GC_1916_1373,(0,9,0):C.R2GC_558_1746,(0,2,2):C.R2GC_1375_699,(0,2,3):C.R2GC_1375_700,(0,2,4):C.R2GC_1375_701,(0,8,0):C.R2GC_1165_268,(0,8,4):C.R2GC_1165_269})

V_79 = CTVertex(name = 'V_79',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.a, P.G__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g] ] ],
                couplings = {(0,1,0):C.R2GC_2018_1430,(0,1,1):C.R2GC_1455_947,(0,2,0):C.R2GC_2020_1432,(0,2,1):C.R2GC_682_1855,(0,0,1):C.R2GC_681_1854})

V_80 = CTVertex(name = 'V_80',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.G0, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFSS30, L.FFSS42 ],
                loop_particles = [ [ [P.c, P.g] ], [ [P.t] ] ],
                couplings = {(0,0,1):C.R2GC_1815_1253,(0,0,0):C.R2GC_678_1851,(0,1,1):C.R2GC_1818_1256,(0,1,0):C.R2GC_679_1852})

V_81 = CTVertex(name = 'V_81',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS45, L.FFS55 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_691_1861,(0,1,0):C.R2GC_692_1862})

V_82 = CTVertex(name = 'V_82',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.G__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFSS24, L.FFSS30, L.FFSS42 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g] ] ],
                couplings = {(0,1,0):C.R2GC_2003_1416,(0,1,1):C.R2GC_1454_946,(0,2,0):C.R2GC_2006_1419,(0,2,1):C.R2GC_680_1853,(0,0,1):C.R2GC_642_1824})

V_83 = CTVertex(name = 'V_83',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.a, P.G__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14, L.FFVSS19 ],
                loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,1,1):C.R2GC_2007_1420,(0,1,2):C.R2GC_1705_1191,(0,2,1):C.R2GC_2019_1431,(0,2,2):C.R2GC_2403_1695,(0,2,0):C.R2GC_2403_1696,(0,0,2):C.R2GC_681_1854,(0,3,1):C.R2GC_841_1925})

V_84 = CTVertex(name = 'V_84',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.a, P.G__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.u] ] ],
                couplings = {(0,1,0):C.R2GC_2018_1430,(0,1,1):C.R2GC_1455_947,(0,2,0):C.R2GC_2020_1432,(0,2,1):C.R2GC_682_1855,(0,0,1):C.R2GC_681_1854})

V_85 = CTVertex(name = 'V_85',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.G0, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFSS30, L.FFSS42 ],
                loop_particles = [ [ [P.g, P.u] ], [ [P.t] ] ],
                couplings = {(0,0,1):C.R2GC_1815_1253,(0,0,0):C.R2GC_678_1851,(0,1,1):C.R2GC_1818_1256,(0,1,0):C.R2GC_679_1852})

V_86 = CTVertex(name = 'V_86',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS45, L.FFS55 ],
                loop_particles = [ [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_691_1861,(0,1,0):C.R2GC_692_1862})

V_87 = CTVertex(name = 'V_87',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.G__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFSS24, L.FFSS30, L.FFSS42 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.u] ] ],
                couplings = {(0,1,0):C.R2GC_2003_1416,(0,1,1):C.R2GC_1454_946,(0,2,0):C.R2GC_2006_1419,(0,2,1):C.R2GC_680_1853,(0,0,1):C.R2GC_642_1824})

V_88 = CTVertex(name = 'V_88',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.W__minus__, P.G0, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS13, L.FFVSS14 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_2040_1449,(0,0,1):C.R2GC_1457_949,(0,1,0):C.R2GC_2046_1455,(0,1,1):C.R2GC_685_1857})

V_89 = CTVertex(name = 'V_89',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.W__minus__, P.G__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS13, L.FFVSS14 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_2039_1448,(0,0,1):C.R2GC_1458_950,(0,1,0):C.R2GC_2045_1454,(0,1,1):C.R2GC_686_1858})

V_90 = CTVertex(name = 'V_90',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.W__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS120, L.FFVS94 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g] ] ],
                couplings = {(0,1,0):C.R2GC_2109_1492,(0,1,1):C.R2GC_1463_955,(0,0,0):C.R2GC_2113_1496,(0,0,1):C.R2GC_695_1865})

V_91 = CTVertex(name = 'V_91',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.W__minus__, P.G0, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS15, L.FFVSS17 ],
                loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,0,1):C.R2GC_2027_1439,(0,0,2):C.R2GC_1644_1163,(0,1,1):C.R2GC_2043_1452,(0,1,2):C.R2GC_2411_1706,(0,3,1):C.R2GC_849_1931,(0,2,0):C.R2GC_1644_1164})

V_92 = CTVertex(name = 'V_92',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.W__minus__, P.G__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS15, L.FFVSS17 ],
                loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,0,1):C.R2GC_2028_1440,(0,0,2):C.R2GC_1643_1161,(0,1,1):C.R2GC_2042_1451,(0,1,2):C.R2GC_2410_1705,(0,3,1):C.R2GC_850_1932,(0,2,0):C.R2GC_1643_1162})

V_93 = CTVertex(name = 'V_93',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.W__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS120, L.FFVS121, L.FFVS122, L.FFVS144, L.FFVS162, L.FFVS94 ],
                loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,5,1):C.R2GC_2101_1486,(0,5,2):C.R2GC_1641_1157,(0,0,1):C.R2GC_2111_1494,(0,0,2):C.R2GC_2391_1685,(0,2,1):C.R2GC_867_1948,(0,3,1):C.R2GC_1548_1031,(0,1,0):C.R2GC_1641_1158,(0,4,0):C.R2GC_961_2012})

V_94 = CTVertex(name = 'V_94',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.W__minus__, P.G0, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS13, L.FFVSS14 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_2040_1449,(0,0,1):C.R2GC_1457_949,(0,1,0):C.R2GC_2046_1455,(0,1,1):C.R2GC_685_1857})

V_95 = CTVertex(name = 'V_95',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.W__minus__, P.G__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS13, L.FFVSS14 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_2039_1448,(0,0,1):C.R2GC_1458_950,(0,1,0):C.R2GC_2045_1454,(0,1,1):C.R2GC_686_1858})

V_96 = CTVertex(name = 'V_96',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.W__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS120, L.FFVS94 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.u] ] ],
                couplings = {(0,1,0):C.R2GC_2109_1492,(0,1,1):C.R2GC_1463_955,(0,0,0):C.R2GC_2113_1496,(0,0,1):C.R2GC_695_1865})

V_97 = CTVertex(name = 'V_97',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.W__plus__, P.G0, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS13, L.FFVSS14 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_2040_1449,(0,0,1):C.R2GC_1457_949,(0,1,0):C.R2GC_2046_1455,(0,1,1):C.R2GC_685_1857})

V_98 = CTVertex(name = 'V_98',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.W__plus__, P.G__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVSS13, L.FFVSS14 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_2041_1450,(0,0,1):C.R2GC_1456_948,(0,1,0):C.R2GC_2047_1456,(0,1,1):C.R2GC_684_1856})

V_99 = CTVertex(name = 'V_99',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS120, L.FFVS94 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g] ] ],
                couplings = {(0,1,0):C.R2GC_2110_1493,(0,1,1):C.R2GC_1462_954,(0,0,0):C.R2GC_2114_1497,(0,0,1):C.R2GC_694_1864})

V_100 = CTVertex(name = 'V_100',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.W__plus__, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS15, L.FFVSS17 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2027_1439,(0,0,2):C.R2GC_1644_1163,(0,1,1):C.R2GC_2043_1452,(0,1,2):C.R2GC_2411_1706,(0,3,1):C.R2GC_849_1931,(0,2,0):C.R2GC_1644_1164})

V_101 = CTVertex(name = 'V_101',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.W__plus__, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS15, L.FFVSS17 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2026_1438,(0,0,2):C.R2GC_1645_1165,(0,1,1):C.R2GC_2044_1453,(0,1,2):C.R2GC_2412_1707,(0,3,1):C.R2GC_848_1930,(0,2,0):C.R2GC_1645_1166})

V_102 = CTVertex(name = 'V_102',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS100, L.FFVS120, L.FFVS121, L.FFVS122, L.FFVS91, L.FFVS94, L.FFVS96 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,5,1):C.R2GC_2100_1485,(0,5,2):C.R2GC_1642_1159,(0,1,1):C.R2GC_2112_1495,(0,1,2):C.R2GC_2392_1686,(0,3,1):C.R2GC_866_1947,(0,6,1):C.R2GC_1549_1032,(0,2,0):C.R2GC_1642_1160,(0,0,0):C.R2GC_960_2011,(0,4,0):C.R2GC_962_2013})

V_103 = CTVertex(name = 'V_103',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.W__plus__, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2040_1449,(0,0,1):C.R2GC_1457_949,(0,1,0):C.R2GC_2046_1455,(0,1,1):C.R2GC_685_1857})

V_104 = CTVertex(name = 'V_104',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.W__plus__, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2041_1450,(0,0,1):C.R2GC_1456_948,(0,1,0):C.R2GC_2047_1456,(0,1,1):C.R2GC_684_1856})

V_105 = CTVertex(name = 'V_105',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,1,0):C.R2GC_2110_1493,(0,1,1):C.R2GC_1462_954,(0,0,0):C.R2GC_2114_1497,(0,0,1):C.R2GC_694_1864})

V_106 = CTVertex(name = 'V_106',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.Z, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1827_1264,(0,0,0):C.R2GC_1459_951,(0,1,1):C.R2GC_1830_1267,(0,1,0):C.R2GC_687_1859,(0,2,0):C.R2GC_689_1860})

V_107 = CTVertex(name = 'V_107',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.Z, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_2060_1467,(0,0,1):C.R2GC_1460_952,(0,1,0):C.R2GC_2062_1469,(0,1,1):C.R2GC_1461_953,(0,2,1):C.R2GC_689_1860})

V_108 = CTVertex(name = 'V_108',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.Z, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1827_1264,(0,0,0):C.R2GC_1459_951,(0,1,1):C.R2GC_1830_1267,(0,1,0):C.R2GC_687_1859,(0,2,0):C.R2GC_689_1860})

V_109 = CTVertex(name = 'V_109',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS125, L.FFVS94 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.t] ] ],
                 couplings = {(0,2,1):C.R2GC_1849_1277,(0,2,0):C.R2GC_1464_956,(0,0,1):C.R2GC_1852_1280,(0,0,0):C.R2GC_696_1866,(0,1,0):C.R2GC_697_1867})

V_110 = CTVertex(name = 'V_110',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.Z, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS16, L.FFVSS17, L.FFVSS20 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1823_1261,(0,0,0):C.R2GC_1712_1194,(0,1,1):C.R2GC_1829_1266,(0,1,0):C.R2GC_2224_1572,(0,3,1):C.R2GC_623_1806,(0,4,1):C.R2GC_624_1807,(0,4,0):C.R2GC_2225_1573,(0,2,0):C.R2GC_908_1986})

V_111 = CTVertex(name = 'V_111',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.Z, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS17, L.FFVSS19, L.FFVSS20 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2049_1457,(0,0,2):C.R2GC_1714_1196,(0,1,1):C.R2GC_2061_1468,(0,1,2):C.R2GC_2416_1714,(0,1,0):C.R2GC_2416_1715,(0,2,1):C.R2GC_851_1933,(0,3,1):C.R2GC_852_1934,(0,4,2):C.R2GC_689_1860})

V_112 = CTVertex(name = 'V_112',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.Z, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS16, L.FFVSS17, L.FFVSS20 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1823_1261,(0,0,0):C.R2GC_1712_1194,(0,1,1):C.R2GC_1829_1266,(0,1,0):C.R2GC_2224_1572,(0,3,1):C.R2GC_623_1806,(0,4,1):C.R2GC_624_1807,(0,4,0):C.R2GC_2225_1573,(0,2,0):C.R2GC_908_1986})

V_113 = CTVertex(name = 'V_113',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS112, L.FFVS120, L.FFVS122, L.FFVS123, L.FFVS125, L.FFVS129, L.FFVS132, L.FFVS160, L.FFVS161, L.FFVS94 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,9,1):C.R2GC_1845_1274,(0,9,0):C.R2GC_1723_1201,(0,1,1):C.R2GC_1851_1279,(0,1,0):C.R2GC_2196_1551,(0,2,1):C.R2GC_636_1819,(0,4,1):C.R2GC_637_1820,(0,4,0):C.R2GC_2212_1558,(0,6,1):C.R2GC_1289_459,(0,5,1):C.R2GC_1290_460,(0,3,0):C.R2GC_894_1973,(0,8,0):C.R2GC_890_1969,(0,7,0):C.R2GC_899_1977,(0,0,0):C.R2GC_901_1979})

V_114 = CTVertex(name = 'V_114',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.Z, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.g, P.u] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1827_1264,(0,0,0):C.R2GC_1459_951,(0,1,1):C.R2GC_1830_1267,(0,1,0):C.R2GC_687_1859,(0,2,0):C.R2GC_689_1860})

V_115 = CTVertex(name = 'V_115',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.Z, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2060_1467,(0,0,1):C.R2GC_1460_952,(0,1,0):C.R2GC_2062_1469,(0,1,1):C.R2GC_1461_953,(0,2,1):C.R2GC_689_1860})

V_116 = CTVertex(name = 'V_116',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.Z, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.g, P.u] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1827_1264,(0,0,0):C.R2GC_1459_951,(0,1,1):C.R2GC_1830_1267,(0,1,0):C.R2GC_687_1859,(0,2,0):C.R2GC_689_1860})

V_117 = CTVertex(name = 'V_117',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS125, L.FFVS94 ],
                 loop_particles = [ [ [P.g, P.u] ], [ [P.t] ] ],
                 couplings = {(0,2,1):C.R2GC_1849_1277,(0,2,0):C.R2GC_1464_956,(0,0,1):C.R2GC_1852_1280,(0,0,0):C.R2GC_696_1866,(0,1,0):C.R2GC_697_1867})

V_118 = CTVertex(name = 'V_118',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.a, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,1,2):C.R2GC_2015_1427,(0,1,0):C.R2GC_2404_1697,(0,1,1):C.R2GC_2404_1698,(0,2,2):C.R2GC_2016_1428,(0,2,0):C.R2GC_651_1827,(0,0,0):C.R2GC_652_1828})

V_119 = CTVertex(name = 'V_119',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.G0, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30, L.FFSS42 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1813_1251,(0,0,0):C.R2GC_1440_932,(0,1,1):C.R2GC_1814_1252,(0,1,0):C.R2GC_640_1822})

V_120 = CTVertex(name = 'V_120',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45, L.FFS55 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1836_1271,(0,0,0):C.R2GC_1449_941,(0,1,0):C.R2GC_665_1841})

V_121 = CTVertex(name = 'V_121',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS24, L.FFSS28, L.FFSS30, L.FFSS32, L.FFSS42 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,2,2):C.R2GC_2000_1413,(0,2,0):C.R2GC_643_1825,(0,4,2):C.R2GC_2002_1415,(0,4,0):C.R2GC_641_1823,(0,0,0):C.R2GC_642_1824,(0,3,1):C.R2GC_978_2024,(0,1,1):C.R2GC_2217_1565})

V_122 = CTVertex(name = 'V_122',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.a, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g] ] ],
                 couplings = {(0,1,0):C.R2GC_2017_1429,(0,1,1):C.R2GC_709_1870,(0,2,0):C.R2GC_2016_1428,(0,2,1):C.R2GC_651_1827,(0,0,1):C.R2GC_652_1828})

V_123 = CTVertex(name = 'V_123',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.G0, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30, L.FFSS42 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1816_1254,(0,0,0):C.R2GC_1468_960,(0,1,1):C.R2GC_1814_1252,(0,1,0):C.R2GC_640_1822})

V_124 = CTVertex(name = 'V_124',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45, L.FFS55 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_1475_963,(0,1,0):C.R2GC_665_1841})

V_125 = CTVertex(name = 'V_125',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS24, L.FFSS30, L.FFSS42 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g] ] ],
                 couplings = {(0,1,0):C.R2GC_2004_1417,(0,1,1):C.R2GC_706_1869,(0,2,0):C.R2GC_2002_1415,(0,2,1):C.R2GC_641_1823,(0,0,1):C.R2GC_642_1824})

V_126 = CTVertex(name = 'V_126',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.a, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.s] ] ],
                 couplings = {(0,1,0):C.R2GC_2017_1429,(0,1,1):C.R2GC_709_1870,(0,2,0):C.R2GC_2016_1428,(0,2,1):C.R2GC_651_1827,(0,0,1):C.R2GC_652_1828})

V_127 = CTVertex(name = 'V_127',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.G0, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30, L.FFSS42 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1816_1254,(0,0,0):C.R2GC_1468_960,(0,1,1):C.R2GC_1814_1252,(0,1,0):C.R2GC_640_1822})

V_128 = CTVertex(name = 'V_128',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45, L.FFS55 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_1475_963,(0,1,0):C.R2GC_665_1841})

V_129 = CTVertex(name = 'V_129',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS24, L.FFSS30, L.FFSS42 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.s] ] ],
                 couplings = {(0,1,0):C.R2GC_2004_1417,(0,1,1):C.R2GC_706_1869,(0,2,0):C.R2GC_2002_1415,(0,2,1):C.R2GC_641_1823,(0,0,1):C.R2GC_642_1824})

V_130 = CTVertex(name = 'V_130',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.W__minus__, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,2):C.R2GC_2032_1443,(0,0,0):C.R2GC_1644_1163,(0,0,1):C.R2GC_1644_1164,(0,1,2):C.R2GC_2037_1446,(0,1,0):C.R2GC_660_1836})

V_131 = CTVertex(name = 'V_131',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.W__minus__, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,2):C.R2GC_2031_1442,(0,0,0):C.R2GC_1643_1161,(0,0,1):C.R2GC_1643_1162,(0,1,2):C.R2GC_2036_1445,(0,1,0):C.R2GC_661_1837})

V_132 = CTVertex(name = 'V_132',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,1,2):C.R2GC_2104_1488,(0,1,0):C.R2GC_1641_1157,(0,1,1):C.R2GC_1641_1158,(0,0,2):C.R2GC_2107_1490,(0,0,0):C.R2GC_669_1845})

V_133 = CTVertex(name = 'V_133',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.W__minus__, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_2040_1449,(0,0,1):C.R2GC_1457_949,(0,1,0):C.R2GC_2037_1446,(0,1,1):C.R2GC_660_1836})

V_134 = CTVertex(name = 'V_134',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.W__minus__, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_2039_1448,(0,0,1):C.R2GC_1458_950,(0,1,0):C.R2GC_2036_1445,(0,1,1):C.R2GC_661_1837})

V_135 = CTVertex(name = 'V_135',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g] ] ],
                 couplings = {(0,1,0):C.R2GC_2109_1492,(0,1,1):C.R2GC_1463_955,(0,0,0):C.R2GC_2107_1490,(0,0,1):C.R2GC_669_1845})

V_136 = CTVertex(name = 'V_136',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.W__minus__, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2040_1449,(0,0,1):C.R2GC_1457_949,(0,1,0):C.R2GC_2037_1446,(0,1,1):C.R2GC_660_1836})

V_137 = CTVertex(name = 'V_137',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.W__minus__, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2039_1448,(0,0,1):C.R2GC_1458_950,(0,1,0):C.R2GC_2036_1445,(0,1,1):C.R2GC_661_1837})

V_138 = CTVertex(name = 'V_138',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.s] ] ],
                 couplings = {(0,1,0):C.R2GC_2109_1492,(0,1,1):C.R2GC_1463_955,(0,0,0):C.R2GC_2107_1490,(0,0,1):C.R2GC_669_1845})

V_139 = CTVertex(name = 'V_139',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.W__plus__, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,2):C.R2GC_2032_1443,(0,0,0):C.R2GC_1644_1163,(0,0,1):C.R2GC_1644_1164,(0,1,2):C.R2GC_2037_1446,(0,1,0):C.R2GC_660_1836})

V_140 = CTVertex(name = 'V_140',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.W__plus__, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,2):C.R2GC_2035_1444,(0,0,0):C.R2GC_1645_1165,(0,0,1):C.R2GC_1645_1166,(0,1,2):C.R2GC_2038_1447,(0,1,0):C.R2GC_659_1835})

V_141 = CTVertex(name = 'V_141',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,1,2):C.R2GC_2106_1489,(0,1,0):C.R2GC_1642_1159,(0,1,1):C.R2GC_1642_1160,(0,0,2):C.R2GC_2108_1491,(0,0,0):C.R2GC_668_1844})

V_142 = CTVertex(name = 'V_142',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.W__plus__, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_2040_1449,(0,0,1):C.R2GC_1457_949,(0,1,0):C.R2GC_2037_1446,(0,1,1):C.R2GC_660_1836})

V_143 = CTVertex(name = 'V_143',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.W__plus__, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_2041_1450,(0,0,1):C.R2GC_1456_948,(0,1,0):C.R2GC_2038_1447,(0,1,1):C.R2GC_659_1835})

V_144 = CTVertex(name = 'V_144',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g] ] ],
                 couplings = {(0,1,0):C.R2GC_2110_1493,(0,1,1):C.R2GC_1462_954,(0,0,0):C.R2GC_2108_1491,(0,0,1):C.R2GC_668_1844})

V_145 = CTVertex(name = 'V_145',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.W__plus__, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2040_1449,(0,0,1):C.R2GC_1457_949,(0,1,0):C.R2GC_2037_1446,(0,1,1):C.R2GC_660_1836})

V_146 = CTVertex(name = 'V_146',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.W__plus__, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2041_1450,(0,0,1):C.R2GC_1456_948,(0,1,0):C.R2GC_2038_1447,(0,1,1):C.R2GC_659_1835})

V_147 = CTVertex(name = 'V_147',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.s] ] ],
                 couplings = {(0,1,0):C.R2GC_2110_1493,(0,1,1):C.R2GC_1462_954,(0,0,0):C.R2GC_2108_1491,(0,0,1):C.R2GC_668_1844})

V_148 = CTVertex(name = 'V_148',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.Z, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1825_1262,(0,0,0):C.R2GC_1444_936,(0,1,1):C.R2GC_1826_1263,(0,1,0):C.R2GC_662_1838,(0,2,0):C.R2GC_664_1840})

V_149 = CTVertex(name = 'V_149',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.Z, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,2):C.R2GC_2057_1464,(0,0,0):C.R2GC_2417_1716,(0,0,1):C.R2GC_2417_1717,(0,1,2):C.R2GC_2058_1465,(0,1,0):C.R2GC_1445_937,(0,2,0):C.R2GC_664_1840})

V_150 = CTVertex(name = 'V_150',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.Z, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1825_1262,(0,0,0):C.R2GC_1444_936,(0,1,1):C.R2GC_1826_1263,(0,1,0):C.R2GC_662_1838,(0,2,0):C.R2GC_664_1840})

V_151 = CTVertex(name = 'V_151',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS125, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.t] ] ],
                 couplings = {(0,2,1):C.R2GC_1847_1275,(0,2,0):C.R2GC_1450_942,(0,0,1):C.R2GC_1848_1276,(0,0,0):C.R2GC_670_1846,(0,1,0):C.R2GC_671_1847})

V_152 = CTVertex(name = 'V_152',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.Z, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1828_1265,(0,0,0):C.R2GC_1472_961,(0,1,1):C.R2GC_1826_1263,(0,1,0):C.R2GC_662_1838,(0,2,0):C.R2GC_664_1840})

V_153 = CTVertex(name = 'V_153',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.Z, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_2059_1466,(0,0,1):C.R2GC_1474_962,(0,1,0):C.R2GC_2058_1465,(0,1,1):C.R2GC_1445_937,(0,2,1):C.R2GC_664_1840})

V_154 = CTVertex(name = 'V_154',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.Z, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1828_1265,(0,0,0):C.R2GC_1472_961,(0,1,1):C.R2GC_1826_1263,(0,1,0):C.R2GC_662_1838,(0,2,0):C.R2GC_664_1840})

V_155 = CTVertex(name = 'V_155',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS125, L.FFVS94 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.t] ] ],
                 couplings = {(0,2,1):C.R2GC_1850_1278,(0,2,0):C.R2GC_1478_964,(0,0,1):C.R2GC_1848_1276,(0,0,0):C.R2GC_670_1846,(0,1,0):C.R2GC_671_1847})

V_156 = CTVertex(name = 'V_156',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.Z, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1828_1265,(0,0,0):C.R2GC_1472_961,(0,1,1):C.R2GC_1826_1263,(0,1,0):C.R2GC_662_1838,(0,2,0):C.R2GC_664_1840})

V_157 = CTVertex(name = 'V_157',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.Z, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2059_1466,(0,0,1):C.R2GC_1474_962,(0,1,0):C.R2GC_2058_1465,(0,1,1):C.R2GC_1445_937,(0,2,1):C.R2GC_664_1840})

V_158 = CTVertex(name = 'V_158',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.Z, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1828_1265,(0,0,0):C.R2GC_1472_961,(0,1,1):C.R2GC_1826_1263,(0,1,0):C.R2GC_662_1838,(0,2,0):C.R2GC_664_1840})

V_159 = CTVertex(name = 'V_159',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS125, L.FFVS94 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.t] ] ],
                 couplings = {(0,2,1):C.R2GC_1850_1278,(0,2,0):C.R2GC_1478_964,(0,0,1):C.R2GC_1848_1276,(0,0,0):C.R2GC_670_1846,(0,1,0):C.R2GC_671_1847})

V_160 = CTVertex(name = 'V_160',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.a, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2009_1422,(0,0,1):C.R2GC_2313_1624})

V_161 = CTVertex(name = 'V_161',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.a, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2010_1423,(0,0,1):C.R2GC_2314_1625})

V_162 = CTVertex(name = 'V_162',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2081_1477,(0,0,1):C.R2GC_2322_1633})

V_163 = CTVertex(name = 'V_163',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_1995_1408,(0,0,1):C.R2GC_2310_1621})

V_164 = CTVertex(name = 'V_164',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_1994_1407,(0,0,1):C.R2GC_2309_1620})

V_165 = CTVertex(name = 'V_165',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2320_1631})

V_166 = CTVertex(name = 'V_166',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.a, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2012_1425,(0,0,2):C.R2GC_2401_1691,(0,0,0):C.R2GC_2401_1692})

V_167 = CTVertex(name = 'V_167',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.a, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2013_1426,(0,0,2):C.R2GC_2402_1693,(0,0,0):C.R2GC_2402_1694})

V_168 = CTVertex(name = 'V_168',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS101, L.FFVS107, L.FFVS94, L.FFVS99 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,2):C.R2GC_2083_1479,(0,2,3):C.R2GC_2422_1725,(0,2,1):C.R2GC_2422_1726,(0,0,0):C.R2GC_654_1830,(0,1,3):C.R2GC_875_1954,(0,3,2):C.R2GC_1525_1008})

V_169 = CTVertex(name = 'V_169',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS27, L.FFSS30, L.FFSS33, L.FFSS34 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_980_2025,(0,2,2):C.R2GC_906_1984,(0,3,0):C.R2GC_906_1984,(0,1,1):C.R2GC_1998_1411,(0,1,0):C.R2GC_2344_1655})

V_170 = CTVertex(name = 'V_170',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.a, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2009_1422,(0,0,1):C.R2GC_2313_1624})

V_171 = CTVertex(name = 'V_171',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.a, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2010_1423,(0,0,1):C.R2GC_2314_1625})

V_172 = CTVertex(name = 'V_172',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2081_1477,(0,0,1):C.R2GC_2322_1633})

V_173 = CTVertex(name = 'V_173',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1995_1408,(0,0,1):C.R2GC_2310_1621})

V_174 = CTVertex(name = 'V_174',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1994_1407,(0,0,1):C.R2GC_2309_1620})

V_175 = CTVertex(name = 'V_175',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45 ],
                 loop_particles = [ [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2320_1631})

V_176 = CTVertex(name = 'V_176',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2025_1437,(0,0,1):C.R2GC_2316_1627})

V_177 = CTVertex(name = 'V_177',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2025_1437,(0,0,1):C.R2GC_2316_1627})

V_178 = CTVertex(name = 'V_178',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2025_1437,(0,0,1):C.R2GC_2316_1627})

V_179 = CTVertex(name = 'V_179',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2099_1484,(0,0,1):C.R2GC_2323_1634})

V_180 = CTVertex(name = 'V_180',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2029_1441,(0,0,0):C.R2GC_2376_1684})

V_181 = CTVertex(name = 'V_181',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2029_1441,(0,0,0):C.R2GC_2376_1684})

V_182 = CTVertex(name = 'V_182',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2029_1441,(0,0,0):C.R2GC_2376_1684})

V_183 = CTVertex(name = 'V_183',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS101, L.FFVS94, L.FFVS96 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,1,1):C.R2GC_2102_1487,(0,1,0):C.R2GC_2427_1732,(0,0,0):C.R2GC_957_2008,(0,2,1):C.R2GC_1551_1034})

V_184 = CTVertex(name = 'V_184',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2025_1437,(0,0,1):C.R2GC_2316_1627})

V_185 = CTVertex(name = 'V_185',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2025_1437,(0,0,1):C.R2GC_2316_1627})

V_186 = CTVertex(name = 'V_186',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2025_1437,(0,0,1):C.R2GC_2316_1627})

V_187 = CTVertex(name = 'V_187',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2099_1484,(0,0,1):C.R2GC_2323_1634})

V_188 = CTVertex(name = 'V_188',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.Z, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2051_1459,(0,0,1):C.R2GC_2318_1629})

V_189 = CTVertex(name = 'V_189',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.Z, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2052_1460,(0,0,1):C.R2GC_2319_1630})

V_190 = CTVertex(name = 'V_190',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2122_1502,(0,0,1):C.R2GC_2325_1636})

V_191 = CTVertex(name = 'V_191',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.Z, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2054_1462,(0,0,2):C.R2GC_2414_1710,(0,0,0):C.R2GC_2414_1711})

V_192 = CTVertex(name = 'V_192',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.Z, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2055_1463,(0,0,2):C.R2GC_2415_1712,(0,0,0):C.R2GC_2415_1713})

V_193 = CTVertex(name = 'V_193',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS101, L.FFVS107, L.FFVS94, L.FFVS96, L.FFVS99 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,2):C.R2GC_2124_1504,(0,2,3):C.R2GC_2433_1735,(0,2,1):C.R2GC_2433_1736,(0,0,0):C.R2GC_1447_939,(0,1,3):C.R2GC_896_1974,(0,3,2):C.R2GC_1555_1038,(0,4,2):C.R2GC_1569_1064})

V_194 = CTVertex(name = 'V_194',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.Z, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2051_1459,(0,0,1):C.R2GC_2318_1629})

V_195 = CTVertex(name = 'V_195',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.Z, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2052_1460,(0,0,1):C.R2GC_2319_1630})

V_196 = CTVertex(name = 'V_196',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2122_1502,(0,0,1):C.R2GC_2325_1636})

V_197 = CTVertex(name = 'V_197',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.a, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2012_1425,(0,0,2):C.R2GC_2401_1691,(0,0,0):C.R2GC_2401_1692})

V_198 = CTVertex(name = 'V_198',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.a, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2011_1424,(0,0,2):C.R2GC_2400_1689,(0,0,0):C.R2GC_2400_1690})

V_199 = CTVertex(name = 'V_199',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS109, L.FFVS138, L.FFVS139, L.FFVS141, L.FFVS162, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,5,2):C.R2GC_2082_1478,(0,5,3):C.R2GC_2421_1723,(0,5,1):C.R2GC_2421_1724,(0,4,0):C.R2GC_653_1829,(0,3,3):C.R2GC_876_1955,(0,1,2):C.R2GC_853_1935,(0,2,2):C.R2GC_854_1936,(0,0,2):C.R2GC_2067_1472})

V_200 = CTVertex(name = 'V_200',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS29, L.FFSS30, L.FFSS31, L.FFSS36 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,3,0):C.R2GC_980_2025,(0,0,2):C.R2GC_906_1984,(0,2,0):C.R2GC_906_1984,(0,1,1):C.R2GC_1999_1412,(0,1,0):C.R2GC_2345_1656})

V_201 = CTVertex(name = 'V_201',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.a, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2009_1422,(0,0,1):C.R2GC_2313_1624})

V_202 = CTVertex(name = 'V_202',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.a, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2008_1421,(0,0,1):C.R2GC_2312_1623})

V_203 = CTVertex(name = 'V_203',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2080_1476,(0,0,1):C.R2GC_2321_1632})

V_204 = CTVertex(name = 'V_204',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1996_1409,(0,0,1):C.R2GC_2311_1622})

V_205 = CTVertex(name = 'V_205',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1994_1407,(0,0,1):C.R2GC_2309_1620})

V_206 = CTVertex(name = 'V_206',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45 ],
                 loop_particles = [ [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2320_1631})

V_207 = CTVertex(name = 'V_207',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.a, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2009_1422,(0,0,1):C.R2GC_2313_1624})

V_208 = CTVertex(name = 'V_208',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.a, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2008_1421,(0,0,1):C.R2GC_2312_1623})

V_209 = CTVertex(name = 'V_209',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2080_1476,(0,0,1):C.R2GC_2321_1632})

V_210 = CTVertex(name = 'V_210',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_1996_1409,(0,0,1):C.R2GC_2311_1622})

V_211 = CTVertex(name = 'V_211',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_1994_1407,(0,0,1):C.R2GC_2309_1620})

V_212 = CTVertex(name = 'V_212',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2320_1631})

V_213 = CTVertex(name = 'V_213',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2029_1441,(0,0,0):C.R2GC_2376_1684})

V_214 = CTVertex(name = 'V_214',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2029_1441,(0,0,0):C.R2GC_2376_1684})

V_215 = CTVertex(name = 'V_215',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2029_1441,(0,0,0):C.R2GC_2376_1684})

V_216 = CTVertex(name = 'V_216',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS144, L.FFVS162, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,2,1):C.R2GC_2102_1487,(0,2,0):C.R2GC_2427_1732,(0,1,0):C.R2GC_958_2009,(0,0,1):C.R2GC_1550_1033})

V_217 = CTVertex(name = 'V_217',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2025_1437,(0,0,1):C.R2GC_2316_1627})

V_218 = CTVertex(name = 'V_218',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2025_1437,(0,0,1):C.R2GC_2316_1627})

V_219 = CTVertex(name = 'V_219',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2025_1437,(0,0,1):C.R2GC_2316_1627})

V_220 = CTVertex(name = 'V_220',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2099_1484,(0,0,1):C.R2GC_2323_1634})

V_221 = CTVertex(name = 'V_221',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2025_1437,(0,0,1):C.R2GC_2316_1627})

V_222 = CTVertex(name = 'V_222',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2025_1437,(0,0,1):C.R2GC_2316_1627})

V_223 = CTVertex(name = 'V_223',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2025_1437,(0,0,1):C.R2GC_2316_1627})

V_224 = CTVertex(name = 'V_224',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2099_1484,(0,0,1):C.R2GC_2323_1634})

V_225 = CTVertex(name = 'V_225',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.Z, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2054_1462,(0,0,2):C.R2GC_2414_1710,(0,0,0):C.R2GC_2414_1711})

V_226 = CTVertex(name = 'V_226',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.Z, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2053_1461,(0,0,2):C.R2GC_2413_1708,(0,0,0):C.R2GC_2413_1709})

V_227 = CTVertex(name = 'V_227',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS109, L.FFVS138, L.FFVS139, L.FFVS141, L.FFVS144, L.FFVS162, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,6,2):C.R2GC_2123_1503,(0,6,3):C.R2GC_2432_1733,(0,6,1):C.R2GC_2432_1734,(0,5,0):C.R2GC_1446_938,(0,3,3):C.R2GC_897_1975,(0,4,2):C.R2GC_1554_1037,(0,1,2):C.R2GC_868_1949,(0,2,2):C.R2GC_869_1950,(0,0,2):C.R2GC_2117_1498})

V_228 = CTVertex(name = 'V_228',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.Z, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2051_1459,(0,0,1):C.R2GC_2318_1629})

V_229 = CTVertex(name = 'V_229',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.Z, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2050_1458,(0,0,1):C.R2GC_2317_1628})

V_230 = CTVertex(name = 'V_230',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2121_1501,(0,0,1):C.R2GC_2324_1635})

V_231 = CTVertex(name = 'V_231',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.Z, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2051_1459,(0,0,1):C.R2GC_2318_1629})

V_232 = CTVertex(name = 'V_232',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.Z, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2050_1458,(0,0,1):C.R2GC_2317_1628})

V_233 = CTVertex(name = 'V_233',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_2121_1501,(0,0,1):C.R2GC_2324_1635})

V_234 = CTVertex(name = 'V_234',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS102, L.FFVS106, L.FFVS94, L.FFVS95 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,2):C.R2GC_1539_1022,(0,2,3):C.R2GC_2367_1677,(0,2,1):C.R2GC_2367_1678,(0,0,0):C.R2GC_655_1831,(0,1,3):C.R2GC_881_1960,(0,3,2):C.R2GC_1531_1014})

V_235 = CTVertex(name = 'V_235',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS159, L.FFVS87 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,1,1):C.R2GC_1267_437,(0,0,0):C.R2GC_884_1963})

V_236 = CTVertex(name = 'V_236',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS120, L.FFVS125, L.FFVS130, L.FFVS158, L.FFVS86, L.FFVS94 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,4,0):C.R2GC_1942_1389,(0,2,1):C.R2GC_1266_436,(0,5,1):C.R2GC_1282_452,(0,1,1):C.R2GC_634_1817,(0,1,0):C.R2GC_2189_1549,(0,0,1):C.R2GC_635_1818,(0,3,0):C.R2GC_883_1962})

V_237 = CTVertex(name = 'V_237',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.g, P.g, P.G__minus__ ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVS87, L.FFVVS89 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.t] ], [ [P.b, P.t] ], [ [P.g] ] ],
                 couplings = {(2,1,2):C.R2GC_2164_1520,(2,1,0):C.R2GC_2165_1522,(2,1,1):C.R2GC_1536_1019,(1,1,2):C.R2GC_2167_1524,(1,1,0):C.R2GC_2168_1526,(1,1,1):C.R2GC_1537_1020,(2,0,2):C.R2GC_2167_1524,(2,0,0):C.R2GC_2167_1525,(1,0,2):C.R2GC_2164_1520,(1,0,0):C.R2GC_2164_1521,(1,0,1):C.R2GC_1535_1018,(0,0,0):C.R2GC_1574_1073,(0,0,1):C.R2GC_856_1938})

V_238 = CTVertex(name = 'V_238',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.g, P.G0 ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVS119, L.FFVVS78, L.FFVVS80, L.FFVVS89 ],
                 loop_particles = [ [ [P.g] ], [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_2182_1542,(2,1,1):C.R2GC_2182_1543,(2,1,2):C.R2GC_1274_444,(1,1,0):C.R2GC_2170_1528,(1,1,1):C.R2GC_2171_1530,(1,1,2):C.R2GC_1270_440,(0,2,2):C.R2GC_627_1810,(0,2,1):C.R2GC_886_1965,(1,2,0):C.R2GC_2182_1542,(1,2,1):C.R2GC_2185_1545,(1,2,2):C.R2GC_1275_445,(2,2,0):C.R2GC_2170_1528,(2,2,1):C.R2GC_2170_1529,(2,3,1):C.R2GC_2171_1530,(1,3,1):C.R2GC_2182_1543,(2,0,1):C.R2GC_2184_1544,(1,0,1):C.R2GC_2173_1531})

V_239 = CTVertex(name = 'V_239',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.g, P.H ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS119, L.FFVVS77, L.FFVVS81, L.FFVVS87, L.FFVVS89 ],
                 loop_particles = [ [ [P.g] ], [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(2,2,0):C.R2GC_2174_1532,(2,2,1):C.R2GC_2174_1533,(2,2,2):C.R2GC_1272_442,(1,2,0):C.R2GC_2177_1536,(1,2,1):C.R2GC_2178_1538,(1,2,2):C.R2GC_1271_441,(0,3,2):C.R2GC_628_1811,(0,3,1):C.R2GC_887_1966,(1,3,2):C.R2GC_1273_443,(1,3,1):C.R2GC_2181_1541,(2,3,1):C.R2GC_2180_1540,(2,5,0):C.R2GC_2174_1532,(2,5,1):C.R2GC_2179_1539,(1,5,0):C.R2GC_2177_1536,(1,5,1):C.R2GC_2177_1537,(2,4,0):C.R2GC_2176_1535,(1,4,0):C.R2GC_2175_1534,(2,1,0):C.R2GC_2174_1532,(2,1,1):C.R2GC_2179_1539,(1,1,0):C.R2GC_2177_1536,(1,1,1):C.R2GC_2177_1537,(2,0,0):C.R2GC_2176_1535,(1,0,0):C.R2GC_2175_1534})

V_240 = CTVertex(name = 'V_240',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV103, L.FFVV104, L.FFVV107, L.FFVV108, L.FFVV109, L.FFVV110, L.FFVV111, L.FFVV112, L.FFVV113, L.FFVV114, L.FFVV115, L.FFVV117, L.FFVV129, L.FFVV130, L.FFVV131, L.FFVV132, L.FFVV141, L.FFVV142, L.FFVV143, L.FFVV154, L.FFVV157, L.FFVV158, L.FFVV159, L.FFVV160, L.FFVV161, L.FFVV162, L.FFVV163, L.FFVV164, L.FFVV165, L.FFVV167, L.FFVV187, L.FFVV188, L.FFVV189, L.FFVV190, L.FFVV193, L.FFVV194, L.FFVV195, L.FFVV197 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.g] ], [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,18,0):C.R2GC_1329_558,(0,18,1):C.R2GC_1329_559,(0,18,2):C.R2GC_1329_560,(0,18,5):C.R2GC_1328_554,(0,37,5):C.R2GC_1047_70,(2,16,0):C.R2GC_1048_71,(2,16,1):C.R2GC_1330_561,(2,16,2):C.R2GC_1330_562,(2,16,5):C.R2GC_1330_563,(2,17,0):C.R2GC_1051_75,(2,17,1):C.R2GC_1327_552,(2,17,2):C.R2GC_1327_553,(2,17,5):C.R2GC_1326_551,(1,16,0):C.R2GC_1048_71,(1,16,1):C.R2GC_1330_561,(1,16,2):C.R2GC_1330_562,(1,16,5):C.R2GC_1330_563,(1,17,0):C.R2GC_1051_75,(1,17,1):C.R2GC_1327_552,(1,17,2):C.R2GC_1327_553,(1,17,5):C.R2GC_1326_551,(2,3,0):C.R2GC_1776_1210,(2,3,1):C.R2GC_1877_1309,(2,3,2):C.R2GC_1877_1310,(2,3,5):C.R2GC_1872_1304,(2,4,0):C.R2GC_1776_1210,(2,4,1):C.R2GC_1877_1309,(2,4,2):C.R2GC_1877_1310,(2,4,5):C.R2GC_1872_1304,(2,5,0):C.R2GC_1866_1292,(2,5,1):C.R2GC_1862_1285,(2,5,2):C.R2GC_1862_1286,(2,5,5):C.R2GC_1866_1293,(2,6,0):C.R2GC_1044_68,(2,6,5):C.R2GC_1044_69,(1,3,0):C.R2GC_1774_1208,(1,3,1):C.R2GC_1862_1285,(1,3,2):C.R2GC_1862_1286,(1,3,5):C.R2GC_1862_1287,(1,4,0):C.R2GC_1774_1208,(1,4,1):C.R2GC_1862_1285,(1,4,2):C.R2GC_1862_1286,(1,4,5):C.R2GC_1862_1287,(1,5,0):C.R2GC_1880_1313,(1,5,1):C.R2GC_1877_1309,(1,5,2):C.R2GC_1877_1310,(1,5,5):C.R2GC_1880_1314,(1,6,0):C.R2GC_1769_1206,(1,6,5):C.R2GC_1769_1207,(2,7,0):C.R2GC_1774_1208,(2,7,1):C.R2GC_1862_1285,(2,7,2):C.R2GC_1862_1286,(2,7,5):C.R2GC_1862_1287,(2,8,0):C.R2GC_1774_1208,(2,8,1):C.R2GC_1862_1285,(2,8,2):C.R2GC_1862_1286,(2,8,5):C.R2GC_1862_1287,(2,9,0):C.R2GC_1769_1206,(2,9,5):C.R2GC_1769_1207,(2,10,0):C.R2GC_1880_1313,(2,10,1):C.R2GC_1877_1309,(2,10,2):C.R2GC_1877_1310,(2,10,5):C.R2GC_1880_1314,(1,7,0):C.R2GC_1776_1210,(1,7,1):C.R2GC_1877_1309,(1,7,2):C.R2GC_1877_1310,(1,7,5):C.R2GC_1872_1304,(1,8,0):C.R2GC_1776_1210,(1,8,1):C.R2GC_1877_1309,(1,8,2):C.R2GC_1877_1310,(1,8,5):C.R2GC_1872_1304,(1,9,0):C.R2GC_1044_68,(1,9,5):C.R2GC_1044_69,(1,10,0):C.R2GC_1866_1292,(1,10,1):C.R2GC_1862_1285,(1,10,2):C.R2GC_1862_1286,(1,10,5):C.R2GC_1866_1293,(2,11,3):C.R2GC_2238_1580,(2,11,4):C.R2GC_2243_1587,(2,11,5):C.R2GC_1148_251,(1,11,3):C.R2GC_2241_1584,(1,11,4):C.R2GC_2241_1585,(1,11,5):C.R2GC_1147_250,(0,20,5):C.R2GC_593_1777,(1,2,3):C.R2GC_2239_1582,(1,2,5):C.R2GC_1149_252,(0,36,0):C.R2GC_1333_569,(0,36,1):C.R2GC_1333_570,(0,36,2):C.R2GC_1333_571,(0,36,5):C.R2GC_1333_572,(2,34,0):C.R2GC_1334_573,(2,34,1):C.R2GC_1334_574,(2,34,2):C.R2GC_1334_575,(2,34,5):C.R2GC_1326_551,(2,35,0):C.R2GC_1332_566,(2,35,1):C.R2GC_1332_567,(2,35,2):C.R2GC_1332_568,(2,35,5):C.R2GC_1330_563,(1,34,0):C.R2GC_1334_573,(1,34,1):C.R2GC_1334_574,(1,34,2):C.R2GC_1334_575,(1,34,5):C.R2GC_1326_551,(1,35,0):C.R2GC_1332_566,(1,35,1):C.R2GC_1332_567,(1,35,2):C.R2GC_1332_568,(1,35,5):C.R2GC_1330_563,(2,21,0):C.R2GC_1887_1321,(2,21,1):C.R2GC_1887_1322,(2,21,2):C.R2GC_1887_1323,(2,21,5):C.R2GC_1872_1304,(2,22,0):C.R2GC_1887_1321,(2,22,1):C.R2GC_1887_1322,(2,22,2):C.R2GC_1887_1323,(2,22,5):C.R2GC_1872_1304,(2,23,0):C.R2GC_1882_1315,(2,23,1):C.R2GC_1882_1316,(2,23,2):C.R2GC_1882_1317,(2,23,5):C.R2GC_1886_1320,(2,24,5):C.R2GC_1782_1220,(1,21,0):C.R2GC_1882_1315,(1,21,1):C.R2GC_1882_1316,(1,21,2):C.R2GC_1882_1317,(1,21,5):C.R2GC_1862_1287,(1,22,0):C.R2GC_1882_1315,(1,22,1):C.R2GC_1882_1316,(1,22,2):C.R2GC_1882_1317,(1,22,5):C.R2GC_1862_1287,(1,23,0):C.R2GC_1887_1321,(1,23,1):C.R2GC_1887_1322,(1,23,2):C.R2GC_1887_1323,(1,23,5):C.R2GC_1890_1325,(1,24,5):C.R2GC_1783_1221,(2,25,0):C.R2GC_1882_1315,(2,25,1):C.R2GC_1882_1316,(2,25,2):C.R2GC_1882_1317,(2,25,5):C.R2GC_1862_1287,(2,26,0):C.R2GC_1882_1315,(2,26,1):C.R2GC_1882_1316,(2,26,2):C.R2GC_1882_1317,(2,26,5):C.R2GC_1862_1287,(2,27,5):C.R2GC_1783_1221,(2,28,0):C.R2GC_1887_1321,(2,28,1):C.R2GC_1887_1322,(2,28,2):C.R2GC_1887_1323,(2,28,5):C.R2GC_1890_1325,(1,25,0):C.R2GC_1887_1321,(1,25,1):C.R2GC_1887_1322,(1,25,2):C.R2GC_1887_1323,(1,25,5):C.R2GC_1872_1304,(1,26,0):C.R2GC_1887_1321,(1,26,1):C.R2GC_1887_1322,(1,26,2):C.R2GC_1887_1323,(1,26,5):C.R2GC_1872_1304,(1,27,5):C.R2GC_1782_1220,(1,28,0):C.R2GC_1882_1315,(1,28,1):C.R2GC_1882_1316,(1,28,2):C.R2GC_1882_1317,(1,28,5):C.R2GC_1886_1320,(2,29,3):C.R2GC_2238_1580,(2,29,4):C.R2GC_2243_1587,(2,29,5):C.R2GC_1148_251,(1,29,3):C.R2GC_2241_1584,(1,29,4):C.R2GC_2241_1585,(1,29,5):C.R2GC_1147_250,(1,19,3):C.R2GC_2239_1582,(1,19,5):C.R2GC_1149_252,(2,14,0):C.R2GC_1880_1313,(2,14,1):C.R2GC_1877_1309,(2,14,2):C.R2GC_1877_1310,(2,14,5):C.R2GC_1880_1314,(1,14,0):C.R2GC_1864_1288,(1,14,1):C.R2GC_1862_1285,(1,14,2):C.R2GC_1862_1286,(1,14,5):C.R2GC_1864_1289,(2,32,0):C.R2GC_1887_1321,(2,32,1):C.R2GC_1887_1322,(2,32,2):C.R2GC_1887_1323,(2,32,5):C.R2GC_1890_1325,(1,32,0):C.R2GC_1882_1315,(1,32,1):C.R2GC_1882_1316,(1,32,2):C.R2GC_1882_1317,(1,32,5):C.R2GC_1884_1318,(2,12,0):C.R2GC_1044_68,(2,12,5):C.R2GC_1044_69,(1,12,0):C.R2GC_1044_68,(1,12,5):C.R2GC_1044_69,(2,30,5):C.R2GC_1782_1220,(1,30,5):C.R2GC_1782_1220,(2,15,0):C.R2GC_1865_1290,(2,15,1):C.R2GC_1862_1285,(2,15,2):C.R2GC_1862_1286,(2,15,5):C.R2GC_1865_1291,(1,15,0):C.R2GC_1879_1311,(1,15,1):C.R2GC_1877_1309,(1,15,2):C.R2GC_1877_1310,(1,15,5):C.R2GC_1879_1312,(2,33,0):C.R2GC_1882_1315,(2,33,1):C.R2GC_1882_1316,(2,33,2):C.R2GC_1882_1317,(2,33,5):C.R2GC_1885_1319,(1,33,0):C.R2GC_1887_1321,(1,33,1):C.R2GC_1887_1322,(1,33,2):C.R2GC_1887_1323,(1,33,5):C.R2GC_1889_1324,(2,13,0):C.R2GC_1043_66,(2,13,5):C.R2GC_1043_67,(1,13,0):C.R2GC_1043_66,(1,13,5):C.R2GC_1043_67,(2,31,5):C.R2GC_587_1772,(1,31,5):C.R2GC_587_1772,(2,0,3):C.R2GC_2238_1580,(2,0,4):C.R2GC_2238_1581,(1,0,3):C.R2GC_2241_1584,(1,0,4):C.R2GC_2242_1586,(0,1,4):C.R2GC_913_1990,(2,1,4):C.R2GC_2244_1588,(1,1,4):C.R2GC_2245_1589,(2,2,3):C.R2GC_2240_1583,(2,19,3):C.R2GC_2240_1583})

V_241 = CTVertex(name = 'V_241',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS111, L.FFVS164, L.FFVS87 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,2,1):C.R2GC_1263_433,(0,1,0):C.R2GC_877_1956,(0,0,0):C.R2GC_879_1958})

V_242 = CTVertex(name = 'V_242',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS111, L.FFVS142, L.FFVS146, L.FFVS163, L.FFVS164 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,1,1):C.R2GC_1288_458,(0,2,1):C.R2GC_1291_461,(0,3,0):C.R2GC_891_1970,(0,4,0):C.R2GC_898_1976,(0,0,0):C.R2GC_900_1978})

V_243 = CTVertex(name = 'V_243',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS112, L.FFVS117, L.FFVS120, L.FFVS125, L.FFVS130, L.FFVS160, L.FFVS86, L.FFVS94 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,6,0):C.R2GC_693_1863,(0,4,1):C.R2GC_626_1809,(0,7,1):C.R2GC_1276_446,(0,3,1):C.R2GC_632_1815,(0,3,0):C.R2GC_2188_1548,(0,1,1):C.R2GC_625_1808,(0,2,1):C.R2GC_633_1816,(0,5,0):C.R2GC_878_1957,(0,0,0):C.R2GC_880_1959})

V_244 = CTVertex(name = 'V_244',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS101, L.FFVS96 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_956_2007,(0,1,1):C.R2GC_1553_1036})

V_245 = CTVertex(name = 'V_245',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS93, L.FFVVS97 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1557_1040,(0,1,0):C.R2GC_967_2018})

V_246 = CTVertex(name = 'V_246',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.a, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS103, L.FFVVS107 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_965_2016,(0,0,1):C.R2GC_1560_1043})

V_247 = CTVertex(name = 'V_247',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.a, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS103, L.FFVVS107 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_964_2015,(0,0,1):C.R2GC_1559_1042})

V_248 = CTVertex(name = 'V_248',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV107, L.FFVV117, L.FFVV124, L.FFVV127, L.FFVV138, L.FFVV149, L.FFVV151 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,3,0):C.R2GC_988_2030,(0,6,2):C.R2GC_1601_1104,(0,5,1):C.R2GC_814_1910,(0,4,1):C.R2GC_812_1909,(0,2,1):C.R2GC_835_1921,(0,1,1):C.R2GC_836_1922,(0,0,1):C.R2GC_1990_1403})

V_249 = CTVertex(name = 'V_249',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS122, L.FFVVS123 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,1,1):C.R2GC_1542_1025,(0,0,0):C.R2GC_954_2005})

V_250 = CTVertex(name = 'V_250',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS124, L.FFVVS84 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,1,1):C.R2GC_1541_1024,(0,0,0):C.R2GC_955_2006})

V_251 = CTVertex(name = 'V_251',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133, L.FFVV143, L.FFVV144, L.FFVV147, L.FFVV170, L.FFVV172, L.FFVV195 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,1,2):C.R2GC_1586_1086,(0,2,1):C.R2GC_778_1887,(0,3,1):C.R2GC_780_1888,(0,0,2):C.R2GC_1965_1398,(0,5,1):C.R2GC_1499_970,(0,6,1):C.R2GC_1587_1087,(0,6,2):C.R2GC_1587_1088,(0,4,0):C.R2GC_985_2027})

V_252 = CTVertex(name = 'V_252',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS100, L.FFVVS87, L.FFVVS88, L.FFVVS98 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_953_2004,(0,3,1):C.R2GC_863_1944,(0,2,1):C.R2GC_865_1946,(0,1,1):C.R2GC_2088_1482})

V_253 = CTVertex(name = 'V_253',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.W__plus__, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS103, L.FFVVS105, L.FFVVS107 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1527_1010,(0,1,1):C.R2GC_1543_1026,(0,1,0):C.R2GC_2373_1682,(0,2,0):C.R2GC_949_2000})

V_254 = CTVertex(name = 'V_254',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS93, L.FFVVS95, L.FFVVS97 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_947_1998,(0,1,1):C.R2GC_1545_1028,(0,1,0):C.R2GC_2370_1679,(0,0,1):C.R2GC_1530_1013})

V_255 = CTVertex(name = 'V_255',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS93, L.FFVVS95, L.FFVVS97 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_946_1997,(0,1,1):C.R2GC_1546_1029,(0,1,0):C.R2GC_2371_1680,(0,0,1):C.R2GC_1529_1012})

V_256 = CTVertex(name = 'V_256',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV107, L.FFVV116, L.FFVV118, L.FFVV120, L.FFVV122, L.FFVV133, L.FFVV136, L.FFVV137, L.FFVV139, L.FFVV150, L.FFVV152 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,4,0):C.R2GC_984_2026,(0,3,1):C.R2GC_1500_971,(0,3,0):C.R2GC_2426_1731,(0,10,2):C.R2GC_1575_1074,(0,9,1):C.R2GC_756_1873,(0,5,2):C.R2GC_1971_1402,(0,6,1):C.R2GC_782_1890,(0,8,1):C.R2GC_753_1871,(0,7,1):C.R2GC_1968_1399,(0,2,1):C.R2GC_762_1876,(0,1,1):C.R2GC_763_1877,(0,0,1):C.R2GC_1958_1396})

V_257 = CTVertex(name = 'V_257',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS128, L.FFVS149, L.FFVS153, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,3,2):C.R2GC_1540_1023,(0,3,3):C.R2GC_2366_1675,(0,3,1):C.R2GC_2366_1676,(0,2,0):C.R2GC_656_1832,(0,1,3):C.R2GC_882_1961,(0,0,2):C.R2GC_1532_1015})

V_258 = CTVertex(name = 'V_258',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.g, P.g, P.G__plus__ ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS119 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.t] ], [ [P.b, P.t] ], [ [P.g] ] ],
                 couplings = {(2,1,2):C.R2GC_2167_1524,(2,1,0):C.R2GC_2168_1526,(2,1,1):C.R2GC_1537_1020,(1,1,2):C.R2GC_2164_1520,(1,1,0):C.R2GC_2165_1522,(1,1,1):C.R2GC_1536_1019,(2,0,2):C.R2GC_2164_1520,(2,0,0):C.R2GC_2166_1523,(1,0,2):C.R2GC_2167_1524,(1,0,0):C.R2GC_2169_1527,(1,0,1):C.R2GC_1538_1021,(0,0,0):C.R2GC_1573_1072,(0,0,1):C.R2GC_855_1937})

V_259 = CTVertex(name = 'V_259',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS144, L.FFVS162 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_959_2010,(0,0,1):C.R2GC_1552_1035})

V_260 = CTVertex(name = 'V_260',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS144, L.FFVVS148 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1556_1039,(0,1,0):C.R2GC_966_2017})

V_261 = CTVertex(name = 'V_261',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.a, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS128, L.FFVVS132 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_963_2014,(0,0,1):C.R2GC_1558_1041})

V_262 = CTVertex(name = 'V_262',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.a, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS128, L.FFVVS132 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_964_2015,(0,0,1):C.R2GC_1559_1042})

V_263 = CTVertex(name = 'V_263',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV139, L.FFVV150, L.FFVV152, L.FFVV154, L.FFVV166, L.FFVV174, L.FFVV177 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,6,0):C.R2GC_988_2030,(0,2,2):C.R2GC_1601_1104,(0,1,1):C.R2GC_814_1910,(0,0,1):C.R2GC_811_1908,(0,5,1):C.R2GC_835_1921,(0,4,1):C.R2GC_836_1922,(0,3,1):C.R2GC_1990_1403})

V_264 = CTVertex(name = 'V_264',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.W__minus__, P.W__plus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS118, L.FFVVS134, L.FFVVS136 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,3,0):C.R2GC_952_2003,(0,2,1):C.R2GC_862_1943,(0,1,1):C.R2GC_864_1945,(0,0,1):C.R2GC_2089_1483})

V_265 = CTVertex(name = 'V_265',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS128, L.FFVVS130, L.FFVVS132 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1526_1009,(0,1,1):C.R2GC_1544_1027,(0,1,0):C.R2GC_2374_1683,(0,2,0):C.R2GC_948_1999})

V_266 = CTVertex(name = 'V_266',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS144, L.FFVVS146, L.FFVVS148 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_945_1996,(0,1,1):C.R2GC_1547_1030,(0,1,0):C.R2GC_2372_1681,(0,0,1):C.R2GC_1528_1011})

V_267 = CTVertex(name = 'V_267',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS144, L.FFVVS146, L.FFVVS148 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_946_1997,(0,1,1):C.R2GC_1546_1029,(0,1,0):C.R2GC_2371_1680,(0,0,1):C.R2GC_1529_1012})

V_268 = CTVertex(name = 'V_268',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133, L.FFVV136, L.FFVV137, L.FFVV138, L.FFVV149, L.FFVV151, L.FFVV154, L.FFVV167, L.FFVV181, L.FFVV183, L.FFVV184 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,10,0):C.R2GC_984_2026,(0,9,1):C.R2GC_1500_971,(0,9,0):C.R2GC_2426_1731,(0,5,2):C.R2GC_1575_1074,(0,4,1):C.R2GC_756_1873,(0,0,2):C.R2GC_1970_1401,(0,1,1):C.R2GC_783_1891,(0,3,1):C.R2GC_754_1872,(0,2,1):C.R2GC_1969_1400,(0,8,1):C.R2GC_762_1876,(0,7,1):C.R2GC_763_1877,(0,6,1):C.R2GC_1958_1396})

V_269 = CTVertex(name = 'V_269',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF4 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1684_1173})

V_270 = CTVertex(name = 'V_270',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF4 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_1684_1173})

V_271 = CTVertex(name = 'V_271',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF4, L.FF5, L.FF6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,1,1):C.R2GC_1678_1168,(0,1,0):[ C.R2GC_1706_1192, C.R2GC_2248_1592 ],(0,0,0):[ C.R2GC_1684_1173, C.R2GC_2246_1590 ],(0,2,0):C.R2GC_2226_1574})

V_272 = CTVertex(name = 'V_272',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF4 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_1684_1173})

V_273 = CTVertex(name = 'V_273',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF4 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_1684_1173})

V_274 = CTVertex(name = 'V_274',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF4 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_1684_1173})

V_275 = CTVertex(name = 'V_275',
                 type = 'R2',
                 particles = [ P.g, P.g ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VV5, L.VV6, L.VV7 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.g] ], [ [P.t] ] ],
                 couplings = {(0,2,1):C.R2GC_567_1754,(0,0,2):C.R2GC_597_1781,(0,1,0):C.R2GC_1299_469})

V_276 = CTVertex(name = 'V_276',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS33 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_666_1842})

V_277 = CTVertex(name = 'V_277',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS33 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_666_1842})

V_278 = CTVertex(name = 'V_278',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS33 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_666_1842})

V_279 = CTVertex(name = 'V_279',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS33 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_666_1842})

V_280 = CTVertex(name = 'V_280',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS33 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_666_1842})

V_281 = CTVertex(name = 'V_281',
                 type = 'R2',
                 particles = [ P.g, P.g, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVS10, L.VVS9 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1421_860,(0,1,1):C.R2GC_1421_861,(0,1,2):C.R2GC_1421_862,(0,1,3):C.R2GC_1421_863,(0,0,3):C.R2GC_594_1778})

V_282 = CTVertex(name = 'V_282',
                 type = 'R2',
                 particles = [ P.g, P.g, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVV6 ],
                 loop_particles = [ [ [P.b] ], [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1369_683,(0,0,2):C.R2GC_1369_684,(0,0,0):C.R2GC_1433_908,(0,0,3):C.R2GC_1433_909,(0,0,4):C.R2GC_1433_910,(0,0,5):C.R2GC_1433_911})

V_283 = CTVertex(name = 'V_283',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS24 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_642_1824})

V_284 = CTVertex(name = 'V_284',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS24 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_642_1824})

V_285 = CTVertex(name = 'V_285',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS24 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_642_1824})

V_286 = CTVertex(name = 'V_286',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS24 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_642_1824})

V_287 = CTVertex(name = 'V_287',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS24 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_642_1824})

V_288 = CTVertex(name = 'V_288',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS24 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_642_1824})

V_289 = CTVertex(name = 'V_289',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS24 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_642_1824})

V_290 = CTVertex(name = 'V_290',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS24 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_642_1824})

V_291 = CTVertex(name = 'V_291',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS24 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_642_1824})

V_292 = CTVertex(name = 'V_292',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS24 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_642_1824})

V_293 = CTVertex(name = 'V_293',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS86, L.FFVS94 ],
                 loop_particles = [ [ [P.g, P.u] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_693_1863,(0,2,1):C.R2GC_1279_449,(0,0,1):C.R2GC_1281_451})

V_294 = CTVertex(name = 'V_294',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS86, L.FFVS94 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_693_1863,(0,2,1):C.R2GC_1279_449,(0,0,1):C.R2GC_1281_451})

V_295 = CTVertex(name = 'V_295',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS86, L.FFVS94 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_667_1843,(0,2,1):C.R2GC_1280_450,(0,0,1):C.R2GC_1278_448})

V_296 = CTVertex(name = 'V_296',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS86, L.FFVS94 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_667_1843,(0,2,1):C.R2GC_1280_450,(0,0,1):C.R2GC_1278_448})

V_297 = CTVertex(name = 'V_297',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS86, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_667_1843,(0,2,1):C.R2GC_1277_447,(0,0,1):C.R2GC_1278_448})

V_298 = CTVertex(name = 'V_298',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS120, L.FFVS86, L.FFVS94 ],
                 loop_particles = [ [ [P.g, P.u] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1942_1389,(0,2,1):C.R2GC_1285_455,(0,0,1):C.R2GC_1287_457})

V_299 = CTVertex(name = 'V_299',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS120, L.FFVS86, L.FFVS94 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1942_1389,(0,2,1):C.R2GC_1285_455,(0,0,1):C.R2GC_1287_457})

V_300 = CTVertex(name = 'V_300',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS120, L.FFVS86, L.FFVS94 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1942_1389,(0,2,1):C.R2GC_1286_456,(0,0,1):C.R2GC_1284_454})

V_301 = CTVertex(name = 'V_301',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS120, L.FFVS86, L.FFVS94 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1942_1389,(0,2,1):C.R2GC_1286_456,(0,0,1):C.R2GC_1284_454})

V_302 = CTVertex(name = 'V_302',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS120, L.FFVS86, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1942_1389,(0,2,1):C.R2GC_1283_453,(0,0,1):C.R2GC_1284_454})

V_303 = CTVertex(name = 'V_303',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV104, L.FFVV143, L.FFVV145, L.FFVV148, L.FFVV157, L.FFVV195, L.FFVV196, L.FFVV197 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1309_491,(0,1,1):C.R2GC_1309_492,(0,1,2):C.R2GC_1309_493,(0,3,0):C.R2GC_553_1745,(0,2,4):C.R2GC_1013_19,(0,7,4):C.R2GC_579_1766,(0,4,4):C.R2GC_1144_247,(0,5,0):C.R2GC_1311_497,(0,5,1):C.R2GC_1311_498,(0,5,2):C.R2GC_1311_499,(0,6,4):C.R2GC_580_1767,(0,0,3):C.R2GC_2235_1578})

V_304 = CTVertex(name = 'V_304',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV104, L.FFVV143, L.FFVV145, L.FFVV148, L.FFVV157, L.FFVV195, L.FFVV196, L.FFVV197 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1378_709,(0,1,1):C.R2GC_1378_710,(0,1,2):C.R2GC_1378_711,(0,3,0):C.R2GC_999_2039,(0,2,4):C.R2GC_1170_277,(0,7,4):C.R2GC_603_1787,(0,4,4):C.R2GC_1225_379,(0,5,0):C.R2GC_1380_715,(0,5,1):C.R2GC_1380_716,(0,5,2):C.R2GC_1380_717,(0,5,4):C.R2GC_1380_718,(0,6,4):C.R2GC_604_1788,(0,0,3):C.R2GC_2254_1593})

V_305 = CTVertex(name = 'V_305',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV104, L.FFVV143, L.FFVV145, L.FFVV148, L.FFVV157, L.FFVV195, L.FFVV196, L.FFVV197 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1409_812,(0,1,1):C.R2GC_1409_813,(0,1,2):C.R2GC_1409_814,(0,3,0):C.R2GC_1002_3,(0,2,4):C.R2GC_1228_382,(0,7,4):C.R2GC_613_1796,(0,4,4):C.R2GC_1245_411,(0,5,0):C.R2GC_1411_818,(0,5,1):C.R2GC_1411_819,(0,5,2):C.R2GC_1411_820,(0,5,4):C.R2GC_1411_821,(0,6,4):C.R2GC_614_1797,(0,0,3):C.R2GC_2257_1596})

V_306 = CTVertex(name = 'V_306',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.g ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVV104, L.FFVV140, L.FFVV143, L.FFVV157, L.FFVV192, L.FFVV195, L.FFVV197 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_1318_520,(0,2,1):C.R2GC_1318_521,(0,2,2):C.R2GC_1318_522,(0,6,4):C.R2GC_1319_525,(0,1,0):C.R2GC_1031_46,(0,1,4):C.R2GC_1031_47,(0,3,4):C.R2GC_1146_249,(0,5,0):C.R2GC_1320_526,(0,5,1):C.R2GC_1320_527,(0,5,2):C.R2GC_1320_528,(0,4,4):C.R2GC_585_1770,(0,0,3):C.R2GC_2237_1579})

V_307 = CTVertex(name = 'V_307',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV134, L.FFVV143, L.FFVV157, L.FFVV169, L.FFVV191, L.FFVV195, L.FFVV197 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1384_731,(0,1,1):C.R2GC_1384_732,(0,1,2):C.R2GC_1384_733,(0,6,4):C.R2GC_1385_736,(0,0,0):C.R2GC_1182_297,(0,0,4):C.R2GC_1182_298,(0,2,4):C.R2GC_1226_380,(0,2,3):C.R2GC_2255_1594,(0,5,0):C.R2GC_1386_737,(0,5,1):C.R2GC_1386_738,(0,5,2):C.R2GC_1386_739,(0,5,4):C.R2GC_1386_740,(0,4,4):C.R2GC_607_1790,(0,3,3):C.R2GC_915_1992})

V_308 = CTVertex(name = 'V_308',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV134, L.FFVV143, L.FFVV154, L.FFVV171 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,3,0):C.R2GC_990_2032,(0,1,2):C.R2GC_1608_1111,(0,0,1):C.R2GC_1504_975,(0,2,1):C.R2GC_1512_983})

V_309 = CTVertex(name = 'V_309',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV107, L.FFVV123, L.FFVV134, L.FFVV143 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,1,0):C.R2GC_989_2031,(0,3,2):C.R2GC_1608_1111,(0,2,1):C.R2GC_1504_975,(0,0,1):C.R2GC_1512_983})

V_310 = CTVertex(name = 'V_310',
                 type = 'R2',
                 particles = [ P.g, P.g, P.G0, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS12, L.VVSS13 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1325_545,(0,1,1):C.R2GC_1325_546,(0,1,2):C.R2GC_1325_547,(0,1,3):C.R2GC_1325_548,(0,0,3):C.R2GC_629_1812})

V_311 = CTVertex(name = 'V_311',
                 type = 'R2',
                 particles = [ P.g, P.g, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVS10 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1426_880,(0,0,1):C.R2GC_1426_881,(0,0,2):C.R2GC_1426_882,(0,0,3):C.R2GC_1426_883})

V_312 = CTVertex(name = 'V_312',
                 type = 'R2',
                 particles = [ P.a, P.g, P.g, P.G0 ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VVVS18 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1422_864,(0,0,1):C.R2GC_1422_865,(0,0,2):C.R2GC_1422_866,(0,0,3):C.R2GC_1422_867})

V_313 = CTVertex(name = 'V_313',
                 type = 'R2',
                 particles = [ P.g, P.g, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVS18 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1432_904,(0,0,1):C.R2GC_1432_905,(0,0,2):C.R2GC_1432_906,(0,0,3):C.R2GC_1432_907})

V_314 = CTVertex(name = 'V_314',
                 type = 'R2',
                 particles = [ P.g, P.g, P.g, P.G0 ],
                 color = [ 'd(1,2,3)', 'f(1,2,3)' ],
                 lorentz = [ L.VVVS11, L.VVVS12, L.VVVS18 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(1,0,0):C.R2GC_1423_868,(1,0,1):C.R2GC_1423_869,(1,0,2):C.R2GC_1423_870,(1,0,3):C.R2GC_1423_871,(0,2,0):C.R2GC_1424_872,(0,2,1):C.R2GC_1424_873,(0,2,2):C.R2GC_1424_874,(0,2,3):C.R2GC_1424_875,(1,1,3):C.R2GC_596_1780})

V_315 = CTVertex(name = 'V_315',
                 type = 'R2',
                 particles = [ P.g, P.g, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVS10, L.VVVS18 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1514_989,(0,0,3):C.R2GC_1514_990,(0,0,4):C.R2GC_1514_991,(0,0,5):C.R2GC_1514_992,(0,0,1):C.R2GC_1514_993,(0,1,1):C.R2GC_1623_1141,(0,1,2):C.R2GC_1623_1142})

V_316 = CTVertex(name = 'V_316',
                 type = 'R2',
                 particles = [ P.g, P.g, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVS10, L.VVVS18 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1513_984,(0,0,3):C.R2GC_1513_985,(0,0,4):C.R2GC_1513_986,(0,0,5):C.R2GC_1513_987,(0,0,1):C.R2GC_1513_988,(0,1,1):C.R2GC_1623_1141,(0,1,2):C.R2GC_1623_1142})

V_317 = CTVertex(name = 'V_317',
                 type = 'R2',
                 particles = [ P.a, P.a, P.g, P.g ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.VVVV22 ],
                 loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                 couplings = {(0,0,0):[ C.R2GC_1300_470, C.R2GC_1434_912 ],(0,0,1):[ C.R2GC_1300_471, C.R2GC_1434_913 ]})

V_318 = CTVertex(name = 'V_318',
                 type = 'R2',
                 particles = [ P.a, P.g, P.g, P.Z ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VVVV22 ],
                 loop_particles = [ [ [P.b] ], [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1373_695,(0,0,2):C.R2GC_1373_696,(0,0,0):C.R2GC_1437_920,(0,0,3):C.R2GC_1437_921,(0,0,4):C.R2GC_1437_922,(0,0,5):C.R2GC_1437_923})

V_319 = CTVertex(name = 'V_319',
                 type = 'R2',
                 particles = [ P.g, P.g, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVV22 ],
                 loop_particles = [ [ [P.b] ], [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1408_810,(0,0,2):C.R2GC_1408_811,(0,0,0):C.R2GC_1439_928,(0,0,3):C.R2GC_1439_929,(0,0,4):C.R2GC_1439_930,(0,0,5):C.R2GC_1439_931})

V_320 = CTVertex(name = 'V_320',
                 type = 'R2',
                 particles = [ P.g, P.g, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVV22 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.b, P.t], [P.c, P.s], [P.d, P.u] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,1):C.R2GC_1585_1085,(0,0,0):C.R2GC_1626_1155,(0,0,2):C.R2GC_1626_1156})

V_321 = CTVertex(name = 'V_321',
                 type = 'R2',
                 particles = [ P.a, P.g, P.g, P.g ],
                 color = [ 'd(2,3,4)' ],
                 lorentz = [ L.VVVV22 ],
                 loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                 couplings = {(0,0,0):[ C.R2GC_1301_472, C.R2GC_1436_918 ],(0,0,1):[ C.R2GC_1301_473, C.R2GC_1436_919 ]})

V_322 = CTVertex(name = 'V_322',
                 type = 'R2',
                 particles = [ P.g, P.g, P.g, P.Z ],
                 color = [ 'd(1,2,3)', 'f(1,2,3)' ],
                 lorentz = [ L.VVVV13, L.VVVV22 ],
                 loop_particles = [ [ [P.b] ], [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(1,0,1):C.R2GC_1370_685,(1,0,2):C.R2GC_1370_686,(1,0,0):C.R2GC_1435_914,(1,0,3):C.R2GC_1435_915,(1,0,4):C.R2GC_1435_916,(1,0,5):C.R2GC_1435_917,(0,1,1):C.R2GC_1374_697,(0,1,2):C.R2GC_1374_698,(0,1,0):C.R2GC_1438_924,(0,1,3):C.R2GC_1438_925,(0,1,4):C.R2GC_1438_926,(0,1,5):C.R2GC_1438_927})

V_323 = CTVertex(name = 'V_323',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.a, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.g, P.u] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_681_1854,(0,1,1):C.R2GC_1254_424,(0,2,1):C.R2GC_1256_426})

V_324 = CTVertex(name = 'V_324',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.a, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_681_1854,(0,1,1):C.R2GC_1254_424,(0,2,1):C.R2GC_1256_426})

V_325 = CTVertex(name = 'V_325',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_619_1802,(0,0,0):C.R2GC_2219_1567,(0,1,1):C.R2GC_1251_421,(0,2,1):C.R2GC_620_1803})

V_326 = CTVertex(name = 'V_326',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.a, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_652_1828,(0,1,1):C.R2GC_1255_425,(0,2,1):C.R2GC_1253_423})

V_327 = CTVertex(name = 'V_327',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.a, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_652_1828,(0,1,1):C.R2GC_1255_425,(0,2,1):C.R2GC_1253_423})

V_328 = CTVertex(name = 'V_328',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.a, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_652_1828,(0,1,1):C.R2GC_1252_422,(0,2,1):C.R2GC_1253_423})

V_329 = CTVertex(name = 'V_329',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.H, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_621_1804,(0,0,0):C.R2GC_2220_1568,(0,1,1):C.R2GC_1257_427,(0,2,1):C.R2GC_622_1805})

V_330 = CTVertex(name = 'V_330',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.g, P.H, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.g, P.u] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1935_1388,(0,1,1):C.R2GC_1260_430,(0,2,1):C.R2GC_1262_432})

V_331 = CTVertex(name = 'V_331',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.g, P.H, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1935_1388,(0,1,1):C.R2GC_1260_430,(0,2,1):C.R2GC_1262_432})

V_332 = CTVertex(name = 'V_332',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.g, P.H, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1935_1388,(0,1,1):C.R2GC_1261_431,(0,2,1):C.R2GC_1259_429})

V_333 = CTVertex(name = 'V_333',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.g, P.H, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1935_1388,(0,1,1):C.R2GC_1261_431,(0,2,1):C.R2GC_1259_429})

V_334 = CTVertex(name = 'V_334',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.g, P.H, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1935_1388,(0,1,1):C.R2GC_1258_428,(0,2,1):C.R2GC_1259_429})

V_335 = CTVertex(name = 'V_335',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.a, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.g, P.u] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_681_1854,(0,1,1):C.R2GC_1254_424,(0,2,1):C.R2GC_1256_426})

V_336 = CTVertex(name = 'V_336',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.a, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_681_1854,(0,1,1):C.R2GC_1254_424,(0,2,1):C.R2GC_1256_426})

V_337 = CTVertex(name = 'V_337',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_619_1802,(0,0,0):C.R2GC_2219_1567,(0,1,1):C.R2GC_1251_421,(0,2,1):C.R2GC_620_1803})

V_338 = CTVertex(name = 'V_338',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.a, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_652_1828,(0,1,1):C.R2GC_1255_425,(0,2,1):C.R2GC_1253_423})

V_339 = CTVertex(name = 'V_339',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.a, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_652_1828,(0,1,1):C.R2GC_1255_425,(0,2,1):C.R2GC_1253_423})

V_340 = CTVertex(name = 'V_340',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.a, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_652_1828,(0,1,1):C.R2GC_1252_422,(0,2,1):C.R2GC_1253_423})

V_341 = CTVertex(name = 'V_341',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.G0, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_621_1804,(0,0,0):C.R2GC_2220_1568,(0,1,1):C.R2GC_1257_427,(0,2,1):C.R2GC_622_1805})

V_342 = CTVertex(name = 'V_342',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.g, P.G0, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.g, P.u] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1935_1388,(0,1,1):C.R2GC_1260_430,(0,2,1):C.R2GC_1262_432})

V_343 = CTVertex(name = 'V_343',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.g, P.G0, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1935_1388,(0,1,1):C.R2GC_1260_430,(0,2,1):C.R2GC_1262_432})

V_344 = CTVertex(name = 'V_344',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.g, P.G0, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1935_1388,(0,1,1):C.R2GC_1261_431,(0,2,1):C.R2GC_1259_429})

V_345 = CTVertex(name = 'V_345',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.g, P.G0, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1935_1388,(0,1,1):C.R2GC_1261_431,(0,2,1):C.R2GC_1259_429})

V_346 = CTVertex(name = 'V_346',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.g, P.G0, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1935_1388,(0,1,1):C.R2GC_1258_428,(0,2,1):C.R2GC_1259_429})

V_347 = CTVertex(name = 'V_347',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.g, P.G__plus__, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.t], [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1519_1006,(0,0,1):C.R2GC_2405_1699})

V_348 = CTVertex(name = 'V_348',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.g, P.G0, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.t], [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1518_1005,(0,0,1):C.R2GC_2406_1700})

V_349 = CTVertex(name = 'V_349',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.g, P.G__minus__, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.t], [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1517_1004,(0,0,1):C.R2GC_2407_1701})

V_350 = CTVertex(name = 'V_350',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.g, P.G0, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.t], [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1518_1005,(0,0,1):C.R2GC_2406_1700})

V_351 = CTVertex(name = 'V_351',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_621_1804,(0,0,2):C.R2GC_1935_1388,(0,1,1):C.R2GC_842_1926,(0,2,1):C.R2GC_622_1805,(0,2,0):C.R2GC_2408_1702})

V_352 = CTVertex(name = 'V_352',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1935_1388,(0,1,2):C.R2GC_1520_1007,(0,1,1):C.R2GC_2408_1702,(0,2,2):C.R2GC_1259_429})

V_353 = CTVertex(name = 'V_353',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,1):C.R2GC_1935_1388,(0,1,0):C.R2GC_1261_431,(0,2,0):C.R2GC_1262_432})

V_354 = CTVertex(name = 'V_354',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g] ] ],
                 couplings = {(0,0,1):C.R2GC_1935_1388,(0,1,0):C.R2GC_1261_431,(0,2,0):C.R2GC_1262_432})

V_355 = CTVertex(name = 'V_355',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g] ] ],
                 couplings = {(0,0,1):C.R2GC_1935_1388,(0,1,0):C.R2GC_1260_430,(0,2,0):C.R2GC_1259_429})

V_356 = CTVertex(name = 'V_356',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.g, P.s] ] ],
                 couplings = {(0,0,1):C.R2GC_1935_1388,(0,1,0):C.R2GC_1260_430,(0,2,0):C.R2GC_1259_429})

V_357 = CTVertex(name = 'V_357',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS81 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1265_435,(0,0,0):C.R2GC_2158_1516})

V_358 = CTVertex(name = 'V_358',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS81 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1293_463,(0,0,0):C.R2GC_2207_1556})

V_359 = CTVertex(name = 'V_359',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.Z, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS81 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1854_1282,(0,0,0):C.R2GC_2215_1563})

V_360 = CTVertex(name = 'V_360',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.g, P.H ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVS81 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1268_438,(0,0,0):C.R2GC_2162_1518})

V_361 = CTVertex(name = 'V_361',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.Z, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS111, L.FFVVS137, L.FFVVS81 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,2,1):C.R2GC_1294_464,(0,1,0):C.R2GC_893_1972,(0,0,0):C.R2GC_903_1981})

V_362 = CTVertex(name = 'V_362',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.g, P.W__minus__, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS87, L.FFVVS99 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_968_2019,(0,0,1):C.R2GC_1564_1047})

V_363 = CTVertex(name = 'V_363',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.g, P.W__plus__, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS125 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_969_2020,(0,0,1):C.R2GC_1564_1047})

V_364 = CTVertex(name = 'V_364',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS80 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1264_434,(0,0,0):C.R2GC_2159_1517})

V_365 = CTVertex(name = 'V_365',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS80 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1292_462,(0,0,0):C.R2GC_2208_1557})

V_366 = CTVertex(name = 'V_366',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.Z, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS80 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1853_1281,(0,0,0):C.R2GC_2216_1564})

V_367 = CTVertex(name = 'V_367',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.g, P.G0 ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVS80 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1269_439,(0,0,0):C.R2GC_2163_1519})

V_368 = CTVertex(name = 'V_368',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.Z, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS112, L.FFVVS138, L.FFVVS80 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,2,1):C.R2GC_1295_465,(0,1,0):C.R2GC_892_1971,(0,0,0):C.R2GC_904_1982})

V_369 = CTVertex(name = 'V_369',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.g, P.W__minus__, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS87, L.FFVVS99 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_970_2021,(0,0,1):C.R2GC_1563_1046})

V_370 = CTVertex(name = 'V_370',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.g, P.W__plus__, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS125 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_970_2021,(0,0,1):C.R2GC_1565_1048})

V_371 = CTVertex(name = 'V_371',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.a, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS110 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2157_1514,(0,0,2):C.R2GC_2157_1515,(0,0,1):C.R2GC_2070_1474})

V_372 = CTVertex(name = 'V_372',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.a, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS110 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2206_1554,(0,0,2):C.R2GC_2206_1555,(0,0,1):C.R2GC_2120_1500})

V_373 = CTVertex(name = 'V_373',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.Z, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS110 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2214_1561,(0,0,2):C.R2GC_2214_1562,(0,0,1):C.R2GC_2126_1506})

V_374 = CTVertex(name = 'V_374',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.a, P.g, P.G__plus__ ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS142, L.FFVVS149 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_657_1833,(0,2,2):C.R2GC_885_1964,(0,0,1):C.R2GC_1534_1017})

V_375 = CTVertex(name = 'V_375',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.g, P.Z, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS125, L.FFVVS135 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1448_940,(0,2,2):C.R2GC_902_1980,(0,0,1):C.R2GC_1571_1066})

V_376 = CTVertex(name = 'V_376',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.W__minus__, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS125 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1562_1045,(0,1,0):C.R2GC_971_2022})

V_377 = CTVertex(name = 'V_377',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.a, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS87 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2156_1512,(0,0,2):C.R2GC_2156_1513,(0,0,1):C.R2GC_2069_1473})

V_378 = CTVertex(name = 'V_378',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.a, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS87 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2205_1552,(0,0,2):C.R2GC_2205_1553,(0,0,1):C.R2GC_2119_1499})

V_379 = CTVertex(name = 'V_379',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.Z, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS87 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2213_1559,(0,0,2):C.R2GC_2213_1560,(0,0,1):C.R2GC_2125_1505})

V_380 = CTVertex(name = 'V_380',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.a, P.g, P.G__minus__ ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVS101, L.FFVVS108, L.FFVVS87 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_657_1833,(0,0,2):C.R2GC_885_1964,(0,2,1):C.R2GC_1533_1016})

V_381 = CTVertex(name = 'V_381',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.g, P.Z, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS87, L.FFVVS90, L.FFVVS99 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_1448_940,(0,1,2):C.R2GC_902_1980,(0,0,1):C.R2GC_1570_1065})

V_382 = CTVertex(name = 'V_382',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.W__plus__, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS87, L.FFVVS99 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1561_1044,(0,1,0):C.R2GC_971_2022})

V_383 = CTVertex(name = 'V_383',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.g, P.g ],
                 color = [ 'd(3,4,5)*Identity(1,2)', 'f(3,4,5)*Identity(1,2)', 'T(3,-1,1)*T(4,-2,-1)*T(5,2,-2)', 'T(3,-1,1)*T(4,2,-2)*T(5,-2,-1)', 'T(3,-2,-1)*T(4,-1,1)*T(5,2,-2)', 'T(3,-2,-1)*T(4,2,-2)*T(5,-1,1)', 'T(3,2,-2)*T(4,-1,1)*T(5,-2,-1)', 'T(3,2,-2)*T(4,-2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV146, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV152, L.FFVVV153, L.FFVVV154, L.FFVVV79, L.FFVVV80, L.FFVVV81, L.FFVVV82, L.FFVVV84, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.G__plus__] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.G0, P.t], [P.H, P.t] ], [ [P.t] ] ],
                 couplings = {(1,1,0):C.R2GC_1362_661,(1,1,2):C.R2GC_1362_662,(1,1,3):C.R2GC_1362_663,(1,1,5):C.R2GC_1361_657,(1,9,5):C.R2GC_1106_177,(0,0,0):C.R2GC_1359_650,(0,0,2):C.R2GC_1359_651,(0,0,3):C.R2GC_1359_652,(0,0,5):C.R2GC_1359_653,(0,7,5):C.R2GC_1105_176,(7,1,0):C.R2GC_1107_178,(7,1,2):C.R2GC_1363_664,(7,1,3):C.R2GC_1363_665,(7,1,5):C.R2GC_1363_666,(7,17,0):C.R2GC_1894_1332,(7,17,2):C.R2GC_1893_1329,(7,17,3):C.R2GC_1893_1330,(7,17,5):C.R2GC_1894_1333,(7,16,0):C.R2GC_1900_1350,(7,16,2):C.R2GC_1899_1347,(7,16,3):C.R2GC_1899_1348,(7,16,5):C.R2GC_1900_1351,(7,15,0):C.R2GC_1894_1332,(7,15,2):C.R2GC_1893_1329,(7,15,3):C.R2GC_1893_1330,(7,15,5):C.R2GC_1894_1333,(6,1,0):C.R2GC_1112_184,(6,1,2):C.R2GC_1358_648,(6,1,3):C.R2GC_1358_649,(6,1,5):C.R2GC_1357_647,(6,17,0):C.R2GC_1893_1328,(6,17,2):C.R2GC_1893_1329,(6,17,3):C.R2GC_1893_1330,(6,17,5):C.R2GC_1893_1331,(6,16,0):C.R2GC_1894_1332,(6,16,2):C.R2GC_1893_1329,(6,16,3):C.R2GC_1893_1330,(6,16,5):C.R2GC_1894_1333,(6,15,0):C.R2GC_1900_1350,(6,15,2):C.R2GC_1899_1347,(6,15,3):C.R2GC_1899_1348,(6,15,5):C.R2GC_1900_1351,(5,1,0):C.R2GC_1112_184,(5,1,2):C.R2GC_1358_648,(5,1,3):C.R2GC_1358_649,(5,1,5):C.R2GC_1357_647,(5,17,0):C.R2GC_1900_1350,(5,17,2):C.R2GC_1899_1347,(5,17,3):C.R2GC_1899_1348,(5,17,5):C.R2GC_1900_1351,(5,16,0):C.R2GC_1894_1332,(5,16,2):C.R2GC_1893_1329,(5,16,3):C.R2GC_1893_1330,(5,16,5):C.R2GC_1894_1333,(5,16,4):C.R2GC_2272_1603,(5,15,0):C.R2GC_1893_1328,(5,15,2):C.R2GC_1893_1329,(5,15,3):C.R2GC_1893_1330,(5,15,5):C.R2GC_1893_1331,(5,15,4):C.R2GC_2278_1605,(3,1,0):C.R2GC_1107_178,(3,1,2):C.R2GC_1363_664,(3,1,3):C.R2GC_1363_665,(3,1,5):C.R2GC_1363_666,(3,17,0):C.R2GC_1894_1332,(3,17,2):C.R2GC_1893_1329,(3,17,3):C.R2GC_1893_1330,(3,17,5):C.R2GC_1894_1333,(3,17,4):C.R2GC_2272_1603,(3,16,0):C.R2GC_1893_1328,(3,16,2):C.R2GC_1893_1329,(3,16,3):C.R2GC_1893_1330,(3,16,5):C.R2GC_1893_1331,(3,16,4):C.R2GC_2278_1605,(3,15,0):C.R2GC_1899_1346,(3,15,2):C.R2GC_1899_1347,(3,15,3):C.R2GC_1899_1348,(3,15,5):C.R2GC_1899_1349,(3,15,4):C.R2GC_1947_1394,(4,1,0):C.R2GC_1107_178,(4,1,2):C.R2GC_1363_664,(4,1,3):C.R2GC_1363_665,(4,1,5):C.R2GC_1363_666,(4,17,0):C.R2GC_1899_1346,(4,17,2):C.R2GC_1899_1347,(4,17,3):C.R2GC_1899_1348,(4,17,5):C.R2GC_1899_1349,(4,17,4):C.R2GC_1947_1394,(4,16,0):C.R2GC_1893_1328,(4,16,2):C.R2GC_1893_1329,(4,16,3):C.R2GC_1893_1330,(4,16,5):C.R2GC_1893_1331,(4,16,4):C.R2GC_2278_1605,(4,15,0):C.R2GC_1894_1332,(4,15,2):C.R2GC_1893_1329,(4,15,3):C.R2GC_1893_1330,(4,15,5):C.R2GC_1894_1333,(4,15,4):C.R2GC_2272_1603,(2,1,0):C.R2GC_1112_184,(2,1,2):C.R2GC_1358_648,(2,1,3):C.R2GC_1358_649,(2,1,5):C.R2GC_1357_647,(2,17,0):C.R2GC_1893_1328,(2,17,2):C.R2GC_1893_1329,(2,17,3):C.R2GC_1893_1330,(2,17,5):C.R2GC_1893_1331,(2,17,4):C.R2GC_2278_1605,(2,16,0):C.R2GC_1899_1346,(2,16,2):C.R2GC_1899_1347,(2,16,3):C.R2GC_1899_1348,(2,16,5):C.R2GC_1899_1349,(2,16,4):C.R2GC_1947_1394,(2,15,0):C.R2GC_1893_1328,(2,15,2):C.R2GC_1893_1329,(2,15,3):C.R2GC_1893_1330,(2,15,5):C.R2GC_1893_1331,(2,15,4):C.R2GC_2278_1605,(7,14,0):C.R2GC_1101_172,(7,14,5):C.R2GC_1101_173,(6,14,0):C.R2GC_1102_174,(6,14,5):C.R2GC_1102_175,(5,14,0):C.R2GC_1102_174,(5,14,5):C.R2GC_1102_175,(5,14,4):C.R2GC_1945_1392,(3,14,0):C.R2GC_1101_172,(3,14,5):C.R2GC_1101_173,(3,14,4):C.R2GC_2274_1604,(4,14,0):C.R2GC_1101_172,(4,14,5):C.R2GC_1101_173,(4,14,4):C.R2GC_2274_1604,(2,14,0):C.R2GC_1102_174,(2,14,5):C.R2GC_1102_175,(2,14,4):C.R2GC_1945_1392,(1,8,0):C.R2GC_1367_676,(1,8,2):C.R2GC_1367_677,(1,8,3):C.R2GC_1367_678,(1,8,5):C.R2GC_1367_679,(0,6,0):C.R2GC_1366_672,(0,6,2):C.R2GC_1366_673,(0,6,3):C.R2GC_1366_674,(0,6,5):C.R2GC_1366_675,(7,8,0):C.R2GC_1368_680,(7,8,2):C.R2GC_1368_681,(7,8,3):C.R2GC_1368_682,(7,8,5):C.R2GC_1357_647,(7,5,0):C.R2GC_1901_1352,(7,5,2):C.R2GC_1901_1353,(7,5,3):C.R2GC_1901_1354,(7,5,5):C.R2GC_1902_1356,(7,5,1):C.R2GC_1945_1392,(7,4,0):C.R2GC_1903_1357,(7,4,2):C.R2GC_1903_1358,(7,4,3):C.R2GC_1903_1359,(7,4,5):C.R2GC_1904_1361,(7,3,0):C.R2GC_1901_1352,(7,3,2):C.R2GC_1901_1353,(7,3,3):C.R2GC_1901_1354,(7,3,5):C.R2GC_1902_1356,(7,3,1):C.R2GC_1945_1392,(6,8,0):C.R2GC_1365_669,(6,8,2):C.R2GC_1365_670,(6,8,3):C.R2GC_1365_671,(6,8,5):C.R2GC_1363_666,(6,5,0):C.R2GC_1901_1352,(6,5,2):C.R2GC_1901_1353,(6,5,3):C.R2GC_1901_1354,(6,5,5):C.R2GC_1901_1355,(6,5,1):C.R2GC_1948_1395,(6,4,0):C.R2GC_1901_1352,(6,4,2):C.R2GC_1901_1353,(6,4,3):C.R2GC_1901_1354,(6,4,5):C.R2GC_1902_1356,(6,4,1):C.R2GC_1945_1392,(6,3,0):C.R2GC_1903_1357,(6,3,2):C.R2GC_1903_1358,(6,3,3):C.R2GC_1903_1359,(6,3,5):C.R2GC_1904_1361,(5,8,0):C.R2GC_1365_669,(5,8,2):C.R2GC_1365_670,(5,8,3):C.R2GC_1365_671,(5,8,5):C.R2GC_1363_666,(5,5,0):C.R2GC_1903_1357,(5,5,2):C.R2GC_1903_1358,(5,5,3):C.R2GC_1903_1359,(5,5,5):C.R2GC_1904_1361,(5,4,0):C.R2GC_1901_1352,(5,4,2):C.R2GC_1901_1353,(5,4,3):C.R2GC_1901_1354,(5,4,5):C.R2GC_1902_1356,(5,4,1):C.R2GC_1945_1392,(5,4,4):C.R2GC_2272_1603,(5,3,0):C.R2GC_1901_1352,(5,3,2):C.R2GC_1901_1353,(5,3,3):C.R2GC_1901_1354,(5,3,5):C.R2GC_1901_1355,(5,3,1):C.R2GC_1948_1395,(5,3,4):C.R2GC_2278_1605,(3,8,0):C.R2GC_1368_680,(3,8,2):C.R2GC_1368_681,(3,8,3):C.R2GC_1368_682,(3,8,5):C.R2GC_1357_647,(3,5,0):C.R2GC_1901_1352,(3,5,2):C.R2GC_1901_1353,(3,5,3):C.R2GC_1901_1354,(3,5,5):C.R2GC_1902_1356,(3,5,1):C.R2GC_1945_1392,(3,5,4):C.R2GC_2272_1603,(3,4,0):C.R2GC_1901_1352,(3,4,2):C.R2GC_1901_1353,(3,4,3):C.R2GC_1901_1354,(3,4,5):C.R2GC_1901_1355,(3,4,1):C.R2GC_1948_1395,(3,4,4):C.R2GC_2278_1605,(3,3,0):C.R2GC_1903_1357,(3,3,2):C.R2GC_1903_1358,(3,3,3):C.R2GC_1903_1359,(3,3,5):C.R2GC_1903_1360,(3,3,1):C.R2GC_2265_1600,(3,3,4):C.R2GC_1947_1394,(4,8,0):C.R2GC_1368_680,(4,8,2):C.R2GC_1368_681,(4,8,3):C.R2GC_1368_682,(4,8,5):C.R2GC_1357_647,(4,5,0):C.R2GC_1903_1357,(4,5,2):C.R2GC_1903_1358,(4,5,3):C.R2GC_1903_1359,(4,5,5):C.R2GC_1903_1360,(4,5,1):C.R2GC_2265_1600,(4,5,4):C.R2GC_1947_1394,(4,4,0):C.R2GC_1901_1352,(4,4,2):C.R2GC_1901_1353,(4,4,3):C.R2GC_1901_1354,(4,4,5):C.R2GC_1901_1355,(4,4,1):C.R2GC_1948_1395,(4,4,4):C.R2GC_2278_1605,(4,3,0):C.R2GC_1901_1352,(4,3,2):C.R2GC_1901_1353,(4,3,3):C.R2GC_1901_1354,(4,3,5):C.R2GC_1902_1356,(4,3,1):C.R2GC_1945_1392,(4,3,4):C.R2GC_2272_1603,(2,8,0):C.R2GC_1365_669,(2,8,2):C.R2GC_1365_670,(2,8,3):C.R2GC_1365_671,(2,8,5):C.R2GC_1363_666,(2,5,0):C.R2GC_1901_1352,(2,5,2):C.R2GC_1901_1353,(2,5,3):C.R2GC_1901_1354,(2,5,5):C.R2GC_1901_1355,(2,5,1):C.R2GC_1948_1395,(2,5,4):C.R2GC_2278_1605,(2,4,0):C.R2GC_1903_1357,(2,4,2):C.R2GC_1903_1358,(2,4,3):C.R2GC_1903_1359,(2,4,5):C.R2GC_1903_1360,(2,4,1):C.R2GC_2265_1600,(2,4,4):C.R2GC_1947_1394,(2,3,0):C.R2GC_1901_1352,(2,3,2):C.R2GC_1901_1353,(2,3,3):C.R2GC_1901_1354,(2,3,5):C.R2GC_1901_1355,(2,3,1):C.R2GC_1948_1395,(2,3,4):C.R2GC_2278_1605,(7,2,5):C.R2GC_591_1775,(7,2,1):C.R2GC_1946_1393,(6,2,5):C.R2GC_592_1776,(6,2,1):C.R2GC_1947_1394,(5,2,5):C.R2GC_592_1776,(5,2,1):C.R2GC_1947_1394,(5,2,4):C.R2GC_1945_1392,(3,2,5):C.R2GC_591_1775,(3,2,1):C.R2GC_1946_1393,(3,2,4):C.R2GC_2274_1604,(4,2,5):C.R2GC_591_1775,(4,2,1):C.R2GC_1946_1393,(4,2,4):C.R2GC_2274_1604,(2,2,5):C.R2GC_592_1776,(2,2,1):C.R2GC_1947_1394,(2,2,4):C.R2GC_1945_1392,(7,10,4):C.R2GC_2274_1604,(6,10,4):C.R2GC_1945_1392,(7,11,4):C.R2GC_2272_1603,(6,12,4):C.R2GC_2272_1603,(7,13,4):C.R2GC_2272_1603,(6,13,4):C.R2GC_2278_1605})

V_384 = CTVertex(name = 'V_384',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.g, P.g, P.g ],
                 color = [ 'd(3,4,5)*Identity(1,2)', 'f(3,4,5)*Identity(1,2)', 'T(3,-1,1)*T(4,-2,-1)*T(5,2,-2)', 'T(3,-1,1)*T(4,2,-2)*T(5,-2,-1)', 'T(3,-2,-1)*T(4,-1,1)*T(5,2,-2)', 'T(3,-2,-1)*T(4,2,-2)*T(5,-1,1)', 'T(3,2,-2)*T(4,-1,1)*T(5,-2,-1)', 'T(3,2,-2)*T(4,-2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV152, L.FFVVV153, L.FFVVV154, L.FFVVV84, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.G__plus__, P.t] ], [ [P.t] ] ],
                 couplings = {(1,8,0):C.R2GC_996_2036,(1,1,0):C.R2GC_1361_657,(1,1,1):C.R2GC_1361_658,(1,1,2):C.R2GC_1361_659,(1,1,4):C.R2GC_1361_660,(0,6,0):C.R2GC_1109_181,(0,0,0):C.R2GC_1359_653,(0,0,1):C.R2GC_1360_654,(0,0,2):C.R2GC_1360_655,(0,0,4):C.R2GC_1360_656,(7,1,0):C.R2GC_1107_178,(7,1,1):C.R2GC_1364_667,(7,1,2):C.R2GC_1364_668,(7,1,4):C.R2GC_1363_666,(7,12,0):C.R2GC_1896_1338,(7,12,1):C.R2GC_1895_1335,(7,12,2):C.R2GC_1895_1336,(7,12,4):C.R2GC_1896_1339,(7,12,3):C.R2GC_1945_1392,(7,11,0):C.R2GC_1898_1344,(7,11,1):C.R2GC_1897_1341,(7,11,2):C.R2GC_1897_1342,(7,11,4):C.R2GC_1898_1345,(7,10,0):C.R2GC_1896_1338,(7,10,1):C.R2GC_1895_1335,(7,10,2):C.R2GC_1895_1336,(7,10,4):C.R2GC_1896_1339,(7,10,3):C.R2GC_1945_1392,(6,1,0):C.R2GC_1112_184,(6,1,1):C.R2GC_1357_645,(6,1,2):C.R2GC_1357_646,(6,1,4):C.R2GC_1357_647,(6,12,0):C.R2GC_1895_1334,(6,12,1):C.R2GC_1895_1335,(6,12,2):C.R2GC_1895_1336,(6,12,4):C.R2GC_1895_1337,(6,12,3):C.R2GC_1948_1395,(6,11,0):C.R2GC_1896_1338,(6,11,1):C.R2GC_1895_1335,(6,11,2):C.R2GC_1895_1336,(6,11,4):C.R2GC_1896_1339,(6,11,3):C.R2GC_1945_1392,(6,10,0):C.R2GC_1898_1344,(6,10,1):C.R2GC_1897_1341,(6,10,2):C.R2GC_1897_1342,(6,10,4):C.R2GC_1898_1345,(5,1,0):C.R2GC_1112_184,(5,1,1):C.R2GC_1357_645,(5,1,2):C.R2GC_1357_646,(5,1,4):C.R2GC_1357_647,(5,12,0):C.R2GC_1898_1344,(5,12,1):C.R2GC_1897_1341,(5,12,2):C.R2GC_1897_1342,(5,12,4):C.R2GC_1898_1345,(5,11,0):C.R2GC_1896_1338,(5,11,1):C.R2GC_1895_1335,(5,11,2):C.R2GC_1895_1336,(5,11,4):C.R2GC_1896_1339,(5,11,3):C.R2GC_1945_1392,(5,10,0):C.R2GC_1895_1334,(5,10,1):C.R2GC_1895_1335,(5,10,2):C.R2GC_1895_1336,(5,10,4):C.R2GC_1895_1337,(5,10,3):C.R2GC_1948_1395,(3,1,0):C.R2GC_1107_178,(3,1,1):C.R2GC_1364_667,(3,1,2):C.R2GC_1364_668,(3,1,4):C.R2GC_1363_666,(3,12,0):C.R2GC_1896_1338,(3,12,1):C.R2GC_1895_1335,(3,12,2):C.R2GC_1895_1336,(3,12,4):C.R2GC_1896_1339,(3,12,3):C.R2GC_1945_1392,(3,11,0):C.R2GC_1895_1334,(3,11,1):C.R2GC_1895_1335,(3,11,2):C.R2GC_1895_1336,(3,11,4):C.R2GC_1895_1337,(3,11,3):C.R2GC_1948_1395,(3,10,0):C.R2GC_1897_1340,(3,10,1):C.R2GC_1897_1341,(3,10,2):C.R2GC_1897_1342,(3,10,4):C.R2GC_1897_1343,(3,10,3):C.R2GC_2265_1600,(4,1,0):C.R2GC_1107_178,(4,1,1):C.R2GC_1364_667,(4,1,2):C.R2GC_1364_668,(4,1,4):C.R2GC_1363_666,(4,12,0):C.R2GC_1897_1340,(4,12,1):C.R2GC_1897_1341,(4,12,2):C.R2GC_1897_1342,(4,12,4):C.R2GC_1897_1343,(4,12,3):C.R2GC_2265_1600,(4,11,0):C.R2GC_1895_1334,(4,11,1):C.R2GC_1895_1335,(4,11,2):C.R2GC_1895_1336,(4,11,4):C.R2GC_1895_1337,(4,11,3):C.R2GC_1948_1395,(4,10,0):C.R2GC_1896_1338,(4,10,1):C.R2GC_1895_1335,(4,10,2):C.R2GC_1895_1336,(4,10,4):C.R2GC_1896_1339,(4,10,3):C.R2GC_1945_1392,(2,1,0):C.R2GC_1112_184,(2,1,1):C.R2GC_1357_645,(2,1,2):C.R2GC_1357_646,(2,1,4):C.R2GC_1357_647,(2,12,0):C.R2GC_1895_1334,(2,12,1):C.R2GC_1895_1335,(2,12,2):C.R2GC_1895_1336,(2,12,4):C.R2GC_1895_1337,(2,12,3):C.R2GC_1948_1395,(2,11,0):C.R2GC_1897_1340,(2,11,1):C.R2GC_1897_1341,(2,11,2):C.R2GC_1897_1342,(2,11,4):C.R2GC_1897_1343,(2,11,3):C.R2GC_2265_1600,(2,10,0):C.R2GC_1895_1334,(2,10,1):C.R2GC_1895_1335,(2,10,2):C.R2GC_1895_1336,(2,10,4):C.R2GC_1895_1337,(2,10,3):C.R2GC_1948_1395,(7,9,0):C.R2GC_1101_173,(7,9,4):C.R2GC_1101_172,(7,9,3):C.R2GC_1946_1393,(6,9,0):C.R2GC_1102_175,(6,9,4):C.R2GC_1102_174,(6,9,3):C.R2GC_1947_1394,(5,9,0):C.R2GC_1102_175,(5,9,4):C.R2GC_1102_174,(5,9,3):C.R2GC_1947_1394,(3,9,0):C.R2GC_1101_173,(3,9,4):C.R2GC_1101_172,(3,9,3):C.R2GC_1946_1393,(4,9,0):C.R2GC_1101_173,(4,9,4):C.R2GC_1101_172,(4,9,3):C.R2GC_1946_1393,(2,9,0):C.R2GC_1102_175,(2,9,4):C.R2GC_1102_174,(2,9,3):C.R2GC_1947_1394,(1,7,4):C.R2GC_1110_182,(0,5,4):C.R2GC_1108_180,(7,7,0):C.R2GC_1112_184,(7,7,4):C.R2GC_1112_185,(7,4,0):C.R2GC_1786_1226,(7,4,4):C.R2GC_1786_1227,(7,3,0):C.R2GC_1787_1228,(7,3,4):C.R2GC_1787_1229,(7,2,0):C.R2GC_1786_1226,(7,2,4):C.R2GC_1786_1227,(6,7,0):C.R2GC_1107_178,(6,7,4):C.R2GC_1107_179,(6,4,0):C.R2GC_1786_1226,(6,4,4):C.R2GC_1786_1227,(6,3,0):C.R2GC_1786_1226,(6,3,4):C.R2GC_1786_1227,(6,2,0):C.R2GC_1787_1228,(6,2,4):C.R2GC_1787_1229,(5,7,0):C.R2GC_1107_178,(5,7,4):C.R2GC_1107_179,(5,4,0):C.R2GC_1787_1228,(5,4,4):C.R2GC_1787_1229,(5,3,0):C.R2GC_1786_1226,(5,3,4):C.R2GC_1786_1227,(5,2,0):C.R2GC_1786_1226,(5,2,4):C.R2GC_1786_1227,(3,7,0):C.R2GC_1112_184,(3,7,4):C.R2GC_1112_185,(3,4,0):C.R2GC_1786_1226,(3,4,4):C.R2GC_1786_1227,(3,3,0):C.R2GC_1786_1226,(3,3,4):C.R2GC_1786_1227,(3,2,0):C.R2GC_1787_1228,(3,2,4):C.R2GC_1787_1229,(4,7,0):C.R2GC_1112_184,(4,7,4):C.R2GC_1112_185,(4,4,0):C.R2GC_1787_1228,(4,4,4):C.R2GC_1787_1229,(4,3,0):C.R2GC_1786_1226,(4,3,4):C.R2GC_1786_1227,(4,2,0):C.R2GC_1786_1226,(4,2,4):C.R2GC_1786_1227,(2,7,0):C.R2GC_1107_178,(2,7,4):C.R2GC_1107_179,(2,4,0):C.R2GC_1786_1226,(2,4,4):C.R2GC_1786_1227,(2,3,0):C.R2GC_1787_1228,(2,3,4):C.R2GC_1787_1229,(2,2,0):C.R2GC_1786_1226,(2,2,4):C.R2GC_1786_1227})

V_385 = CTVertex(name = 'V_385',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(4,5)', 'T(4,-1,1)*T(5,2,-1)', 'T(4,2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV146, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV152, L.FFVVV153, L.FFVVV79, L.FFVVV80, L.FFVVV81, L.FFVVV84, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.G__plus__] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.G0, P.t], [P.H, P.t] ], [ [P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_1075_122,(2,1,2):C.R2GC_1346_609,(2,1,3):C.R2GC_1346_610,(2,1,5):C.R2GC_1346_611,(1,1,0):C.R2GC_1079_128,(1,1,2):C.R2GC_1339_587,(1,1,3):C.R2GC_1339_588,(1,1,5):C.R2GC_1338_586,(0,0,0):C.R2GC_1340_589,(0,0,2):C.R2GC_1340_590,(0,0,3):C.R2GC_1340_591,(0,0,5):C.R2GC_1340_592,(0,7,5):C.R2GC_1074_121,(2,15,0):C.R2GC_1345_607,(2,15,2):C.R2GC_1344_604,(2,15,3):C.R2GC_1344_605,(2,15,5):C.R2GC_1345_608,(2,15,4):C.R2GC_1944_1391,(1,15,0):C.R2GC_1344_603,(1,15,2):C.R2GC_1344_604,(1,15,3):C.R2GC_1344_605,(1,15,5):C.R2GC_1344_606,(1,15,4):C.R2GC_1943_1390,(2,14,0):C.R2GC_1344_603,(2,14,2):C.R2GC_1344_604,(2,14,3):C.R2GC_1344_605,(2,14,5):C.R2GC_1344_606,(1,14,0):C.R2GC_1345_607,(1,14,2):C.R2GC_1344_604,(1,14,3):C.R2GC_1344_605,(1,14,5):C.R2GC_1345_608,(2,13,0):C.R2GC_1345_607,(2,13,2):C.R2GC_1344_604,(2,13,3):C.R2GC_1344_605,(2,13,5):C.R2GC_1345_608,(1,13,0):C.R2GC_1344_603,(1,13,2):C.R2GC_1344_604,(1,13,3):C.R2GC_1344_605,(1,13,5):C.R2GC_1344_606,(2,12,0):C.R2GC_1070_113,(2,12,5):C.R2GC_1070_114,(1,12,0):C.R2GC_1071_115,(1,12,5):C.R2GC_1071_116,(2,8,0):C.R2GC_1352_626,(2,8,2):C.R2GC_1352_627,(2,8,3):C.R2GC_1352_628,(2,8,5):C.R2GC_1338_586,(1,8,0):C.R2GC_1348_614,(1,8,2):C.R2GC_1348_615,(1,8,3):C.R2GC_1348_616,(1,8,5):C.R2GC_1346_611,(0,6,0):C.R2GC_1349_617,(0,6,2):C.R2GC_1349_618,(0,6,3):C.R2GC_1349_619,(0,6,5):C.R2GC_1349_620,(2,5,0):C.R2GC_1350_621,(2,5,2):C.R2GC_1350_622,(2,5,3):C.R2GC_1350_623,(2,5,5):C.R2GC_1351_625,(2,5,1):C.R2GC_1943_1390,(2,5,4):C.R2GC_1944_1391,(1,5,0):C.R2GC_1350_621,(1,5,2):C.R2GC_1350_622,(1,5,3):C.R2GC_1350_623,(1,5,5):C.R2GC_1350_624,(1,5,1):C.R2GC_1944_1391,(1,5,4):C.R2GC_1943_1390,(2,4,0):C.R2GC_1350_621,(2,4,2):C.R2GC_1350_622,(2,4,3):C.R2GC_1350_623,(2,4,5):C.R2GC_1350_624,(2,4,1):C.R2GC_1944_1391,(1,4,0):C.R2GC_1350_621,(1,4,2):C.R2GC_1350_622,(1,4,3):C.R2GC_1350_623,(1,4,5):C.R2GC_1351_625,(1,4,1):C.R2GC_1943_1390,(2,3,0):C.R2GC_1350_621,(2,3,2):C.R2GC_1350_622,(2,3,3):C.R2GC_1350_623,(2,3,5):C.R2GC_1351_625,(2,3,1):C.R2GC_1943_1390,(1,3,0):C.R2GC_1350_621,(1,3,2):C.R2GC_1350_622,(1,3,3):C.R2GC_1350_623,(1,3,5):C.R2GC_1350_624,(1,3,1):C.R2GC_1944_1391,(2,2,5):C.R2GC_589_1773,(2,2,1):C.R2GC_1944_1391,(1,2,5):C.R2GC_590_1774,(1,2,1):C.R2GC_1943_1390,(2,9,4):C.R2GC_1943_1390,(1,9,4):C.R2GC_1944_1391,(2,10,4):C.R2GC_1944_1391,(1,10,4):C.R2GC_1943_1390,(2,11,4):C.R2GC_1943_1390,(1,11,4):C.R2GC_1944_1391})

V_386 = CTVertex(name = 'V_386',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.a, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(4,5)', 'T(4,-1,1)*T(5,2,-1)', 'T(4,2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV152, L.FFVVV153, L.FFVVV84, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.G__plus__, P.t] ], [ [P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_1075_122,(2,1,1):C.R2GC_1347_612,(2,1,2):C.R2GC_1347_613,(2,1,4):C.R2GC_1346_611,(1,1,0):C.R2GC_1079_128,(1,1,1):C.R2GC_1338_584,(1,1,2):C.R2GC_1338_585,(1,1,4):C.R2GC_1338_586,(0,6,0):C.R2GC_1077_125,(0,0,0):C.R2GC_1341_593,(0,0,1):C.R2GC_1341_594,(0,0,2):C.R2GC_1341_595,(0,0,4):C.R2GC_1341_596,(2,11,0):C.R2GC_1343_601,(2,11,1):C.R2GC_1342_598,(2,11,2):C.R2GC_1342_599,(2,11,4):C.R2GC_1343_602,(2,11,3):C.R2GC_2260_1599,(1,11,0):C.R2GC_1342_597,(1,11,1):C.R2GC_1342_598,(1,11,2):C.R2GC_1342_599,(1,11,4):C.R2GC_1342_600,(1,11,3):C.R2GC_2259_1598,(2,10,0):C.R2GC_1342_597,(2,10,1):C.R2GC_1342_598,(2,10,2):C.R2GC_1342_599,(2,10,4):C.R2GC_1342_600,(2,10,3):C.R2GC_2259_1598,(1,10,0):C.R2GC_1343_601,(1,10,1):C.R2GC_1342_598,(1,10,2):C.R2GC_1342_599,(1,10,4):C.R2GC_1343_602,(1,10,3):C.R2GC_2260_1599,(2,9,0):C.R2GC_1343_601,(2,9,1):C.R2GC_1342_598,(2,9,2):C.R2GC_1342_599,(2,9,4):C.R2GC_1343_602,(2,9,3):C.R2GC_2260_1599,(1,9,0):C.R2GC_1342_597,(1,9,1):C.R2GC_1342_598,(1,9,2):C.R2GC_1342_599,(1,9,4):C.R2GC_1342_600,(1,9,3):C.R2GC_2259_1598,(2,8,0):C.R2GC_1072_117,(2,8,4):C.R2GC_1072_118,(2,8,3):C.R2GC_2259_1598,(1,8,0):C.R2GC_1073_119,(1,8,4):C.R2GC_1073_120,(1,8,3):C.R2GC_2260_1599,(2,7,0):C.R2GC_1079_128,(2,7,4):C.R2GC_1079_129,(1,7,0):C.R2GC_1075_122,(1,7,4):C.R2GC_1075_123,(0,5,4):C.R2GC_1076_124,(2,4,0):C.R2GC_1078_126,(2,4,4):C.R2GC_1078_127,(1,4,0):C.R2GC_1078_126,(1,4,4):C.R2GC_1078_127,(2,3,0):C.R2GC_1078_126,(2,3,4):C.R2GC_1078_127,(1,3,0):C.R2GC_1078_126,(1,3,4):C.R2GC_1078_127,(2,2,0):C.R2GC_1078_126,(2,2,4):C.R2GC_1078_127,(1,2,0):C.R2GC_1078_126,(1,2,4):C.R2GC_1078_127})

V_387 = CTVertex(name = 'V_387',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.g, P.Z ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV145, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV152, L.FFVVV153, L.FFVVV83, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.G__plus__] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.G0, P.t], [P.H, P.t] ], [ [P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_1401_787,(2,1,2):C.R2GC_1401_788,(2,1,3):C.R2GC_1401_789,(2,1,5):C.R2GC_1401_790,(1,1,0):C.R2GC_1393_761,(1,1,2):C.R2GC_1394_765,(1,1,3):C.R2GC_1394_766,(1,1,5):C.R2GC_1393_764,(0,0,0):C.R2GC_1395_767,(0,0,2):C.R2GC_1395_768,(0,0,3):C.R2GC_1395_769,(0,0,5):C.R2GC_1395_770,(0,7,5):C.R2GC_1198_327,(2,12,0):C.R2GC_1399_781,(2,12,2):C.R2GC_1399_782,(2,12,3):C.R2GC_1399_783,(2,12,5):C.R2GC_1399_784,(2,12,4):C.R2GC_2282_1606,(1,12,0):C.R2GC_1400_785,(1,12,2):C.R2GC_1399_782,(1,12,3):C.R2GC_1399_783,(1,12,5):C.R2GC_1400_786,(1,12,4):C.R2GC_2284_1609,(2,11,0):C.R2GC_1400_785,(2,11,2):C.R2GC_1399_782,(2,11,3):C.R2GC_1399_783,(2,11,5):C.R2GC_1400_786,(2,11,4):C.R2GC_2284_1609,(1,11,0):C.R2GC_1399_781,(1,11,2):C.R2GC_1399_782,(1,11,3):C.R2GC_1399_783,(1,11,5):C.R2GC_1399_784,(1,11,4):C.R2GC_2282_1606,(2,10,0):C.R2GC_1400_785,(2,10,2):C.R2GC_1399_782,(2,10,3):C.R2GC_1399_783,(2,10,5):C.R2GC_1400_786,(2,10,4):C.R2GC_2284_1609,(1,10,0):C.R2GC_1399_781,(1,10,2):C.R2GC_1399_782,(1,10,3):C.R2GC_1399_783,(1,10,5):C.R2GC_1399_784,(1,10,4):C.R2GC_2282_1606,(2,9,0):C.R2GC_1194_319,(2,9,5):C.R2GC_1194_320,(2,9,4):C.R2GC_2282_1606,(1,9,0):C.R2GC_1195_321,(1,9,5):C.R2GC_1195_322,(1,9,4):C.R2GC_2284_1609,(2,8,0):C.R2GC_1407_806,(2,8,2):C.R2GC_1407_807,(2,8,3):C.R2GC_1407_808,(2,8,5):C.R2GC_1407_809,(1,8,0):C.R2GC_1403_793,(1,8,2):C.R2GC_1403_794,(1,8,3):C.R2GC_1403_795,(1,8,5):C.R2GC_1403_796,(0,6,0):C.R2GC_1404_797,(0,6,2):C.R2GC_1404_798,(0,6,3):C.R2GC_1404_799,(0,6,5):C.R2GC_1404_800,(2,5,0):C.R2GC_1405_801,(2,5,2):C.R2GC_1405_802,(2,5,3):C.R2GC_1405_803,(2,5,5):C.R2GC_1405_804,(2,5,1):C.R2GC_2283_1607,(2,5,4):C.R2GC_2283_1608,(1,5,0):C.R2GC_1405_801,(1,5,2):C.R2GC_1405_802,(1,5,3):C.R2GC_1405_803,(1,5,5):C.R2GC_1406_805,(1,5,1):C.R2GC_2285_1610,(1,5,4):C.R2GC_2285_1611,(2,4,0):C.R2GC_1405_801,(2,4,2):C.R2GC_1405_802,(2,4,3):C.R2GC_1405_803,(2,4,5):C.R2GC_1406_805,(2,4,1):C.R2GC_2285_1610,(2,4,4):C.R2GC_2285_1611,(1,4,0):C.R2GC_1405_801,(1,4,2):C.R2GC_1405_802,(1,4,3):C.R2GC_1405_803,(1,4,5):C.R2GC_1405_804,(1,4,1):C.R2GC_2283_1607,(1,4,4):C.R2GC_2283_1608,(2,3,0):C.R2GC_1405_801,(2,3,2):C.R2GC_1405_802,(2,3,3):C.R2GC_1405_803,(2,3,5):C.R2GC_1406_805,(2,3,1):C.R2GC_2285_1610,(2,3,4):C.R2GC_2285_1611,(1,3,0):C.R2GC_1405_801,(1,3,2):C.R2GC_1405_802,(1,3,3):C.R2GC_1405_803,(1,3,5):C.R2GC_1405_804,(1,3,1):C.R2GC_2283_1607,(1,3,4):C.R2GC_2283_1608,(2,2,5):C.R2GC_609_1792,(2,2,1):C.R2GC_2283_1607,(2,2,4):C.R2GC_2283_1608,(1,2,5):C.R2GC_610_1793,(1,2,1):C.R2GC_2285_1610,(1,2,4):C.R2GC_2285_1611})

V_388 = CTVertex(name = 'V_388',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.g, P.g, P.Z ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV152, L.FFVVV153, L.FFVVV83, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.G__plus__, P.t] ], [ [P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_1401_787,(2,1,1):C.R2GC_1402_791,(2,1,2):C.R2GC_1402_792,(2,1,4):C.R2GC_1401_790,(1,1,0):C.R2GC_1393_761,(1,1,1):C.R2GC_1393_762,(1,1,2):C.R2GC_1393_763,(1,1,4):C.R2GC_1393_764,(0,0,0):C.R2GC_1396_771,(0,0,1):C.R2GC_1396_772,(0,0,2):C.R2GC_1396_773,(0,0,4):C.R2GC_1396_774,(0,6,0):C.R2GC_1000_1,(2,11,0):C.R2GC_1397_775,(2,11,1):C.R2GC_1397_776,(2,11,2):C.R2GC_1397_777,(2,11,4):C.R2GC_1397_778,(2,11,3):C.R2GC_2266_1601,(1,11,0):C.R2GC_1398_779,(1,11,1):C.R2GC_1397_776,(1,11,2):C.R2GC_1397_777,(1,11,4):C.R2GC_1398_780,(1,11,3):C.R2GC_2267_1602,(2,10,0):C.R2GC_1398_779,(2,10,1):C.R2GC_1397_776,(2,10,2):C.R2GC_1397_777,(2,10,4):C.R2GC_1398_780,(2,10,3):C.R2GC_2267_1602,(1,10,0):C.R2GC_1397_775,(1,10,1):C.R2GC_1397_776,(1,10,2):C.R2GC_1397_777,(1,10,4):C.R2GC_1397_778,(1,10,3):C.R2GC_2266_1601,(2,9,0):C.R2GC_1398_779,(2,9,1):C.R2GC_1397_776,(2,9,2):C.R2GC_1397_777,(2,9,4):C.R2GC_1398_780,(2,9,3):C.R2GC_2267_1602,(1,9,0):C.R2GC_1397_775,(1,9,1):C.R2GC_1397_776,(1,9,2):C.R2GC_1397_777,(1,9,4):C.R2GC_1397_778,(1,9,3):C.R2GC_2266_1601,(2,8,0):C.R2GC_1196_323,(2,8,4):C.R2GC_1196_324,(2,8,3):C.R2GC_2266_1601,(1,8,0):C.R2GC_1197_325,(1,8,4):C.R2GC_1197_326,(1,8,3):C.R2GC_2267_1602,(2,7,0):C.R2GC_1203_335,(2,7,4):C.R2GC_1203_336,(1,7,0):C.R2GC_1199_328,(1,7,4):C.R2GC_1199_329,(0,5,0):C.R2GC_1200_330,(0,5,4):C.R2GC_1200_331,(2,4,0):C.R2GC_1202_333,(2,4,4):C.R2GC_1202_334,(1,4,0):C.R2GC_1202_333,(1,4,4):C.R2GC_1202_334,(2,3,0):C.R2GC_1202_333,(2,3,4):C.R2GC_1202_334,(1,3,0):C.R2GC_1202_333,(1,3,4):C.R2GC_1202_334,(2,2,0):C.R2GC_1202_333,(2,2,4):C.R2GC_1202_334,(1,2,0):C.R2GC_1202_333,(1,2,4):C.R2GC_1202_334})

V_389 = CTVertex(name = 'V_389',
                 type = 'R2',
                 particles = [ P.g, P.g, P.Z, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVSS18 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1372_691,(0,0,1):C.R2GC_1372_692,(0,0,2):C.R2GC_1372_693,(0,0,3):C.R2GC_1372_694})

V_390 = CTVertex(name = 'V_390',
                 type = 'R2',
                 particles = [ P.a, P.g, P.g, P.G0, P.H ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VVVSS26 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1336_578,(0,0,1):C.R2GC_1336_579,(0,0,2):C.R2GC_1336_580,(0,0,3):C.R2GC_1336_581})

V_391 = CTVertex(name = 'V_391',
                 type = 'R2',
                 particles = [ P.g, P.g, P.Z, P.G0, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVSS26 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1391_755,(0,0,1):C.R2GC_1391_756,(0,0,2):C.R2GC_1391_757,(0,0,3):C.R2GC_1391_758})

V_392 = CTVertex(name = 'V_392',
                 type = 'R2',
                 particles = [ P.g, P.g, P.g, P.G0, P.H ],
                 color = [ 'd(1,2,3)', 'f(1,2,3)' ],
                 lorentz = [ L.VVVSS20, L.VVVSS26 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(1,0,0):C.R2GC_1353_629,(1,0,1):C.R2GC_1353_630,(1,0,2):C.R2GC_1353_631,(1,0,3):C.R2GC_1353_632,(0,1,0):C.R2GC_1356_641,(0,1,1):C.R2GC_1356_642,(0,1,2):C.R2GC_1356_643,(0,1,3):C.R2GC_1356_644})

V_393 = CTVertex(name = 'V_393',
                 type = 'R2',
                 particles = [ P.g, P.g, P.Z, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVSS18 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1372_691,(0,0,1):C.R2GC_1372_692,(0,0,2):C.R2GC_1372_693,(0,0,3):C.R2GC_1372_694})

V_394 = CTVertex(name = 'V_394',
                 type = 'R2',
                 particles = [ P.g, P.g, P.W__minus__, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVSS18, L.VVVSS26 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1568_1059,(0,0,3):C.R2GC_1568_1060,(0,0,4):C.R2GC_1568_1061,(0,0,5):C.R2GC_1568_1062,(0,0,1):C.R2GC_1568_1063,(0,1,1):C.R2GC_1611_1114,(0,1,2):C.R2GC_1611_1115})

V_395 = CTVertex(name = 'V_395',
                 type = 'R2',
                 particles = [ P.g, P.g, P.W__minus__, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVSS18, L.VVVSS26 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1566_1049,(0,0,3):C.R2GC_1566_1050,(0,0,4):C.R2GC_1566_1051,(0,0,5):C.R2GC_1566_1052,(0,0,1):C.R2GC_1566_1053,(0,1,1):C.R2GC_1613_1118,(0,1,2):C.R2GC_1613_1119})

V_396 = CTVertex(name = 'V_396',
                 type = 'R2',
                 particles = [ P.g, P.g, P.W__plus__, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVSS18, L.VVVSS26 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1567_1054,(0,0,3):C.R2GC_1567_1055,(0,0,4):C.R2GC_1567_1056,(0,0,5):C.R2GC_1567_1057,(0,0,1):C.R2GC_1567_1058,(0,1,1):C.R2GC_1611_1114,(0,1,2):C.R2GC_1611_1115})

V_397 = CTVertex(name = 'V_397',
                 type = 'R2',
                 particles = [ P.g, P.g, P.W__plus__, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVSS18, L.VVVSS26 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1566_1049,(0,0,3):C.R2GC_1566_1050,(0,0,4):C.R2GC_1566_1051,(0,0,5):C.R2GC_1566_1052,(0,0,1):C.R2GC_1566_1053,(0,1,1):C.R2GC_1612_1116,(0,1,2):C.R2GC_1612_1117})

V_398 = CTVertex(name = 'V_398',
                 type = 'R2',
                 particles = [ P.a, P.g, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VVVSS19, L.VVVSS27, L.VVVSS28, L.VVVSS30, L.VVVSS32 ],
                 loop_particles = [ [ [P.b] ], [ [P.b], [P.d], [P.s] ], [ [P.b, P.t] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,1,1):C.R2GC_1003_4,(0,2,0):C.R2GC_1004_5,(0,2,4):C.R2GC_1004_6,(0,3,3):C.R2GC_1335_576,(0,3,5):C.R2GC_1335_577,(0,4,3):C.R2GC_1337_582,(0,4,5):C.R2GC_1337_583,(0,0,2):C.R2GC_859_1940})

V_399 = CTVertex(name = 'V_399',
                 type = 'R2',
                 particles = [ P.g, P.g, P.Z, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVSS18, L.VVVSS26, L.VVVSS29, L.VVVSS31, L.VVVSS33, L.VVVSS34 ],
                 loop_particles = [ [ [P.b] ], [ [P.b], [P.d], [P.s] ], [ [P.b, P.t] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1572_1067,(0,0,3):C.R2GC_1572_1068,(0,0,4):C.R2GC_1572_1069,(0,0,5):C.R2GC_1572_1070,(0,0,2):C.R2GC_1572_1071,(0,2,1):C.R2GC_1005_7,(0,3,0):C.R2GC_1006_8,(0,3,4):C.R2GC_1006_9,(0,4,3):C.R2GC_1390_753,(0,4,5):C.R2GC_1390_754,(0,5,3):C.R2GC_1392_759,(0,5,5):C.R2GC_1392_760,(0,1,0):C.R2GC_1371_687,(0,1,3):C.R2GC_1371_688,(0,1,4):C.R2GC_1371_689,(0,1,5):C.R2GC_1371_690})

V_400 = CTVertex(name = 'V_400',
                 type = 'R2',
                 particles = [ P.a, P.g, P.g, P.Z, P.H ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VVVVS20 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1427_884,(0,0,1):C.R2GC_1427_885,(0,0,2):C.R2GC_1427_886,(0,0,3):C.R2GC_1427_887})

V_401 = CTVertex(name = 'V_401',
                 type = 'R2',
                 particles = [ P.g, P.g, P.Z, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVVS20 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1425_876,(0,0,1):C.R2GC_1425_877,(0,0,2):C.R2GC_1425_878,(0,0,3):C.R2GC_1425_879})

V_402 = CTVertex(name = 'V_402',
                 type = 'R2',
                 particles = [ P.g, P.g, P.W__minus__, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVVS20 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1622_1139,(0,0,1):C.R2GC_1622_1140})

V_403 = CTVertex(name = 'V_403',
                 type = 'R2',
                 particles = [ P.g, P.g, P.g, P.Z, P.H ],
                 color = [ 'd(1,2,3)', 'f(1,2,3)' ],
                 lorentz = [ L.VVVVS11, L.VVVVS20 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(1,0,0):C.R2GC_1431_900,(1,0,1):C.R2GC_1431_901,(1,0,2):C.R2GC_1431_902,(1,0,3):C.R2GC_1431_903,(0,1,0):C.R2GC_1430_896,(0,1,1):C.R2GC_1430_897,(0,1,2):C.R2GC_1430_898,(0,1,3):C.R2GC_1430_899})

V_404 = CTVertex(name = 'V_404',
                 type = 'R2',
                 particles = [ P.a, P.g, P.g, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VVVVS20 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1624_1143,(0,0,3):C.R2GC_1624_1144,(0,0,4):C.R2GC_1624_1145,(0,0,5):C.R2GC_1624_1146,(0,0,1):C.R2GC_1624_1147,(0,0,2):C.R2GC_1624_1148})

V_405 = CTVertex(name = 'V_405',
                 type = 'R2',
                 particles = [ P.g, P.g, P.W__minus__, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVVS20 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1620_1127,(0,0,3):C.R2GC_1620_1128,(0,0,4):C.R2GC_1620_1129,(0,0,5):C.R2GC_1620_1130,(0,0,1):C.R2GC_1620_1131,(0,0,2):C.R2GC_1620_1132})

V_406 = CTVertex(name = 'V_406',
                 type = 'R2',
                 particles = [ P.g, P.g, P.g, P.W__minus__, P.G__plus__ ],
                 color = [ 'd(1,2,3)', 'f(1,2,3)' ],
                 lorentz = [ L.VVVVS11, L.VVVVS20 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.t] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(1,0,0):C.R2GC_1516_999,(1,0,2):C.R2GC_1516_1000,(1,0,3):C.R2GC_1516_1001,(1,0,4):C.R2GC_1516_1002,(1,0,1):C.R2GC_1516_1003,(0,1,0):C.R2GC_1428_888,(0,1,2):C.R2GC_1428_889,(0,1,3):C.R2GC_1428_890,(0,1,4):C.R2GC_1428_891})

V_407 = CTVertex(name = 'V_407',
                 type = 'R2',
                 particles = [ P.a, P.g, P.g, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(2,3)' ],
                 lorentz = [ L.VVVVS20 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1625_1149,(0,0,3):C.R2GC_1625_1150,(0,0,4):C.R2GC_1625_1151,(0,0,5):C.R2GC_1625_1152,(0,0,1):C.R2GC_1625_1153,(0,0,2):C.R2GC_1625_1154})

V_408 = CTVertex(name = 'V_408',
                 type = 'R2',
                 particles = [ P.g, P.g, P.W__plus__, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVVVS20 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1621_1133,(0,0,3):C.R2GC_1621_1134,(0,0,4):C.R2GC_1621_1135,(0,0,5):C.R2GC_1621_1136,(0,0,1):C.R2GC_1621_1137,(0,0,2):C.R2GC_1621_1138})

V_409 = CTVertex(name = 'V_409',
                 type = 'R2',
                 particles = [ P.g, P.g, P.g, P.W__plus__, P.G__minus__ ],
                 color = [ 'd(1,2,3)', 'f(1,2,3)' ],
                 lorentz = [ L.VVVVS11, L.VVVVS20 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.t] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(1,0,0):C.R2GC_1515_994,(1,0,2):C.R2GC_1515_995,(1,0,3):C.R2GC_1515_996,(1,0,4):C.R2GC_1515_997,(1,0,1):C.R2GC_1515_998,(0,1,0):C.R2GC_1429_892,(0,1,2):C.R2GC_1429_893,(0,1,3):C.R2GC_1429_894,(0,1,4):C.R2GC_1429_895})

V_410 = CTVertex(name = 'V_410',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_860_1941})

V_411 = CTVertex(name = 'V_411',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_860_1941})

V_412 = CTVertex(name = 'V_412',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_861_1942})

V_413 = CTVertex(name = 'V_413',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_861_1942})

V_414 = CTVertex(name = 'V_414',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV143, L.FFVV145, L.FFVV148, L.FFVV195, L.FFVV197 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,4,0):C.R2GC_552_1744,(0,1,0):C.R2GC_992_2034,(0,0,1):C.R2GC_1310_494,(0,0,2):C.R2GC_1310_495,(0,0,3):C.R2GC_1310_496,(0,2,3):C.R2GC_578_1765,(0,3,3):C.R2GC_1014_20})

V_415 = CTVertex(name = 'V_415',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV143, L.FFVV145, L.FFVV148, L.FFVV195, L.FFVV197 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_998_2038,(0,4,0):C.R2GC_559_1747,(0,0,1):C.R2GC_1379_712,(0,0,2):C.R2GC_1379_713,(0,0,3):C.R2GC_1379_714,(0,2,3):C.R2GC_1171_278,(0,3,0):C.R2GC_1172_279,(0,3,3):C.R2GC_1172_280})

V_416 = CTVertex(name = 'V_416',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV143, L.FFVV145, L.FFVV148, L.FFVV195, L.FFVV197 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1001_2,(0,4,0):C.R2GC_563_1750,(0,0,1):C.R2GC_1410_815,(0,0,2):C.R2GC_1410_816,(0,0,3):C.R2GC_1410_817,(0,2,3):C.R2GC_1229_383,(0,3,0):C.R2GC_1230_384,(0,3,3):C.R2GC_1230_385})

V_417 = CTVertex(name = 'V_417',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133, L.FFVV143, L.FFVV146, L.FFVV153, L.FFVV195 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,1,1):C.R2GC_1586_1086,(0,2,0):C.R2GC_778_1887,(0,3,0):C.R2GC_780_1888,(0,0,1):C.R2GC_1964_1397,(0,4,0):C.R2GC_777_1886})

V_418 = CTVertex(name = 'V_418',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1017_24,(0,0,1):C.R2GC_1017_25,(0,1,0):C.R2GC_1015_21,(0,1,1):C.R2GC_1014_20})

V_419 = CTVertex(name = 'V_419',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1017_24,(0,0,1):C.R2GC_1017_25,(0,1,0):C.R2GC_1015_21,(0,1,1):C.R2GC_1014_20})

V_420 = CTVertex(name = 'V_420',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1016_22,(0,0,1):C.R2GC_1016_23,(0,1,0):C.R2GC_1018_26,(0,1,1):C.R2GC_1018_27})

V_421 = CTVertex(name = 'V_421',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1016_22,(0,0,1):C.R2GC_1016_23,(0,1,0):C.R2GC_1018_26,(0,1,1):C.R2GC_1018_27})

V_422 = CTVertex(name = 'V_422',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1175_284,(0,0,1):C.R2GC_1175_285,(0,1,0):C.R2GC_1173_281,(0,1,1):C.R2GC_1172_280})

V_423 = CTVertex(name = 'V_423',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1175_284,(0,0,1):C.R2GC_1175_285,(0,1,0):C.R2GC_1173_281,(0,1,1):C.R2GC_1172_280})

V_424 = CTVertex(name = 'V_424',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1174_282,(0,0,1):C.R2GC_1174_283,(0,1,0):C.R2GC_1176_286,(0,1,1):C.R2GC_1176_287})

V_425 = CTVertex(name = 'V_425',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1174_282,(0,0,1):C.R2GC_1174_283,(0,1,0):C.R2GC_1176_286,(0,1,1):C.R2GC_1176_287})

V_426 = CTVertex(name = 'V_426',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1233_389,(0,0,1):C.R2GC_1233_390,(0,1,0):C.R2GC_1231_386,(0,1,1):C.R2GC_1230_385})

V_427 = CTVertex(name = 'V_427',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1233_389,(0,0,1):C.R2GC_1233_390,(0,1,0):C.R2GC_1231_386,(0,1,1):C.R2GC_1230_385})

V_428 = CTVertex(name = 'V_428',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1232_387,(0,0,1):C.R2GC_1232_388,(0,1,0):C.R2GC_1234_391,(0,1,1):C.R2GC_1234_392})

V_429 = CTVertex(name = 'V_429',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1232_387,(0,0,1):C.R2GC_1232_388,(0,1,0):C.R2GC_1234_391,(0,1,1):C.R2GC_1234_392})

V_430 = CTVertex(name = 'V_430',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV152 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1601_1104})

V_431 = CTVertex(name = 'V_431',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV152 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1601_1104})

V_432 = CTVertex(name = 'V_432',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133, L.FFVV151 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1575_1074,(0,0,0):C.R2GC_1970_1401})

V_433 = CTVertex(name = 'V_433',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133, L.FFVV151 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1575_1074,(0,0,0):C.R2GC_1970_1401})

V_434 = CTVertex(name = 'V_434',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV151 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1601_1104})

V_435 = CTVertex(name = 'V_435',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV151 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1601_1104})

V_436 = CTVertex(name = 'V_436',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133, L.FFVV152 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1575_1074,(0,0,0):C.R2GC_1971_1402})

V_437 = CTVertex(name = 'V_437',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133, L.FFVV152 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1575_1074,(0,0,0):C.R2GC_1971_1402})

V_438 = CTVertex(name = 'V_438',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133, L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1586_1086,(0,0,0):C.R2GC_1964_1397,(0,2,0):C.R2GC_777_1886})

V_439 = CTVertex(name = 'V_439',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133, L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1586_1086,(0,0,0):C.R2GC_1964_1397,(0,2,0):C.R2GC_777_1886})

V_440 = CTVertex(name = 'V_440',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133, L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1586_1086,(0,0,0):C.R2GC_1965_1398,(0,2,0):C.R2GC_781_1889})

V_441 = CTVertex(name = 'V_441',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133, L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1586_1086,(0,0,0):C.R2GC_1965_1398,(0,2,0):C.R2GC_781_1889})

V_442 = CTVertex(name = 'V_442',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.a, P.g ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVV140, L.FFVV143, L.FFVV195, L.FFVV197 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,3,0):C.R2GC_1318_520,(0,1,1):C.R2GC_1319_523,(0,1,2):C.R2GC_1319_524,(0,1,3):C.R2GC_1319_525,(0,0,0):C.R2GC_1032_48,(0,0,3):C.R2GC_1032_49,(0,2,3):C.R2GC_1033_50})

V_443 = CTVertex(name = 'V_443',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV134, L.FFVV143, L.FFVV195, L.FFVV197 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,3,0):C.R2GC_1384_731,(0,1,1):C.R2GC_1385_734,(0,1,2):C.R2GC_1385_735,(0,1,3):C.R2GC_1385_736,(0,0,0):C.R2GC_1183_299,(0,0,3):C.R2GC_1183_300,(0,2,0):C.R2GC_1184_301,(0,2,3):C.R2GC_1184_302})

V_444 = CTVertex(name = 'V_444',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV108, L.FFVV109, L.FFVV110, L.FFVV111, L.FFVV112, L.FFVV113, L.FFVV114, L.FFVV115, L.FFVV129, L.FFVV130, L.FFVV131, L.FFVV132, L.FFVV141, L.FFVV142, L.FFVV143, L.FFVV158, L.FFVV159, L.FFVV160, L.FFVV162, L.FFVV163, L.FFVV165, L.FFVV189, L.FFVV190, L.FFVV193, L.FFVV194, L.FFVV195, L.FFVV197 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,26,0):C.R2GC_993_2035,(0,14,0):C.R2GC_1328_554,(0,14,1):C.R2GC_1328_555,(0,14,2):C.R2GC_1328_556,(0,14,3):C.R2GC_1328_557,(2,12,0):C.R2GC_1048_71,(2,12,1):C.R2GC_1331_564,(2,12,2):C.R2GC_1331_565,(2,12,3):C.R2GC_1330_563,(2,13,0):C.R2GC_1051_75,(2,13,1):C.R2GC_1326_549,(2,13,2):C.R2GC_1326_550,(2,13,3):C.R2GC_1326_551,(1,12,0):C.R2GC_1048_71,(1,12,1):C.R2GC_1331_564,(1,12,2):C.R2GC_1331_565,(1,12,3):C.R2GC_1330_563,(1,13,0):C.R2GC_1051_75,(1,13,1):C.R2GC_1326_549,(1,13,2):C.R2GC_1326_550,(1,13,3):C.R2GC_1326_551,(2,0,0):C.R2GC_1776_1210,(2,0,1):C.R2GC_1872_1302,(2,0,2):C.R2GC_1872_1303,(2,0,3):C.R2GC_1872_1304,(2,1,0):C.R2GC_1776_1210,(2,1,1):C.R2GC_1872_1302,(2,1,2):C.R2GC_1872_1303,(2,1,3):C.R2GC_1872_1304,(2,2,0):C.R2GC_1871_1300,(2,2,1):C.R2GC_1867_1294,(2,2,2):C.R2GC_1867_1295,(2,2,3):C.R2GC_1871_1301,(2,3,0):C.R2GC_1044_69,(2,3,3):C.R2GC_1044_68,(1,0,0):C.R2GC_1774_1208,(1,0,1):C.R2GC_1867_1294,(1,0,2):C.R2GC_1867_1295,(1,0,3):C.R2GC_1862_1287,(1,1,0):C.R2GC_1774_1208,(1,1,1):C.R2GC_1867_1294,(1,1,2):C.R2GC_1867_1295,(1,1,3):C.R2GC_1862_1287,(1,2,0):C.R2GC_1875_1307,(1,2,1):C.R2GC_1872_1302,(1,2,2):C.R2GC_1872_1303,(1,2,3):C.R2GC_1875_1308,(1,3,0):C.R2GC_1769_1207,(1,3,3):C.R2GC_1769_1206,(2,4,0):C.R2GC_1774_1208,(2,4,1):C.R2GC_1867_1294,(2,4,2):C.R2GC_1867_1295,(2,4,3):C.R2GC_1862_1287,(2,5,0):C.R2GC_1774_1208,(2,5,1):C.R2GC_1867_1294,(2,5,2):C.R2GC_1867_1295,(2,5,3):C.R2GC_1862_1287,(2,6,0):C.R2GC_1769_1207,(2,6,3):C.R2GC_1769_1206,(2,7,0):C.R2GC_1875_1307,(2,7,1):C.R2GC_1872_1302,(2,7,2):C.R2GC_1872_1303,(2,7,3):C.R2GC_1875_1308,(1,4,0):C.R2GC_1776_1210,(1,4,1):C.R2GC_1872_1302,(1,4,2):C.R2GC_1872_1303,(1,4,3):C.R2GC_1872_1304,(1,5,0):C.R2GC_1776_1210,(1,5,1):C.R2GC_1872_1302,(1,5,2):C.R2GC_1872_1303,(1,5,3):C.R2GC_1872_1304,(1,6,0):C.R2GC_1044_69,(1,6,3):C.R2GC_1044_68,(1,7,0):C.R2GC_1871_1300,(1,7,1):C.R2GC_1867_1294,(1,7,2):C.R2GC_1867_1295,(1,7,3):C.R2GC_1871_1301,(0,25,3):C.R2GC_1049_73,(2,23,0):C.R2GC_1051_75,(2,23,3):C.R2GC_1051_76,(2,24,0):C.R2GC_1048_71,(2,24,3):C.R2GC_1048_72,(1,23,0):C.R2GC_1051_75,(1,23,3):C.R2GC_1051_76,(1,24,0):C.R2GC_1048_71,(1,24,3):C.R2GC_1048_72,(2,15,0):C.R2GC_1776_1210,(2,15,3):C.R2GC_1776_1211,(2,16,0):C.R2GC_1776_1210,(2,16,3):C.R2GC_1776_1211,(2,17,0):C.R2GC_1774_1208,(2,17,3):C.R2GC_1774_1209,(1,15,0):C.R2GC_1774_1208,(1,15,3):C.R2GC_1774_1209,(1,16,0):C.R2GC_1774_1208,(1,16,3):C.R2GC_1774_1209,(1,17,0):C.R2GC_1776_1210,(1,17,3):C.R2GC_1776_1211,(2,18,0):C.R2GC_1774_1208,(2,18,3):C.R2GC_1774_1209,(2,19,0):C.R2GC_1774_1208,(2,19,3):C.R2GC_1774_1209,(2,20,0):C.R2GC_1776_1210,(2,20,3):C.R2GC_1776_1211,(1,18,0):C.R2GC_1776_1210,(1,18,3):C.R2GC_1776_1211,(1,19,0):C.R2GC_1776_1210,(1,19,3):C.R2GC_1776_1211,(1,20,0):C.R2GC_1774_1208,(1,20,3):C.R2GC_1774_1209,(2,10,0):C.R2GC_1875_1307,(2,10,1):C.R2GC_1872_1302,(2,10,2):C.R2GC_1872_1303,(2,10,3):C.R2GC_1875_1308,(1,10,0):C.R2GC_1869_1296,(1,10,1):C.R2GC_1867_1294,(1,10,2):C.R2GC_1867_1295,(1,10,3):C.R2GC_1869_1297,(2,21,0):C.R2GC_1776_1210,(2,21,3):C.R2GC_1776_1211,(1,21,0):C.R2GC_1774_1208,(1,21,3):C.R2GC_1774_1209,(2,8,0):C.R2GC_1044_69,(2,8,3):C.R2GC_1044_68,(1,8,0):C.R2GC_1044_69,(1,8,3):C.R2GC_1044_68,(2,11,0):C.R2GC_1870_1298,(2,11,1):C.R2GC_1867_1294,(2,11,2):C.R2GC_1867_1295,(2,11,3):C.R2GC_1870_1299,(1,11,0):C.R2GC_1874_1305,(1,11,1):C.R2GC_1872_1302,(1,11,2):C.R2GC_1872_1303,(1,11,3):C.R2GC_1874_1306,(2,22,0):C.R2GC_1774_1208,(2,22,3):C.R2GC_1774_1209,(1,22,0):C.R2GC_1776_1210,(1,22,3):C.R2GC_1776_1211,(2,9,0):C.R2GC_1043_67,(2,9,3):C.R2GC_1043_66,(1,9,0):C.R2GC_1043_67,(1,9,3):C.R2GC_1043_66})

V_445 = CTVertex(name = 'V_445',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.a, P.g ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1035_52,(0,0,1):C.R2GC_1035_53,(0,1,0):C.R2GC_1037_56,(0,1,1):C.R2GC_1037_57})

V_446 = CTVertex(name = 'V_446',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.a, P.g ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1035_52,(0,0,1):C.R2GC_1035_53,(0,1,0):C.R2GC_1037_56,(0,1,1):C.R2GC_1037_57})

V_447 = CTVertex(name = 'V_447',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.a, P.g ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1036_54,(0,0,1):C.R2GC_1036_55,(0,1,0):C.R2GC_1034_51,(0,1,1):C.R2GC_1033_50})

V_448 = CTVertex(name = 'V_448',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.a, P.g ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1036_54,(0,0,1):C.R2GC_1036_55,(0,1,0):C.R2GC_1034_51,(0,1,1):C.R2GC_1033_50})

V_449 = CTVertex(name = 'V_449',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1186_304,(0,0,1):C.R2GC_1186_305,(0,1,0):C.R2GC_1188_308,(0,1,1):C.R2GC_1188_309})

V_450 = CTVertex(name = 'V_450',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1186_304,(0,0,1):C.R2GC_1186_305,(0,1,0):C.R2GC_1188_308,(0,1,1):C.R2GC_1188_309})

V_451 = CTVertex(name = 'V_451',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1187_306,(0,0,1):C.R2GC_1187_307,(0,1,0):C.R2GC_1185_303,(0,1,1):C.R2GC_1184_302})

V_452 = CTVertex(name = 'V_452',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1187_306,(0,0,1):C.R2GC_1187_307,(0,1,0):C.R2GC_1185_303,(0,1,1):C.R2GC_1184_302})

V_453 = CTVertex(name = 'V_453',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV143 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1608_1111})

V_454 = CTVertex(name = 'V_454',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV143 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1608_1111})

V_455 = CTVertex(name = 'V_455',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV143 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1608_1111})

V_456 = CTVertex(name = 'V_456',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV143 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1608_1111})

V_457 = CTVertex(name = 'V_457',
                 type = 'R2',
                 particles = [ P.ve__tilde__, P.ve, P.g, P.g ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVV143 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1056_85,(0,0,1):C.R2GC_1056_86})

V_458 = CTVertex(name = 'V_458',
                 type = 'R2',
                 particles = [ P.vm__tilde__, P.vm, P.g, P.g ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVV143 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1058_89,(0,0,1):C.R2GC_1058_90})

V_459 = CTVertex(name = 'V_459',
                 type = 'R2',
                 particles = [ P.vt__tilde__, P.vt, P.g, P.g ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVV143 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1060_93,(0,0,1):C.R2GC_1060_94})

V_460 = CTVertex(name = 'V_460',
                 type = 'R2',
                 particles = [ P.e__plus__, P.e__minus__, P.g, P.g ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1055_83,(0,0,1):C.R2GC_1055_84,(0,1,0):C.R2GC_1052_77,(0,1,1):C.R2GC_1052_78})

V_461 = CTVertex(name = 'V_461',
                 type = 'R2',
                 particles = [ P.mu__plus__, P.mu__minus__, P.g, P.g ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1057_87,(0,0,1):C.R2GC_1057_88,(0,1,0):C.R2GC_1053_79,(0,1,1):C.R2GC_1053_80})

V_462 = CTVertex(name = 'V_462',
                 type = 'R2',
                 particles = [ P.ta__plus__, P.ta__minus__, P.g, P.g ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVV143, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1059_91,(0,0,1):C.R2GC_1059_92,(0,1,0):C.R2GC_1054_81,(0,1,1):C.R2GC_1054_82})

V_463 = CTVertex(name = 'V_463',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV108, L.FFVV109, L.FFVV110, L.FFVV112, L.FFVV113, L.FFVV115, L.FFVV131, L.FFVV132, L.FFVV141, L.FFVV142, L.FFVV143, L.FFVV158, L.FFVV159, L.FFVV160, L.FFVV162, L.FFVV163, L.FFVV165, L.FFVV189, L.FFVV190, L.FFVV193, L.FFVV194, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,10,0):C.R2GC_1064_101,(0,10,1):C.R2GC_1064_102,(2,8,0):C.R2GC_1065_103,(2,8,1):C.R2GC_1065_104,(2,9,0):C.R2GC_1062_97,(2,9,1):C.R2GC_1062_98,(1,8,0):C.R2GC_1065_103,(1,8,1):C.R2GC_1065_104,(1,9,0):C.R2GC_1062_97,(1,9,1):C.R2GC_1062_98,(2,0,0):C.R2GC_1781_1218,(2,0,1):C.R2GC_1781_1219,(2,1,0):C.R2GC_1781_1218,(2,1,1):C.R2GC_1781_1219,(2,2,0):C.R2GC_1778_1212,(2,2,1):C.R2GC_1778_1213,(1,0,0):C.R2GC_1778_1212,(1,0,1):C.R2GC_1778_1213,(1,1,0):C.R2GC_1778_1212,(1,1,1):C.R2GC_1778_1213,(1,2,0):C.R2GC_1781_1218,(1,2,1):C.R2GC_1781_1219,(2,3,0):C.R2GC_1778_1212,(2,3,1):C.R2GC_1778_1213,(2,4,0):C.R2GC_1778_1212,(2,4,1):C.R2GC_1778_1213,(2,5,0):C.R2GC_1781_1218,(2,5,1):C.R2GC_1781_1219,(1,3,0):C.R2GC_1781_1218,(1,3,1):C.R2GC_1781_1219,(1,4,0):C.R2GC_1781_1218,(1,4,1):C.R2GC_1781_1219,(1,5,0):C.R2GC_1778_1212,(1,5,1):C.R2GC_1778_1213,(0,21,0):C.R2GC_1068_109,(0,21,1):C.R2GC_1068_110,(2,19,0):C.R2GC_1069_111,(2,19,1):C.R2GC_1069_112,(2,20,0):C.R2GC_1067_107,(2,20,1):C.R2GC_1067_108,(1,19,0):C.R2GC_1069_111,(1,19,1):C.R2GC_1069_112,(1,20,0):C.R2GC_1067_107,(1,20,1):C.R2GC_1067_108,(2,11,0):C.R2GC_1785_1224,(2,11,1):C.R2GC_1785_1225,(2,12,0):C.R2GC_1785_1224,(2,12,1):C.R2GC_1785_1225,(2,13,0):C.R2GC_1784_1222,(2,13,1):C.R2GC_1784_1223,(1,11,0):C.R2GC_1784_1222,(1,11,1):C.R2GC_1784_1223,(1,12,0):C.R2GC_1784_1222,(1,12,1):C.R2GC_1784_1223,(1,13,0):C.R2GC_1785_1224,(1,13,1):C.R2GC_1785_1225,(2,14,0):C.R2GC_1784_1222,(2,14,1):C.R2GC_1784_1223,(2,15,0):C.R2GC_1784_1222,(2,15,1):C.R2GC_1784_1223,(2,16,0):C.R2GC_1785_1224,(2,16,1):C.R2GC_1785_1225,(1,14,0):C.R2GC_1785_1224,(1,14,1):C.R2GC_1785_1225,(1,15,0):C.R2GC_1785_1224,(1,15,1):C.R2GC_1785_1225,(1,16,0):C.R2GC_1784_1222,(1,16,1):C.R2GC_1784_1223,(2,6,0):C.R2GC_1781_1218,(2,6,1):C.R2GC_1781_1219,(1,6,0):C.R2GC_1778_1212,(1,6,1):C.R2GC_1778_1213,(2,17,0):C.R2GC_1785_1224,(2,17,1):C.R2GC_1785_1225,(1,17,0):C.R2GC_1784_1222,(1,17,1):C.R2GC_1784_1223,(2,7,0):C.R2GC_1778_1212,(2,7,1):C.R2GC_1778_1213,(1,7,0):C.R2GC_1781_1218,(1,7,1):C.R2GC_1781_1219,(2,18,0):C.R2GC_1784_1222,(2,18,1):C.R2GC_1784_1223,(1,18,0):C.R2GC_1785_1224,(1,18,1):C.R2GC_1785_1225})

V_464 = CTVertex(name = 'V_464',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV108, L.FFVV109, L.FFVV110, L.FFVV112, L.FFVV113, L.FFVV115, L.FFVV131, L.FFVV132, L.FFVV141, L.FFVV142, L.FFVV143, L.FFVV158, L.FFVV159, L.FFVV160, L.FFVV162, L.FFVV163, L.FFVV165, L.FFVV189, L.FFVV190, L.FFVV193, L.FFVV194, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,10,0):C.R2GC_1064_101,(0,10,1):C.R2GC_1064_102,(2,8,0):C.R2GC_1065_103,(2,8,1):C.R2GC_1065_104,(2,9,0):C.R2GC_1062_97,(2,9,1):C.R2GC_1062_98,(1,8,0):C.R2GC_1065_103,(1,8,1):C.R2GC_1065_104,(1,9,0):C.R2GC_1062_97,(1,9,1):C.R2GC_1062_98,(2,0,0):C.R2GC_1781_1218,(2,0,1):C.R2GC_1781_1219,(2,1,0):C.R2GC_1781_1218,(2,1,1):C.R2GC_1781_1219,(2,2,0):C.R2GC_1778_1212,(2,2,1):C.R2GC_1778_1213,(1,0,0):C.R2GC_1778_1212,(1,0,1):C.R2GC_1778_1213,(1,1,0):C.R2GC_1778_1212,(1,1,1):C.R2GC_1778_1213,(1,2,0):C.R2GC_1781_1218,(1,2,1):C.R2GC_1781_1219,(2,3,0):C.R2GC_1778_1212,(2,3,1):C.R2GC_1778_1213,(2,4,0):C.R2GC_1778_1212,(2,4,1):C.R2GC_1778_1213,(2,5,0):C.R2GC_1781_1218,(2,5,1):C.R2GC_1781_1219,(1,3,0):C.R2GC_1781_1218,(1,3,1):C.R2GC_1781_1219,(1,4,0):C.R2GC_1781_1218,(1,4,1):C.R2GC_1781_1219,(1,5,0):C.R2GC_1778_1212,(1,5,1):C.R2GC_1778_1213,(0,21,0):C.R2GC_1068_109,(0,21,1):C.R2GC_1068_110,(2,19,0):C.R2GC_1069_111,(2,19,1):C.R2GC_1069_112,(2,20,0):C.R2GC_1067_107,(2,20,1):C.R2GC_1067_108,(1,19,0):C.R2GC_1069_111,(1,19,1):C.R2GC_1069_112,(1,20,0):C.R2GC_1067_107,(1,20,1):C.R2GC_1067_108,(2,11,0):C.R2GC_1785_1224,(2,11,1):C.R2GC_1785_1225,(2,12,0):C.R2GC_1785_1224,(2,12,1):C.R2GC_1785_1225,(2,13,0):C.R2GC_1784_1222,(2,13,1):C.R2GC_1784_1223,(1,11,0):C.R2GC_1784_1222,(1,11,1):C.R2GC_1784_1223,(1,12,0):C.R2GC_1784_1222,(1,12,1):C.R2GC_1784_1223,(1,13,0):C.R2GC_1785_1224,(1,13,1):C.R2GC_1785_1225,(2,14,0):C.R2GC_1784_1222,(2,14,1):C.R2GC_1784_1223,(2,15,0):C.R2GC_1784_1222,(2,15,1):C.R2GC_1784_1223,(2,16,0):C.R2GC_1785_1224,(2,16,1):C.R2GC_1785_1225,(1,14,0):C.R2GC_1785_1224,(1,14,1):C.R2GC_1785_1225,(1,15,0):C.R2GC_1785_1224,(1,15,1):C.R2GC_1785_1225,(1,16,0):C.R2GC_1784_1222,(1,16,1):C.R2GC_1784_1223,(2,6,0):C.R2GC_1781_1218,(2,6,1):C.R2GC_1781_1219,(1,6,0):C.R2GC_1778_1212,(1,6,1):C.R2GC_1778_1213,(2,17,0):C.R2GC_1785_1224,(2,17,1):C.R2GC_1785_1225,(1,17,0):C.R2GC_1784_1222,(1,17,1):C.R2GC_1784_1223,(2,7,0):C.R2GC_1778_1212,(2,7,1):C.R2GC_1778_1213,(1,7,0):C.R2GC_1781_1218,(1,7,1):C.R2GC_1781_1219,(2,18,0):C.R2GC_1784_1222,(2,18,1):C.R2GC_1784_1223,(1,18,0):C.R2GC_1785_1224,(1,18,1):C.R2GC_1785_1225})

V_465 = CTVertex(name = 'V_465',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV108, L.FFVV109, L.FFVV110, L.FFVV112, L.FFVV113, L.FFVV115, L.FFVV131, L.FFVV132, L.FFVV141, L.FFVV142, L.FFVV143, L.FFVV158, L.FFVV159, L.FFVV160, L.FFVV162, L.FFVV163, L.FFVV165, L.FFVV189, L.FFVV190, L.FFVV193, L.FFVV194, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,10,0):C.R2GC_1063_99,(0,10,1):C.R2GC_1063_100,(2,8,0):C.R2GC_1066_105,(2,8,1):C.R2GC_1066_106,(2,9,0):C.R2GC_1061_95,(2,9,1):C.R2GC_1061_96,(1,8,0):C.R2GC_1066_105,(1,8,1):C.R2GC_1066_106,(1,9,0):C.R2GC_1061_95,(1,9,1):C.R2GC_1061_96,(2,0,0):C.R2GC_1780_1216,(2,0,1):C.R2GC_1780_1217,(2,1,0):C.R2GC_1780_1216,(2,1,1):C.R2GC_1780_1217,(2,2,0):C.R2GC_1779_1214,(2,2,1):C.R2GC_1779_1215,(1,0,0):C.R2GC_1779_1214,(1,0,1):C.R2GC_1779_1215,(1,1,0):C.R2GC_1779_1214,(1,1,1):C.R2GC_1779_1215,(1,2,0):C.R2GC_1780_1216,(1,2,1):C.R2GC_1780_1217,(2,3,0):C.R2GC_1779_1214,(2,3,1):C.R2GC_1779_1215,(2,4,0):C.R2GC_1779_1214,(2,4,1):C.R2GC_1779_1215,(2,5,0):C.R2GC_1780_1216,(2,5,1):C.R2GC_1780_1217,(1,3,0):C.R2GC_1780_1216,(1,3,1):C.R2GC_1780_1217,(1,4,0):C.R2GC_1780_1216,(1,4,1):C.R2GC_1780_1217,(1,5,0):C.R2GC_1779_1214,(1,5,1):C.R2GC_1779_1215,(0,21,0):C.R2GC_1050_74,(0,21,1):C.R2GC_1049_73,(2,19,0):C.R2GC_1051_75,(2,19,1):C.R2GC_1051_76,(2,20,0):C.R2GC_1048_71,(2,20,1):C.R2GC_1048_72,(1,19,0):C.R2GC_1051_75,(1,19,1):C.R2GC_1051_76,(1,20,0):C.R2GC_1048_71,(1,20,1):C.R2GC_1048_72,(2,11,0):C.R2GC_1776_1210,(2,11,1):C.R2GC_1776_1211,(2,12,0):C.R2GC_1776_1210,(2,12,1):C.R2GC_1776_1211,(2,13,0):C.R2GC_1774_1208,(2,13,1):C.R2GC_1774_1209,(1,11,0):C.R2GC_1774_1208,(1,11,1):C.R2GC_1774_1209,(1,12,0):C.R2GC_1774_1208,(1,12,1):C.R2GC_1774_1209,(1,13,0):C.R2GC_1776_1210,(1,13,1):C.R2GC_1776_1211,(2,14,0):C.R2GC_1774_1208,(2,14,1):C.R2GC_1774_1209,(2,15,0):C.R2GC_1774_1208,(2,15,1):C.R2GC_1774_1209,(2,16,0):C.R2GC_1776_1210,(2,16,1):C.R2GC_1776_1211,(1,14,0):C.R2GC_1776_1210,(1,14,1):C.R2GC_1776_1211,(1,15,0):C.R2GC_1776_1210,(1,15,1):C.R2GC_1776_1211,(1,16,0):C.R2GC_1774_1208,(1,16,1):C.R2GC_1774_1209,(2,6,0):C.R2GC_1780_1216,(2,6,1):C.R2GC_1780_1217,(1,6,0):C.R2GC_1779_1214,(1,6,1):C.R2GC_1779_1215,(2,17,0):C.R2GC_1776_1210,(2,17,1):C.R2GC_1776_1211,(1,17,0):C.R2GC_1774_1208,(1,17,1):C.R2GC_1774_1209,(2,7,0):C.R2GC_1779_1214,(2,7,1):C.R2GC_1779_1215,(1,7,0):C.R2GC_1780_1216,(1,7,1):C.R2GC_1780_1217,(2,18,0):C.R2GC_1774_1208,(2,18,1):C.R2GC_1774_1209,(1,18,0):C.R2GC_1776_1210,(1,18,1):C.R2GC_1776_1211})

V_466 = CTVertex(name = 'V_466',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV108, L.FFVV109, L.FFVV110, L.FFVV112, L.FFVV113, L.FFVV115, L.FFVV131, L.FFVV132, L.FFVV141, L.FFVV142, L.FFVV143, L.FFVV158, L.FFVV159, L.FFVV160, L.FFVV162, L.FFVV163, L.FFVV165, L.FFVV189, L.FFVV190, L.FFVV193, L.FFVV194, L.FFVV195 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,10,0):C.R2GC_1063_99,(0,10,1):C.R2GC_1063_100,(2,8,0):C.R2GC_1066_105,(2,8,1):C.R2GC_1066_106,(2,9,0):C.R2GC_1061_95,(2,9,1):C.R2GC_1061_96,(1,8,0):C.R2GC_1066_105,(1,8,1):C.R2GC_1066_106,(1,9,0):C.R2GC_1061_95,(1,9,1):C.R2GC_1061_96,(2,0,0):C.R2GC_1780_1216,(2,0,1):C.R2GC_1780_1217,(2,1,0):C.R2GC_1780_1216,(2,1,1):C.R2GC_1780_1217,(2,2,0):C.R2GC_1779_1214,(2,2,1):C.R2GC_1779_1215,(1,0,0):C.R2GC_1779_1214,(1,0,1):C.R2GC_1779_1215,(1,1,0):C.R2GC_1779_1214,(1,1,1):C.R2GC_1779_1215,(1,2,0):C.R2GC_1780_1216,(1,2,1):C.R2GC_1780_1217,(2,3,0):C.R2GC_1779_1214,(2,3,1):C.R2GC_1779_1215,(2,4,0):C.R2GC_1779_1214,(2,4,1):C.R2GC_1779_1215,(2,5,0):C.R2GC_1780_1216,(2,5,1):C.R2GC_1780_1217,(1,3,0):C.R2GC_1780_1216,(1,3,1):C.R2GC_1780_1217,(1,4,0):C.R2GC_1780_1216,(1,4,1):C.R2GC_1780_1217,(1,5,0):C.R2GC_1779_1214,(1,5,1):C.R2GC_1779_1215,(0,21,0):C.R2GC_1050_74,(0,21,1):C.R2GC_1049_73,(2,19,0):C.R2GC_1051_75,(2,19,1):C.R2GC_1051_76,(2,20,0):C.R2GC_1048_71,(2,20,1):C.R2GC_1048_72,(1,19,0):C.R2GC_1051_75,(1,19,1):C.R2GC_1051_76,(1,20,0):C.R2GC_1048_71,(1,20,1):C.R2GC_1048_72,(2,11,0):C.R2GC_1776_1210,(2,11,1):C.R2GC_1776_1211,(2,12,0):C.R2GC_1776_1210,(2,12,1):C.R2GC_1776_1211,(2,13,0):C.R2GC_1774_1208,(2,13,1):C.R2GC_1774_1209,(1,11,0):C.R2GC_1774_1208,(1,11,1):C.R2GC_1774_1209,(1,12,0):C.R2GC_1774_1208,(1,12,1):C.R2GC_1774_1209,(1,13,0):C.R2GC_1776_1210,(1,13,1):C.R2GC_1776_1211,(2,14,0):C.R2GC_1774_1208,(2,14,1):C.R2GC_1774_1209,(2,15,0):C.R2GC_1774_1208,(2,15,1):C.R2GC_1774_1209,(2,16,0):C.R2GC_1776_1210,(2,16,1):C.R2GC_1776_1211,(1,14,0):C.R2GC_1776_1210,(1,14,1):C.R2GC_1776_1211,(1,15,0):C.R2GC_1776_1210,(1,15,1):C.R2GC_1776_1211,(1,16,0):C.R2GC_1774_1208,(1,16,1):C.R2GC_1774_1209,(2,6,0):C.R2GC_1780_1216,(2,6,1):C.R2GC_1780_1217,(1,6,0):C.R2GC_1779_1214,(1,6,1):C.R2GC_1779_1215,(2,17,0):C.R2GC_1776_1210,(2,17,1):C.R2GC_1776_1211,(1,17,0):C.R2GC_1774_1208,(1,17,1):C.R2GC_1774_1209,(2,7,0):C.R2GC_1779_1214,(2,7,1):C.R2GC_1779_1215,(1,7,0):C.R2GC_1780_1216,(1,7,1):C.R2GC_1780_1217,(2,18,0):C.R2GC_1774_1208,(2,18,1):C.R2GC_1774_1209,(1,18,0):C.R2GC_1776_1210,(1,18,1):C.R2GC_1776_1211})

V_467 = CTVertex(name = 'V_467',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.g, P.G__plus__, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_843_1927})

V_468 = CTVertex(name = 'V_468',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.g, P.G__plus__, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_843_1927})

V_469 = CTVertex(name = 'V_469',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.g, P.G0, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_844_1928})

V_470 = CTVertex(name = 'V_470',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.g, P.G0, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_844_1928})

V_471 = CTVertex(name = 'V_471',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.g, P.G__minus__, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_845_1929})

V_472 = CTVertex(name = 'V_472',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.g, P.G__minus__, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_845_1929})

V_473 = CTVertex(name = 'V_473',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.g, P.G0, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_844_1928})

V_474 = CTVertex(name = 'V_474',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.g, P.G0, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_844_1928})

V_475 = CTVertex(name = 'V_475',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151, L.FFVVV152 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1313_504,(0,0,1):C.R2GC_1313_505,(0,0,2):C.R2GC_1313_506,(0,0,3):C.R2GC_1313_507,(0,2,3):C.R2GC_581_1768,(0,1,0):C.R2GC_1314_508,(0,1,1):C.R2GC_1314_509,(0,1,2):C.R2GC_1314_510,(0,1,3):C.R2GC_1314_511})

V_476 = CTVertex(name = 'V_476',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.a, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151, L.FFVVV152 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_1020_29,(0,0,0):C.R2GC_1312_500,(0,0,1):C.R2GC_1312_501,(0,0,2):C.R2GC_1312_502,(0,0,3):C.R2GC_1312_503,(0,1,3):C.R2GC_1019_28})

V_477 = CTVertex(name = 'V_477',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151, L.FFVVV152 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1382_723,(0,0,1):C.R2GC_1382_724,(0,0,2):C.R2GC_1382_725,(0,0,3):C.R2GC_1382_726,(0,2,3):C.R2GC_605_1789,(0,1,0):C.R2GC_1383_727,(0,1,1):C.R2GC_1383_728,(0,1,2):C.R2GC_1383_729,(0,1,3):C.R2GC_1383_730})

V_478 = CTVertex(name = 'V_478',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.a, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151, L.FFVVV152 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1381_719,(0,0,1):C.R2GC_1381_720,(0,0,2):C.R2GC_1381_721,(0,0,3):C.R2GC_1381_722,(0,2,0):C.R2GC_560_1748,(0,1,0):C.R2GC_1177_288,(0,1,3):C.R2GC_1177_289})

V_479 = CTVertex(name = 'V_479',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151, L.FFVVV152 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1413_826,(0,0,1):C.R2GC_1413_827,(0,0,2):C.R2GC_1413_828,(0,0,3):C.R2GC_1413_829,(0,2,3):C.R2GC_615_1798,(0,1,0):C.R2GC_1414_830,(0,1,1):C.R2GC_1414_831,(0,1,2):C.R2GC_1414_832,(0,1,3):C.R2GC_1414_833})

V_480 = CTVertex(name = 'V_480',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.a, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151, L.FFVVV152 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1412_822,(0,0,1):C.R2GC_1412_823,(0,0,2):C.R2GC_1412_824,(0,0,3):C.R2GC_1412_825,(0,2,0):C.R2GC_564_1751,(0,1,0):C.R2GC_1235_393,(0,1,3):C.R2GC_1235_394})

V_481 = CTVertex(name = 'V_481',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.Z, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV150, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d], [P.s] ], [ [P.t] ], [ [P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1419_850,(0,0,1):C.R2GC_1419_851,(0,0,2):C.R2GC_1419_852,(0,0,3):C.R2GC_1419_853,(0,0,4):C.R2GC_1419_854,(0,1,3):C.R2GC_617_1800,(0,2,0):C.R2GC_1420_855,(0,2,1):C.R2GC_1420_856,(0,2,2):C.R2GC_1420_857,(0,2,3):C.R2GC_1420_858,(0,2,4):C.R2GC_1420_859})

V_482 = CTVertex(name = 'V_482',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.Z, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151, L.FFVVV152 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1418_846,(0,0,1):C.R2GC_1418_847,(0,0,2):C.R2GC_1418_848,(0,0,3):C.R2GC_1418_849,(0,2,0):C.R2GC_566_1753,(0,1,0):C.R2GC_1246_412,(0,1,3):C.R2GC_1246_413})

V_483 = CTVertex(name = 'V_483',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.a, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV110, L.FFVVV111, L.FFVVV113 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,1,1):C.R2GC_1602_1105,(0,2,0):C.R2GC_815_1911,(0,0,0):C.R2GC_817_1912})

V_484 = CTVertex(name = 'V_484',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.a, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV138, L.FFVVV142, L.FFVVV89, L.FFVVV90, L.FFVVV91, L.FFVVV92 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,1,1):C.R2GC_1593_1095,(0,0,0):C.R2GC_796_1899,(0,3,1):C.R2GC_1576_1075,(0,5,0):C.R2GC_757_1874,(0,2,0):C.R2GC_759_1875,(0,4,0):C.R2GC_792_1897})

V_485 = CTVertex(name = 'V_485',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV103, L.FFVVV104, L.FFVVV105, L.FFVVV106, L.FFVVV107, L.FFVVV108 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,2,1):C.R2GC_1502_973,(0,4,1):C.R2GC_1619_1126,(0,0,0):C.R2GC_1501_972,(0,3,0):C.R2GC_1503_974,(0,1,0):C.R2GC_837_1923,(0,5,0):C.R2GC_839_1924})

V_486 = CTVertex(name = 'V_486',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.a, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV110, L.FFVVV111, L.FFVVV113 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,1,1):C.R2GC_1602_1105,(0,2,0):C.R2GC_815_1911,(0,0,0):C.R2GC_817_1912})

V_487 = CTVertex(name = 'V_487',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.a, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV137, L.FFVVV140, L.FFVVV88, L.FFVVV89, L.FFVVV90, L.FFVVV92 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,1,1):C.R2GC_1592_1094,(0,0,0):C.R2GC_795_1898,(0,4,1):C.R2GC_1576_1075,(0,5,0):C.R2GC_757_1874,(0,3,0):C.R2GC_759_1875,(0,2,0):C.R2GC_792_1897})

V_488 = CTVertex(name = 'V_488',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV103, L.FFVVV104, L.FFVVV105, L.FFVVV106, L.FFVVV107, L.FFVVV108 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,2,1):C.R2GC_1502_973,(0,4,1):C.R2GC_1619_1126,(0,0,0):C.R2GC_1501_972,(0,3,0):C.R2GC_1503_974,(0,1,0):C.R2GC_837_1923,(0,5,0):C.R2GC_839_1924})

V_489 = CTVertex(name = 'V_489',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV125, L.FFVVV129, L.FFVVV135, L.FFVVV144, L.FFVVV155 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,1):C.R2GC_1588_1089,(0,3,1):C.R2GC_1589_1090,(0,1,0):C.R2GC_785_1893,(0,2,0):C.R2GC_789_1894,(0,4,0):C.R2GC_1591_1092,(0,4,1):C.R2GC_1591_1093})

V_490 = CTVertex(name = 'V_490',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.a, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV125, L.FFVVV134, L.FFVVV143, L.FFVVV144, L.FFVVV155 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,1):C.R2GC_1588_1089,(0,3,1):C.R2GC_1590_1091,(0,1,0):C.R2GC_785_1893,(0,2,0):C.R2GC_790_1895,(0,4,0):C.R2GC_784_1892})

V_491 = CTVertex(name = 'V_491',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV112, L.FFVVV123, L.FFVVV124, L.FFVVV125, L.FFVVV126, L.FFVVV130, L.FFVVV132, L.FFVVV141, L.FFVVV153, L.FFVVV155 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,1,1):C.R2GC_1581_1080,(0,3,1):C.R2GC_1604_1106,(0,7,1):C.R2GC_1605_1107,(0,2,0):C.R2GC_771_1883,(0,6,0):C.R2GC_775_1884,(0,4,0):C.R2GC_819_1914,(0,5,0):C.R2GC_823_1915,(0,0,1):C.R2GC_1583_1082,(0,8,0):C.R2GC_1584_1083,(0,8,1):C.R2GC_1584_1084,(0,9,0):C.R2GC_1607_1109,(0,9,1):C.R2GC_1607_1110})

V_492 = CTVertex(name = 'V_492',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.W__minus__, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV112, L.FFVVV123, L.FFVVV125, L.FFVVV131, L.FFVVV133, L.FFVVV136, L.FFVVV139, L.FFVVV141, L.FFVVV153, L.FFVVV155 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,1,1):C.R2GC_1581_1080,(0,2,1):C.R2GC_1604_1106,(0,7,1):C.R2GC_1606_1108,(0,3,0):C.R2GC_771_1883,(0,5,0):C.R2GC_775_1884,(0,4,0):C.R2GC_819_1914,(0,6,0):C.R2GC_824_1916,(0,0,1):C.R2GC_1582_1081,(0,8,0):C.R2GC_770_1882,(0,9,0):C.R2GC_818_1913})

V_493 = CTVertex(name = 'V_493',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.W__minus__, P.W__plus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV100, L.FFVVV98, L.FFVVV99 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,2,1):C.R2GC_1580_1079,(0,0,0):C.R2GC_765_1879,(0,1,0):C.R2GC_769_1881})

V_494 = CTVertex(name = 'V_494',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV115, L.FFVVV116, L.FFVVV117 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,1,1):C.R2GC_1579_1078,(0,0,0):C.R2GC_764_1878,(0,2,0):C.R2GC_768_1880})

V_495 = CTVertex(name = 'V_495',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.a, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1021_30,(0,0,1):C.R2GC_1021_31,(0,1,0):C.R2GC_1020_29,(0,1,1):C.R2GC_1019_28})

V_496 = CTVertex(name = 'V_496',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.a, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1021_30,(0,0,1):C.R2GC_1021_31,(0,1,0):C.R2GC_1020_29,(0,1,1):C.R2GC_1019_28})

V_497 = CTVertex(name = 'V_497',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.a, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1022_32,(0,0,1):C.R2GC_1022_33,(0,1,0):C.R2GC_1023_34,(0,1,1):C.R2GC_1023_35})

V_498 = CTVertex(name = 'V_498',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.a, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1022_32,(0,0,1):C.R2GC_1022_33,(0,1,0):C.R2GC_1023_34,(0,1,1):C.R2GC_1023_35})

V_499 = CTVertex(name = 'V_499',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.a, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1179_291,(0,0,1):C.R2GC_1179_292,(0,1,0):C.R2GC_1178_290,(0,1,1):C.R2GC_1177_289})

V_500 = CTVertex(name = 'V_500',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.a, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1179_291,(0,0,1):C.R2GC_1179_292,(0,1,0):C.R2GC_1178_290,(0,1,1):C.R2GC_1177_289})

V_501 = CTVertex(name = 'V_501',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.a, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1180_293,(0,0,1):C.R2GC_1180_294,(0,1,0):C.R2GC_1181_295,(0,1,1):C.R2GC_1181_296})

V_502 = CTVertex(name = 'V_502',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.a, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1180_293,(0,0,1):C.R2GC_1180_294,(0,1,0):C.R2GC_1181_295,(0,1,1):C.R2GC_1181_296})

V_503 = CTVertex(name = 'V_503',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.a, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1237_396,(0,0,1):C.R2GC_1237_397,(0,1,0):C.R2GC_1236_395,(0,1,1):C.R2GC_1235_394})

V_504 = CTVertex(name = 'V_504',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.a, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1237_396,(0,0,1):C.R2GC_1237_397,(0,1,0):C.R2GC_1236_395,(0,1,1):C.R2GC_1235_394})

V_505 = CTVertex(name = 'V_505',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.a, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1238_398,(0,0,1):C.R2GC_1238_399,(0,1,0):C.R2GC_1239_400,(0,1,1):C.R2GC_1239_401})

V_506 = CTVertex(name = 'V_506',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.a, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1238_398,(0,0,1):C.R2GC_1238_399,(0,1,0):C.R2GC_1239_400,(0,1,1):C.R2GC_1239_401})

V_507 = CTVertex(name = 'V_507',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.Z, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1248_415,(0,0,1):C.R2GC_1248_416,(0,1,0):C.R2GC_1247_414,(0,1,1):C.R2GC_1246_413})

V_508 = CTVertex(name = 'V_508',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.Z, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1248_415,(0,0,1):C.R2GC_1248_416,(0,1,0):C.R2GC_1247_414,(0,1,1):C.R2GC_1246_413})

V_509 = CTVertex(name = 'V_509',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.Z, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1249_417,(0,0,1):C.R2GC_1249_418,(0,1,0):C.R2GC_1250_419,(0,1,1):C.R2GC_1250_420})

V_510 = CTVertex(name = 'V_510',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.Z, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1249_417,(0,0,1):C.R2GC_1249_418,(0,1,0):C.R2GC_1250_419,(0,1,1):C.R2GC_1250_420})

V_511 = CTVertex(name = 'V_511',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.a, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV111 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1602_1105})

V_512 = CTVertex(name = 'V_512',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.a, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV111 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1602_1105})

V_513 = CTVertex(name = 'V_513',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.a, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV142, L.FFVVV90 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1593_1095,(0,1,0):C.R2GC_1576_1075})

V_514 = CTVertex(name = 'V_514',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.a, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV142, L.FFVVV90 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1593_1095,(0,1,0):C.R2GC_1576_1075})

V_515 = CTVertex(name = 'V_515',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV105, L.FFVVV107 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1502_973,(0,1,0):C.R2GC_1619_1126})

V_516 = CTVertex(name = 'V_516',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV105, L.FFVVV107 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1502_973,(0,1,0):C.R2GC_1619_1126})

V_517 = CTVertex(name = 'V_517',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.a, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV111 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1602_1105})

V_518 = CTVertex(name = 'V_518',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.a, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV111 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1602_1105})

V_519 = CTVertex(name = 'V_519',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.a, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV140, L.FFVVV90 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1592_1094,(0,1,0):C.R2GC_1576_1075})

V_520 = CTVertex(name = 'V_520',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.a, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV140, L.FFVVV90 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1592_1094,(0,1,0):C.R2GC_1576_1075})

V_521 = CTVertex(name = 'V_521',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV105, L.FFVVV107 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1502_973,(0,1,0):C.R2GC_1619_1126})

V_522 = CTVertex(name = 'V_522',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV105, L.FFVVV107 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1502_973,(0,1,0):C.R2GC_1619_1126})

V_523 = CTVertex(name = 'V_523',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.a, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV125, L.FFVVV144, L.FFVVV155 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1588_1089,(0,1,0):C.R2GC_1590_1091,(0,2,0):C.R2GC_784_1892})

V_524 = CTVertex(name = 'V_524',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.a, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV125, L.FFVVV144, L.FFVVV155 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1588_1089,(0,1,0):C.R2GC_1590_1091,(0,2,0):C.R2GC_784_1892})

V_525 = CTVertex(name = 'V_525',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.a, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV125, L.FFVVV144, L.FFVVV155 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1588_1089,(0,1,0):C.R2GC_1589_1090,(0,2,0):C.R2GC_791_1896})

V_526 = CTVertex(name = 'V_526',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.a, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV125, L.FFVVV144, L.FFVVV155 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1588_1089,(0,1,0):C.R2GC_1589_1090,(0,2,0):C.R2GC_791_1896})

V_527 = CTVertex(name = 'V_527',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.W__minus__, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV112, L.FFVVV123, L.FFVVV125, L.FFVVV141, L.FFVVV153, L.FFVVV155 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1581_1080,(0,2,0):C.R2GC_1604_1106,(0,3,0):C.R2GC_1606_1108,(0,0,0):C.R2GC_1582_1081,(0,4,0):C.R2GC_770_1882,(0,5,0):C.R2GC_818_1913})

V_528 = CTVertex(name = 'V_528',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.W__minus__, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV112, L.FFVVV123, L.FFVVV125, L.FFVVV141, L.FFVVV153, L.FFVVV155 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1581_1080,(0,2,0):C.R2GC_1604_1106,(0,3,0):C.R2GC_1606_1108,(0,0,0):C.R2GC_1582_1081,(0,4,0):C.R2GC_770_1882,(0,5,0):C.R2GC_818_1913})

V_529 = CTVertex(name = 'V_529',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.W__minus__, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV112, L.FFVVV123, L.FFVVV125, L.FFVVV141, L.FFVVV153, L.FFVVV155 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1581_1080,(0,2,0):C.R2GC_1604_1106,(0,3,0):C.R2GC_1605_1107,(0,0,0):C.R2GC_1583_1082,(0,4,0):C.R2GC_776_1885,(0,5,0):C.R2GC_825_1917})

V_530 = CTVertex(name = 'V_530',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.W__minus__, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV112, L.FFVVV123, L.FFVVV125, L.FFVVV141, L.FFVVV153, L.FFVVV155 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1581_1080,(0,2,0):C.R2GC_1604_1106,(0,3,0):C.R2GC_1605_1107,(0,0,0):C.R2GC_1583_1082,(0,4,0):C.R2GC_776_1885,(0,5,0):C.R2GC_825_1917})

V_531 = CTVertex(name = 'V_531',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.W__minus__, P.W__plus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV99 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1580_1079})

V_532 = CTVertex(name = 'V_532',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.W__minus__, P.W__plus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV99 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1580_1079})

V_533 = CTVertex(name = 'V_533',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV116 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1579_1078})

V_534 = CTVertex(name = 'V_534',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV116 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1579_1078})

V_535 = CTVertex(name = 'V_535',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.a, P.g ],
                 color = [ 'T(5,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151, L.FFVVV152 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1322_533,(0,0,1):C.R2GC_1322_534,(0,0,2):C.R2GC_1322_535,(0,0,3):C.R2GC_1322_536,(0,2,3):C.R2GC_586_1771,(0,1,0):C.R2GC_1323_537,(0,1,1):C.R2GC_1323_538,(0,1,2):C.R2GC_1323_539,(0,1,3):C.R2GC_1323_540})

V_536 = CTVertex(name = 'V_536',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.a, P.a, P.g ],
                 color = [ 'T(5,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151, L.FFVVV152 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_1039_59,(0,0,0):C.R2GC_1321_529,(0,0,1):C.R2GC_1321_530,(0,0,2):C.R2GC_1321_531,(0,0,3):C.R2GC_1321_532,(0,1,3):C.R2GC_1038_58})

V_537 = CTVertex(name = 'V_537',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.g, P.Z ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151, L.FFVVV152 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1388_745,(0,0,1):C.R2GC_1388_746,(0,0,2):C.R2GC_1388_747,(0,0,3):C.R2GC_1388_748,(0,2,3):C.R2GC_608_1791,(0,1,0):C.R2GC_1389_749,(0,1,1):C.R2GC_1389_750,(0,1,2):C.R2GC_1389_751,(0,1,3):C.R2GC_1389_752})

V_538 = CTVertex(name = 'V_538',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.a, P.g, P.Z ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151, L.FFVVV152 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1387_741,(0,0,1):C.R2GC_1387_742,(0,0,2):C.R2GC_1387_743,(0,0,3):C.R2GC_1387_744,(0,2,0):C.R2GC_562_1749,(0,1,0):C.R2GC_1189_310,(0,1,3):C.R2GC_1189_311})

V_539 = CTVertex(name = 'V_539',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.Z, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151, L.FFVVV152 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1416_838,(0,0,1):C.R2GC_1416_839,(0,0,2):C.R2GC_1416_840,(0,0,3):C.R2GC_1416_841,(0,2,3):C.R2GC_616_1799,(0,1,0):C.R2GC_1417_842,(0,1,1):C.R2GC_1417_843,(0,1,2):C.R2GC_1417_844,(0,1,3):C.R2GC_1417_845})

V_540 = CTVertex(name = 'V_540',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.g, P.Z, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151, L.FFVVV152 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1415_834,(0,0,1):C.R2GC_1415_835,(0,0,2):C.R2GC_1415_836,(0,0,3):C.R2GC_1415_837,(0,2,0):C.R2GC_565_1752,(0,1,0):C.R2GC_1240_402,(0,1,3):C.R2GC_1240_403})

V_541 = CTVertex(name = 'V_541',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.a, P.g, P.W__plus__ ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVV127, L.FFVVV95 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,1):C.R2GC_1609_1112,(0,1,0):C.R2GC_1505_976})

V_542 = CTVertex(name = 'V_542',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.g, P.W__plus__, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV118, L.FFVVV119, L.FFVVV123, L.FFVVV127 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,3,1):C.R2GC_1577_1076,(0,2,1):C.R2GC_1599_1102,(0,1,0):C.R2GC_1496_966,(0,0,0):C.R2GC_1498_969})

V_543 = CTVertex(name = 'V_543',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.a, P.g, P.W__minus__ ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVV120, L.FFVVV128 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,1,1):C.R2GC_1610_1113,(0,0,0):C.R2GC_1506_977})

V_544 = CTVertex(name = 'V_544',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.g, P.W__minus__, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV123, L.FFVVV128, L.FFVVV96, L.FFVVV97 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,1,1):C.R2GC_1578_1077,(0,0,1):C.R2GC_1598_1101,(0,3,0):C.R2GC_1496_966,(0,2,0):C.R2GC_1498_969})

V_545 = CTVertex(name = 'V_545',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.W__minus__, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV109, L.FFVVV121, L.FFVVV123, L.FFVVV151, L.FFVVV94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,2,1):C.R2GC_1595_1097,(0,1,1):C.R2GC_1594_1096,(0,4,0):C.R2GC_798_1901,(0,0,0):C.R2GC_801_1904,(0,3,0):C.R2GC_1597_1099,(0,3,1):C.R2GC_1597_1100})

V_546 = CTVertex(name = 'V_546',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b, P.g, P.W__minus__, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV102, L.FFVVV121, L.FFVVV122, L.FFVVV123, L.FFVVV151 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,3,1):C.R2GC_1596_1098,(0,1,1):C.R2GC_1594_1096,(0,0,0):C.R2GC_799_1902,(0,2,0):C.R2GC_800_1903,(0,4,0):C.R2GC_797_1900})

V_547 = CTVertex(name = 'V_547',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.g, P.g, P.W__plus__ ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV83, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(2,1,1):C.R2GC_1614_1120,(1,1,1):C.R2GC_1618_1125,(0,0,0):C.R2GC_1615_1121,(0,0,1):C.R2GC_1509_980,(2,5,0):C.R2GC_1616_1122,(2,5,1):C.R2GC_1616_1123,(1,5,0):C.R2GC_1617_1124,(1,5,1):C.R2GC_1616_1123,(2,4,0):C.R2GC_1617_1124,(2,4,1):C.R2GC_1616_1123,(1,4,0):C.R2GC_1616_1122,(1,4,1):C.R2GC_1616_1123,(2,3,0):C.R2GC_1617_1124,(2,3,1):C.R2GC_1616_1123,(1,3,0):C.R2GC_1616_1122,(1,3,1):C.R2GC_1616_1123,(2,2,0):C.R2GC_1508_979,(1,2,0):C.R2GC_1507_978})

V_548 = CTVertex(name = 'V_548',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.g, P.g, P.W__minus__ ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV83, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(2,1,1):C.R2GC_1614_1120,(1,1,1):C.R2GC_1618_1125,(0,0,0):C.R2GC_1615_1121,(0,0,1):C.R2GC_1509_980,(2,5,0):C.R2GC_1616_1122,(2,5,1):C.R2GC_1616_1123,(1,5,0):C.R2GC_1617_1124,(1,5,1):C.R2GC_1616_1123,(2,4,0):C.R2GC_1617_1124,(2,4,1):C.R2GC_1616_1123,(1,4,0):C.R2GC_1616_1122,(1,4,1):C.R2GC_1616_1123,(2,3,0):C.R2GC_1617_1124,(2,3,1):C.R2GC_1616_1123,(1,3,0):C.R2GC_1616_1122,(1,3,1):C.R2GC_1616_1123,(2,2,0):C.R2GC_1508_979,(1,2,0):C.R2GC_1507_978})

V_549 = CTVertex(name = 'V_549',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.a, P.a, P.g ],
                 color = [ 'T(5,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1041_62,(0,0,1):C.R2GC_1041_63,(0,1,0):C.R2GC_1042_64,(0,1,1):C.R2GC_1042_65})

V_550 = CTVertex(name = 'V_550',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.a, P.a, P.g ],
                 color = [ 'T(5,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1041_62,(0,0,1):C.R2GC_1041_63,(0,1,0):C.R2GC_1042_64,(0,1,1):C.R2GC_1042_65})

V_551 = CTVertex(name = 'V_551',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.a, P.a, P.g ],
                 color = [ 'T(5,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1040_60,(0,0,1):C.R2GC_1040_61,(0,1,0):C.R2GC_1039_59,(0,1,1):C.R2GC_1038_58})

V_552 = CTVertex(name = 'V_552',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.a, P.a, P.g ],
                 color = [ 'T(5,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1040_60,(0,0,1):C.R2GC_1040_61,(0,1,0):C.R2GC_1039_59,(0,1,1):C.R2GC_1038_58})

V_553 = CTVertex(name = 'V_553',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.a, P.g, P.Z ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1192_315,(0,0,1):C.R2GC_1192_316,(0,1,0):C.R2GC_1193_317,(0,1,1):C.R2GC_1193_318})

V_554 = CTVertex(name = 'V_554',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.a, P.g, P.Z ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1192_315,(0,0,1):C.R2GC_1192_316,(0,1,0):C.R2GC_1193_317,(0,1,1):C.R2GC_1193_318})

V_555 = CTVertex(name = 'V_555',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.a, P.g, P.Z ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1191_313,(0,0,1):C.R2GC_1191_314,(0,1,0):C.R2GC_1190_312,(0,1,1):C.R2GC_1189_311})

V_556 = CTVertex(name = 'V_556',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.a, P.g, P.Z ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1191_313,(0,0,1):C.R2GC_1191_314,(0,1,0):C.R2GC_1190_312,(0,1,1):C.R2GC_1189_311})

V_557 = CTVertex(name = 'V_557',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.g, P.Z, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1243_407,(0,0,1):C.R2GC_1243_408,(0,1,0):C.R2GC_1244_409,(0,1,1):C.R2GC_1244_410})

V_558 = CTVertex(name = 'V_558',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.g, P.Z, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1243_407,(0,0,1):C.R2GC_1243_408,(0,1,0):C.R2GC_1244_409,(0,1,1):C.R2GC_1244_410})

V_559 = CTVertex(name = 'V_559',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.g, P.Z, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1242_405,(0,0,1):C.R2GC_1242_406,(0,1,0):C.R2GC_1241_404,(0,1,1):C.R2GC_1240_403})

V_560 = CTVertex(name = 'V_560',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.g, P.Z, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1242_405,(0,0,1):C.R2GC_1242_406,(0,1,0):C.R2GC_1241_404,(0,1,1):C.R2GC_1240_403})

V_561 = CTVertex(name = 'V_561',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.a, P.g, P.W__plus__ ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVV127 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1609_1112})

V_562 = CTVertex(name = 'V_562',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.a, P.g, P.W__plus__ ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVV127 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1609_1112})

V_563 = CTVertex(name = 'V_563',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.g, P.W__plus__, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV123, L.FFVVV127 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1577_1076,(0,0,0):C.R2GC_1599_1102})

V_564 = CTVertex(name = 'V_564',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.g, P.W__plus__, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV123, L.FFVVV127 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1577_1076,(0,0,0):C.R2GC_1599_1102})

V_565 = CTVertex(name = 'V_565',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.a, P.g, P.W__minus__ ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVV128 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1610_1113})

V_566 = CTVertex(name = 'V_566',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.a, P.g, P.W__minus__ ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVV128 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1610_1113})

V_567 = CTVertex(name = 'V_567',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.g, P.W__minus__, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV123, L.FFVVV128 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1578_1077,(0,0,0):C.R2GC_1598_1101})

V_568 = CTVertex(name = 'V_568',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.g, P.W__minus__, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV123, L.FFVVV128 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1578_1077,(0,0,0):C.R2GC_1598_1101})

V_569 = CTVertex(name = 'V_569',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.g, P.W__minus__, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV151 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1596_1098,(0,0,0):C.R2GC_1594_1096,(0,2,0):C.R2GC_797_1900})

V_570 = CTVertex(name = 'V_570',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.g, P.W__minus__, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV151 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1596_1098,(0,0,0):C.R2GC_1594_1096,(0,2,0):C.R2GC_797_1900})

V_571 = CTVertex(name = 'V_571',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.g, P.W__minus__, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV151 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1595_1097,(0,0,0):C.R2GC_1594_1096,(0,2,0):C.R2GC_805_1905})

V_572 = CTVertex(name = 'V_572',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.g, P.W__minus__, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV151 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1595_1097,(0,0,0):C.R2GC_1594_1096,(0,2,0):C.R2GC_805_1905})

V_573 = CTVertex(name = 'V_573',
                 type = 'R2',
                 particles = [ P.ve__tilde__, P.ve, P.a, P.g, P.g ],
                 color = [ 'Identity(4,5)' ],
                 lorentz = [ L.FFVVV121 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1084_138,(0,0,1):C.R2GC_1084_139})

V_574 = CTVertex(name = 'V_574',
                 type = 'R2',
                 particles = [ P.vm__tilde__, P.vm, P.a, P.g, P.g ],
                 color = [ 'Identity(4,5)' ],
                 lorentz = [ L.FFVVV121 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1086_142,(0,0,1):C.R2GC_1086_143})

V_575 = CTVertex(name = 'V_575',
                 type = 'R2',
                 particles = [ P.vt__tilde__, P.vt, P.a, P.g, P.g ],
                 color = [ 'Identity(4,5)' ],
                 lorentz = [ L.FFVVV121 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1088_146,(0,0,1):C.R2GC_1088_147})

V_576 = CTVertex(name = 'V_576',
                 type = 'R2',
                 particles = [ P.e__plus__, P.e__minus__, P.a, P.g, P.g ],
                 color = [ 'Identity(4,5)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1083_136,(0,0,1):C.R2GC_1083_137,(0,1,0):C.R2GC_1080_130,(0,1,1):C.R2GC_1080_131})

V_577 = CTVertex(name = 'V_577',
                 type = 'R2',
                 particles = [ P.mu__plus__, P.mu__minus__, P.a, P.g, P.g ],
                 color = [ 'Identity(4,5)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1085_140,(0,0,1):C.R2GC_1085_141,(0,1,0):C.R2GC_1081_132,(0,1,1):C.R2GC_1081_133})

V_578 = CTVertex(name = 'V_578',
                 type = 'R2',
                 particles = [ P.ta__plus__, P.ta__minus__, P.a, P.g, P.g ],
                 color = [ 'Identity(4,5)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1087_144,(0,0,1):C.R2GC_1087_145,(0,1,0):C.R2GC_1082_134,(0,1,1):C.R2GC_1082_135})

V_579 = CTVertex(name = 'V_579',
                 type = 'R2',
                 particles = [ P.ve__tilde__, P.ve, P.g, P.g, P.Z ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVVV121 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1208_345,(0,0,1):C.R2GC_1208_346})

V_580 = CTVertex(name = 'V_580',
                 type = 'R2',
                 particles = [ P.vm__tilde__, P.vm, P.g, P.g, P.Z ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVVV121 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1210_349,(0,0,1):C.R2GC_1210_350})

V_581 = CTVertex(name = 'V_581',
                 type = 'R2',
                 particles = [ P.vt__tilde__, P.vt, P.g, P.g, P.Z ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVVV121 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1212_353,(0,0,1):C.R2GC_1212_354})

V_582 = CTVertex(name = 'V_582',
                 type = 'R2',
                 particles = [ P.e__plus__, P.e__minus__, P.g, P.g, P.Z ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1207_343,(0,0,1):C.R2GC_1207_344,(0,1,0):C.R2GC_1204_337,(0,1,1):C.R2GC_1204_338})

V_583 = CTVertex(name = 'V_583',
                 type = 'R2',
                 particles = [ P.mu__plus__, P.mu__minus__, P.g, P.g, P.Z ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1209_347,(0,0,1):C.R2GC_1209_348,(0,1,0):C.R2GC_1205_339,(0,1,1):C.R2GC_1205_340})

V_584 = CTVertex(name = 'V_584',
                 type = 'R2',
                 particles = [ P.ta__plus__, P.ta__minus__, P.g, P.g, P.Z ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV151 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1211_351,(0,0,1):C.R2GC_1211_352,(0,1,0):C.R2GC_1206_341,(0,1,1):C.R2GC_1206_342})

V_585 = CTVertex(name = 'V_585',
                 type = 'R2',
                 particles = [ P.ve__tilde__, P.e__minus__, P.g, P.g, P.W__plus__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVVV121 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_829_1918})

V_586 = CTVertex(name = 'V_586',
                 type = 'R2',
                 particles = [ P.vm__tilde__, P.mu__minus__, P.g, P.g, P.W__plus__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVVV121 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_830_1919})

V_587 = CTVertex(name = 'V_587',
                 type = 'R2',
                 particles = [ P.vt__tilde__, P.ta__minus__, P.g, P.g, P.W__plus__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVVV121 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_831_1920})

V_588 = CTVertex(name = 'V_588',
                 type = 'R2',
                 particles = [ P.e__plus__, P.ve, P.g, P.g, P.W__minus__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVVV121 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_829_1918})

V_589 = CTVertex(name = 'V_589',
                 type = 'R2',
                 particles = [ P.mu__plus__, P.vm, P.g, P.g, P.W__minus__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVVV121 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_830_1919})

V_590 = CTVertex(name = 'V_590',
                 type = 'R2',
                 particles = [ P.ta__plus__, P.vt, P.g, P.g, P.W__minus__ ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFVVV121 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_831_1920})

V_591 = CTVertex(name = 'V_591',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.a, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(4,5)', 'T(4,-1,1)*T(5,2,-1)', 'T(4,2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV153, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_1095_160,(2,1,1):C.R2GC_1095_161,(1,1,0):C.R2GC_1090_150,(1,1,1):C.R2GC_1090_151,(0,0,0):C.R2GC_1091_152,(0,0,1):C.R2GC_1091_153,(2,9,0):C.R2GC_1094_158,(2,9,1):C.R2GC_1094_159,(1,9,0):C.R2GC_1094_158,(1,9,1):C.R2GC_1094_159,(2,8,0):C.R2GC_1094_158,(2,8,1):C.R2GC_1094_159,(1,8,0):C.R2GC_1094_158,(1,8,1):C.R2GC_1094_159,(2,7,0):C.R2GC_1094_158,(2,7,1):C.R2GC_1094_159,(1,7,0):C.R2GC_1094_158,(1,7,1):C.R2GC_1094_159,(2,6,0):C.R2GC_1100_170,(2,6,1):C.R2GC_1100_171,(1,6,0):C.R2GC_1097_164,(1,6,1):C.R2GC_1097_165,(0,5,0):C.R2GC_1098_166,(0,5,1):C.R2GC_1098_167,(2,4,0):C.R2GC_1099_168,(2,4,1):C.R2GC_1099_169,(1,4,0):C.R2GC_1099_168,(1,4,1):C.R2GC_1099_169,(2,3,0):C.R2GC_1099_168,(2,3,1):C.R2GC_1099_169,(1,3,0):C.R2GC_1099_168,(1,3,1):C.R2GC_1099_169,(2,2,0):C.R2GC_1099_168,(2,2,1):C.R2GC_1099_169,(1,2,0):C.R2GC_1099_168,(1,2,1):C.R2GC_1099_169})

V_592 = CTVertex(name = 'V_592',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.a, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(4,5)', 'T(4,-1,1)*T(5,2,-1)', 'T(4,2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV153, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_1095_160,(2,1,1):C.R2GC_1095_161,(1,1,0):C.R2GC_1090_150,(1,1,1):C.R2GC_1090_151,(0,0,0):C.R2GC_1091_152,(0,0,1):C.R2GC_1091_153,(2,9,0):C.R2GC_1094_158,(2,9,1):C.R2GC_1094_159,(1,9,0):C.R2GC_1094_158,(1,9,1):C.R2GC_1094_159,(2,8,0):C.R2GC_1094_158,(2,8,1):C.R2GC_1094_159,(1,8,0):C.R2GC_1094_158,(1,8,1):C.R2GC_1094_159,(2,7,0):C.R2GC_1094_158,(2,7,1):C.R2GC_1094_159,(1,7,0):C.R2GC_1094_158,(1,7,1):C.R2GC_1094_159,(2,6,0):C.R2GC_1100_170,(2,6,1):C.R2GC_1100_171,(1,6,0):C.R2GC_1097_164,(1,6,1):C.R2GC_1097_165,(0,5,0):C.R2GC_1098_166,(0,5,1):C.R2GC_1098_167,(2,4,0):C.R2GC_1099_168,(2,4,1):C.R2GC_1099_169,(1,4,0):C.R2GC_1099_168,(1,4,1):C.R2GC_1099_169,(2,3,0):C.R2GC_1099_168,(2,3,1):C.R2GC_1099_169,(1,3,0):C.R2GC_1099_168,(1,3,1):C.R2GC_1099_169,(2,2,0):C.R2GC_1099_168,(2,2,1):C.R2GC_1099_169,(1,2,0):C.R2GC_1099_168,(1,2,1):C.R2GC_1099_169})

V_593 = CTVertex(name = 'V_593',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.a, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(4,5)', 'T(4,-1,1)*T(5,2,-1)', 'T(4,2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV153, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_1096_162,(2,1,1):C.R2GC_1096_163,(1,1,0):C.R2GC_1089_148,(1,1,1):C.R2GC_1089_149,(0,0,0):C.R2GC_1092_154,(0,0,1):C.R2GC_1092_155,(2,9,0):C.R2GC_1093_156,(2,9,1):C.R2GC_1093_157,(1,9,0):C.R2GC_1093_156,(1,9,1):C.R2GC_1093_157,(2,8,0):C.R2GC_1093_156,(2,8,1):C.R2GC_1093_157,(1,8,0):C.R2GC_1093_156,(1,8,1):C.R2GC_1093_157,(2,7,0):C.R2GC_1093_156,(2,7,1):C.R2GC_1093_157,(1,7,0):C.R2GC_1093_156,(1,7,1):C.R2GC_1093_157,(2,6,0):C.R2GC_1079_128,(2,6,1):C.R2GC_1079_129,(1,6,0):C.R2GC_1075_122,(1,6,1):C.R2GC_1075_123,(0,5,0):C.R2GC_1077_125,(0,5,1):C.R2GC_1076_124,(2,4,0):C.R2GC_1078_126,(2,4,1):C.R2GC_1078_127,(1,4,0):C.R2GC_1078_126,(1,4,1):C.R2GC_1078_127,(2,3,0):C.R2GC_1078_126,(2,3,1):C.R2GC_1078_127,(1,3,0):C.R2GC_1078_126,(1,3,1):C.R2GC_1078_127,(2,2,0):C.R2GC_1078_126,(2,2,1):C.R2GC_1078_127,(1,2,0):C.R2GC_1078_126,(1,2,1):C.R2GC_1078_127})

V_594 = CTVertex(name = 'V_594',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.a, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(4,5)', 'T(4,-1,1)*T(5,2,-1)', 'T(4,2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV153, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_1096_162,(2,1,1):C.R2GC_1096_163,(1,1,0):C.R2GC_1089_148,(1,1,1):C.R2GC_1089_149,(0,0,0):C.R2GC_1092_154,(0,0,1):C.R2GC_1092_155,(2,9,0):C.R2GC_1093_156,(2,9,1):C.R2GC_1093_157,(1,9,0):C.R2GC_1093_156,(1,9,1):C.R2GC_1093_157,(2,8,0):C.R2GC_1093_156,(2,8,1):C.R2GC_1093_157,(1,8,0):C.R2GC_1093_156,(1,8,1):C.R2GC_1093_157,(2,7,0):C.R2GC_1093_156,(2,7,1):C.R2GC_1093_157,(1,7,0):C.R2GC_1093_156,(1,7,1):C.R2GC_1093_157,(2,6,0):C.R2GC_1079_128,(2,6,1):C.R2GC_1079_129,(1,6,0):C.R2GC_1075_122,(1,6,1):C.R2GC_1075_123,(0,5,0):C.R2GC_1077_125,(0,5,1):C.R2GC_1076_124,(2,4,0):C.R2GC_1078_126,(2,4,1):C.R2GC_1078_127,(1,4,0):C.R2GC_1078_126,(1,4,1):C.R2GC_1078_127,(2,3,0):C.R2GC_1078_126,(2,3,1):C.R2GC_1078_127,(1,3,0):C.R2GC_1078_126,(1,3,1):C.R2GC_1078_127,(2,2,0):C.R2GC_1078_126,(2,2,1):C.R2GC_1078_127,(1,2,0):C.R2GC_1078_126,(1,2,1):C.R2GC_1078_127})

V_595 = CTVertex(name = 'V_595',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.g, P.g, P.Z ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV153, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_1219_367,(2,1,1):C.R2GC_1219_368,(1,1,0):C.R2GC_1214_357,(1,1,1):C.R2GC_1214_358,(0,0,0):C.R2GC_1215_359,(0,0,1):C.R2GC_1215_360,(2,9,0):C.R2GC_1218_365,(2,9,1):C.R2GC_1218_366,(1,9,0):C.R2GC_1218_365,(1,9,1):C.R2GC_1218_366,(2,8,0):C.R2GC_1218_365,(2,8,1):C.R2GC_1218_366,(1,8,0):C.R2GC_1218_365,(1,8,1):C.R2GC_1218_366,(2,7,0):C.R2GC_1218_365,(2,7,1):C.R2GC_1218_366,(1,7,0):C.R2GC_1218_365,(1,7,1):C.R2GC_1218_366,(2,6,0):C.R2GC_1224_377,(2,6,1):C.R2GC_1224_378,(1,6,0):C.R2GC_1221_371,(1,6,1):C.R2GC_1221_372,(0,5,0):C.R2GC_1222_373,(0,5,1):C.R2GC_1222_374,(2,4,0):C.R2GC_1223_375,(2,4,1):C.R2GC_1223_376,(1,4,0):C.R2GC_1223_375,(1,4,1):C.R2GC_1223_376,(2,3,0):C.R2GC_1223_375,(2,3,1):C.R2GC_1223_376,(1,3,0):C.R2GC_1223_375,(1,3,1):C.R2GC_1223_376,(2,2,0):C.R2GC_1223_375,(2,2,1):C.R2GC_1223_376,(1,2,0):C.R2GC_1223_375,(1,2,1):C.R2GC_1223_376})

V_596 = CTVertex(name = 'V_596',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.g, P.g, P.Z ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV153, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_1219_367,(2,1,1):C.R2GC_1219_368,(1,1,0):C.R2GC_1214_357,(1,1,1):C.R2GC_1214_358,(0,0,0):C.R2GC_1215_359,(0,0,1):C.R2GC_1215_360,(2,9,0):C.R2GC_1218_365,(2,9,1):C.R2GC_1218_366,(1,9,0):C.R2GC_1218_365,(1,9,1):C.R2GC_1218_366,(2,8,0):C.R2GC_1218_365,(2,8,1):C.R2GC_1218_366,(1,8,0):C.R2GC_1218_365,(1,8,1):C.R2GC_1218_366,(2,7,0):C.R2GC_1218_365,(2,7,1):C.R2GC_1218_366,(1,7,0):C.R2GC_1218_365,(1,7,1):C.R2GC_1218_366,(2,6,0):C.R2GC_1224_377,(2,6,1):C.R2GC_1224_378,(1,6,0):C.R2GC_1221_371,(1,6,1):C.R2GC_1221_372,(0,5,0):C.R2GC_1222_373,(0,5,1):C.R2GC_1222_374,(2,4,0):C.R2GC_1223_375,(2,4,1):C.R2GC_1223_376,(1,4,0):C.R2GC_1223_375,(1,4,1):C.R2GC_1223_376,(2,3,0):C.R2GC_1223_375,(2,3,1):C.R2GC_1223_376,(1,3,0):C.R2GC_1223_375,(1,3,1):C.R2GC_1223_376,(2,2,0):C.R2GC_1223_375,(2,2,1):C.R2GC_1223_376,(1,2,0):C.R2GC_1223_375,(1,2,1):C.R2GC_1223_376})

V_597 = CTVertex(name = 'V_597',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.g, P.g, P.Z ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV153, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_1220_369,(2,1,1):C.R2GC_1220_370,(1,1,0):C.R2GC_1213_355,(1,1,1):C.R2GC_1213_356,(0,0,0):C.R2GC_1216_361,(0,0,1):C.R2GC_1216_362,(2,9,0):C.R2GC_1217_363,(2,9,1):C.R2GC_1217_364,(1,9,0):C.R2GC_1217_363,(1,9,1):C.R2GC_1217_364,(2,8,0):C.R2GC_1217_363,(2,8,1):C.R2GC_1217_364,(1,8,0):C.R2GC_1217_363,(1,8,1):C.R2GC_1217_364,(2,7,0):C.R2GC_1217_363,(2,7,1):C.R2GC_1217_364,(1,7,0):C.R2GC_1217_363,(1,7,1):C.R2GC_1217_364,(2,6,0):C.R2GC_1203_335,(2,6,1):C.R2GC_1203_336,(1,6,0):C.R2GC_1199_328,(1,6,1):C.R2GC_1199_329,(0,5,0):C.R2GC_1201_332,(0,5,1):C.R2GC_1200_331,(2,4,0):C.R2GC_1202_333,(2,4,1):C.R2GC_1202_334,(1,4,0):C.R2GC_1202_333,(1,4,1):C.R2GC_1202_334,(2,3,0):C.R2GC_1202_333,(2,3,1):C.R2GC_1202_334,(1,3,0):C.R2GC_1202_333,(1,3,1):C.R2GC_1202_334,(2,2,0):C.R2GC_1202_333,(2,2,1):C.R2GC_1202_334,(1,2,0):C.R2GC_1202_333,(1,2,1):C.R2GC_1202_334})

V_598 = CTVertex(name = 'V_598',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.g, P.g, P.Z ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV153, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_1220_369,(2,1,1):C.R2GC_1220_370,(1,1,0):C.R2GC_1213_355,(1,1,1):C.R2GC_1213_356,(0,0,0):C.R2GC_1216_361,(0,0,1):C.R2GC_1216_362,(2,9,0):C.R2GC_1217_363,(2,9,1):C.R2GC_1217_364,(1,9,0):C.R2GC_1217_363,(1,9,1):C.R2GC_1217_364,(2,8,0):C.R2GC_1217_363,(2,8,1):C.R2GC_1217_364,(1,8,0):C.R2GC_1217_363,(1,8,1):C.R2GC_1217_364,(2,7,0):C.R2GC_1217_363,(2,7,1):C.R2GC_1217_364,(1,7,0):C.R2GC_1217_363,(1,7,1):C.R2GC_1217_364,(2,6,0):C.R2GC_1203_335,(2,6,1):C.R2GC_1203_336,(1,6,0):C.R2GC_1199_328,(1,6,1):C.R2GC_1199_329,(0,5,0):C.R2GC_1201_332,(0,5,1):C.R2GC_1200_331,(2,4,0):C.R2GC_1202_333,(2,4,1):C.R2GC_1202_334,(1,4,0):C.R2GC_1202_333,(1,4,1):C.R2GC_1202_334,(2,3,0):C.R2GC_1202_333,(2,3,1):C.R2GC_1202_334,(1,3,0):C.R2GC_1202_333,(1,3,1):C.R2GC_1202_334,(2,2,0):C.R2GC_1202_333,(2,2,1):C.R2GC_1202_334,(1,2,0):C.R2GC_1202_333,(1,2,1):C.R2GC_1202_334})

V_599 = CTVertex(name = 'V_599',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.g, P.g, P.W__plus__ ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_1614_1120,(1,1,0):C.R2GC_1618_1125,(0,0,0):C.R2GC_1509_980,(2,4,0):C.R2GC_1616_1123,(1,4,0):C.R2GC_1616_1123,(2,3,0):C.R2GC_1616_1123,(1,3,0):C.R2GC_1616_1123,(2,2,0):C.R2GC_1616_1123,(1,2,0):C.R2GC_1616_1123})

V_600 = CTVertex(name = 'V_600',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.g, P.g, P.W__plus__ ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_1614_1120,(1,1,0):C.R2GC_1618_1125,(0,0,0):C.R2GC_1509_980,(2,4,0):C.R2GC_1616_1123,(1,4,0):C.R2GC_1616_1123,(2,3,0):C.R2GC_1616_1123,(1,3,0):C.R2GC_1616_1123,(2,2,0):C.R2GC_1616_1123,(1,2,0):C.R2GC_1616_1123})

V_601 = CTVertex(name = 'V_601',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.g, P.g, P.W__minus__ ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_1614_1120,(1,1,0):C.R2GC_1618_1125,(0,0,0):C.R2GC_1509_980,(2,4,0):C.R2GC_1616_1123,(1,4,0):C.R2GC_1616_1123,(2,3,0):C.R2GC_1616_1123,(1,3,0):C.R2GC_1616_1123,(2,2,0):C.R2GC_1616_1123,(1,2,0):C.R2GC_1616_1123})

V_602 = CTVertex(name = 'V_602',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.g, P.g, P.W__minus__ ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(2,1,0):C.R2GC_1614_1120,(1,1,0):C.R2GC_1618_1125,(0,0,0):C.R2GC_1509_980,(2,4,0):C.R2GC_1616_1123,(1,4,0):C.R2GC_1616_1123,(2,3,0):C.R2GC_1616_1123,(1,3,0):C.R2GC_1616_1123,(2,2,0):C.R2GC_1616_1123,(1,2,0):C.R2GC_1616_1123})

V_603 = CTVertex(name = 'V_603',
                 type = 'R2',
                 particles = [ P.ve__tilde__, P.ve, P.g, P.g, P.g ],
                 color = [ 'd(3,4,5)', 'f(3,4,5)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(1,1,0):C.R2GC_1122_204,(1,1,1):C.R2GC_1122_205,(0,0,0):C.R2GC_1120_200,(0,0,1):C.R2GC_1120_201})

V_604 = CTVertex(name = 'V_604',
                 type = 'R2',
                 particles = [ P.vm__tilde__, P.vm, P.g, P.g, P.g ],
                 color = [ 'd(3,4,5)', 'f(3,4,5)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(1,1,0):C.R2GC_1126_212,(1,1,1):C.R2GC_1126_213,(0,0,0):C.R2GC_1124_208,(0,0,1):C.R2GC_1124_209})

V_605 = CTVertex(name = 'V_605',
                 type = 'R2',
                 particles = [ P.vt__tilde__, P.vt, P.g, P.g, P.g ],
                 color = [ 'd(3,4,5)', 'f(3,4,5)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(1,1,0):C.R2GC_1130_220,(1,1,1):C.R2GC_1130_221,(0,0,0):C.R2GC_1128_216,(0,0,1):C.R2GC_1128_217})

V_606 = CTVertex(name = 'V_606',
                 type = 'R2',
                 particles = [ P.e__plus__, P.e__minus__, P.g, P.g, P.g ],
                 color = [ 'd(3,4,5)', 'f(3,4,5)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV151, L.FFVVV153 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(1,1,0):C.R2GC_1121_202,(1,1,1):C.R2GC_1121_203,(0,0,0):C.R2GC_1119_198,(0,0,1):C.R2GC_1119_199,(1,3,0):C.R2GC_1114_188,(1,3,1):C.R2GC_1114_189,(0,2,0):C.R2GC_1113_186,(0,2,1):C.R2GC_1113_187})

V_607 = CTVertex(name = 'V_607',
                 type = 'R2',
                 particles = [ P.mu__plus__, P.mu__minus__, P.g, P.g, P.g ],
                 color = [ 'd(3,4,5)', 'f(3,4,5)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV151, L.FFVVV153 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(1,1,0):C.R2GC_1125_210,(1,1,1):C.R2GC_1125_211,(0,0,0):C.R2GC_1123_206,(0,0,1):C.R2GC_1123_207,(1,3,0):C.R2GC_1116_192,(1,3,1):C.R2GC_1116_193,(0,2,0):C.R2GC_1115_190,(0,2,1):C.R2GC_1115_191})

V_608 = CTVertex(name = 'V_608',
                 type = 'R2',
                 particles = [ P.ta__plus__, P.ta__minus__, P.g, P.g, P.g ],
                 color = [ 'd(3,4,5)', 'f(3,4,5)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV151, L.FFVVV153 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(1,1,0):C.R2GC_1129_218,(1,1,1):C.R2GC_1129_219,(0,0,0):C.R2GC_1127_214,(0,0,1):C.R2GC_1127_215,(1,3,0):C.R2GC_1118_196,(1,3,1):C.R2GC_1118_197,(0,2,0):C.R2GC_1117_194,(0,2,1):C.R2GC_1117_195})

V_609 = CTVertex(name = 'V_609',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.g, P.g, P.g ],
                 color = [ 'd(3,4,5)*Identity(1,2)', 'f(3,4,5)*Identity(1,2)', 'T(3,-1,1)*T(4,-2,-1)*T(5,2,-2)', 'T(3,-1,1)*T(4,2,-2)*T(5,-2,-1)', 'T(3,-2,-1)*T(4,-1,1)*T(5,2,-2)', 'T(3,-2,-1)*T(4,2,-2)*T(5,-1,1)', 'T(3,2,-2)*T(4,-1,1)*T(5,-2,-1)', 'T(3,2,-2)*T(4,-2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV153, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(1,1,0):C.R2GC_1135_230,(1,1,1):C.R2GC_1135_231,(0,0,0):C.R2GC_1134_228,(0,0,1):C.R2GC_1134_229,(7,1,0):C.R2GC_1138_236,(7,1,1):C.R2GC_1138_237,(7,9,0):C.R2GC_1789_1232,(7,9,1):C.R2GC_1789_1233,(7,8,0):C.R2GC_1790_1234,(7,8,1):C.R2GC_1790_1235,(7,7,0):C.R2GC_1789_1232,(7,7,1):C.R2GC_1789_1233,(6,1,0):C.R2GC_1131_222,(6,1,1):C.R2GC_1131_223,(6,9,0):C.R2GC_1789_1232,(6,9,1):C.R2GC_1789_1233,(6,8,0):C.R2GC_1789_1232,(6,8,1):C.R2GC_1789_1233,(6,7,0):C.R2GC_1790_1234,(6,7,1):C.R2GC_1790_1235,(5,1,0):C.R2GC_1131_222,(5,1,1):C.R2GC_1131_223,(5,9,0):C.R2GC_1790_1234,(5,9,1):C.R2GC_1790_1235,(5,8,0):C.R2GC_1789_1232,(5,8,1):C.R2GC_1789_1233,(5,7,0):C.R2GC_1789_1232,(5,7,1):C.R2GC_1789_1233,(3,1,0):C.R2GC_1138_236,(3,1,1):C.R2GC_1138_237,(3,9,0):C.R2GC_1789_1232,(3,9,1):C.R2GC_1789_1233,(3,8,0):C.R2GC_1789_1232,(3,8,1):C.R2GC_1789_1233,(3,7,0):C.R2GC_1790_1234,(3,7,1):C.R2GC_1790_1235,(4,1,0):C.R2GC_1138_236,(4,1,1):C.R2GC_1138_237,(4,9,0):C.R2GC_1790_1234,(4,9,1):C.R2GC_1790_1235,(4,8,0):C.R2GC_1789_1232,(4,8,1):C.R2GC_1789_1233,(4,7,0):C.R2GC_1789_1232,(4,7,1):C.R2GC_1789_1233,(2,1,0):C.R2GC_1131_222,(2,1,1):C.R2GC_1131_223,(2,9,0):C.R2GC_1789_1232,(2,9,1):C.R2GC_1789_1233,(2,8,0):C.R2GC_1790_1234,(2,8,1):C.R2GC_1790_1235,(2,7,0):C.R2GC_1789_1232,(2,7,1):C.R2GC_1789_1233,(1,6,0):C.R2GC_1111_183,(1,6,1):C.R2GC_1110_182,(0,5,0):C.R2GC_1109_181,(0,5,1):C.R2GC_1108_180,(7,6,0):C.R2GC_1112_184,(7,6,1):C.R2GC_1112_185,(7,4,0):C.R2GC_1786_1226,(7,4,1):C.R2GC_1786_1227,(7,3,0):C.R2GC_1787_1228,(7,3,1):C.R2GC_1787_1229,(7,2,0):C.R2GC_1786_1226,(7,2,1):C.R2GC_1786_1227,(6,6,0):C.R2GC_1107_178,(6,6,1):C.R2GC_1107_179,(6,4,0):C.R2GC_1786_1226,(6,4,1):C.R2GC_1786_1227,(6,3,0):C.R2GC_1786_1226,(6,3,1):C.R2GC_1786_1227,(6,2,0):C.R2GC_1787_1228,(6,2,1):C.R2GC_1787_1229,(5,6,0):C.R2GC_1107_178,(5,6,1):C.R2GC_1107_179,(5,4,0):C.R2GC_1787_1228,(5,4,1):C.R2GC_1787_1229,(5,3,0):C.R2GC_1786_1226,(5,3,1):C.R2GC_1786_1227,(5,2,0):C.R2GC_1786_1226,(5,2,1):C.R2GC_1786_1227,(3,6,0):C.R2GC_1112_184,(3,6,1):C.R2GC_1112_185,(3,4,0):C.R2GC_1786_1226,(3,4,1):C.R2GC_1786_1227,(3,3,0):C.R2GC_1786_1226,(3,3,1):C.R2GC_1786_1227,(3,2,0):C.R2GC_1787_1228,(3,2,1):C.R2GC_1787_1229,(4,6,0):C.R2GC_1112_184,(4,6,1):C.R2GC_1112_185,(4,4,0):C.R2GC_1787_1228,(4,4,1):C.R2GC_1787_1229,(4,3,0):C.R2GC_1786_1226,(4,3,1):C.R2GC_1786_1227,(4,2,0):C.R2GC_1786_1226,(4,2,1):C.R2GC_1786_1227,(2,6,0):C.R2GC_1107_178,(2,6,1):C.R2GC_1107_179,(2,4,0):C.R2GC_1786_1226,(2,4,1):C.R2GC_1786_1227,(2,3,0):C.R2GC_1787_1228,(2,3,1):C.R2GC_1787_1229,(2,2,0):C.R2GC_1786_1226,(2,2,1):C.R2GC_1786_1227})

V_610 = CTVertex(name = 'V_610',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.g, P.g, P.g ],
                 color = [ 'd(3,4,5)*Identity(1,2)', 'f(3,4,5)*Identity(1,2)', 'T(3,-1,1)*T(4,-2,-1)*T(5,2,-2)', 'T(3,-1,1)*T(4,2,-2)*T(5,-2,-1)', 'T(3,-2,-1)*T(4,-1,1)*T(5,2,-2)', 'T(3,-2,-1)*T(4,2,-2)*T(5,-1,1)', 'T(3,2,-2)*T(4,-1,1)*T(5,-2,-1)', 'T(3,2,-2)*T(4,-2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV153, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(1,1,0):C.R2GC_1135_230,(1,1,1):C.R2GC_1135_231,(0,0,0):C.R2GC_1134_228,(0,0,1):C.R2GC_1134_229,(7,1,0):C.R2GC_1138_236,(7,1,1):C.R2GC_1138_237,(7,9,0):C.R2GC_1789_1232,(7,9,1):C.R2GC_1789_1233,(7,8,0):C.R2GC_1790_1234,(7,8,1):C.R2GC_1790_1235,(7,7,0):C.R2GC_1789_1232,(7,7,1):C.R2GC_1789_1233,(6,1,0):C.R2GC_1131_222,(6,1,1):C.R2GC_1131_223,(6,9,0):C.R2GC_1789_1232,(6,9,1):C.R2GC_1789_1233,(6,8,0):C.R2GC_1789_1232,(6,8,1):C.R2GC_1789_1233,(6,7,0):C.R2GC_1790_1234,(6,7,1):C.R2GC_1790_1235,(5,1,0):C.R2GC_1131_222,(5,1,1):C.R2GC_1131_223,(5,9,0):C.R2GC_1790_1234,(5,9,1):C.R2GC_1790_1235,(5,8,0):C.R2GC_1789_1232,(5,8,1):C.R2GC_1789_1233,(5,7,0):C.R2GC_1789_1232,(5,7,1):C.R2GC_1789_1233,(3,1,0):C.R2GC_1138_236,(3,1,1):C.R2GC_1138_237,(3,9,0):C.R2GC_1789_1232,(3,9,1):C.R2GC_1789_1233,(3,8,0):C.R2GC_1789_1232,(3,8,1):C.R2GC_1789_1233,(3,7,0):C.R2GC_1790_1234,(3,7,1):C.R2GC_1790_1235,(4,1,0):C.R2GC_1138_236,(4,1,1):C.R2GC_1138_237,(4,9,0):C.R2GC_1790_1234,(4,9,1):C.R2GC_1790_1235,(4,8,0):C.R2GC_1789_1232,(4,8,1):C.R2GC_1789_1233,(4,7,0):C.R2GC_1789_1232,(4,7,1):C.R2GC_1789_1233,(2,1,0):C.R2GC_1131_222,(2,1,1):C.R2GC_1131_223,(2,9,0):C.R2GC_1789_1232,(2,9,1):C.R2GC_1789_1233,(2,8,0):C.R2GC_1790_1234,(2,8,1):C.R2GC_1790_1235,(2,7,0):C.R2GC_1789_1232,(2,7,1):C.R2GC_1789_1233,(1,6,0):C.R2GC_1111_183,(1,6,1):C.R2GC_1110_182,(0,5,0):C.R2GC_1109_181,(0,5,1):C.R2GC_1108_180,(7,6,0):C.R2GC_1112_184,(7,6,1):C.R2GC_1112_185,(7,4,0):C.R2GC_1786_1226,(7,4,1):C.R2GC_1786_1227,(7,3,0):C.R2GC_1787_1228,(7,3,1):C.R2GC_1787_1229,(7,2,0):C.R2GC_1786_1226,(7,2,1):C.R2GC_1786_1227,(6,6,0):C.R2GC_1107_178,(6,6,1):C.R2GC_1107_179,(6,4,0):C.R2GC_1786_1226,(6,4,1):C.R2GC_1786_1227,(6,3,0):C.R2GC_1786_1226,(6,3,1):C.R2GC_1786_1227,(6,2,0):C.R2GC_1787_1228,(6,2,1):C.R2GC_1787_1229,(5,6,0):C.R2GC_1107_178,(5,6,1):C.R2GC_1107_179,(5,4,0):C.R2GC_1787_1228,(5,4,1):C.R2GC_1787_1229,(5,3,0):C.R2GC_1786_1226,(5,3,1):C.R2GC_1786_1227,(5,2,0):C.R2GC_1786_1226,(5,2,1):C.R2GC_1786_1227,(3,6,0):C.R2GC_1112_184,(3,6,1):C.R2GC_1112_185,(3,4,0):C.R2GC_1786_1226,(3,4,1):C.R2GC_1786_1227,(3,3,0):C.R2GC_1786_1226,(3,3,1):C.R2GC_1786_1227,(3,2,0):C.R2GC_1787_1228,(3,2,1):C.R2GC_1787_1229,(4,6,0):C.R2GC_1112_184,(4,6,1):C.R2GC_1112_185,(4,4,0):C.R2GC_1787_1228,(4,4,1):C.R2GC_1787_1229,(4,3,0):C.R2GC_1786_1226,(4,3,1):C.R2GC_1786_1227,(4,2,0):C.R2GC_1786_1226,(4,2,1):C.R2GC_1786_1227,(2,6,0):C.R2GC_1107_178,(2,6,1):C.R2GC_1107_179,(2,4,0):C.R2GC_1786_1226,(2,4,1):C.R2GC_1786_1227,(2,3,0):C.R2GC_1787_1228,(2,3,1):C.R2GC_1787_1229,(2,2,0):C.R2GC_1786_1226,(2,2,1):C.R2GC_1786_1227})

V_611 = CTVertex(name = 'V_611',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.g, P.g, P.g ],
                 color = [ 'd(3,4,5)*Identity(1,2)', 'f(3,4,5)*Identity(1,2)', 'T(3,-1,1)*T(4,-2,-1)*T(5,2,-2)', 'T(3,-1,1)*T(4,2,-2)*T(5,-2,-1)', 'T(3,-2,-1)*T(4,-1,1)*T(5,2,-2)', 'T(3,-2,-1)*T(4,2,-2)*T(5,-1,1)', 'T(3,2,-2)*T(4,-1,1)*T(5,-2,-1)', 'T(3,2,-2)*T(4,-2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV153, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(1,1,0):C.R2GC_1136_232,(1,1,1):C.R2GC_1136_233,(0,0,0):C.R2GC_1133_226,(0,0,1):C.R2GC_1133_227,(7,1,0):C.R2GC_1137_234,(7,1,1):C.R2GC_1137_235,(7,9,0):C.R2GC_1788_1230,(7,9,1):C.R2GC_1788_1231,(7,8,0):C.R2GC_1791_1236,(7,8,1):C.R2GC_1791_1237,(7,7,0):C.R2GC_1788_1230,(7,7,1):C.R2GC_1788_1231,(6,1,0):C.R2GC_1132_224,(6,1,1):C.R2GC_1132_225,(6,9,0):C.R2GC_1788_1230,(6,9,1):C.R2GC_1788_1231,(6,8,0):C.R2GC_1788_1230,(6,8,1):C.R2GC_1788_1231,(6,7,0):C.R2GC_1791_1236,(6,7,1):C.R2GC_1791_1237,(5,1,0):C.R2GC_1132_224,(5,1,1):C.R2GC_1132_225,(5,9,0):C.R2GC_1791_1236,(5,9,1):C.R2GC_1791_1237,(5,8,0):C.R2GC_1788_1230,(5,8,1):C.R2GC_1788_1231,(5,7,0):C.R2GC_1788_1230,(5,7,1):C.R2GC_1788_1231,(3,1,0):C.R2GC_1137_234,(3,1,1):C.R2GC_1137_235,(3,9,0):C.R2GC_1788_1230,(3,9,1):C.R2GC_1788_1231,(3,8,0):C.R2GC_1788_1230,(3,8,1):C.R2GC_1788_1231,(3,7,0):C.R2GC_1791_1236,(3,7,1):C.R2GC_1791_1237,(4,1,0):C.R2GC_1137_234,(4,1,1):C.R2GC_1137_235,(4,9,0):C.R2GC_1791_1236,(4,9,1):C.R2GC_1791_1237,(4,8,0):C.R2GC_1788_1230,(4,8,1):C.R2GC_1788_1231,(4,7,0):C.R2GC_1788_1230,(4,7,1):C.R2GC_1788_1231,(2,1,0):C.R2GC_1132_224,(2,1,1):C.R2GC_1132_225,(2,9,0):C.R2GC_1788_1230,(2,9,1):C.R2GC_1788_1231,(2,8,0):C.R2GC_1791_1236,(2,8,1):C.R2GC_1791_1237,(2,7,0):C.R2GC_1788_1230,(2,7,1):C.R2GC_1788_1231,(1,6,0):C.R2GC_1141_242,(1,6,1):C.R2GC_1141_243,(0,5,0):C.R2GC_1140_240,(0,5,1):C.R2GC_1140_241,(7,6,0):C.R2GC_1142_244,(7,6,1):C.R2GC_1142_245,(7,4,0):C.R2GC_1792_1238,(7,4,1):C.R2GC_1792_1239,(7,3,0):C.R2GC_1793_1240,(7,3,1):C.R2GC_1793_1241,(7,2,0):C.R2GC_1792_1238,(7,2,1):C.R2GC_1792_1239,(6,6,0):C.R2GC_1139_238,(6,6,1):C.R2GC_1139_239,(6,4,0):C.R2GC_1792_1238,(6,4,1):C.R2GC_1792_1239,(6,3,0):C.R2GC_1792_1238,(6,3,1):C.R2GC_1792_1239,(6,2,0):C.R2GC_1793_1240,(6,2,1):C.R2GC_1793_1241,(5,6,0):C.R2GC_1139_238,(5,6,1):C.R2GC_1139_239,(5,4,0):C.R2GC_1793_1240,(5,4,1):C.R2GC_1793_1241,(5,3,0):C.R2GC_1792_1238,(5,3,1):C.R2GC_1792_1239,(5,2,0):C.R2GC_1792_1238,(5,2,1):C.R2GC_1792_1239,(3,6,0):C.R2GC_1142_244,(3,6,1):C.R2GC_1142_245,(3,4,0):C.R2GC_1792_1238,(3,4,1):C.R2GC_1792_1239,(3,3,0):C.R2GC_1792_1238,(3,3,1):C.R2GC_1792_1239,(3,2,0):C.R2GC_1793_1240,(3,2,1):C.R2GC_1793_1241,(4,6,0):C.R2GC_1142_244,(4,6,1):C.R2GC_1142_245,(4,4,0):C.R2GC_1793_1240,(4,4,1):C.R2GC_1793_1241,(4,3,0):C.R2GC_1792_1238,(4,3,1):C.R2GC_1792_1239,(4,2,0):C.R2GC_1792_1238,(4,2,1):C.R2GC_1792_1239,(2,6,0):C.R2GC_1139_238,(2,6,1):C.R2GC_1139_239,(2,4,0):C.R2GC_1792_1238,(2,4,1):C.R2GC_1792_1239,(2,3,0):C.R2GC_1793_1240,(2,3,1):C.R2GC_1793_1241,(2,2,0):C.R2GC_1792_1238,(2,2,1):C.R2GC_1792_1239})

V_612 = CTVertex(name = 'V_612',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.g, P.g, P.g ],
                 color = [ 'd(3,4,5)*Identity(1,2)', 'f(3,4,5)*Identity(1,2)', 'T(3,-1,1)*T(4,-2,-1)*T(5,2,-2)', 'T(3,-1,1)*T(4,2,-2)*T(5,-2,-1)', 'T(3,-2,-1)*T(4,-1,1)*T(5,2,-2)', 'T(3,-2,-1)*T(4,2,-2)*T(5,-1,1)', 'T(3,2,-2)*T(4,-1,1)*T(5,-2,-1)', 'T(3,2,-2)*T(4,-2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV121, L.FFVVV123, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV151, L.FFVVV153, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(1,1,0):C.R2GC_1136_232,(1,1,1):C.R2GC_1136_233,(0,0,0):C.R2GC_1133_226,(0,0,1):C.R2GC_1133_227,(7,1,0):C.R2GC_1137_234,(7,1,1):C.R2GC_1137_235,(7,9,0):C.R2GC_1788_1230,(7,9,1):C.R2GC_1788_1231,(7,8,0):C.R2GC_1791_1236,(7,8,1):C.R2GC_1791_1237,(7,7,0):C.R2GC_1788_1230,(7,7,1):C.R2GC_1788_1231,(6,1,0):C.R2GC_1132_224,(6,1,1):C.R2GC_1132_225,(6,9,0):C.R2GC_1788_1230,(6,9,1):C.R2GC_1788_1231,(6,8,0):C.R2GC_1788_1230,(6,8,1):C.R2GC_1788_1231,(6,7,0):C.R2GC_1791_1236,(6,7,1):C.R2GC_1791_1237,(5,1,0):C.R2GC_1132_224,(5,1,1):C.R2GC_1132_225,(5,9,0):C.R2GC_1791_1236,(5,9,1):C.R2GC_1791_1237,(5,8,0):C.R2GC_1788_1230,(5,8,1):C.R2GC_1788_1231,(5,7,0):C.R2GC_1788_1230,(5,7,1):C.R2GC_1788_1231,(3,1,0):C.R2GC_1137_234,(3,1,1):C.R2GC_1137_235,(3,9,0):C.R2GC_1788_1230,(3,9,1):C.R2GC_1788_1231,(3,8,0):C.R2GC_1788_1230,(3,8,1):C.R2GC_1788_1231,(3,7,0):C.R2GC_1791_1236,(3,7,1):C.R2GC_1791_1237,(4,1,0):C.R2GC_1137_234,(4,1,1):C.R2GC_1137_235,(4,9,0):C.R2GC_1791_1236,(4,9,1):C.R2GC_1791_1237,(4,8,0):C.R2GC_1788_1230,(4,8,1):C.R2GC_1788_1231,(4,7,0):C.R2GC_1788_1230,(4,7,1):C.R2GC_1788_1231,(2,1,0):C.R2GC_1132_224,(2,1,1):C.R2GC_1132_225,(2,9,0):C.R2GC_1788_1230,(2,9,1):C.R2GC_1788_1231,(2,8,0):C.R2GC_1791_1236,(2,8,1):C.R2GC_1791_1237,(2,7,0):C.R2GC_1788_1230,(2,7,1):C.R2GC_1788_1231,(1,6,0):C.R2GC_1141_242,(1,6,1):C.R2GC_1141_243,(0,5,0):C.R2GC_1140_240,(0,5,1):C.R2GC_1140_241,(7,6,0):C.R2GC_1142_244,(7,6,1):C.R2GC_1142_245,(7,4,0):C.R2GC_1792_1238,(7,4,1):C.R2GC_1792_1239,(7,3,0):C.R2GC_1793_1240,(7,3,1):C.R2GC_1793_1241,(7,2,0):C.R2GC_1792_1238,(7,2,1):C.R2GC_1792_1239,(6,6,0):C.R2GC_1139_238,(6,6,1):C.R2GC_1139_239,(6,4,0):C.R2GC_1792_1238,(6,4,1):C.R2GC_1792_1239,(6,3,0):C.R2GC_1792_1238,(6,3,1):C.R2GC_1792_1239,(6,2,0):C.R2GC_1793_1240,(6,2,1):C.R2GC_1793_1241,(5,6,0):C.R2GC_1139_238,(5,6,1):C.R2GC_1139_239,(5,4,0):C.R2GC_1793_1240,(5,4,1):C.R2GC_1793_1241,(5,3,0):C.R2GC_1792_1238,(5,3,1):C.R2GC_1792_1239,(5,2,0):C.R2GC_1792_1238,(5,2,1):C.R2GC_1792_1239,(3,6,0):C.R2GC_1142_244,(3,6,1):C.R2GC_1142_245,(3,4,0):C.R2GC_1792_1238,(3,4,1):C.R2GC_1792_1239,(3,3,0):C.R2GC_1792_1238,(3,3,1):C.R2GC_1792_1239,(3,2,0):C.R2GC_1793_1240,(3,2,1):C.R2GC_1793_1241,(4,6,0):C.R2GC_1142_244,(4,6,1):C.R2GC_1142_245,(4,4,0):C.R2GC_1793_1240,(4,4,1):C.R2GC_1793_1241,(4,3,0):C.R2GC_1792_1238,(4,3,1):C.R2GC_1792_1239,(4,2,0):C.R2GC_1792_1238,(4,2,1):C.R2GC_1792_1239,(2,6,0):C.R2GC_1139_238,(2,6,1):C.R2GC_1139_239,(2,4,0):C.R2GC_1792_1238,(2,4,1):C.R2GC_1792_1239,(2,3,0):C.R2GC_1793_1240,(2,3,1):C.R2GC_1793_1241,(2,2,0):C.R2GC_1792_1238,(2,2,1):C.R2GC_1792_1239})

V_613 = CTVertex(name = 'V_613',
                 type = 'UV',
                 particles = [ P.g, P.g, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS15, L.VVSS18, L.VVSS19 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1861_253,(0,0,2):C.UVGC_1861_254,(0,2,1):C.UVGC_1753_108,(0,1,2):C.UVGC_1833_208})

V_614 = CTVertex(name = 'V_614',
                 type = 'UV',
                 particles = [ P.g, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS15, L.VVSS18, L.VVSS19 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b, P.t] ], [ [P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1861_253,(0,0,3):C.UVGC_1861_254,(0,2,2):C.UVGC_1753_108,(0,1,1):C.UVGC_1833_208})

V_615 = CTVertex(name = 'V_615',
                 type = 'UV',
                 particles = [ P.g, P.g, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVSS15, L.VVSS18, L.VVSS19 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1861_253,(0,0,2):C.UVGC_1861_254,(0,2,1):C.UVGC_1753_108,(0,1,2):C.UVGC_1833_208})

V_616 = CTVertex(name = 'V_616',
                 type = 'UV',
                 particles = [ P.g, P.g, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVS12, L.VVS13, L.VVS15 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1918_351,(0,0,2):C.UVGC_1918_352,(0,2,1):C.UVGC_1755_110,(0,1,2):C.UVGC_1794_165})

V_617 = CTVertex(name = 'V_617',
                 type = 'UV',
                 particles = [ P.g, P.g, P.g ],
                 color = [ 'f(1,2,3)' ],
                 lorentz = [ L.VVV8 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1905_316,(0,0,1):C.UVGC_1905_317,(0,0,2):[ C.UVGC_1905_318, C.UVGC_1681_36 ]})

V_618 = CTVertex(name = 'V_618',
                 type = 'UV',
                 particles = [ P.g, P.g, P.g, P.g ],
                 color = [ 'f(-1,1,2)*f(-1,3,4)', 'f(-1,1,3)*f(-1,2,4)', 'f(-1,1,4)*f(-1,2,3)' ],
                 lorentz = [ L.VVVV14, L.VVVV15, L.VVVV17 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1909_324,(0,0,1):C.UVGC_1909_325,(0,0,2):[ C.UVGC_1909_326, C.UVGC_1683_38 ],(1,0,0):C.UVGC_1909_324,(1,0,1):C.UVGC_1909_325,(1,0,2):[ C.UVGC_1909_326, C.UVGC_1683_38 ],(0,1,0):C.UVGC_1907_321,(0,1,1):C.UVGC_1907_322,(0,1,2):[ C.UVGC_1907_323, C.UVGC_1682_37 ],(2,1,0):C.UVGC_1909_324,(2,1,1):C.UVGC_1909_325,(2,1,2):[ C.UVGC_1909_326, C.UVGC_1683_38 ],(1,2,0):C.UVGC_1907_321,(1,2,1):C.UVGC_1907_322,(1,2,2):[ C.UVGC_1907_323, C.UVGC_1682_37 ],(2,2,0):C.UVGC_1907_321,(2,2,1):C.UVGC_1907_322,(2,2,2):[ C.UVGC_1907_323, C.UVGC_1682_37 ]})

V_619 = CTVertex(name = 'V_619',
                 type = 'UV',
                 particles = [ P.g, P.g, P.g, P.G0, P.G0 ],
                 color = [ 'f(1,2,3)' ],
                 lorentz = [ L.VVVSS25 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1892_289,(0,0,1):C.UVGC_1892_290,(0,0,2):C.UVGC_1892_291})

V_620 = CTVertex(name = 'V_620',
                 type = 'UV',
                 particles = [ P.g, P.g, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'f(1,2,3)' ],
                 lorentz = [ L.VVVSS25 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b, P.t] ], [ [P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1892_289,(0,0,2):C.UVGC_1892_290,(0,0,3):C.UVGC_2077_511,(0,0,1):C.UVGC_2077_512})

V_621 = CTVertex(name = 'V_621',
                 type = 'UV',
                 particles = [ P.g, P.g, P.g, P.H, P.H ],
                 color = [ 'f(1,2,3)' ],
                 lorentz = [ L.VVVSS25 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1892_289,(0,0,1):C.UVGC_1892_290,(0,0,2):C.UVGC_1892_291})

V_622 = CTVertex(name = 'V_622',
                 type = 'UV',
                 particles = [ P.g, P.g, P.g, P.H ],
                 color = [ 'f(1,2,3)' ],
                 lorentz = [ L.VVVS17 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1919_353,(0,0,1):C.UVGC_1919_354,(0,0,2):C.UVGC_1919_355})

V_623 = CTVertex(name = 'V_623',
                 type = 'UV',
                 particles = [ P.g, P.g, P.g, P.g, P.H ],
                 color = [ 'f(-1,1,2)*f(-1,3,4)', 'f(-1,1,3)*f(-1,2,4)', 'f(-1,1,4)*f(-1,2,3)' ],
                 lorentz = [ L.VVVVS12, L.VVVVS13, L.VVVVS15 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1923_362,(0,0,1):C.UVGC_1923_363,(0,0,2):C.UVGC_1923_364,(1,0,0):C.UVGC_1923_362,(1,0,1):C.UVGC_1923_363,(1,0,2):C.UVGC_1923_364,(0,1,0):C.UVGC_1921_359,(0,1,1):C.UVGC_1921_360,(0,1,2):C.UVGC_1921_361,(2,1,0):C.UVGC_1923_362,(2,1,1):C.UVGC_1923_363,(2,1,2):C.UVGC_1923_364,(1,2,0):C.UVGC_1921_359,(1,2,1):C.UVGC_1921_360,(1,2,2):C.UVGC_1921_361,(2,2,0):C.UVGC_1921_359,(2,2,1):C.UVGC_1921_360,(2,2,2):C.UVGC_1921_361})

V_624 = CTVertex(name = 'V_624',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS41, L.FFS42, L.FFS43, L.FFS45, L.FFS49, L.FFS57 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t], [P.b, P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,5):C.UVGC_2085_521,(0,0,2):C.UVGC_2085_522,(0,0,0):[ C.UVGC_2425_988, C.UVGC_2394_931 ],(0,0,3):[ C.UVGC_2425_989, C.UVGC_2394_932 ],(0,0,1):[ C.UVGC_2425_990, C.UVGC_2394_933 ],(0,1,0):C.UVGC_1931_374,(0,5,4):C.UVGC_2365_888,(0,4,3):C.UVGC_2132_567,(0,3,2):C.UVGC_2079_514,(0,3,0):C.UVGC_2418_969,(0,3,3):C.UVGC_2418_970,(0,3,1):C.UVGC_2418_971,(0,2,2):C.UVGC_2063_499})

V_625 = CTVertex(name = 'V_625',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS31, L.FFS32, L.FFS34, L.FFS35, L.FFS45, L.FFS55 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,4,1):C.UVGC_1834_209,(0,4,0):C.UVGC_1717_72,(0,5,1):C.UVGC_1840_215,(0,5,0):C.UVGC_1718_73,(0,0,1):C.UVGC_1679_34,(0,0,0):[ C.UVGC_2190_660, C.UVGC_1715_70 ],(0,1,1):C.UVGC_1832_207,(0,2,0):C.UVGC_2187_657,(0,2,1):C.UVGC_1835_210,(0,3,0):C.UVGC_2134_569})

V_626 = CTVertex(name = 'V_626',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS33, L.FFS37, L.FFS38, L.FFS39, L.FFS58 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,1,1):C.UVGC_1842_217,(0,1,0):[ C.UVGC_2258_749, C.UVGC_1716_71 ],(0,0,0):C.UVGC_2186_656,(0,2,1):C.UVGC_1831_206,(0,4,0):C.UVGC_1649_4,(0,3,0):C.UVGC_1729_84})

V_627 = CTVertex(name = 'V_627',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS27, L.FFSS30, L.FFSS39 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t], [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2024_460,(0,0,0):C.UVGC_2419_972,(0,0,3):C.UVGC_2419_973,(0,0,1):C.UVGC_2419_974,(0,2,4):C.UVGC_2397_936,(0,1,2):C.UVGC_1997_433,(0,1,0):C.UVGC_2343_842,(0,1,3):C.UVGC_2343_843,(0,1,1):C.UVGC_2343_844})

V_628 = CTVertex(name = 'V_628',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS24, L.FFSS26, L.FFSS37 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,1,1):C.UVGC_1821_196,(0,1,0):C.UVGC_2227_705,(0,2,0):C.UVGC_2221_699,(0,0,0):C.UVGC_2217_695})

V_629 = CTVertex(name = 'V_629',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS25, L.FFSS26, L.FFSS30, L.FFSS37, L.FFSS42, L.FFSS44 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_1993_429,(0,2,2):C.UVGC_1686_41,(0,4,1):C.UVGC_2005_441,(0,4,2):C.UVGC_1688_43,(0,1,1):C.UVGC_2022_458,(0,1,2):C.UVGC_2227_705,(0,3,1):C.UVGC_2021_457,(0,3,0):C.UVGC_2221_699,(0,0,1):C.UVGC_2001_437,(0,5,0):C.UVGC_2217_695})

V_630 = CTVertex(name = 'V_630',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.G0, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS23, L.FFSS25, L.FFSS30, L.FFSS35, L.FFSS42 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_1811_186,(0,2,0):C.UVGC_1687_42,(0,4,1):C.UVGC_1817_192,(0,4,0):C.UVGC_1689_44,(0,0,1):C.UVGC_1820_195,(0,0,0):C.UVGC_2229_707,(0,3,1):C.UVGC_1819_194,(0,3,0):C.UVGC_2223_701,(0,1,1):C.UVGC_1812_187,(0,1,0):C.UVGC_2218_696})

V_631 = CTVertex(name = 'V_631',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS24, L.FFSS26, L.FFSS37 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,1,1):C.UVGC_1822_197,(0,1,0):C.UVGC_2228_706,(0,2,0):C.UVGC_2222_700,(0,0,0):C.UVGC_2217_695})

V_632 = CTVertex(name = 'V_632',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45, L.FFS52, L.FFS53, L.FFS54, L.FFS56, L.FFS59 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t], [P.b, P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,1,5):C.UVGC_2084_519,(0,1,2):C.UVGC_2084_520,(0,1,0):[ C.UVGC_2424_985, C.UVGC_2393_928 ],(0,1,3):[ C.UVGC_2424_986, C.UVGC_2393_929 ],(0,1,1):[ C.UVGC_2424_987, C.UVGC_2393_930 ],(0,2,3):C.UVGC_2133_568,(0,4,4):C.UVGC_2364_887,(0,5,0):C.UVGC_1932_375,(0,0,2):C.UVGC_2079_514,(0,0,0):C.UVGC_2418_969,(0,0,3):C.UVGC_2418_970,(0,0,1):C.UVGC_2418_971,(0,3,2):C.UVGC_2064_500})

V_633 = CTVertex(name = 'V_633',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30, L.FFSS36, L.FFSS40 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t], [P.b, P.g, P.t] ] ],
                 couplings = {(0,1,2):C.UVGC_2023_459,(0,1,0):C.UVGC_2420_975,(0,1,3):C.UVGC_2420_976,(0,1,1):C.UVGC_2420_977,(0,2,4):C.UVGC_2395_934,(0,0,2):C.UVGC_1997_433,(0,0,0):C.UVGC_2343_842,(0,0,3):C.UVGC_2343_843,(0,0,1):C.UVGC_2343_844})

V_634 = CTVertex(name = 'V_634',
                 type = 'UV',
                 particles = [ P.vt__tilde__, P.ta__minus__, P.b__tilde__, P.t ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFFF18 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2350_859,(0,0,2):C.UVGC_2350_860,(0,0,1):C.UVGC_2350_861})

V_635 = CTVertex(name = 'V_635',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.ta__plus__, P.vt ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFFF18 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2350_859,(0,0,2):C.UVGC_2350_860,(0,0,1):C.UVGC_2350_861})

V_636 = CTVertex(name = 'V_636',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.u__tilde__, P.u ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2327_819,(0,0,2):C.UVGC_2327_818,(0,0,1):C.UVGC_2327_820,(0,2,0):C.UVGC_2329_824,(0,2,2):C.UVGC_2307_778,(0,2,1):C.UVGC_2307_779,(0,3,0):C.UVGC_2333_833,(0,3,2):C.UVGC_2333_832,(0,3,1):C.UVGC_2333_834,(0,1,0):C.UVGC_2331_827,(0,1,2):C.UVGC_2331_826,(0,1,1):C.UVGC_2331_828,(1,0,0):C.UVGC_2328_822,(1,0,2):C.UVGC_2328_821,(1,0,1):C.UVGC_2328_823,(1,2,0):C.UVGC_2330_825,(1,2,2):C.UVGC_2308_780,(1,2,1):C.UVGC_2308_781,(1,3,0):C.UVGC_2334_836,(1,3,2):C.UVGC_2334_835,(1,3,1):C.UVGC_2334_837,(1,1,0):C.UVGC_2332_830,(1,1,2):C.UVGC_2332_829,(1,1,1):C.UVGC_2332_831})

V_637 = CTVertex(name = 'V_637',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.u__tilde__, P.u ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF18, L.FFFF28 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2305_774,(0,0,1):C.UVGC_2305_775,(0,1,0):C.UVGC_2307_778,(0,1,1):C.UVGC_2307_779,(1,0,0):C.UVGC_2306_776,(1,0,1):C.UVGC_2306_777,(1,1,0):C.UVGC_2308_780,(1,1,1):C.UVGC_2308_781})

V_638 = CTVertex(name = 'V_638',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.d__tilde__, P.u ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF18 ],
                 loop_particles = [ [ [P.b, P.d, P.g], [P.g, P.t, P.u] ], [ [P.b, P.g], [P.d, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t], [P.d, P.g, P.u] ], [ [P.b, P.g, P.u], [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_2447_1017,(0,0,4):C.UVGC_2447_1018,(0,0,0):C.UVGC_2447_1020,(0,0,2):C.UVGC_2447_1021,(0,0,3):C.UVGC_2447_1019,(1,0,1):C.UVGC_2445_1012,(1,0,4):C.UVGC_2445_1013,(1,0,0):C.UVGC_2445_1015,(1,0,2):C.UVGC_2445_1016,(1,0,3):C.UVGC_2445_1014})

V_639 = CTVertex(name = 'V_639',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.t__tilde__, P.t ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2327_818,(0,0,2):C.UVGC_2327_819,(0,0,1):C.UVGC_2327_820,(0,1,0):C.UVGC_2307_778,(0,1,2):C.UVGC_2329_824,(0,1,1):C.UVGC_2307_779,(0,2,0):C.UVGC_2331_826,(0,2,2):C.UVGC_2331_827,(0,2,1):C.UVGC_2331_828,(0,3,0):C.UVGC_2333_832,(0,3,2):C.UVGC_2333_833,(0,3,1):C.UVGC_2333_834,(1,0,0):C.UVGC_2328_821,(1,0,2):C.UVGC_2328_822,(1,0,1):C.UVGC_2328_823,(1,1,0):C.UVGC_2308_780,(1,1,2):C.UVGC_2330_825,(1,1,1):C.UVGC_2308_781,(1,2,0):C.UVGC_2332_829,(1,2,2):C.UVGC_2332_830,(1,2,1):C.UVGC_2332_831,(1,3,0):C.UVGC_2334_835,(1,3,2):C.UVGC_2334_836,(1,3,1):C.UVGC_2334_837})

V_640 = CTVertex(name = 'V_640',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.c__tilde__, P.c ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF18, L.FFFF28 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g], [P.c, P.g] ] ],
                 couplings = {(0,0,1):C.UVGC_2305_774,(0,0,0):C.UVGC_2305_775,(0,1,1):C.UVGC_2307_778,(0,1,0):C.UVGC_2307_779,(1,0,1):C.UVGC_2306_776,(1,0,0):C.UVGC_2306_777,(1,1,1):C.UVGC_2308_780,(1,1,0):C.UVGC_2308_781})

V_641 = CTVertex(name = 'V_641',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.s__tilde__, P.c ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF18 ],
                 loop_particles = [ [ [P.b, P.c, P.g], [P.g, P.s, P.t] ], [ [P.b, P.g], [P.c, P.g], [P.g, P.s] ], [ [P.b, P.g, P.s], [P.c, P.g, P.t] ], [ [P.b, P.g, P.t], [P.c, P.g, P.s] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_2447_1017,(0,0,4):C.UVGC_2447_1018,(0,0,0):C.UVGC_2447_1019,(0,0,2):C.UVGC_2447_1020,(0,0,3):C.UVGC_2447_1021,(1,0,1):C.UVGC_2445_1012,(1,0,4):C.UVGC_2445_1013,(1,0,0):C.UVGC_2445_1014,(1,0,2):C.UVGC_2445_1015,(1,0,3):C.UVGC_2445_1016})

V_642 = CTVertex(name = 'V_642',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.b__tilde__, P.t ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF18 ],
                 loop_particles = [ [ [P.b, P.d, P.g], [P.g, P.t, P.u] ], [ [P.b, P.g], [P.d, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t], [P.d, P.g, P.u] ], [ [P.b, P.g, P.u], [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_2447_1017,(0,0,4):C.UVGC_2447_1018,(0,0,0):C.UVGC_2447_1020,(0,0,2):C.UVGC_2447_1021,(0,0,3):C.UVGC_2447_1019,(1,0,1):C.UVGC_2445_1012,(1,0,4):C.UVGC_2445_1013,(1,0,0):C.UVGC_2445_1015,(1,0,2):C.UVGC_2445_1016,(1,0,3):C.UVGC_2445_1014})

V_643 = CTVertex(name = 'V_643',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.t__tilde__, P.t ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2305_774,(0,0,2):C.UVGC_2437_1010,(0,0,1):C.UVGC_2305_775,(0,1,0):C.UVGC_2335_838,(0,1,2):C.UVGC_2346_851,(0,1,1):C.UVGC_2335_839,(0,2,0):C.UVGC_2331_826,(0,2,2):C.UVGC_2331_827,(0,2,1):C.UVGC_2331_828,(0,3,0):C.UVGC_2357_876,(0,3,2):C.UVGC_2357_877,(0,3,1):C.UVGC_2357_878,(1,0,0):C.UVGC_2306_776,(1,0,2):C.UVGC_2438_1011,(1,0,1):C.UVGC_2306_777,(1,1,0):C.UVGC_2336_840,(1,1,2):C.UVGC_2347_852,(1,1,1):C.UVGC_2336_841,(1,2,0):C.UVGC_2332_829,(1,2,2):C.UVGC_2332_830,(1,2,1):C.UVGC_2332_831,(1,3,0):C.UVGC_2358_879,(1,3,2):C.UVGC_2358_880,(1,3,1):C.UVGC_2358_881})

V_644 = CTVertex(name = 'V_644',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.d__tilde__, P.d ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF18, L.FFFF28 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.g], [P.d, P.g] ] ],
                 couplings = {(0,0,1):C.UVGC_2327_818,(0,0,0):C.UVGC_2327_820,(0,1,1):C.UVGC_2335_838,(0,1,0):C.UVGC_2335_839,(1,0,1):C.UVGC_2328_821,(1,0,0):C.UVGC_2328_823,(1,1,1):C.UVGC_2336_840,(1,1,0):C.UVGC_2336_841})

V_645 = CTVertex(name = 'V_645',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.b__tilde__, P.t ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF18 ],
                 loop_particles = [ [ [P.b, P.c, P.g], [P.g, P.s, P.t] ], [ [P.b, P.g], [P.c, P.g], [P.g, P.s] ], [ [P.b, P.g, P.s], [P.c, P.g, P.t] ], [ [P.b, P.g, P.t], [P.c, P.g, P.s] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_2447_1017,(0,0,4):C.UVGC_2447_1018,(0,0,0):C.UVGC_2447_1019,(0,0,2):C.UVGC_2447_1020,(0,0,3):C.UVGC_2447_1021,(1,0,1):C.UVGC_2445_1012,(1,0,4):C.UVGC_2445_1013,(1,0,0):C.UVGC_2445_1014,(1,0,2):C.UVGC_2445_1015,(1,0,3):C.UVGC_2445_1016})

V_646 = CTVertex(name = 'V_646',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.t__tilde__, P.t ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2305_774,(0,0,2):C.UVGC_2437_1010,(0,0,1):C.UVGC_2305_775,(0,1,0):C.UVGC_2335_838,(0,1,2):C.UVGC_2346_851,(0,1,1):C.UVGC_2335_839,(0,2,0):C.UVGC_2331_826,(0,2,2):C.UVGC_2331_827,(0,2,1):C.UVGC_2331_828,(0,3,0):C.UVGC_2357_876,(0,3,2):C.UVGC_2357_877,(0,3,1):C.UVGC_2357_878,(1,0,0):C.UVGC_2306_776,(1,0,2):C.UVGC_2438_1011,(1,0,1):C.UVGC_2306_777,(1,1,0):C.UVGC_2336_840,(1,1,2):C.UVGC_2347_852,(1,1,1):C.UVGC_2336_841,(1,2,0):C.UVGC_2332_829,(1,2,2):C.UVGC_2332_830,(1,2,1):C.UVGC_2332_831,(1,3,0):C.UVGC_2358_879,(1,3,2):C.UVGC_2358_880,(1,3,1):C.UVGC_2358_881})

V_647 = CTVertex(name = 'V_647',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.s__tilde__, P.s ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF18, L.FFFF28 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.s] ], [ [P.b, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2327_818,(0,0,1):C.UVGC_2327_820,(0,1,0):C.UVGC_2335_838,(0,1,1):C.UVGC_2335_839,(1,0,0):C.UVGC_2328_821,(1,0,1):C.UVGC_2328_823,(1,1,0):C.UVGC_2336_840,(1,1,1):C.UVGC_2336_841})

V_648 = CTVertex(name = 'V_648',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.t__tilde__, P.t ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF17, L.FFFF18, L.FFFF24, L.FFFF25, L.FFFF26, L.FFFF27, L.FFFF28, L.FFFF29 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(1,0,0):C.UVGC_2128_564,(0,1,0):C.UVGC_2128_564,(0,2,0):C.UVGC_2130_565,(1,5,0):C.UVGC_2130_565,(1,3,0):C.UVGC_2130_565,(1,4,0):C.UVGC_2135_570,(0,6,0):C.UVGC_2130_565,(0,7,0):C.UVGC_2135_570,(0,0,0):C.UVGC_1930_373,(1,1,0):C.UVGC_1930_373,(1,2,0):C.UVGC_2131_566,(0,5,0):C.UVGC_2131_566,(0,3,0):C.UVGC_2131_566,(0,4,0):C.UVGC_2136_571,(1,6,0):C.UVGC_2131_566,(1,7,0):C.UVGC_2136_571})

V_649 = CTVertex(name = 'V_649',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.b__tilde__, P.t ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF17, L.FFFF18, L.FFFF25, L.FFFF26, L.FFFF27 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g], [P.g, P.t] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(1,0,3):C.UVGC_2351_862,(1,0,2):C.UVGC_2351_863,(0,1,0):C.UVGC_2354_867,(0,1,3):C.UVGC_2354_868,(0,1,2):C.UVGC_2354_869,(1,4,0):C.UVGC_2335_838,(1,4,3):C.UVGC_2346_851,(1,4,2):C.UVGC_2335_839,(1,2,0):C.UVGC_2355_870,(1,2,3):C.UVGC_2355_871,(1,2,2):C.UVGC_2355_872,(1,3,0):C.UVGC_2357_876,(1,3,3):C.UVGC_2357_877,(1,3,2):C.UVGC_2357_878,(0,0,2):C.UVGC_2352_864,(1,1,1):C.UVGC_2353_865,(1,1,2):C.UVGC_2353_866,(0,4,0):C.UVGC_2336_840,(0,4,3):C.UVGC_2347_852,(0,4,2):C.UVGC_2336_841,(0,2,0):C.UVGC_2356_873,(0,2,3):C.UVGC_2356_874,(0,2,2):C.UVGC_2356_875,(0,3,0):C.UVGC_2358_879,(0,3,3):C.UVGC_2358_880,(0,3,2):C.UVGC_2358_881})

V_650 = CTVertex(name = 'V_650',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.b__tilde__, P.b ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF17, L.FFFF18, L.FFFF24, L.FFFF25, L.FFFF27, L.FFFF28 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(1,0,0):C.UVGC_1929_372,(0,1,0):C.UVGC_1929_372,(0,2,0):C.UVGC_1927_370,(1,4,0):C.UVGC_1927_370,(1,3,0):C.UVGC_1927_370,(0,5,0):C.UVGC_1927_370,(0,0,0):C.UVGC_1930_373,(1,1,0):C.UVGC_1930_373,(1,2,0):C.UVGC_1928_371,(0,4,0):C.UVGC_1928_371,(0,3,0):C.UVGC_1928_371,(1,5,0):C.UVGC_1928_371})

V_651 = CTVertex(name = 'V_651',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.ve__tilde__, P.ve ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFFF18, L.FFFF24 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1694_49,(0,1,0):C.UVGC_1702_57})

V_652 = CTVertex(name = 'V_652',
                 type = 'UV',
                 particles = [ P.ve__tilde__, P.e__minus__, P.b__tilde__, P.t ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFFF18 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2348_853,(0,0,2):C.UVGC_2348_854,(0,0,1):C.UVGC_2348_855})

V_653 = CTVertex(name = 'V_653',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.e__plus__, P.ve ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFFF18 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2348_853,(0,0,2):C.UVGC_2348_854,(0,0,1):C.UVGC_2348_855})

V_654 = CTVertex(name = 'V_654',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.vm__tilde__, P.vm ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFFF18, L.FFFF24 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1696_51,(0,1,0):C.UVGC_1703_58})

V_655 = CTVertex(name = 'V_655',
                 type = 'UV',
                 particles = [ P.vm__tilde__, P.mu__minus__, P.b__tilde__, P.t ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFFF18 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2349_856,(0,0,2):C.UVGC_2349_857,(0,0,1):C.UVGC_2349_858})

V_656 = CTVertex(name = 'V_656',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.mu__plus__, P.vm ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFFF18 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2349_856,(0,0,2):C.UVGC_2349_857,(0,0,1):C.UVGC_2349_858})

V_657 = CTVertex(name = 'V_657',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.vt__tilde__, P.vt ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFFF18, L.FFFF24 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1698_53,(0,1,0):C.UVGC_1704_59})

V_658 = CTVertex(name = 'V_658',
                 type = 'UV',
                 particles = [ P.e__plus__, P.e__minus__, P.t__tilde__, P.t ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1693_48,(0,1,0):C.UVGC_1690_45,(0,2,0):C.UVGC_1702_57,(0,3,0):C.UVGC_1699_54})

V_659 = CTVertex(name = 'V_659',
                 type = 'UV',
                 particles = [ P.mu__plus__, P.mu__minus__, P.t__tilde__, P.t ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1695_50,(0,1,0):C.UVGC_1691_46,(0,2,0):C.UVGC_1703_58,(0,3,0):C.UVGC_1700_55})

V_660 = CTVertex(name = 'V_660',
                 type = 'UV',
                 particles = [ P.ta__plus__, P.ta__minus__, P.t__tilde__, P.t ],
                 color = [ 'Identity(3,4)' ],
                 lorentz = [ L.FFFF18, L.FFFF24, L.FFFF28, L.FFFF29 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1697_52,(0,1,0):C.UVGC_1692_47,(0,2,0):C.UVGC_1704_59,(0,3,0):C.UVGC_1701_56})

V_661 = CTVertex(name = 'V_661',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV73, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1759_113,(0,0,1):C.UVGC_1759_114,(0,1,0):C.UVGC_1761_117,(0,1,1):C.UVGC_1761_118})

V_662 = CTVertex(name = 'V_662',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV73, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1759_113,(0,0,1):C.UVGC_1759_114,(0,1,0):C.UVGC_1761_117,(0,1,1):C.UVGC_1761_118})

V_663 = CTVertex(name = 'V_663',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV102, L.FFV108, L.FFV57, L.FFV59, L.FFV60, L.FFV62, L.FFV65, L.FFV73, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,2,3):[ C.UVGC_1685_40, C.UVGC_2247_738 ],(0,7,0):C.UVGC_1855_230,(0,7,1):C.UVGC_1855_231,(0,7,2):C.UVGC_1855_232,(0,7,4):C.UVGC_1855_233,(0,8,0):C.UVGC_1857_238,(0,8,1):C.UVGC_1857_239,(0,8,2):C.UVGC_1857_240,(0,8,4):C.UVGC_1857_241,(0,3,4):C.UVGC_1756_111,(0,0,3):C.UVGC_2251_742,(0,4,3):C.UVGC_2232_711,(0,1,3):C.UVGC_1675_30,(0,6,3):C.UVGC_2233_712,(0,5,3):C.UVGC_1676_31})

V_664 = CTVertex(name = 'V_664',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV73, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1760_115,(0,0,1):C.UVGC_1760_116,(0,1,0):C.UVGC_1748_103,(0,1,1):C.UVGC_1757_112})

V_665 = CTVertex(name = 'V_665',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV73, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1760_115,(0,0,1):C.UVGC_1760_116,(0,1,0):C.UVGC_1748_103,(0,1,1):C.UVGC_1757_112})

V_666 = CTVertex(name = 'V_666',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV59, L.FFV73, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1856_234,(0,1,1):C.UVGC_1856_235,(0,1,2):C.UVGC_1856_236,(0,1,3):C.UVGC_1856_237,(0,2,3):C.UVGC_1757_112,(0,0,0):C.UVGC_1748_103})

V_667 = CTVertex(name = 'V_667',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV57, L.FFV73, L.FFV87, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.g, P.u] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_1926_367,(0,0,2):C.UVGC_1926_368,(0,0,3):C.UVGC_1926_369,(0,1,0):C.UVGC_1765_121,(0,1,4):C.UVGC_1765_122,(0,3,0):C.UVGC_1767_125,(0,3,4):C.UVGC_1767_126,(0,2,1):C.UVGC_1906_319,(0,2,2):C.UVGC_1906_320})

V_668 = CTVertex(name = 'V_668',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV57, L.FFV73, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.c, P.g] ], [ [P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_1949_392,(0,0,3):C.UVGC_1949_393,(0,0,2):C.UVGC_1926_369,(0,1,0):C.UVGC_1765_121,(0,1,4):C.UVGC_1765_122,(0,2,0):C.UVGC_1767_125,(0,2,4):C.UVGC_1767_126})

V_669 = CTVertex(name = 'V_669',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV100, L.FFV101, L.FFV102, L.FFV108, L.FFV57, L.FFV59, L.FFV62, L.FFV63, L.FFV73, L.FFV87, L.FFV90, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.g] ], [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,9,5):C.UVGC_1719_74,(0,9,1):C.UVGC_1906_319,(0,9,4):C.UVGC_1906_320,(0,4,1):C.UVGC_1926_367,(0,4,4):C.UVGC_1926_368,(0,4,5):[ C.UVGC_2127_563, C.UVGC_1674_29 ],(0,8,0):C.UVGC_1858_242,(0,8,2):C.UVGC_1858_243,(0,8,3):C.UVGC_1858_244,(0,8,6):C.UVGC_1858_245,(0,11,0):C.UVGC_1860_249,(0,11,2):C.UVGC_1860_250,(0,11,3):C.UVGC_1860_251,(0,11,6):C.UVGC_1860_252,(0,5,6):C.UVGC_1762_119,(0,7,5):C.UVGC_1666_21,(0,2,1):C.UVGC_1920_356,(0,2,4):C.UVGC_1920_357,(0,2,5):C.UVGC_1920_358,(0,3,5):C.UVGC_1664_19,(0,6,5):C.UVGC_1665_20,(0,10,5):C.UVGC_1743_98,(0,1,5):C.UVGC_1741_96,(0,0,1):C.UVGC_1925_365,(0,0,4):C.UVGC_1925_366})

V_670 = CTVertex(name = 'V_670',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV57, L.FFV73, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.d, P.g] ], [ [P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_1949_392,(0,0,3):C.UVGC_1949_393,(0,0,2):C.UVGC_1926_369,(0,1,0):C.UVGC_1766_123,(0,1,4):C.UVGC_1766_124,(0,2,0):C.UVGC_1749_104,(0,2,4):C.UVGC_1763_120})

V_671 = CTVertex(name = 'V_671',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV57, L.FFV73, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.g, P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_1949_392,(0,0,2):C.UVGC_1949_393,(0,0,3):C.UVGC_1926_369,(0,1,0):C.UVGC_1766_123,(0,1,4):C.UVGC_1766_124,(0,2,0):C.UVGC_1749_104,(0,2,4):C.UVGC_1763_120})

V_672 = CTVertex(name = 'V_672',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV57, L.FFV59, L.FFV73, L.FFV87, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b, P.g] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_1926_367,(0,0,5):C.UVGC_1926_368,(0,0,2):C.UVGC_1926_369,(0,2,0):C.UVGC_1858_245,(0,2,3):C.UVGC_1859_246,(0,2,4):C.UVGC_1859_247,(0,2,6):C.UVGC_1859_248,(0,4,6):C.UVGC_1763_120,(0,1,0):C.UVGC_1749_104,(0,3,1):C.UVGC_1906_319,(0,3,5):C.UVGC_1906_320})

V_673 = CTVertex(name = 'V_673',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV70, L.FFV73 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1991_427,(0,0,1):[ C.UVGC_2315_794, C.UVGC_2326_816 ],(0,0,2):[ C.UVGC_2315_795, C.UVGC_2326_817 ],(0,1,0):C.UVGC_1978_414})

V_674 = CTVertex(name = 'V_674',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV70, L.FFV73 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_1991_427,(0,0,1):[ C.UVGC_2315_794, C.UVGC_2326_816 ],(0,0,2):[ C.UVGC_2315_795, C.UVGC_2326_817 ],(0,1,0):C.UVGC_1978_414})

V_675 = CTVertex(name = 'V_675',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV69, L.FFV70, L.FFV73, L.FFV76, L.FFV78 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,2):C.UVGC_1992_428,(0,1,0):[ C.UVGC_2315_794, C.UVGC_2434_1007 ],(0,1,4):[ C.UVGC_2375_906, C.UVGC_2434_1008 ],(0,1,1):[ C.UVGC_2315_795, C.UVGC_2434_1009 ],(0,4,1):C.UVGC_2428_997,(0,3,0):C.UVGC_2230_708,(0,3,4):C.UVGC_2230_709,(0,0,2):C.UVGC_1989_425,(0,2,2):C.UVGC_2296_770,(0,2,3):C.UVGC_1978_414})

V_676 = CTVertex(name = 'V_676',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV70, L.FFV73 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1991_427,(0,0,1):[ C.UVGC_2315_794, C.UVGC_2326_816 ],(0,0,2):[ C.UVGC_2315_795, C.UVGC_2326_817 ],(0,1,0):C.UVGC_1978_414})

V_677 = CTVertex(name = 'V_677',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV70, L.FFV73 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_1991_427,(0,0,1):[ C.UVGC_2315_794, C.UVGC_2326_816 ],(0,0,2):[ C.UVGC_2315_795, C.UVGC_2326_817 ],(0,1,0):C.UVGC_1978_414})

V_678 = CTVertex(name = 'V_678',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV105, L.FFV70, L.FFV73, L.FFV81, L.FFV99 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,2):C.UVGC_1992_428,(0,1,0):[ C.UVGC_2315_794, C.UVGC_2434_1007 ],(0,1,4):[ C.UVGC_2375_906, C.UVGC_2434_1008 ],(0,1,1):[ C.UVGC_2315_795, C.UVGC_2434_1009 ],(0,0,1):C.UVGC_2428_997,(0,4,0):C.UVGC_2230_708,(0,4,4):C.UVGC_2230_709,(0,3,2):C.UVGC_1988_424,(0,2,2):C.UVGC_2296_770,(0,2,3):C.UVGC_1978_414})

V_679 = CTVertex(name = 'V_679',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV70, L.FFV73, L.FFV82, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_1801_172,(0,2,1):C.UVGC_1804_175,(0,1,0):C.UVGC_1808_180,(0,1,1):C.UVGC_1808_181,(0,3,0):C.UVGC_1810_184,(0,3,1):C.UVGC_1810_185})

V_680 = CTVertex(name = 'V_680',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV70, L.FFV73, L.FFV82, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_1801_172,(0,2,1):C.UVGC_1804_175,(0,1,0):C.UVGC_1808_180,(0,1,1):C.UVGC_1808_181,(0,3,0):C.UVGC_1810_184,(0,3,1):C.UVGC_1810_185})

V_681 = CTVertex(name = 'V_681',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV102, L.FFV108, L.FFV56, L.FFV57, L.FFV60, L.FFV62, L.FFV65, L.FFV66, L.FFV70, L.FFV73, L.FFV82, L.FFV85, L.FFV87, L.FFV88, L.FFV95, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,8,4):C.UVGC_1797_168,(0,8,3):[ C.UVGC_1707_62, C.UVGC_1726_81 ],(0,12,3):[ C.UVGC_1713_68, C.UVGC_2256_747 ],(0,10,4):C.UVGC_1803_174,(0,10,3):C.UVGC_1727_82,(0,3,3):C.UVGC_1724_79,(0,13,3):C.UVGC_1725_80,(0,9,0):C.UVGC_1915_339,(0,9,1):C.UVGC_1915_340,(0,9,2):C.UVGC_1915_341,(0,9,4):C.UVGC_1915_342,(0,2,4):C.UVGC_1796_167,(0,11,4):C.UVGC_1798_169,(0,15,0):C.UVGC_1917_347,(0,15,1):C.UVGC_1917_348,(0,15,2):C.UVGC_1917_349,(0,15,4):C.UVGC_1917_350,(0,14,4):C.UVGC_1805_176,(0,7,3):C.UVGC_2249_740,(0,0,3):C.UVGC_2231_710,(0,1,3):C.UVGC_1662_17,(0,4,3):C.UVGC_2252_743,(0,6,3):C.UVGC_2253_744,(0,5,3):C.UVGC_1663_18})

V_682 = CTVertex(name = 'V_682',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV70, L.FFV73, L.FFV82, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_1802_173,(0,2,1):C.UVGC_1800_171,(0,1,0):C.UVGC_1809_182,(0,1,1):C.UVGC_1809_183,(0,3,0):C.UVGC_1807_179,(0,3,1):C.UVGC_1806_178})

V_683 = CTVertex(name = 'V_683',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV70, L.FFV73, L.FFV82, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_1802_173,(0,2,1):C.UVGC_1800_171,(0,1,0):C.UVGC_1809_182,(0,1,1):C.UVGC_1809_183,(0,3,0):C.UVGC_1807_179,(0,3,1):C.UVGC_1806_178})

V_684 = CTVertex(name = 'V_684',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV70, L.FFV73, L.FFV82, L.FFV95, L.FFV96 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(0,0,3):C.UVGC_1799_170,(0,2,3):C.UVGC_1800_171,(0,1,0):C.UVGC_1916_343,(0,1,1):C.UVGC_1916_344,(0,1,2):C.UVGC_1916_345,(0,1,3):C.UVGC_1916_346,(0,4,0):C.UVGC_1806_177,(0,4,3):C.UVGC_1806_178,(0,3,0):C.UVGC_1752_107})

V_685 = CTVertex(name = 'V_685',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.a, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2018_454,(0,1,0):C.UVGC_2020_456})

V_686 = CTVertex(name = 'V_686',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.G0, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30, L.FFSS42 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1815_190,(0,1,0):C.UVGC_1818_193})

V_687 = CTVertex(name = 'V_687',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45, L.FFS55 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1838_213,(0,1,0):C.UVGC_1841_216})

V_688 = CTVertex(name = 'V_688',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30, L.FFSS42 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2003_439,(0,1,0):C.UVGC_2006_442})

V_689 = CTVertex(name = 'V_689',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS12, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,1):C.UVGC_2007_443,(0,1,2):C.UVGC_1705_60,(0,2,1):C.UVGC_2019_455,(0,2,2):C.UVGC_2403_946,(0,2,0):C.UVGC_2403_947,(0,0,1):C.UVGC_2014_450})

V_690 = CTVertex(name = 'V_690',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.a, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2018_454,(0,1,0):C.UVGC_2020_456})

V_691 = CTVertex(name = 'V_691',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.G0, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30, L.FFSS42 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1815_190,(0,1,0):C.UVGC_1818_193})

V_692 = CTVertex(name = 'V_692',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45, L.FFS55 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1838_213,(0,1,0):C.UVGC_1841_216})

V_693 = CTVertex(name = 'V_693',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30, L.FFSS42 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2003_439,(0,1,0):C.UVGC_2006_442})

V_694 = CTVertex(name = 'V_694',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.W__minus__, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2040_476,(0,1,0):C.UVGC_2046_482})

V_695 = CTVertex(name = 'V_695',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.W__minus__, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2039_475,(0,1,0):C.UVGC_2045_481})

V_696 = CTVertex(name = 'V_696',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2109_546,(0,0,0):C.UVGC_2113_550})

V_697 = CTVertex(name = 'V_697',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS12, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,1):C.UVGC_2027_463,(0,1,2):C.UVGC_1709_64,(0,2,1):C.UVGC_2043_479,(0,2,2):C.UVGC_2411_954,(0,2,0):C.UVGC_2411_955,(0,0,1):C.UVGC_2033_469})

V_698 = CTVertex(name = 'V_698',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS12, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,1):C.UVGC_2028_464,(0,1,2):C.UVGC_1708_63,(0,2,1):C.UVGC_2042_478,(0,2,2):C.UVGC_2410_952,(0,2,0):C.UVGC_2410_953,(0,0,1):C.UVGC_2034_470})

V_699 = CTVertex(name = 'V_699',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS119, L.FFVS120, L.FFVS124, L.FFVS128, L.FFVS136, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,5,1):C.UVGC_2101_538,(0,5,2):C.UVGC_1720_75,(0,1,1):C.UVGC_2111_548,(0,1,2):C.UVGC_2391_924,(0,1,0):C.UVGC_2391_925,(0,2,1):C.UVGC_2103_540,(0,0,1):C.UVGC_2090_527,(0,4,0):C.UVGC_2380_913,(0,3,2):C.UVGC_2140_578})

V_700 = CTVertex(name = 'V_700',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.W__minus__, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2040_476,(0,1,0):C.UVGC_2046_482})

V_701 = CTVertex(name = 'V_701',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.W__minus__, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2039_475,(0,1,0):C.UVGC_2045_481})

V_702 = CTVertex(name = 'V_702',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2109_546,(0,0,0):C.UVGC_2113_550})

V_703 = CTVertex(name = 'V_703',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.W__plus__, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2040_476,(0,1,0):C.UVGC_2046_482})

V_704 = CTVertex(name = 'V_704',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.W__plus__, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2041_477,(0,1,0):C.UVGC_2047_483})

V_705 = CTVertex(name = 'V_705',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2110_547,(0,0,0):C.UVGC_2114_551})

V_706 = CTVertex(name = 'V_706',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.W__plus__, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS12, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,1):C.UVGC_2027_463,(0,1,2):C.UVGC_1709_64,(0,2,1):C.UVGC_2043_479,(0,2,2):C.UVGC_2411_954,(0,2,0):C.UVGC_2411_955,(0,0,1):C.UVGC_2033_469})

V_707 = CTVertex(name = 'V_707',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.W__plus__, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS12, L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,1):C.UVGC_2026_462,(0,1,2):C.UVGC_1710_65,(0,2,1):C.UVGC_2044_480,(0,2,2):C.UVGC_2412_956,(0,2,0):C.UVGC_2412_957,(0,0,1):C.UVGC_2030_466})

V_708 = CTVertex(name = 'V_708',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS124, L.FFVS93, L.FFVS94, L.FFVS95, L.FFVS97 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,3,1):C.UVGC_2100_537,(0,3,2):C.UVGC_1721_76,(0,0,1):C.UVGC_2112_549,(0,0,2):C.UVGC_2392_926,(0,0,0):C.UVGC_2392_927,(0,1,1):C.UVGC_2105_542,(0,2,1):C.UVGC_2090_527,(0,5,0):C.UVGC_2381_914,(0,4,2):C.UVGC_2141_579})

V_709 = CTVertex(name = 'V_709',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.W__plus__, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2040_476,(0,1,0):C.UVGC_2046_482})

V_710 = CTVertex(name = 'V_710',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.W__plus__, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2041_477,(0,1,0):C.UVGC_2047_483})

V_711 = CTVertex(name = 'V_711',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2110_547,(0,0,0):C.UVGC_2114_551})

V_712 = CTVertex(name = 'V_712',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.Z, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1827_202,(0,1,0):C.UVGC_1830_205})

V_713 = CTVertex(name = 'V_713',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.Z, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2060_496,(0,1,0):C.UVGC_2062_498})

V_714 = CTVertex(name = 'V_714',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.Z, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1827_202,(0,1,0):C.UVGC_1830_205})

V_715 = CTVertex(name = 'V_715',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1849_224,(0,0,0):C.UVGC_1852_227})

V_716 = CTVertex(name = 'V_716',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.Z, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS12, L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,1,1):C.UVGC_1823_198,(0,1,0):C.UVGC_1712_67,(0,2,1):C.UVGC_1829_204,(0,2,0):C.UVGC_2224_702,(0,3,0):C.UVGC_2225_703,(0,0,1):C.UVGC_1824_199})

V_717 = CTVertex(name = 'V_717',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.Z, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS12, L.FFVSS13, L.FFVSS14, L.FFVSS18 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,1):C.UVGC_2049_485,(0,1,2):C.UVGC_1714_69,(0,2,1):C.UVGC_2061_497,(0,2,2):C.UVGC_2416_967,(0,2,0):C.UVGC_2416_968,(0,0,1):C.UVGC_2048_484,(0,3,1):C.UVGC_2056_492})

V_718 = CTVertex(name = 'V_718',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.Z, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS12, L.FFVSS13, L.FFVSS14, L.FFVSS20 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,1,1):C.UVGC_1823_198,(0,1,0):C.UVGC_1712_67,(0,2,1):C.UVGC_1829_204,(0,2,0):C.UVGC_2224_702,(0,3,0):C.UVGC_2225_703,(0,0,1):C.UVGC_1824_199})

V_719 = CTVertex(name = 'V_719',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS113, L.FFVS116, L.FFVS118, L.FFVS120, L.FFVS124, L.FFVS125, L.FFVS130, L.FFVS131, L.FFVS133, L.FFVS137, L.FFVS145, L.FFVS85, L.FFVS89, L.FFVS94 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,13,1):C.UVGC_1845_220,(0,13,0):C.UVGC_1723_78,(0,3,1):C.UVGC_1851_226,(0,3,0):C.UVGC_2196_668,(0,5,0):C.UVGC_2212_688,(0,2,1):C.UVGC_1671_26,(0,4,1):C.UVGC_1846_221,(0,6,0):C.UVGC_2142_580,(0,9,0):C.UVGC_1658_13,(0,7,0):C.UVGC_2203_677,(0,10,0):C.UVGC_1647_2,(0,12,0):C.UVGC_1648_3,(0,1,0):C.UVGC_1661_16,(0,11,1):C.UVGC_1843_218,(0,8,0):C.UVGC_1737_92,(0,0,0):C.UVGC_1740_95})

V_720 = CTVertex(name = 'V_720',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.Z, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1827_202,(0,1,0):C.UVGC_1830_205})

V_721 = CTVertex(name = 'V_721',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.Z, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2060_496,(0,1,0):C.UVGC_2062_498})

V_722 = CTVertex(name = 'V_722',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.Z, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1827_202,(0,1,0):C.UVGC_1830_205})

V_723 = CTVertex(name = 'V_723',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1849_224,(0,0,0):C.UVGC_1852_227})

V_724 = CTVertex(name = 'V_724',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.a, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_2015_451,(0,0,0):C.UVGC_2219_697,(0,1,1):C.UVGC_2016_452})

V_725 = CTVertex(name = 'V_725',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.G0, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30, L.FFSS42 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1813_188,(0,1,0):C.UVGC_1814_189})

V_726 = CTVertex(name = 'V_726',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45, L.FFS55 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1836_211,(0,1,0):C.UVGC_1837_212})

V_727 = CTVertex(name = 'V_727',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30, L.FFSS39, L.FFSS42 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_2000_436,(0,2,1):C.UVGC_2002_438,(0,1,0):C.UVGC_1833_208})

V_728 = CTVertex(name = 'V_728',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.a, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2017_453,(0,1,0):C.UVGC_2016_452})

V_729 = CTVertex(name = 'V_729',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.G0, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30, L.FFSS42 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1816_191,(0,1,0):C.UVGC_1814_189})

V_730 = CTVertex(name = 'V_730',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45, L.FFS55 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1839_214,(0,1,0):C.UVGC_1837_212})

V_731 = CTVertex(name = 'V_731',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30, L.FFSS42 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2004_440,(0,1,0):C.UVGC_2002_438})

V_732 = CTVertex(name = 'V_732',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.a, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2017_453,(0,1,0):C.UVGC_2016_452})

V_733 = CTVertex(name = 'V_733',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.G0, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30, L.FFSS42 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1816_191,(0,1,0):C.UVGC_1814_189})

V_734 = CTVertex(name = 'V_734',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45, L.FFS55 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1839_214,(0,1,0):C.UVGC_1837_212})

V_735 = CTVertex(name = 'V_735',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30, L.FFSS42 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2004_440,(0,1,0):C.UVGC_2002_438})

V_736 = CTVertex(name = 'V_736',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.W__minus__, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2032_468,(0,1,0):C.UVGC_2037_473})

V_737 = CTVertex(name = 'V_737',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.W__minus__, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2031_467,(0,1,0):C.UVGC_2036_472})

V_738 = CTVertex(name = 'V_738',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2104_541,(0,0,0):C.UVGC_2107_544})

V_739 = CTVertex(name = 'V_739',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.W__minus__, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2040_476,(0,1,0):C.UVGC_2037_473})

V_740 = CTVertex(name = 'V_740',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.W__minus__, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2039_475,(0,1,0):C.UVGC_2036_472})

V_741 = CTVertex(name = 'V_741',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2109_546,(0,0,0):C.UVGC_2107_544})

V_742 = CTVertex(name = 'V_742',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.W__minus__, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2040_476,(0,1,0):C.UVGC_2037_473})

V_743 = CTVertex(name = 'V_743',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.W__minus__, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2039_475,(0,1,0):C.UVGC_2036_472})

V_744 = CTVertex(name = 'V_744',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2109_546,(0,0,0):C.UVGC_2107_544})

V_745 = CTVertex(name = 'V_745',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.W__plus__, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2032_468,(0,1,0):C.UVGC_2037_473})

V_746 = CTVertex(name = 'V_746',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.W__plus__, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2035_471,(0,1,0):C.UVGC_2038_474})

V_747 = CTVertex(name = 'V_747',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2106_543,(0,0,0):C.UVGC_2108_545})

V_748 = CTVertex(name = 'V_748',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.W__plus__, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2040_476,(0,1,0):C.UVGC_2037_473})

V_749 = CTVertex(name = 'V_749',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.W__plus__, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2041_477,(0,1,0):C.UVGC_2038_474})

V_750 = CTVertex(name = 'V_750',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2110_547,(0,0,0):C.UVGC_2108_545})

V_751 = CTVertex(name = 'V_751',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.W__plus__, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2040_476,(0,1,0):C.UVGC_2037_473})

V_752 = CTVertex(name = 'V_752',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.W__plus__, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2041_477,(0,1,0):C.UVGC_2038_474})

V_753 = CTVertex(name = 'V_753',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2110_547,(0,0,0):C.UVGC_2108_545})

V_754 = CTVertex(name = 'V_754',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.Z, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1825_200,(0,1,0):C.UVGC_1826_201})

V_755 = CTVertex(name = 'V_755',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.Z, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_2057_493,(0,0,0):C.UVGC_2225_703,(0,1,1):C.UVGC_2058_494})

V_756 = CTVertex(name = 'V_756',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.Z, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1825_200,(0,1,0):C.UVGC_1826_201})

V_757 = CTVertex(name = 'V_757',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1847_222,(0,0,0):C.UVGC_1848_223})

V_758 = CTVertex(name = 'V_758',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.Z, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1828_203,(0,1,0):C.UVGC_1826_201})

V_759 = CTVertex(name = 'V_759',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.Z, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2059_495,(0,1,0):C.UVGC_2058_494})

V_760 = CTVertex(name = 'V_760',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.Z, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1828_203,(0,1,0):C.UVGC_1826_201})

V_761 = CTVertex(name = 'V_761',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1850_225,(0,0,0):C.UVGC_1848_223})

V_762 = CTVertex(name = 'V_762',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.Z, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1828_203,(0,1,0):C.UVGC_1826_201})

V_763 = CTVertex(name = 'V_763',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.Z, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2059_495,(0,1,0):C.UVGC_2058_494})

V_764 = CTVertex(name = 'V_764',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.Z, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13, L.FFVSS14 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1828_203,(0,1,0):C.UVGC_1826_201})

V_765 = CTVertex(name = 'V_765',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS120, L.FFVS94 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1850_225,(0,0,0):C.UVGC_1848_223})

V_766 = CTVertex(name = 'V_766',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.a, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2009_445,(0,0,1):C.UVGC_2313_790,(0,0,2):C.UVGC_2313_791})

V_767 = CTVertex(name = 'V_767',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.a, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2010_446,(0,0,1):C.UVGC_2314_792,(0,0,2):C.UVGC_2314_793})

V_768 = CTVertex(name = 'V_768',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2081_516,(0,0,1):C.UVGC_2322_808,(0,0,2):C.UVGC_2322_809})

V_769 = CTVertex(name = 'V_769',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_1995_431,(0,0,1):C.UVGC_2310_784,(0,0,2):C.UVGC_2310_785})

V_770 = CTVertex(name = 'V_770',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_1994_430,(0,0,1):C.UVGC_2309_782,(0,0,2):C.UVGC_2309_783})

V_771 = CTVertex(name = 'V_771',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2078_513,(0,0,1):C.UVGC_2320_804,(0,0,2):C.UVGC_2320_805})

V_772 = CTVertex(name = 'V_772',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.a, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2012_448,(0,0,0):C.UVGC_2401_940,(0,0,3):C.UVGC_2401_941,(0,0,1):C.UVGC_2401_942})

V_773 = CTVertex(name = 'V_773',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.a, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2013_449,(0,0,0):C.UVGC_2402_943,(0,0,3):C.UVGC_2402_944,(0,0,1):C.UVGC_2402_945})

V_774 = CTVertex(name = 'V_774',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS91, L.FFVS92, L.FFVS93, L.FFVS94, L.FFVS95, L.FFVS97, L.FFVS98 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,3,2):C.UVGC_2083_518,(0,3,0):C.UVGC_2422_981,(0,3,3):C.UVGC_2422_982,(0,3,1):C.UVGC_2422_983,(0,5,0):C.UVGC_1933_376,(0,6,3):C.UVGC_2147_587,(0,4,0):C.UVGC_2197_669,(0,4,3):C.UVGC_2197_670,(0,2,2):C.UVGC_2065_501,(0,0,2):C.UVGC_2066_502,(0,1,2):C.UVGC_2068_503})

V_775 = CTVertex(name = 'V_775',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30, L.FFSS40 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t], [P.b, P.g, P.t] ] ],
                 couplings = {(0,1,4):C.UVGC_2396_935,(0,0,2):C.UVGC_1998_434,(0,0,0):C.UVGC_2344_845,(0,0,3):C.UVGC_2344_846,(0,0,1):C.UVGC_2344_847})

V_776 = CTVertex(name = 'V_776',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.a, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2009_445,(0,0,1):C.UVGC_2313_790,(0,0,2):C.UVGC_2313_791})

V_777 = CTVertex(name = 'V_777',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.a, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2010_446,(0,0,1):C.UVGC_2314_792,(0,0,2):C.UVGC_2314_793})

V_778 = CTVertex(name = 'V_778',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2081_516,(0,0,1):C.UVGC_2322_808,(0,0,2):C.UVGC_2322_809})

V_779 = CTVertex(name = 'V_779',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1995_431,(0,0,1):C.UVGC_2310_784,(0,0,2):C.UVGC_2310_785})

V_780 = CTVertex(name = 'V_780',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1994_430,(0,0,1):C.UVGC_2309_782,(0,0,2):C.UVGC_2309_783})

V_781 = CTVertex(name = 'V_781',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2078_513,(0,0,1):C.UVGC_2320_804,(0,0,2):C.UVGC_2320_805})

V_782 = CTVertex(name = 'V_782',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2025_461,(0,0,1):C.UVGC_2316_796,(0,0,2):C.UVGC_2316_797})

V_783 = CTVertex(name = 'V_783',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2025_461,(0,0,1):C.UVGC_2316_796,(0,0,2):C.UVGC_2316_797})

V_784 = CTVertex(name = 'V_784',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2025_461,(0,0,1):C.UVGC_2316_796,(0,0,2):C.UVGC_2316_797})

V_785 = CTVertex(name = 'V_785',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2099_536,(0,0,1):C.UVGC_2323_810,(0,0,2):C.UVGC_2323_811})

V_786 = CTVertex(name = 'V_786',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2029_465,(0,0,0):C.UVGC_2376_907,(0,0,3):C.UVGC_2376_908,(0,0,1):C.UVGC_2376_909})

V_787 = CTVertex(name = 'V_787',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2029_465,(0,0,0):C.UVGC_2376_907,(0,0,3):C.UVGC_2376_908,(0,0,1):C.UVGC_2376_909})

V_788 = CTVertex(name = 'V_788',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2029_465,(0,0,0):C.UVGC_2376_907,(0,0,3):C.UVGC_2376_908,(0,0,1):C.UVGC_2376_909})

V_789 = CTVertex(name = 'V_789',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS93, L.FFVS94, L.FFVS95, L.FFVS97 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,2):C.UVGC_2102_539,(0,1,0):C.UVGC_2427_994,(0,1,3):C.UVGC_2427_995,(0,1,1):C.UVGC_2427_996,(0,3,1):C.UVGC_2378_911,(0,2,0):C.UVGC_2138_574,(0,2,3):C.UVGC_2138_575,(0,0,2):C.UVGC_2093_530})

V_790 = CTVertex(name = 'V_790',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2025_461,(0,0,1):C.UVGC_2316_796,(0,0,2):C.UVGC_2316_797})

V_791 = CTVertex(name = 'V_791',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2025_461,(0,0,1):C.UVGC_2316_796,(0,0,2):C.UVGC_2316_797})

V_792 = CTVertex(name = 'V_792',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2025_461,(0,0,1):C.UVGC_2316_796,(0,0,2):C.UVGC_2316_797})

V_793 = CTVertex(name = 'V_793',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2099_536,(0,0,1):C.UVGC_2323_810,(0,0,2):C.UVGC_2323_811})

V_794 = CTVertex(name = 'V_794',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.Z, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2051_487,(0,0,1):C.UVGC_2318_800,(0,0,2):C.UVGC_2318_801})

V_795 = CTVertex(name = 'V_795',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.Z, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2052_488,(0,0,1):C.UVGC_2319_802,(0,0,2):C.UVGC_2319_803})

V_796 = CTVertex(name = 'V_796',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2122_558,(0,0,1):C.UVGC_2325_814,(0,0,2):C.UVGC_2325_815})

V_797 = CTVertex(name = 'V_797',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.Z, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2054_490,(0,0,0):C.UVGC_2414_961,(0,0,3):C.UVGC_2414_962,(0,0,1):C.UVGC_2414_963})

V_798 = CTVertex(name = 'V_798',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.Z, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2055_491,(0,0,0):C.UVGC_2415_964,(0,0,3):C.UVGC_2415_965,(0,0,1):C.UVGC_2415_966})

V_799 = CTVertex(name = 'V_799',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS91, L.FFVS92, L.FFVS93, L.FFVS94, L.FFVS95, L.FFVS97, L.FFVS98 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,3,2):C.UVGC_2124_560,(0,3,0):C.UVGC_2433_1004,(0,3,3):C.UVGC_2433_1005,(0,3,1):C.UVGC_2433_1006,(0,5,0):C.UVGC_1939_382,(0,6,3):C.UVGC_2202_676,(0,4,0):C.UVGC_2145_584,(0,4,3):C.UVGC_2145_585,(0,2,2):C.UVGC_2115_552,(0,0,2):C.UVGC_2116_553,(0,1,2):C.UVGC_2118_554})

V_800 = CTVertex(name = 'V_800',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.Z, P.G0, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2051_487,(0,0,1):C.UVGC_2318_800,(0,0,2):C.UVGC_2318_801})

V_801 = CTVertex(name = 'V_801',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.Z, P.G__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2052_488,(0,0,1):C.UVGC_2319_802,(0,0,2):C.UVGC_2319_803})

V_802 = CTVertex(name = 'V_802',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2122_558,(0,0,1):C.UVGC_2325_814,(0,0,2):C.UVGC_2325_815})

V_803 = CTVertex(name = 'V_803',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.a, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2012_448,(0,0,0):C.UVGC_2401_940,(0,0,3):C.UVGC_2401_941,(0,0,1):C.UVGC_2401_942})

V_804 = CTVertex(name = 'V_804',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.a, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2011_447,(0,0,0):C.UVGC_2400_937,(0,0,3):C.UVGC_2400_938,(0,0,1):C.UVGC_2400_939})

V_805 = CTVertex(name = 'V_805',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS109, L.FFVS110, L.FFVS119, L.FFVS128, L.FFVS134, L.FFVS136, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,6,2):C.UVGC_2082_517,(0,6,0):C.UVGC_2421_978,(0,6,3):C.UVGC_2421_979,(0,6,1):C.UVGC_2421_980,(0,5,0):C.UVGC_1934_377,(0,4,3):C.UVGC_2146_586,(0,3,0):C.UVGC_2198_671,(0,3,3):C.UVGC_2198_672,(0,2,2):C.UVGC_2065_501,(0,0,2):C.UVGC_2066_502,(0,1,2):C.UVGC_2068_503})

V_806 = CTVertex(name = 'V_806',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30, L.FFSS39 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t], [P.b, P.g, P.t] ] ],
                 couplings = {(0,1,4):C.UVGC_2396_935,(0,0,2):C.UVGC_1999_435,(0,0,0):C.UVGC_2345_848,(0,0,3):C.UVGC_2345_849,(0,0,1):C.UVGC_2345_850})

V_807 = CTVertex(name = 'V_807',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.a, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2009_445,(0,0,1):C.UVGC_2313_790,(0,0,2):C.UVGC_2313_791})

V_808 = CTVertex(name = 'V_808',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.a, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2008_444,(0,0,1):C.UVGC_2312_788,(0,0,2):C.UVGC_2312_789})

V_809 = CTVertex(name = 'V_809',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2080_515,(0,0,1):C.UVGC_2321_806,(0,0,2):C.UVGC_2321_807})

V_810 = CTVertex(name = 'V_810',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1996_432,(0,0,1):C.UVGC_2311_786,(0,0,2):C.UVGC_2311_787})

V_811 = CTVertex(name = 'V_811',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1994_430,(0,0,1):C.UVGC_2309_782,(0,0,2):C.UVGC_2309_783})

V_812 = CTVertex(name = 'V_812',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2078_513,(0,0,1):C.UVGC_2320_804,(0,0,2):C.UVGC_2320_805})

V_813 = CTVertex(name = 'V_813',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.a, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2009_445,(0,0,1):C.UVGC_2313_790,(0,0,2):C.UVGC_2313_791})

V_814 = CTVertex(name = 'V_814',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.a, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2008_444,(0,0,1):C.UVGC_2312_788,(0,0,2):C.UVGC_2312_789})

V_815 = CTVertex(name = 'V_815',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2080_515,(0,0,1):C.UVGC_2321_806,(0,0,2):C.UVGC_2321_807})

V_816 = CTVertex(name = 'V_816',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_1996_432,(0,0,1):C.UVGC_2311_786,(0,0,2):C.UVGC_2311_787})

V_817 = CTVertex(name = 'V_817',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFSS30 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_1994_430,(0,0,1):C.UVGC_2309_782,(0,0,2):C.UVGC_2309_783})

V_818 = CTVertex(name = 'V_818',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS45 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2078_513,(0,0,1):C.UVGC_2320_804,(0,0,2):C.UVGC_2320_805})

V_819 = CTVertex(name = 'V_819',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2029_465,(0,0,0):C.UVGC_2376_907,(0,0,3):C.UVGC_2376_908,(0,0,1):C.UVGC_2376_909})

V_820 = CTVertex(name = 'V_820',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2029_465,(0,0,0):C.UVGC_2376_907,(0,0,3):C.UVGC_2376_908,(0,0,1):C.UVGC_2376_909})

V_821 = CTVertex(name = 'V_821',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2029_465,(0,0,0):C.UVGC_2376_907,(0,0,3):C.UVGC_2376_908,(0,0,1):C.UVGC_2376_909})

V_822 = CTVertex(name = 'V_822',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS119, L.FFVS128, L.FFVS136, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,3,2):C.UVGC_2102_539,(0,3,0):C.UVGC_2427_994,(0,3,3):C.UVGC_2427_995,(0,3,1):C.UVGC_2427_996,(0,2,1):C.UVGC_2378_911,(0,1,0):C.UVGC_2138_574,(0,1,3):C.UVGC_2138_575,(0,0,2):C.UVGC_2092_529})

V_823 = CTVertex(name = 'V_823',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2025_461,(0,0,1):C.UVGC_2316_796,(0,0,2):C.UVGC_2316_797})

V_824 = CTVertex(name = 'V_824',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2025_461,(0,0,1):C.UVGC_2316_796,(0,0,2):C.UVGC_2316_797})

V_825 = CTVertex(name = 'V_825',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2025_461,(0,0,1):C.UVGC_2316_796,(0,0,2):C.UVGC_2316_797})

V_826 = CTVertex(name = 'V_826',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2099_536,(0,0,1):C.UVGC_2323_810,(0,0,2):C.UVGC_2323_811})

V_827 = CTVertex(name = 'V_827',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2025_461,(0,0,1):C.UVGC_2316_796,(0,0,2):C.UVGC_2316_797})

V_828 = CTVertex(name = 'V_828',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.G__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2025_461,(0,0,1):C.UVGC_2316_796,(0,0,2):C.UVGC_2316_797})

V_829 = CTVertex(name = 'V_829',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2025_461,(0,0,1):C.UVGC_2316_796,(0,0,2):C.UVGC_2316_797})

V_830 = CTVertex(name = 'V_830',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2099_536,(0,0,1):C.UVGC_2323_810,(0,0,2):C.UVGC_2323_811})

V_831 = CTVertex(name = 'V_831',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.Z, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2054_490,(0,0,0):C.UVGC_2414_961,(0,0,3):C.UVGC_2414_962,(0,0,1):C.UVGC_2414_963})

V_832 = CTVertex(name = 'V_832',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.Z, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2053_489,(0,0,0):C.UVGC_2413_958,(0,0,3):C.UVGC_2413_959,(0,0,1):C.UVGC_2413_960})

V_833 = CTVertex(name = 'V_833',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS109, L.FFVS110, L.FFVS119, L.FFVS128, L.FFVS134, L.FFVS136, L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,6,2):C.UVGC_2123_559,(0,6,0):C.UVGC_2432_1001,(0,6,3):C.UVGC_2432_1002,(0,6,1):C.UVGC_2432_1003,(0,5,0):C.UVGC_1940_383,(0,4,3):C.UVGC_2201_675,(0,3,0):C.UVGC_2144_582,(0,3,3):C.UVGC_2144_583,(0,2,2):C.UVGC_2115_552,(0,0,2):C.UVGC_2116_553,(0,1,2):C.UVGC_2118_554})

V_834 = CTVertex(name = 'V_834',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.Z, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2051_487,(0,0,1):C.UVGC_2318_800,(0,0,2):C.UVGC_2318_801})

V_835 = CTVertex(name = 'V_835',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.Z, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2050_486,(0,0,1):C.UVGC_2317_798,(0,0,2):C.UVGC_2317_799})

V_836 = CTVertex(name = 'V_836',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2121_557,(0,0,1):C.UVGC_2324_812,(0,0,2):C.UVGC_2324_813})

V_837 = CTVertex(name = 'V_837',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.Z, P.G0, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2051_487,(0,0,1):C.UVGC_2318_800,(0,0,2):C.UVGC_2318_801})

V_838 = CTVertex(name = 'V_838',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.Z, P.G__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2050_486,(0,0,1):C.UVGC_2317_798,(0,0,2):C.UVGC_2317_799})

V_839 = CTVertex(name = 'V_839',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS94 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2121_557,(0,0,1):C.UVGC_2324_812,(0,0,2):C.UVGC_2324_813})

V_840 = CTVertex(name = 'V_840',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS103, L.FFVS104, L.FFVS105, L.FFVS94, L.FFVS95 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b, P.g] ], [ [P.g] ], [ [P.g, P.t] ], [ [P.g, P.t], [P.b, P.g, P.t] ] ],
                 couplings = {(0,3,4):C.UVGC_2367_890,(0,0,1):C.UVGC_1937_380,(0,2,3):C.UVGC_1653_8,(0,4,0):C.UVGC_1911_327,(0,4,2):C.UVGC_1911_328,(0,4,3):C.UVGC_1911_329,(0,1,3):C.UVGC_1732_87})

V_841 = CTVertex(name = 'V_841',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS126, L.FFVS148, L.FFVS155, L.FFVS156, L.FFVS87 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,0):C.UVGC_1914_336,(0,4,1):C.UVGC_1914_337,(0,4,2):C.UVGC_1914_338,(0,3,2):C.UVGC_1656_11,(0,1,1):C.UVGC_1646_1,(0,2,2):C.UVGC_1733_88,(0,0,1):C.UVGC_1728_83})

V_842 = CTVertex(name = 'V_842',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS125, L.FFVS130, L.FFVS145, L.FFVS147, L.FFVS154, L.FFVS157, L.FFVS86, L.FFVS89 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,6,2):C.UVGC_1942_385,(0,1,0):C.UVGC_1913_333,(0,1,1):C.UVGC_1913_334,(0,1,2):C.UVGC_1913_335,(0,0,2):C.UVGC_2189_659,(0,5,2):C.UVGC_1655_10,(0,3,1):C.UVGC_1754_109,(0,2,2):C.UVGC_1651_6,(0,7,2):C.UVGC_1652_7,(0,4,2):C.UVGC_1734_89})

V_843 = CTVertex(name = 'V_843',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.g, P.g, P.G__minus__ ],
                 color = [ 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVS87, L.FFVVS89 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b, P.g] ], [ [P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(1,1,0):C.UVGC_2164_608,(1,1,2):C.UVGC_2164_609,(1,1,1):C.UVGC_2165_612,(1,1,3):C.UVGC_2165_613,(0,1,0):C.UVGC_2167_616,(0,1,2):C.UVGC_2167_617,(0,1,1):C.UVGC_2168_620,(0,1,3):C.UVGC_2168_621,(1,0,0):C.UVGC_2167_616,(1,0,2):C.UVGC_2167_617,(1,0,1):C.UVGC_2167_618,(1,0,3):C.UVGC_2167_619,(0,0,0):C.UVGC_2164_608,(0,0,2):C.UVGC_2164_609,(0,0,1):C.UVGC_2164_610,(0,0,3):C.UVGC_2164_611})

V_844 = CTVertex(name = 'V_844',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.g, P.G0 ],
                 color = [ 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVS119, L.FFVVS78, L.FFVVS80, L.FFVVS89 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(1,1,0):C.UVGC_2182_650,(1,1,1):C.UVGC_2182_651,(1,1,2):C.UVGC_2182_652,(0,1,0):C.UVGC_2170_624,(0,1,1):C.UVGC_2170_625,(0,1,2):C.UVGC_2172_628,(0,2,0):C.UVGC_2182_650,(0,2,1):C.UVGC_2182_651,(0,2,2):C.UVGC_2185_655,(1,2,0):C.UVGC_2170_624,(1,2,1):C.UVGC_2170_625,(1,2,2):C.UVGC_2170_626,(1,3,2):C.UVGC_2171_627,(0,3,2):C.UVGC_2183_653,(1,0,2):C.UVGC_2184_654,(0,0,2):C.UVGC_2173_629})

V_845 = CTVertex(name = 'V_845',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.g, P.H ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS119, L.FFVVS139, L.FFVVS140, L.FFVVS77, L.FFVVS81, L.FFVVS87, L.FFVVS89 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(2,4,0):C.UVGC_2174_630,(2,4,1):C.UVGC_2174_631,(2,4,2):C.UVGC_2174_632,(1,4,0):C.UVGC_2178_642,(1,4,1):C.UVGC_2178_643,(1,4,2):C.UVGC_2178_644,(1,5,2):C.UVGC_2181_649,(2,5,2):C.UVGC_2180_648,(2,7,0):C.UVGC_2179_645,(2,7,1):C.UVGC_2179_646,(2,7,2):C.UVGC_2179_647,(1,7,0):C.UVGC_2177_639,(1,7,1):C.UVGC_2177_640,(1,7,2):C.UVGC_2177_641,(2,6,0):C.UVGC_2176_636,(2,6,1):C.UVGC_2176_637,(2,6,2):C.UVGC_2176_638,(1,6,0):C.UVGC_2175_633,(1,6,1):C.UVGC_2175_634,(1,6,2):C.UVGC_2175_635,(2,1,0):C.UVGC_2179_645,(2,1,1):C.UVGC_2179_646,(2,1,2):C.UVGC_2179_647,(1,1,0):C.UVGC_2177_639,(1,1,1):C.UVGC_2177_640,(1,1,2):C.UVGC_2177_641,(2,0,0):C.UVGC_2176_636,(2,0,1):C.UVGC_2176_637,(2,0,2):C.UVGC_2176_638,(1,0,0):C.UVGC_2175_633,(1,0,1):C.UVGC_2175_634,(1,0,2):C.UVGC_2175_635,(0,3,2):C.UVGC_1657_12,(0,2,2):C.UVGC_1735_90})

V_846 = CTVertex(name = 'V_846',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV101, L.FFVV102, L.FFVV103, L.FFVV104, L.FFVV107, L.FFVV108, L.FFVV109, L.FFVV110, L.FFVV111, L.FFVV112, L.FFVV113, L.FFVV114, L.FFVV115, L.FFVV117, L.FFVV131, L.FFVV132, L.FFVV154, L.FFVV158, L.FFVV159, L.FFVV160, L.FFVV161, L.FFVV162, L.FFVV163, L.FFVV164, L.FFVV165, L.FFVV167, L.FFVV179, L.FFVV180, L.FFVV189, L.FFVV190 ],
                 loop_particles = [ [ [P.b] ], [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.g] ], [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(2,5,0):C.UVGC_1751_106,(2,5,2):C.UVGC_1877_273,(2,5,3):C.UVGC_1877_274,(2,6,0):C.UVGC_1751_106,(2,6,2):C.UVGC_1877_273,(2,6,3):C.UVGC_1877_274,(2,7,0):C.UVGC_1866_259,(2,7,2):C.UVGC_1862_255,(2,7,3):C.UVGC_1862_256,(2,7,6):C.UVGC_1866_260,(2,8,0):C.UVGC_1768_127,(2,8,6):C.UVGC_1768_128,(1,5,0):C.UVGC_1750_105,(1,5,2):C.UVGC_1862_255,(1,5,3):C.UVGC_1862_256,(1,6,0):C.UVGC_1750_105,(1,6,2):C.UVGC_1862_255,(1,6,3):C.UVGC_1862_256,(1,7,0):C.UVGC_1881_277,(1,7,2):C.UVGC_1877_273,(1,7,3):C.UVGC_1877_274,(1,7,6):C.UVGC_1881_278,(1,8,0):C.UVGC_1769_129,(1,8,6):C.UVGC_1769_130,(2,9,0):C.UVGC_1750_105,(2,9,2):C.UVGC_1862_255,(2,9,3):C.UVGC_1862_256,(2,9,6):C.UVGC_1772_131,(2,10,0):C.UVGC_1750_105,(2,10,2):C.UVGC_1862_255,(2,10,3):C.UVGC_1862_256,(2,10,6):C.UVGC_1772_131,(2,11,0):C.UVGC_1769_129,(2,11,6):C.UVGC_1769_130,(2,12,0):C.UVGC_1881_277,(2,12,2):C.UVGC_1877_273,(2,12,3):C.UVGC_1877_274,(2,12,6):C.UVGC_1881_278,(1,9,0):C.UVGC_1751_106,(1,9,2):C.UVGC_1877_273,(1,9,3):C.UVGC_1877_274,(1,9,6):C.UVGC_1773_132,(1,10,0):C.UVGC_1751_106,(1,10,2):C.UVGC_1877_273,(1,10,3):C.UVGC_1877_274,(1,10,6):C.UVGC_1773_132,(1,11,0):C.UVGC_1768_127,(1,11,6):C.UVGC_1768_128,(1,12,0):C.UVGC_1866_259,(1,12,2):C.UVGC_1862_255,(1,12,3):C.UVGC_1862_256,(1,12,6):C.UVGC_1866_260,(2,13,1):C.UVGC_2243_732,(2,13,4):C.UVGC_2243_733,(2,13,5):C.UVGC_2243_734,(1,13,1):C.UVGC_2241_726,(1,13,4):C.UVGC_2241_727,(1,13,5):C.UVGC_2241_728,(1,4,1):C.UVGC_2239_720,(1,4,4):C.UVGC_2239_721,(1,4,5):C.UVGC_2239_722,(2,17,0):C.UVGC_1887_284,(2,17,2):C.UVGC_1887_285,(2,17,3):C.UVGC_1887_286,(2,18,0):C.UVGC_1887_284,(2,18,2):C.UVGC_1887_285,(2,18,3):C.UVGC_1887_286,(2,19,0):C.UVGC_1882_279,(2,19,2):C.UVGC_1882_280,(2,19,3):C.UVGC_1882_281,(2,19,6):C.UVGC_1886_283,(2,20,6):C.UVGC_1782_143,(1,17,0):C.UVGC_1882_279,(1,17,2):C.UVGC_1882_280,(1,17,3):C.UVGC_1882_281,(1,18,0):C.UVGC_1882_279,(1,18,2):C.UVGC_1882_280,(1,18,3):C.UVGC_1882_281,(1,19,0):C.UVGC_1887_284,(1,19,2):C.UVGC_1887_285,(1,19,3):C.UVGC_1887_286,(1,19,6):C.UVGC_1891_288,(1,20,6):C.UVGC_1783_144,(2,21,0):C.UVGC_1882_279,(2,21,2):C.UVGC_1882_280,(2,21,3):C.UVGC_1882_281,(2,21,6):C.UVGC_1772_131,(2,22,0):C.UVGC_1882_279,(2,22,2):C.UVGC_1882_280,(2,22,3):C.UVGC_1882_281,(2,22,6):C.UVGC_1772_131,(2,23,6):C.UVGC_1783_144,(2,24,0):C.UVGC_1887_284,(2,24,2):C.UVGC_1887_285,(2,24,3):C.UVGC_1887_286,(2,24,6):C.UVGC_1891_288,(1,21,0):C.UVGC_1887_284,(1,21,2):C.UVGC_1887_285,(1,21,3):C.UVGC_1887_286,(1,21,6):C.UVGC_1773_132,(1,22,0):C.UVGC_1887_284,(1,22,2):C.UVGC_1887_285,(1,22,3):C.UVGC_1887_286,(1,22,6):C.UVGC_1773_132,(1,23,6):C.UVGC_1782_143,(1,24,0):C.UVGC_1882_279,(1,24,2):C.UVGC_1882_280,(1,24,3):C.UVGC_1882_281,(1,24,6):C.UVGC_1886_283,(2,25,1):C.UVGC_2243_732,(2,25,4):C.UVGC_2243_733,(2,25,5):C.UVGC_2243_734,(1,25,1):C.UVGC_2241_726,(1,25,4):C.UVGC_2241_727,(1,25,5):C.UVGC_2241_728,(1,16,1):C.UVGC_2239_720,(1,16,4):C.UVGC_2239_721,(1,16,5):C.UVGC_2239_722,(2,14,0):C.UVGC_1879_275,(2,14,2):C.UVGC_1877_273,(2,14,3):C.UVGC_1877_274,(2,14,6):C.UVGC_1879_276,(1,14,0):C.UVGC_1864_257,(1,14,2):C.UVGC_1862_255,(1,14,3):C.UVGC_1862_256,(1,14,6):C.UVGC_1864_258,(2,28,0):C.UVGC_1887_284,(2,28,2):C.UVGC_1887_285,(2,28,3):C.UVGC_1887_286,(2,28,6):C.UVGC_1889_287,(1,28,0):C.UVGC_1882_279,(1,28,2):C.UVGC_1882_280,(1,28,3):C.UVGC_1882_281,(1,28,6):C.UVGC_1884_282,(2,15,0):C.UVGC_1864_257,(2,15,2):C.UVGC_1862_255,(2,15,3):C.UVGC_1862_256,(2,15,6):C.UVGC_1864_258,(1,15,0):C.UVGC_1879_275,(1,15,2):C.UVGC_1877_273,(1,15,3):C.UVGC_1877_274,(1,15,6):C.UVGC_1879_276,(2,29,0):C.UVGC_1882_279,(2,29,2):C.UVGC_1882_280,(2,29,3):C.UVGC_1882_281,(2,29,6):C.UVGC_1884_282,(1,29,0):C.UVGC_1887_284,(1,29,2):C.UVGC_1887_285,(1,29,3):C.UVGC_1887_286,(1,29,6):C.UVGC_1889_287,(2,0,6):C.UVGC_1773_132,(1,0,6):C.UVGC_1772_131,(2,1,6):C.UVGC_1773_132,(1,1,6):C.UVGC_1772_131,(2,2,1):C.UVGC_2238_717,(2,2,4):C.UVGC_2238_718,(2,2,5):C.UVGC_2238_719,(1,2,1):C.UVGC_2242_729,(1,2,4):C.UVGC_2242_730,(1,2,5):C.UVGC_2242_731,(2,3,5):C.UVGC_2244_735,(1,3,5):C.UVGC_2245_736,(2,4,1):C.UVGC_2240_723,(2,4,4):C.UVGC_2240_724,(2,4,5):C.UVGC_2240_725,(2,16,1):C.UVGC_2240_723,(2,16,4):C.UVGC_2240_724,(2,16,5):C.UVGC_2240_725,(0,27,5):C.UVGC_1667_22,(0,26,5):C.UVGC_1742_97})

V_847 = CTVertex(name = 'V_847',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS143, L.FFVS84, L.FFVS87 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2199_673,(0,0,0):C.UVGC_2149_589,(0,1,0):C.UVGC_2150_590})

V_848 = CTVertex(name = 'V_848',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS114, L.FFVS135, L.FFVS140, L.FFVS143, L.FFVS84, L.FFVS87, L.FFVS90 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,6,1):C.UVGC_1844_219,(0,5,0):C.UVGC_2143_581,(0,2,0):C.UVGC_1659_14,(0,3,0):C.UVGC_2204_678,(0,0,0):C.UVGC_1660_15,(0,1,0):C.UVGC_1738_93,(0,4,0):C.UVGC_1739_94})

V_849 = CTVertex(name = 'V_849',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS115, L.FFVS116, L.FFVS125, L.FFVS130, L.FFVS131, L.FFVS145, L.FFVS89 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,3,0):C.UVGC_2200_674,(0,2,0):C.UVGC_2188_658,(0,4,0):C.UVGC_2148_588,(0,5,0):C.UVGC_1672_27,(0,6,0):C.UVGC_1673_28,(0,1,0):C.UVGC_1650_5,(0,0,0):C.UVGC_1730_85})

V_850 = CTVertex(name = 'V_850',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS93, L.FFVS95, L.FFVS97 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_2379_912,(0,1,0):C.UVGC_2139_576,(0,1,3):C.UVGC_2139_577,(0,0,2):C.UVGC_2091_528})

V_851 = CTVertex(name = 'V_851',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS87, L.FFVVS94, L.FFVVS95 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_2095_532,(0,1,0):C.UVGC_2386_919,(0,2,2):C.UVGC_2155_598})

V_852 = CTVertex(name = 'V_852',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.a, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS104, L.FFVVS105, L.FFVVS87 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2153_595,(0,1,3):C.UVGC_2153_596,(0,0,1):C.UVGC_2384_917,(0,2,2):C.UVGC_2098_535})

V_853 = CTVertex(name = 'V_853',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.a, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS104, L.FFVVS105, L.FFVVS87 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2152_593,(0,1,3):C.UVGC_2152_594,(0,0,1):C.UVGC_2383_916,(0,2,2):C.UVGC_2097_534})

V_854 = CTVertex(name = 'V_854',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV107, L.FFVV125, L.FFVV126, L.FFVV133, L.FFVV135, L.FFVV137 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2234_713,(0,2,4):C.UVGC_2234_714,(0,1,1):C.UVGC_2429_998,(0,3,3):C.UVGC_1982_418,(0,4,2):C.UVGC_1984_420,(0,5,2):C.UVGC_1979_415,(0,0,2):C.UVGC_1990_426})

V_855 = CTVertex(name = 'V_855',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS121, L.FFVVS80, L.FFVVS82, L.FFVVS83 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,1):C.UVGC_2087_524,(0,2,2):C.UVGC_2193_665,(0,0,0):C.UVGC_1669_24,(0,3,0):C.UVGC_1746_101})

V_856 = CTVertex(name = 'V_856',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS117, L.FFVVS133, L.FFVVS81, L.FFVVS85, L.FFVVS86 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_2086_523,(0,0,2):C.UVGC_1711_66,(0,1,0):C.UVGC_1668_23,(0,3,2):C.UVGC_1736_91,(0,4,0):C.UVGC_1745_100})

V_857 = CTVertex(name = 'V_857',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV104, L.FFVV105, L.FFVV106, L.FFVV133, L.FFVV135, L.FFVV137, L.FFVV156, L.FFVV178 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,3,2):C.UVGC_1965_401,(0,0,1):C.UVGC_1977_413,(0,4,1):C.UVGC_1967_403,(0,5,1):C.UVGC_1962_398,(0,6,3):C.UVGC_1722_77,(0,7,0):C.UVGC_1670_25,(0,1,3):C.UVGC_1744_99,(0,2,0):C.UVGC_1747_102})

V_858 = CTVertex(name = 'V_858',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS87, L.FFVVS91, L.FFVVS95 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2192_663,(0,2,3):C.UVGC_2192_664,(0,1,1):C.UVGC_2369_892,(0,0,2):C.UVGC_2088_525})

V_859 = CTVertex(name = 'V_859',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.W__plus__, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS104, L.FFVVS105, L.FFVVS87 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,2):C.UVGC_2373_902,(0,1,0):C.UVGC_2373_903,(0,2,1):C.UVGC_2072_507,(0,0,0):C.UVGC_2363_886})

V_860 = CTVertex(name = 'V_860',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS87, L.FFVVS94, L.FFVVS95 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2370_893,(0,2,3):C.UVGC_2370_894,(0,2,1):C.UVGC_2370_895,(0,1,1):C.UVGC_2361_884,(0,0,2):C.UVGC_2075_510})

V_861 = CTVertex(name = 'V_861',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS87, L.FFVVS94, L.FFVVS95 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2371_896,(0,2,3):C.UVGC_2371_897,(0,2,1):C.UVGC_2371_898,(0,1,1):C.UVGC_2360_883,(0,0,2):C.UVGC_2074_509})

V_862 = CTVertex(name = 'V_862',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV107, L.FFVV119, L.FFVV120, L.FFVV133, L.FFVV135, L.FFVV137 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2426_991,(0,2,4):C.UVGC_2426_992,(0,2,1):C.UVGC_2426_993,(0,1,1):C.UVGC_2423_984,(0,3,3):C.UVGC_1971_407,(0,5,2):C.UVGC_1968_404,(0,4,2):C.UVGC_1973_409,(0,0,2):C.UVGC_1958_394})

V_863 = CTVertex(name = 'V_863',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS128, L.FFVS150, L.FFVS151, L.FFVS152, L.FFVS94 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b, P.g] ], [ [P.g] ], [ [P.g, P.t] ], [ [P.g, P.t], [P.b, P.g, P.t] ] ],
                 couplings = {(0,4,4):C.UVGC_2366_889,(0,3,1):C.UVGC_1936_379,(0,2,3):C.UVGC_1654_9,(0,0,0):C.UVGC_1912_330,(0,0,2):C.UVGC_1912_331,(0,0,3):C.UVGC_1912_332,(0,1,3):C.UVGC_1731_86})

V_864 = CTVertex(name = 'V_864',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.g, P.g, P.G__plus__ ],
                 color = [ 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS119 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b, P.g] ], [ [P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(1,1,0):C.UVGC_2167_616,(1,1,2):C.UVGC_2167_617,(1,1,1):C.UVGC_2168_620,(1,1,3):C.UVGC_2168_621,(0,1,0):C.UVGC_2164_608,(0,1,2):C.UVGC_2164_609,(0,1,1):C.UVGC_2165_612,(0,1,3):C.UVGC_2165_613,(1,0,0):C.UVGC_2164_608,(1,0,2):C.UVGC_2164_609,(1,0,1):C.UVGC_2166_614,(1,0,3):C.UVGC_2166_615,(0,0,0):C.UVGC_2167_616,(0,0,2):C.UVGC_2167_617,(0,0,1):C.UVGC_2169_622,(0,0,3):C.UVGC_2169_623})

V_865 = CTVertex(name = 'V_865',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS119, L.FFVS128, L.FFVS136 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_2377_910,(0,1,0):C.UVGC_2137_572,(0,1,3):C.UVGC_2137_573,(0,0,2):C.UVGC_2091_528})

V_866 = CTVertex(name = 'V_866',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS145, L.FFVVS146 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_2094_531,(0,1,0):C.UVGC_2385_918,(0,2,2):C.UVGC_2154_597})

V_867 = CTVertex(name = 'V_867',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.a, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS129, L.FFVVS130 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2151_591,(0,2,3):C.UVGC_2151_592,(0,1,1):C.UVGC_2382_915,(0,0,2):C.UVGC_2096_533})

V_868 = CTVertex(name = 'V_868',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.a, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS129, L.FFVVS130 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2152_593,(0,2,3):C.UVGC_2152_594,(0,1,1):C.UVGC_2383_916,(0,0,2):C.UVGC_2097_534})

V_869 = CTVertex(name = 'V_869',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133, L.FFVV135, L.FFVV137, L.FFVV154, L.FFVV175, L.FFVV176 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,5,0):C.UVGC_2234_713,(0,5,4):C.UVGC_2234_714,(0,4,1):C.UVGC_2429_998,(0,0,3):C.UVGC_1981_417,(0,1,2):C.UVGC_1983_419,(0,2,2):C.UVGC_1980_416,(0,3,2):C.UVGC_1990_426})

V_870 = CTVertex(name = 'V_870',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__minus__, P.W__plus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS126, L.FFVVS130 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2191_661,(0,2,3):C.UVGC_2191_662,(0,1,1):C.UVGC_2368_891,(0,0,2):C.UVGC_2089_526})

V_871 = CTVertex(name = 'V_871',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS129, L.FFVVS130 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,2):C.UVGC_2374_904,(0,2,0):C.UVGC_2374_905,(0,0,1):C.UVGC_2071_506,(0,1,0):C.UVGC_2362_885})

V_872 = CTVertex(name = 'V_872',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS145, L.FFVVS146 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2372_899,(0,2,3):C.UVGC_2372_900,(0,2,1):C.UVGC_2372_901,(0,1,1):C.UVGC_2359_882,(0,0,2):C.UVGC_2073_508})

V_873 = CTVertex(name = 'V_873',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS110, L.FFVVS145, L.FFVVS146 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2371_896,(0,2,3):C.UVGC_2371_897,(0,2,1):C.UVGC_2371_898,(0,1,1):C.UVGC_2360_883,(0,0,2):C.UVGC_2074_509})

V_874 = CTVertex(name = 'V_874',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133, L.FFVV135, L.FFVV137, L.FFVV154, L.FFVV182, L.FFVV183 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,5,0):C.UVGC_2426_991,(0,5,4):C.UVGC_2426_992,(0,5,1):C.UVGC_2426_993,(0,4,1):C.UVGC_2423_984,(0,0,3):C.UVGC_1970_406,(0,2,2):C.UVGC_1969_405,(0,1,2):C.UVGC_1972_408,(0,3,2):C.UVGC_1958_394})

V_875 = CTVertex(name = 'V_875',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF4, L.FF5, L.FF6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,1,1):C.UVGC_1678_33,(0,1,0):[ C.UVGC_1706_61, C.UVGC_2248_739 ],(0,0,0):[ C.UVGC_1684_39, C.UVGC_2246_737 ],(0,2,0):C.UVGC_2226_704})

V_876 = CTVertex(name = 'V_876',
                 type = 'UV',
                 particles = [ P.g, P.g ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VV8 ],
                 loop_particles = [ [ [P.t] ] ],
                 couplings = {(0,0,0):[ C.UVGC_1677_32, C.UVGC_1680_35 ]})

V_877 = CTVertex(name = 'V_877',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS86 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1942_385})

V_878 = CTVertex(name = 'V_878',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS86 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_1942_385})

V_879 = CTVertex(name = 'V_879',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS86 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_1942_385})

V_880 = CTVertex(name = 'V_880',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS86 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_1942_385})

V_881 = CTVertex(name = 'V_881',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS86 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_1942_385})

V_882 = CTVertex(name = 'V_882',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV104 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2235_715})

V_883 = CTVertex(name = 'V_883',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV104 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2254_745})

V_884 = CTVertex(name = 'V_884',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV104 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2257_748,(0,0,1):C.UVGC_1795_166})

V_885 = CTVertex(name = 'V_885',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.g ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVV104 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2237_716})

V_886 = CTVertex(name = 'V_886',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV155, L.FFVV157 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2255_746,(0,0,0):C.UVGC_2250_741})

V_887 = CTVertex(name = 'V_887',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV173 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2430_999})

V_888 = CTVertex(name = 'V_888',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV121 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2431_1000})

V_889 = CTVertex(name = 'V_889',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.H, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2219_697})

V_890 = CTVertex(name = 'V_890',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.H, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2220_698})

V_891 = CTVertex(name = 'V_891',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.g, P.H, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1935_378})

V_892 = CTVertex(name = 'V_892',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.g, P.H, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_1935_378})

V_893 = CTVertex(name = 'V_893',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.g, P.H, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_1935_378})

V_894 = CTVertex(name = 'V_894',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.g, P.H, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_1935_378})

V_895 = CTVertex(name = 'V_895',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.g, P.H, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_1935_378})

V_896 = CTVertex(name = 'V_896',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.G0, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2219_697})

V_897 = CTVertex(name = 'V_897',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.G0, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2220_698})

V_898 = CTVertex(name = 'V_898',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.g, P.G0, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1935_378})

V_899 = CTVertex(name = 'V_899',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.g, P.G0, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_1935_378})

V_900 = CTVertex(name = 'V_900',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.g, P.G0, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_1935_378})

V_901 = CTVertex(name = 'V_901',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.g, P.G0, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_1935_378})

V_902 = CTVertex(name = 'V_902',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.g, P.G0, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_1935_378})

V_903 = CTVertex(name = 'V_903',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.g, P.G__plus__, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.g, P.t], [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2405_948})

V_904 = CTVertex(name = 'V_904',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.g, P.G0, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.g, P.t], [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2406_949})

V_905 = CTVertex(name = 'V_905',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.g, P.G__minus__, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.g, P.t], [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2407_950})

V_906 = CTVertex(name = 'V_906',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.g, P.G0, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS13 ],
                 loop_particles = [ [ [P.g, P.t], [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2406_949})

V_907 = CTVertex(name = 'V_907',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS14 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_1935_378,(0,1,0):C.UVGC_2408_951})

V_908 = CTVertex(name = 'V_908',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11, L.FFVSS13 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1935_378,(0,1,1):C.UVGC_2408_951})

V_909 = CTVertex(name = 'V_909',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1935_378})

V_910 = CTVertex(name = 'V_910',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_1935_378})

V_911 = CTVertex(name = 'V_911',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_1935_378})

V_912 = CTVertex(name = 'V_912',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.g, P.G__minus__, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVSS11 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_1935_378})

V_913 = CTVertex(name = 'V_913',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS81 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2158_603})

V_914 = CTVertex(name = 'V_914',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS81 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2207_683})

V_915 = CTVertex(name = 'V_915',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.Z, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS81 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_1854_229,(0,0,0):C.UVGC_2215_693})

V_916 = CTVertex(name = 'V_916',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.g, P.H ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVS81 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2162_606})

V_917 = CTVertex(name = 'V_917',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.Z, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS113, L.FFVVS115 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2194_666,(0,0,0):C.UVGC_2211_687})

V_918 = CTVertex(name = 'V_918',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.g, P.W__minus__, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS96 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2389_922})

V_919 = CTVertex(name = 'V_919',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.g, P.W__plus__, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS127 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2388_921})

V_920 = CTVertex(name = 'V_920',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS80 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2159_604})

V_921 = CTVertex(name = 'V_921',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS80 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2208_684})

V_922 = CTVertex(name = 'V_922',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.Z, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS80 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_1853_228,(0,0,0):C.UVGC_2216_694})

V_923 = CTVertex(name = 'V_923',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.g, P.G0 ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVS80 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2163_607})

V_924 = CTVertex(name = 'V_924',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.Z, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS114, L.FFVVS116 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2195_667,(0,0,0):C.UVGC_2210_686})

V_925 = CTVertex(name = 'V_925',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.g, P.W__minus__, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS96 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2387_920})

V_926 = CTVertex(name = 'V_926',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.g, P.W__plus__, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS127 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2387_920})

V_927 = CTVertex(name = 'V_927',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.a, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS110 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2157_601,(0,0,2):C.UVGC_2157_602,(0,0,1):C.UVGC_2070_505})

V_928 = CTVertex(name = 'V_928',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.a, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS110 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2206_681,(0,0,2):C.UVGC_2206_682,(0,0,1):C.UVGC_2120_556})

V_929 = CTVertex(name = 'V_929',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.Z, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS110 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2214_691,(0,0,2):C.UVGC_2214_692,(0,0,1):C.UVGC_2126_562})

V_930 = CTVertex(name = 'V_930',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.a, P.g, P.G__plus__ ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVS143, L.FFVVS147 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1938_381,(0,1,1):C.UVGC_2161_605})

V_931 = CTVertex(name = 'V_931',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.g, P.Z, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS127, L.FFVVS131 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1941_384,(0,1,1):C.UVGC_2209_685})

V_932 = CTVertex(name = 'V_932',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.W__minus__, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS127 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2390_923})

V_933 = CTVertex(name = 'V_933',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.a, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS87 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2156_599,(0,0,2):C.UVGC_2156_600,(0,0,1):C.UVGC_2069_504})

V_934 = CTVertex(name = 'V_934',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.a, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS87 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2205_679,(0,0,2):C.UVGC_2205_680,(0,0,1):C.UVGC_2119_555})

V_935 = CTVertex(name = 'V_935',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.Z, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVS87 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2213_689,(0,0,2):C.UVGC_2213_690,(0,0,1):C.UVGC_2125_561})

V_936 = CTVertex(name = 'V_936',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.a, P.g, P.G__minus__ ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVVS102, L.FFVVS106 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1938_381,(0,0,1):C.UVGC_2161_605})

V_937 = CTVertex(name = 'V_937',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.g, P.Z, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS92, L.FFVVS96 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1941_384,(0,0,1):C.UVGC_2209_685})

V_938 = CTVertex(name = 'V_938',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.W__plus__, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVVS96 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2390_923})

V_939 = CTVertex(name = 'V_939',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.g, P.g ],
                 color = [ 'T(3,-1,1)*T(4,-2,-1)*T(5,2,-2)', 'T(3,-1,1)*T(4,2,-2)*T(5,-2,-1)', 'T(3,-2,-1)*T(4,-1,1)*T(5,2,-2)', 'T(3,-2,-1)*T(4,2,-2)*T(5,-1,1)', 'T(3,2,-2)*T(4,-1,1)*T(5,-2,-1)', 'T(3,2,-2)*T(4,-2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV146, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV79, L.FFVVV80, L.FFVVV81, L.FFVVV82, L.FFVVV84, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.G__plus__] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.G0, P.t], [P.H, P.t] ], [ [P.t] ] ],
                 couplings = {(5,11,0):C.UVGC_1893_292,(5,11,2):C.UVGC_1893_293,(5,11,3):C.UVGC_1893_294,(5,11,5):C.UVGC_1893_295,(5,10,0):C.UVGC_1899_304,(5,10,2):C.UVGC_1899_305,(5,10,3):C.UVGC_1899_306,(5,10,5):C.UVGC_1899_307,(5,9,0):C.UVGC_1893_292,(5,9,2):C.UVGC_1893_293,(5,9,3):C.UVGC_1893_294,(5,9,5):C.UVGC_1893_295,(4,11,0):C.UVGC_1893_292,(4,11,2):C.UVGC_1893_293,(4,11,3):C.UVGC_1893_294,(4,11,5):C.UVGC_1893_295,(4,10,0):C.UVGC_1893_292,(4,10,2):C.UVGC_1893_293,(4,10,3):C.UVGC_1893_294,(4,10,5):C.UVGC_1893_295,(4,9,0):C.UVGC_1899_304,(4,9,2):C.UVGC_1899_305,(4,9,3):C.UVGC_1899_306,(4,9,5):C.UVGC_1899_307,(3,11,0):C.UVGC_1899_304,(3,11,2):C.UVGC_1899_305,(3,11,3):C.UVGC_1899_306,(3,11,5):C.UVGC_1899_307,(3,10,0):C.UVGC_1893_292,(3,10,2):C.UVGC_1893_293,(3,10,3):C.UVGC_1893_294,(3,10,5):C.UVGC_1893_295,(3,10,4):C.UVGC_2272_755,(3,9,0):C.UVGC_1893_292,(3,9,2):C.UVGC_1893_293,(3,9,3):C.UVGC_1893_294,(3,9,5):C.UVGC_1893_295,(3,9,4):C.UVGC_2278_757,(1,11,0):C.UVGC_1893_292,(1,11,2):C.UVGC_1893_293,(1,11,3):C.UVGC_1893_294,(1,11,5):C.UVGC_1893_295,(1,11,4):C.UVGC_2272_755,(1,10,0):C.UVGC_1893_292,(1,10,2):C.UVGC_1893_293,(1,10,3):C.UVGC_1893_294,(1,10,5):C.UVGC_1893_295,(1,10,4):C.UVGC_2278_757,(1,9,0):C.UVGC_1899_304,(1,9,2):C.UVGC_1899_305,(1,9,3):C.UVGC_1899_306,(1,9,5):C.UVGC_1899_307,(1,9,4):C.UVGC_1947_390,(2,11,0):C.UVGC_1899_304,(2,11,2):C.UVGC_1899_305,(2,11,3):C.UVGC_1899_306,(2,11,5):C.UVGC_1899_307,(2,11,4):C.UVGC_1947_390,(2,10,0):C.UVGC_1893_292,(2,10,2):C.UVGC_1893_293,(2,10,3):C.UVGC_1893_294,(2,10,5):C.UVGC_1893_295,(2,10,4):C.UVGC_2278_757,(2,9,0):C.UVGC_1893_292,(2,9,2):C.UVGC_1893_293,(2,9,3):C.UVGC_1893_294,(2,9,5):C.UVGC_1893_295,(2,9,4):C.UVGC_2272_755,(0,11,0):C.UVGC_1893_292,(0,11,2):C.UVGC_1893_293,(0,11,3):C.UVGC_1893_294,(0,11,5):C.UVGC_1893_295,(0,11,4):C.UVGC_2278_757,(0,10,0):C.UVGC_1899_304,(0,10,2):C.UVGC_1899_305,(0,10,3):C.UVGC_1899_306,(0,10,5):C.UVGC_1899_307,(0,10,4):C.UVGC_1947_390,(0,9,0):C.UVGC_1893_292,(0,9,2):C.UVGC_1893_293,(0,9,3):C.UVGC_1893_294,(0,9,5):C.UVGC_1893_295,(0,9,4):C.UVGC_2278_757,(3,8,4):C.UVGC_1945_388,(1,8,4):C.UVGC_2274_756,(2,8,4):C.UVGC_2274_756,(0,8,4):C.UVGC_1945_388,(5,3,0):C.UVGC_1901_308,(5,3,2):C.UVGC_1901_309,(5,3,3):C.UVGC_1901_310,(5,3,5):C.UVGC_1901_311,(5,3,1):C.UVGC_1945_388,(5,2,0):C.UVGC_1903_312,(5,2,2):C.UVGC_1903_313,(5,2,3):C.UVGC_1903_314,(5,2,5):C.UVGC_1903_315,(5,1,0):C.UVGC_1901_308,(5,1,2):C.UVGC_1901_309,(5,1,3):C.UVGC_1901_310,(5,1,5):C.UVGC_1901_311,(5,1,1):C.UVGC_1945_388,(4,3,0):C.UVGC_1901_308,(4,3,2):C.UVGC_1901_309,(4,3,3):C.UVGC_1901_310,(4,3,5):C.UVGC_1901_311,(4,3,1):C.UVGC_1948_391,(4,2,0):C.UVGC_1901_308,(4,2,2):C.UVGC_1901_309,(4,2,3):C.UVGC_1901_310,(4,2,5):C.UVGC_1901_311,(4,2,1):C.UVGC_1945_388,(4,1,0):C.UVGC_1903_312,(4,1,2):C.UVGC_1903_313,(4,1,3):C.UVGC_1903_314,(4,1,5):C.UVGC_1903_315,(3,3,0):C.UVGC_1903_312,(3,3,2):C.UVGC_1903_313,(3,3,3):C.UVGC_1903_314,(3,3,5):C.UVGC_1903_315,(3,2,0):C.UVGC_1901_308,(3,2,2):C.UVGC_1901_309,(3,2,3):C.UVGC_1901_310,(3,2,5):C.UVGC_1901_311,(3,2,1):C.UVGC_1945_388,(3,2,4):C.UVGC_2272_755,(3,1,0):C.UVGC_1901_308,(3,1,2):C.UVGC_1901_309,(3,1,3):C.UVGC_1901_310,(3,1,5):C.UVGC_1901_311,(3,1,1):C.UVGC_1948_391,(3,1,4):C.UVGC_2278_757,(1,3,0):C.UVGC_1901_308,(1,3,2):C.UVGC_1901_309,(1,3,3):C.UVGC_1901_310,(1,3,5):C.UVGC_1901_311,(1,3,1):C.UVGC_1945_388,(1,3,4):C.UVGC_2272_755,(1,2,0):C.UVGC_1901_308,(1,2,2):C.UVGC_1901_309,(1,2,3):C.UVGC_1901_310,(1,2,5):C.UVGC_1901_311,(1,2,1):C.UVGC_1948_391,(1,2,4):C.UVGC_2278_757,(1,1,0):C.UVGC_1903_312,(1,1,2):C.UVGC_1903_313,(1,1,3):C.UVGC_1903_314,(1,1,5):C.UVGC_1903_315,(1,1,1):C.UVGC_2265_752,(1,1,4):C.UVGC_1947_390,(2,3,0):C.UVGC_1903_312,(2,3,2):C.UVGC_1903_313,(2,3,3):C.UVGC_1903_314,(2,3,5):C.UVGC_1903_315,(2,3,1):C.UVGC_2265_752,(2,3,4):C.UVGC_1947_390,(2,2,0):C.UVGC_1901_308,(2,2,2):C.UVGC_1901_309,(2,2,3):C.UVGC_1901_310,(2,2,5):C.UVGC_1901_311,(2,2,1):C.UVGC_1948_391,(2,2,4):C.UVGC_2278_757,(2,1,0):C.UVGC_1901_308,(2,1,2):C.UVGC_1901_309,(2,1,3):C.UVGC_1901_310,(2,1,5):C.UVGC_1901_311,(2,1,1):C.UVGC_1945_388,(2,1,4):C.UVGC_2272_755,(0,3,0):C.UVGC_1901_308,(0,3,2):C.UVGC_1901_309,(0,3,3):C.UVGC_1901_310,(0,3,5):C.UVGC_1901_311,(0,3,1):C.UVGC_1948_391,(0,3,4):C.UVGC_2278_757,(0,2,0):C.UVGC_1903_312,(0,2,2):C.UVGC_1903_313,(0,2,3):C.UVGC_1903_314,(0,2,5):C.UVGC_1903_315,(0,2,1):C.UVGC_2265_752,(0,2,4):C.UVGC_1947_390,(0,1,0):C.UVGC_1901_308,(0,1,2):C.UVGC_1901_309,(0,1,3):C.UVGC_1901_310,(0,1,5):C.UVGC_1901_311,(0,1,1):C.UVGC_1948_391,(0,1,4):C.UVGC_2278_757,(5,0,1):C.UVGC_1946_389,(4,0,1):C.UVGC_1947_390,(3,0,1):C.UVGC_1947_390,(3,0,4):C.UVGC_1945_388,(1,0,1):C.UVGC_1946_389,(1,0,4):C.UVGC_2274_756,(2,0,1):C.UVGC_1946_389,(2,0,4):C.UVGC_2274_756,(0,0,1):C.UVGC_1947_390,(0,0,4):C.UVGC_1945_388,(5,4,4):C.UVGC_2274_756,(4,4,4):C.UVGC_1945_388,(5,5,4):C.UVGC_2272_755,(4,6,4):C.UVGC_2272_755,(5,7,4):C.UVGC_2272_755,(4,7,4):C.UVGC_2278_757})

V_940 = CTVertex(name = 'V_940',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.g, P.g, P.g ],
                 color = [ 'T(3,-1,1)*T(4,-2,-1)*T(5,2,-2)', 'T(3,-1,1)*T(4,2,-2)*T(5,-2,-1)', 'T(3,-2,-1)*T(4,-1,1)*T(5,2,-2)', 'T(3,-2,-1)*T(4,2,-2)*T(5,-1,1)', 'T(3,2,-2)*T(4,-1,1)*T(5,-2,-1)', 'T(3,2,-2)*T(4,-2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV84, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.G__plus__, P.t] ], [ [P.t] ] ],
                 couplings = {(5,6,0):C.UVGC_1895_296,(5,6,1):C.UVGC_1895_297,(5,6,2):C.UVGC_1895_298,(5,6,4):C.UVGC_1895_299,(5,6,3):C.UVGC_1945_388,(5,5,0):C.UVGC_1897_300,(5,5,1):C.UVGC_1897_301,(5,5,2):C.UVGC_1897_302,(5,5,4):C.UVGC_1897_303,(5,4,0):C.UVGC_1895_296,(5,4,1):C.UVGC_1895_297,(5,4,2):C.UVGC_1895_298,(5,4,4):C.UVGC_1895_299,(5,4,3):C.UVGC_1945_388,(4,6,0):C.UVGC_1895_296,(4,6,1):C.UVGC_1895_297,(4,6,2):C.UVGC_1895_298,(4,6,4):C.UVGC_1895_299,(4,6,3):C.UVGC_1948_391,(4,5,0):C.UVGC_1895_296,(4,5,1):C.UVGC_1895_297,(4,5,2):C.UVGC_1895_298,(4,5,4):C.UVGC_1895_299,(4,5,3):C.UVGC_1945_388,(4,4,0):C.UVGC_1897_300,(4,4,1):C.UVGC_1897_301,(4,4,2):C.UVGC_1897_302,(4,4,4):C.UVGC_1897_303,(3,6,0):C.UVGC_1897_300,(3,6,1):C.UVGC_1897_301,(3,6,2):C.UVGC_1897_302,(3,6,4):C.UVGC_1897_303,(3,5,0):C.UVGC_1895_296,(3,5,1):C.UVGC_1895_297,(3,5,2):C.UVGC_1895_298,(3,5,4):C.UVGC_1895_299,(3,5,3):C.UVGC_1945_388,(3,4,0):C.UVGC_1895_296,(3,4,1):C.UVGC_1895_297,(3,4,2):C.UVGC_1895_298,(3,4,4):C.UVGC_1895_299,(3,4,3):C.UVGC_1948_391,(1,6,0):C.UVGC_1895_296,(1,6,1):C.UVGC_1895_297,(1,6,2):C.UVGC_1895_298,(1,6,4):C.UVGC_1895_299,(1,6,3):C.UVGC_1945_388,(1,5,0):C.UVGC_1895_296,(1,5,1):C.UVGC_1895_297,(1,5,2):C.UVGC_1895_298,(1,5,4):C.UVGC_1895_299,(1,5,3):C.UVGC_1948_391,(1,4,0):C.UVGC_1897_300,(1,4,1):C.UVGC_1897_301,(1,4,2):C.UVGC_1897_302,(1,4,4):C.UVGC_1897_303,(1,4,3):C.UVGC_2265_752,(2,6,0):C.UVGC_1897_300,(2,6,1):C.UVGC_1897_301,(2,6,2):C.UVGC_1897_302,(2,6,4):C.UVGC_1897_303,(2,6,3):C.UVGC_2265_752,(2,5,0):C.UVGC_1895_296,(2,5,1):C.UVGC_1895_297,(2,5,2):C.UVGC_1895_298,(2,5,4):C.UVGC_1895_299,(2,5,3):C.UVGC_1948_391,(2,4,0):C.UVGC_1895_296,(2,4,1):C.UVGC_1895_297,(2,4,2):C.UVGC_1895_298,(2,4,4):C.UVGC_1895_299,(2,4,3):C.UVGC_1945_388,(0,6,0):C.UVGC_1895_296,(0,6,1):C.UVGC_1895_297,(0,6,2):C.UVGC_1895_298,(0,6,4):C.UVGC_1895_299,(0,6,3):C.UVGC_1948_391,(0,5,0):C.UVGC_1897_300,(0,5,1):C.UVGC_1897_301,(0,5,2):C.UVGC_1897_302,(0,5,4):C.UVGC_1897_303,(0,5,3):C.UVGC_2265_752,(0,4,0):C.UVGC_1895_296,(0,4,1):C.UVGC_1895_297,(0,4,2):C.UVGC_1895_298,(0,4,4):C.UVGC_1895_299,(0,4,3):C.UVGC_1948_391,(5,3,3):C.UVGC_1946_389,(4,3,3):C.UVGC_1947_390,(3,3,3):C.UVGC_1947_390,(1,3,3):C.UVGC_1946_389,(2,3,3):C.UVGC_1946_389,(0,3,3):C.UVGC_1947_390,(5,2,0):C.UVGC_1786_149,(5,2,4):C.UVGC_1786_150,(5,1,0):C.UVGC_1787_151,(5,1,4):C.UVGC_1787_152,(5,0,0):C.UVGC_1786_149,(5,0,4):C.UVGC_1786_150,(4,2,0):C.UVGC_1786_149,(4,2,4):C.UVGC_1786_150,(4,1,0):C.UVGC_1786_149,(4,1,4):C.UVGC_1786_150,(4,0,0):C.UVGC_1787_151,(4,0,4):C.UVGC_1787_152,(3,2,0):C.UVGC_1787_151,(3,2,4):C.UVGC_1787_152,(3,1,0):C.UVGC_1786_149,(3,1,4):C.UVGC_1786_150,(3,0,0):C.UVGC_1786_149,(3,0,4):C.UVGC_1786_150,(1,2,0):C.UVGC_1786_149,(1,2,4):C.UVGC_1786_150,(1,1,0):C.UVGC_1786_149,(1,1,4):C.UVGC_1786_150,(1,0,0):C.UVGC_1787_151,(1,0,4):C.UVGC_1787_152,(2,2,0):C.UVGC_1787_151,(2,2,4):C.UVGC_1787_152,(2,1,0):C.UVGC_1786_149,(2,1,4):C.UVGC_1786_150,(2,0,0):C.UVGC_1786_149,(2,0,4):C.UVGC_1786_150,(0,2,0):C.UVGC_1786_149,(0,2,4):C.UVGC_1786_150,(0,1,0):C.UVGC_1787_151,(0,1,4):C.UVGC_1787_152,(0,0,0):C.UVGC_1786_149,(0,0,4):C.UVGC_1786_150})

V_941 = CTVertex(name = 'V_941',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.g, P.g ],
                 color = [ 'T(4,-1,1)*T(5,2,-1)', 'T(4,2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV146, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV79, L.FFVVV80, L.FFVVV81, L.FFVVV87 ],
                 loop_particles = [ [ [P.b, P.G__plus__] ], [ [P.G0, P.t], [P.H, P.t] ] ],
                 couplings = {(1,7,1):C.UVGC_1944_387,(0,7,1):C.UVGC_1943_386,(1,3,0):C.UVGC_1943_386,(1,3,1):C.UVGC_1944_387,(0,3,0):C.UVGC_1944_387,(0,3,1):C.UVGC_1943_386,(1,2,0):C.UVGC_1944_387,(0,2,0):C.UVGC_1943_386,(1,1,0):C.UVGC_1943_386,(0,1,0):C.UVGC_1944_387,(1,0,0):C.UVGC_1944_387,(0,0,0):C.UVGC_1943_386,(1,4,1):C.UVGC_1943_386,(0,4,1):C.UVGC_1944_387,(1,5,1):C.UVGC_1944_387,(0,5,1):C.UVGC_1943_386,(1,6,1):C.UVGC_1943_386,(0,6,1):C.UVGC_1944_387})

V_942 = CTVertex(name = 'V_942',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.a, P.g, P.g ],
                 color = [ 'T(4,-1,1)*T(5,2,-1)', 'T(4,2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV84, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.G__plus__, P.t] ] ],
                 couplings = {(1,3,0):C.UVGC_2260_751,(0,3,0):C.UVGC_2259_750,(1,2,0):C.UVGC_2259_750,(0,2,0):C.UVGC_2260_751,(1,1,0):C.UVGC_2260_751,(0,1,0):C.UVGC_2259_750,(1,0,0):C.UVGC_2259_750,(0,0,0):C.UVGC_2260_751})

V_943 = CTVertex(name = 'V_943',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.g, P.Z ],
                 color = [ 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVV145, L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV83, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b, P.G__plus__] ], [ [P.G0, P.t], [P.H, P.t] ] ],
                 couplings = {(1,7,1):C.UVGC_2282_758,(0,7,1):C.UVGC_2284_761,(1,6,1):C.UVGC_2284_761,(0,6,1):C.UVGC_2282_758,(1,5,1):C.UVGC_2284_761,(0,5,1):C.UVGC_2282_758,(1,4,1):C.UVGC_2282_758,(0,4,1):C.UVGC_2284_761,(1,3,0):C.UVGC_2283_759,(1,3,1):C.UVGC_2283_760,(0,3,0):C.UVGC_2285_762,(0,3,1):C.UVGC_2285_763,(1,2,0):C.UVGC_2285_762,(1,2,1):C.UVGC_2285_763,(0,2,0):C.UVGC_2283_759,(0,2,1):C.UVGC_2283_760,(1,1,0):C.UVGC_2285_762,(1,1,1):C.UVGC_2285_763,(0,1,0):C.UVGC_2283_759,(0,1,1):C.UVGC_2283_760,(1,0,0):C.UVGC_2283_759,(1,0,1):C.UVGC_2283_760,(0,0,0):C.UVGC_2285_762,(0,0,1):C.UVGC_2285_763})

V_944 = CTVertex(name = 'V_944',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.g, P.g, P.Z ],
                 color = [ 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVVV83, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.G__plus__, P.t] ] ],
                 couplings = {(1,3,0):C.UVGC_2266_753,(0,3,0):C.UVGC_2267_754,(1,2,0):C.UVGC_2267_754,(0,2,0):C.UVGC_2266_753,(1,1,0):C.UVGC_2267_754,(0,1,0):C.UVGC_2266_753,(1,0,0):C.UVGC_2266_753,(0,0,0):C.UVGC_2267_754})

V_945 = CTVertex(name = 'V_945',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133, L.FFVV135, L.FFVV137 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,1):C.UVGC_1964_400,(0,1,0):C.UVGC_1966_402,(0,2,0):C.UVGC_1963_399})

V_946 = CTVertex(name = 'V_946',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1981_417})

V_947 = CTVertex(name = 'V_947',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1981_417})

V_948 = CTVertex(name = 'V_948',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1970_406})

V_949 = CTVertex(name = 'V_949',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1970_406})

V_950 = CTVertex(name = 'V_950',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1982_418})

V_951 = CTVertex(name = 'V_951',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1982_418})

V_952 = CTVertex(name = 'V_952',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1971_407})

V_953 = CTVertex(name = 'V_953',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1971_407})

V_954 = CTVertex(name = 'V_954',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1964_400})

V_955 = CTVertex(name = 'V_955',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1964_400})

V_956 = CTVertex(name = 'V_956',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1965_401})

V_957 = CTVertex(name = 'V_957',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV133 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1965_401})

V_958 = CTVertex(name = 'V_958',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.g, P.g ],
                 color = [ 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV101, L.FFVV102, L.FFVV108, L.FFVV109, L.FFVV110, L.FFVV111, L.FFVV112, L.FFVV113, L.FFVV114, L.FFVV115, L.FFVV131, L.FFVV132, L.FFVV158, L.FFVV159, L.FFVV160, L.FFVV162, L.FFVV163, L.FFVV165, L.FFVV189, L.FFVV190 ],
                 loop_particles = [ [ [P.b] ], [ [P.c], [P.u] ], [ [P.d], [P.s] ], [ [P.t] ] ],
                 couplings = {(1,2,1):C.UVGC_1872_267,(1,2,2):C.UVGC_1872_268,(1,2,3):C.UVGC_1773_132,(1,3,1):C.UVGC_1872_267,(1,3,2):C.UVGC_1872_268,(1,3,3):C.UVGC_1773_132,(1,4,0):C.UVGC_1871_265,(1,4,1):C.UVGC_1867_261,(1,4,2):C.UVGC_1867_262,(1,4,3):C.UVGC_1871_266,(1,5,0):C.UVGC_1768_128,(1,5,3):C.UVGC_1768_127,(0,2,1):C.UVGC_1867_261,(0,2,2):C.UVGC_1867_262,(0,2,3):C.UVGC_1772_131,(0,3,1):C.UVGC_1867_261,(0,3,2):C.UVGC_1867_262,(0,3,3):C.UVGC_1772_131,(0,4,0):C.UVGC_1876_271,(0,4,1):C.UVGC_1872_267,(0,4,2):C.UVGC_1872_268,(0,4,3):C.UVGC_1876_272,(0,5,0):C.UVGC_1769_130,(0,5,3):C.UVGC_1769_129,(1,6,0):C.UVGC_1750_105,(1,6,1):C.UVGC_1867_261,(1,6,2):C.UVGC_1867_262,(1,6,3):C.UVGC_1772_131,(1,7,0):C.UVGC_1750_105,(1,7,1):C.UVGC_1867_261,(1,7,2):C.UVGC_1867_262,(1,7,3):C.UVGC_1772_131,(1,8,0):C.UVGC_1769_130,(1,8,3):C.UVGC_1769_129,(1,9,0):C.UVGC_1876_271,(1,9,1):C.UVGC_1872_267,(1,9,2):C.UVGC_1872_268,(1,9,3):C.UVGC_1876_272,(0,6,0):C.UVGC_1751_106,(0,6,1):C.UVGC_1872_267,(0,6,2):C.UVGC_1872_268,(0,6,3):C.UVGC_1773_132,(0,7,0):C.UVGC_1751_106,(0,7,1):C.UVGC_1872_267,(0,7,2):C.UVGC_1872_268,(0,7,3):C.UVGC_1773_132,(0,8,0):C.UVGC_1768_128,(0,8,3):C.UVGC_1768_127,(0,9,0):C.UVGC_1871_265,(0,9,1):C.UVGC_1867_261,(0,9,2):C.UVGC_1867_262,(0,9,3):C.UVGC_1871_266,(1,12,3):C.UVGC_1776_134,(1,13,3):C.UVGC_1776_134,(1,14,0):C.UVGC_1750_105,(1,14,3):C.UVGC_1774_133,(0,12,3):C.UVGC_1774_133,(0,13,3):C.UVGC_1774_133,(0,14,0):C.UVGC_1751_106,(0,14,3):C.UVGC_1776_134,(1,15,0):C.UVGC_1750_105,(1,15,3):C.UVGC_1774_133,(1,16,0):C.UVGC_1750_105,(1,16,3):C.UVGC_1774_133,(1,17,0):C.UVGC_1751_106,(1,17,3):C.UVGC_1776_134,(0,15,0):C.UVGC_1751_106,(0,15,3):C.UVGC_1776_134,(0,16,0):C.UVGC_1751_106,(0,16,3):C.UVGC_1776_134,(0,17,0):C.UVGC_1750_105,(0,17,3):C.UVGC_1774_133,(1,10,0):C.UVGC_1874_269,(1,10,1):C.UVGC_1872_267,(1,10,2):C.UVGC_1872_268,(1,10,3):C.UVGC_1874_270,(0,10,0):C.UVGC_1869_263,(0,10,1):C.UVGC_1867_261,(0,10,2):C.UVGC_1867_262,(0,10,3):C.UVGC_1869_264,(1,18,0):C.UVGC_1751_106,(1,18,3):C.UVGC_1776_134,(0,18,0):C.UVGC_1750_105,(0,18,3):C.UVGC_1774_133,(1,11,0):C.UVGC_1869_263,(1,11,1):C.UVGC_1867_261,(1,11,2):C.UVGC_1867_262,(1,11,3):C.UVGC_1869_264,(0,11,0):C.UVGC_1874_269,(0,11,1):C.UVGC_1872_267,(0,11,2):C.UVGC_1872_268,(0,11,3):C.UVGC_1874_270,(1,19,0):C.UVGC_1750_105,(1,19,3):C.UVGC_1774_133,(0,19,0):C.UVGC_1751_106,(0,19,3):C.UVGC_1776_134,(1,0,0):C.UVGC_1751_106,(0,0,0):C.UVGC_1750_105,(1,1,0):C.UVGC_1751_106,(0,1,0):C.UVGC_1750_105})

V_959 = CTVertex(name = 'V_959',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.g, P.g ],
                 color = [ 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV108, L.FFVV109, L.FFVV110, L.FFVV112, L.FFVV113, L.FFVV115, L.FFVV131, L.FFVV132, L.FFVV158, L.FFVV159, L.FFVV160, L.FFVV162, L.FFVV163, L.FFVV165, L.FFVV189, L.FFVV190 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(1,0,0):C.UVGC_1781_141,(1,0,1):C.UVGC_1781_142,(1,1,0):C.UVGC_1781_141,(1,1,1):C.UVGC_1781_142,(1,2,0):C.UVGC_1778_135,(1,2,1):C.UVGC_1778_136,(0,0,0):C.UVGC_1778_135,(0,0,1):C.UVGC_1778_136,(0,1,0):C.UVGC_1778_135,(0,1,1):C.UVGC_1778_136,(0,2,0):C.UVGC_1781_141,(0,2,1):C.UVGC_1781_142,(1,3,0):C.UVGC_1778_135,(1,3,1):C.UVGC_1778_136,(1,4,0):C.UVGC_1778_135,(1,4,1):C.UVGC_1778_136,(1,5,0):C.UVGC_1781_141,(1,5,1):C.UVGC_1781_142,(0,3,0):C.UVGC_1781_141,(0,3,1):C.UVGC_1781_142,(0,4,0):C.UVGC_1781_141,(0,4,1):C.UVGC_1781_142,(0,5,0):C.UVGC_1778_135,(0,5,1):C.UVGC_1778_136,(1,8,0):C.UVGC_1785_147,(1,8,1):C.UVGC_1785_148,(1,9,0):C.UVGC_1785_147,(1,9,1):C.UVGC_1785_148,(1,10,0):C.UVGC_1784_145,(1,10,1):C.UVGC_1784_146,(0,8,0):C.UVGC_1784_145,(0,8,1):C.UVGC_1784_146,(0,9,0):C.UVGC_1784_145,(0,9,1):C.UVGC_1784_146,(0,10,0):C.UVGC_1785_147,(0,10,1):C.UVGC_1785_148,(1,11,0):C.UVGC_1784_145,(1,11,1):C.UVGC_1784_146,(1,12,0):C.UVGC_1784_145,(1,12,1):C.UVGC_1784_146,(1,13,0):C.UVGC_1785_147,(1,13,1):C.UVGC_1785_148,(0,11,0):C.UVGC_1785_147,(0,11,1):C.UVGC_1785_148,(0,12,0):C.UVGC_1785_147,(0,12,1):C.UVGC_1785_148,(0,13,0):C.UVGC_1784_145,(0,13,1):C.UVGC_1784_146,(1,6,0):C.UVGC_1781_141,(1,6,1):C.UVGC_1781_142,(0,6,0):C.UVGC_1778_135,(0,6,1):C.UVGC_1778_136,(1,14,0):C.UVGC_1785_147,(1,14,1):C.UVGC_1785_148,(0,14,0):C.UVGC_1784_145,(0,14,1):C.UVGC_1784_146,(1,7,0):C.UVGC_1778_135,(1,7,1):C.UVGC_1778_136,(0,7,0):C.UVGC_1781_141,(0,7,1):C.UVGC_1781_142,(1,15,0):C.UVGC_1784_145,(1,15,1):C.UVGC_1784_146,(0,15,0):C.UVGC_1785_147,(0,15,1):C.UVGC_1785_148})

V_960 = CTVertex(name = 'V_960',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.g, P.g ],
                 color = [ 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV108, L.FFVV109, L.FFVV110, L.FFVV112, L.FFVV113, L.FFVV115, L.FFVV131, L.FFVV132, L.FFVV158, L.FFVV159, L.FFVV160, L.FFVV162, L.FFVV163, L.FFVV165, L.FFVV189, L.FFVV190 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(1,0,0):C.UVGC_1781_141,(1,0,1):C.UVGC_1781_142,(1,1,0):C.UVGC_1781_141,(1,1,1):C.UVGC_1781_142,(1,2,0):C.UVGC_1778_135,(1,2,1):C.UVGC_1778_136,(0,0,0):C.UVGC_1778_135,(0,0,1):C.UVGC_1778_136,(0,1,0):C.UVGC_1778_135,(0,1,1):C.UVGC_1778_136,(0,2,0):C.UVGC_1781_141,(0,2,1):C.UVGC_1781_142,(1,3,0):C.UVGC_1778_135,(1,3,1):C.UVGC_1778_136,(1,4,0):C.UVGC_1778_135,(1,4,1):C.UVGC_1778_136,(1,5,0):C.UVGC_1781_141,(1,5,1):C.UVGC_1781_142,(0,3,0):C.UVGC_1781_141,(0,3,1):C.UVGC_1781_142,(0,4,0):C.UVGC_1781_141,(0,4,1):C.UVGC_1781_142,(0,5,0):C.UVGC_1778_135,(0,5,1):C.UVGC_1778_136,(1,8,0):C.UVGC_1785_147,(1,8,1):C.UVGC_1785_148,(1,9,0):C.UVGC_1785_147,(1,9,1):C.UVGC_1785_148,(1,10,0):C.UVGC_1784_145,(1,10,1):C.UVGC_1784_146,(0,8,0):C.UVGC_1784_145,(0,8,1):C.UVGC_1784_146,(0,9,0):C.UVGC_1784_145,(0,9,1):C.UVGC_1784_146,(0,10,0):C.UVGC_1785_147,(0,10,1):C.UVGC_1785_148,(1,11,0):C.UVGC_1784_145,(1,11,1):C.UVGC_1784_146,(1,12,0):C.UVGC_1784_145,(1,12,1):C.UVGC_1784_146,(1,13,0):C.UVGC_1785_147,(1,13,1):C.UVGC_1785_148,(0,11,0):C.UVGC_1785_147,(0,11,1):C.UVGC_1785_148,(0,12,0):C.UVGC_1785_147,(0,12,1):C.UVGC_1785_148,(0,13,0):C.UVGC_1784_145,(0,13,1):C.UVGC_1784_146,(1,6,0):C.UVGC_1781_141,(1,6,1):C.UVGC_1781_142,(0,6,0):C.UVGC_1778_135,(0,6,1):C.UVGC_1778_136,(1,14,0):C.UVGC_1785_147,(1,14,1):C.UVGC_1785_148,(0,14,0):C.UVGC_1784_145,(0,14,1):C.UVGC_1784_146,(1,7,0):C.UVGC_1778_135,(1,7,1):C.UVGC_1778_136,(0,7,0):C.UVGC_1781_141,(0,7,1):C.UVGC_1781_142,(1,15,0):C.UVGC_1784_145,(1,15,1):C.UVGC_1784_146,(0,15,0):C.UVGC_1785_147,(0,15,1):C.UVGC_1785_148})

V_961 = CTVertex(name = 'V_961',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.g, P.g ],
                 color = [ 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV108, L.FFVV109, L.FFVV110, L.FFVV112, L.FFVV113, L.FFVV115, L.FFVV131, L.FFVV132, L.FFVV158, L.FFVV159, L.FFVV160, L.FFVV162, L.FFVV163, L.FFVV165, L.FFVV189, L.FFVV190 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(1,0,0):C.UVGC_1780_139,(1,0,1):C.UVGC_1780_140,(1,1,0):C.UVGC_1780_139,(1,1,1):C.UVGC_1780_140,(1,2,0):C.UVGC_1779_137,(1,2,1):C.UVGC_1779_138,(0,0,0):C.UVGC_1779_137,(0,0,1):C.UVGC_1779_138,(0,1,0):C.UVGC_1779_137,(0,1,1):C.UVGC_1779_138,(0,2,0):C.UVGC_1780_139,(0,2,1):C.UVGC_1780_140,(1,3,0):C.UVGC_1779_137,(1,3,1):C.UVGC_1779_138,(1,4,0):C.UVGC_1779_137,(1,4,1):C.UVGC_1779_138,(1,5,0):C.UVGC_1780_139,(1,5,1):C.UVGC_1780_140,(0,3,0):C.UVGC_1780_139,(0,3,1):C.UVGC_1780_140,(0,4,0):C.UVGC_1780_139,(0,4,1):C.UVGC_1780_140,(0,5,0):C.UVGC_1779_137,(0,5,1):C.UVGC_1779_138,(1,8,0):C.UVGC_1751_106,(1,8,1):C.UVGC_1776_134,(1,9,0):C.UVGC_1751_106,(1,9,1):C.UVGC_1776_134,(1,10,0):C.UVGC_1750_105,(1,10,1):C.UVGC_1774_133,(0,8,0):C.UVGC_1750_105,(0,8,1):C.UVGC_1774_133,(0,9,0):C.UVGC_1750_105,(0,9,1):C.UVGC_1774_133,(0,10,0):C.UVGC_1751_106,(0,10,1):C.UVGC_1776_134,(1,11,0):C.UVGC_1750_105,(1,11,1):C.UVGC_1774_133,(1,12,0):C.UVGC_1750_105,(1,12,1):C.UVGC_1774_133,(1,13,0):C.UVGC_1751_106,(1,13,1):C.UVGC_1776_134,(0,11,0):C.UVGC_1751_106,(0,11,1):C.UVGC_1776_134,(0,12,0):C.UVGC_1751_106,(0,12,1):C.UVGC_1776_134,(0,13,0):C.UVGC_1750_105,(0,13,1):C.UVGC_1774_133,(1,6,0):C.UVGC_1780_139,(1,6,1):C.UVGC_1780_140,(0,6,0):C.UVGC_1779_137,(0,6,1):C.UVGC_1779_138,(1,14,0):C.UVGC_1751_106,(1,14,1):C.UVGC_1776_134,(0,14,0):C.UVGC_1750_105,(0,14,1):C.UVGC_1774_133,(1,7,0):C.UVGC_1779_137,(1,7,1):C.UVGC_1779_138,(0,7,0):C.UVGC_1780_139,(0,7,1):C.UVGC_1780_140,(1,15,0):C.UVGC_1750_105,(1,15,1):C.UVGC_1774_133,(0,15,0):C.UVGC_1751_106,(0,15,1):C.UVGC_1776_134})

V_962 = CTVertex(name = 'V_962',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.g, P.g ],
                 color = [ 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV108, L.FFVV109, L.FFVV110, L.FFVV112, L.FFVV113, L.FFVV115, L.FFVV131, L.FFVV132, L.FFVV158, L.FFVV159, L.FFVV160, L.FFVV162, L.FFVV163, L.FFVV165, L.FFVV189, L.FFVV190 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(1,0,0):C.UVGC_1780_139,(1,0,1):C.UVGC_1780_140,(1,1,0):C.UVGC_1780_139,(1,1,1):C.UVGC_1780_140,(1,2,0):C.UVGC_1779_137,(1,2,1):C.UVGC_1779_138,(0,0,0):C.UVGC_1779_137,(0,0,1):C.UVGC_1779_138,(0,1,0):C.UVGC_1779_137,(0,1,1):C.UVGC_1779_138,(0,2,0):C.UVGC_1780_139,(0,2,1):C.UVGC_1780_140,(1,3,0):C.UVGC_1779_137,(1,3,1):C.UVGC_1779_138,(1,4,0):C.UVGC_1779_137,(1,4,1):C.UVGC_1779_138,(1,5,0):C.UVGC_1780_139,(1,5,1):C.UVGC_1780_140,(0,3,0):C.UVGC_1780_139,(0,3,1):C.UVGC_1780_140,(0,4,0):C.UVGC_1780_139,(0,4,1):C.UVGC_1780_140,(0,5,0):C.UVGC_1779_137,(0,5,1):C.UVGC_1779_138,(1,8,0):C.UVGC_1751_106,(1,8,1):C.UVGC_1776_134,(1,9,0):C.UVGC_1751_106,(1,9,1):C.UVGC_1776_134,(1,10,0):C.UVGC_1750_105,(1,10,1):C.UVGC_1774_133,(0,8,0):C.UVGC_1750_105,(0,8,1):C.UVGC_1774_133,(0,9,0):C.UVGC_1750_105,(0,9,1):C.UVGC_1774_133,(0,10,0):C.UVGC_1751_106,(0,10,1):C.UVGC_1776_134,(1,11,0):C.UVGC_1750_105,(1,11,1):C.UVGC_1774_133,(1,12,0):C.UVGC_1750_105,(1,12,1):C.UVGC_1774_133,(1,13,0):C.UVGC_1751_106,(1,13,1):C.UVGC_1776_134,(0,11,0):C.UVGC_1751_106,(0,11,1):C.UVGC_1776_134,(0,12,0):C.UVGC_1751_106,(0,12,1):C.UVGC_1776_134,(0,13,0):C.UVGC_1750_105,(0,13,1):C.UVGC_1774_133,(1,6,0):C.UVGC_1780_139,(1,6,1):C.UVGC_1780_140,(0,6,0):C.UVGC_1779_137,(0,6,1):C.UVGC_1779_138,(1,14,0):C.UVGC_1751_106,(1,14,1):C.UVGC_1776_134,(0,14,0):C.UVGC_1750_105,(0,14,1):C.UVGC_1774_133,(1,7,0):C.UVGC_1779_137,(1,7,1):C.UVGC_1779_138,(0,7,0):C.UVGC_1780_139,(0,7,1):C.UVGC_1780_140,(1,15,0):C.UVGC_1750_105,(1,15,1):C.UVGC_1774_133,(0,15,0):C.UVGC_1751_106,(0,15,1):C.UVGC_1776_134})

V_963 = CTVertex(name = 'V_963',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.a, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV114 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2299_771,(0,0,1):C.UVGC_1985_421})

V_964 = CTVertex(name = 'V_964',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.a, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV93 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2295_769,(0,0,1):C.UVGC_1976_412})

V_965 = CTVertex(name = 'V_965',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV101 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2288_766,(0,0,1):C.UVGC_1961_397})

V_966 = CTVertex(name = 'V_966',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.a, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV114 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2299_771,(0,0,1):C.UVGC_1985_421})

V_967 = CTVertex(name = 'V_967',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.a, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV93 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2295_769,(0,0,1):C.UVGC_1976_412})

V_968 = CTVertex(name = 'V_968',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV101 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2288_766,(0,0,1):C.UVGC_1961_397})

V_969 = CTVertex(name = 'V_969',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV101 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2293_767,(0,0,1):C.UVGC_1974_410})

V_970 = CTVertex(name = 'V_970',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.a, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV101 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2294_768,(0,0,1):C.UVGC_1975_411})

V_971 = CTVertex(name = 'V_971',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV114 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2301_773,(0,0,1):C.UVGC_1987_423})

V_972 = CTVertex(name = 'V_972',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.W__minus__, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV114 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2300_772,(0,0,1):C.UVGC_1986_422})

V_973 = CTVertex(name = 'V_973',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__minus__, P.W__plus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV101 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2287_765,(0,0,1):C.UVGC_1960_396})

V_974 = CTVertex(name = 'V_974',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV114 ],
                 loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.s], [P.d, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_2286_764,(0,0,1):C.UVGC_1959_395})

V_975 = CTVertex(name = 'V_975',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.a, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV114 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1985_421})

V_976 = CTVertex(name = 'V_976',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.a, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV114 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1985_421})

V_977 = CTVertex(name = 'V_977',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.a, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV93 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1976_412})

V_978 = CTVertex(name = 'V_978',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.a, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV93 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1976_412})

V_979 = CTVertex(name = 'V_979',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV101 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1961_397})

V_980 = CTVertex(name = 'V_980',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV101 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1961_397})

V_981 = CTVertex(name = 'V_981',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.a, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV114 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1985_421})

V_982 = CTVertex(name = 'V_982',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.a, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV114 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1985_421})

V_983 = CTVertex(name = 'V_983',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.a, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV93 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1976_412})

V_984 = CTVertex(name = 'V_984',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.a, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV93 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1976_412})

V_985 = CTVertex(name = 'V_985',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV101 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1961_397})

V_986 = CTVertex(name = 'V_986',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV101 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1961_397})

V_987 = CTVertex(name = 'V_987',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.a, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV101 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1975_411})

V_988 = CTVertex(name = 'V_988',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.a, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV101 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1975_411})

V_989 = CTVertex(name = 'V_989',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.a, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV101 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1974_410})

V_990 = CTVertex(name = 'V_990',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.a, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV101 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1974_410})

V_991 = CTVertex(name = 'V_991',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.W__minus__, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV114 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1986_422})

V_992 = CTVertex(name = 'V_992',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.W__minus__, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV114 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1986_422})

V_993 = CTVertex(name = 'V_993',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.W__minus__, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV114 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1987_423})

V_994 = CTVertex(name = 'V_994',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.W__minus__, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV114 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1987_423})

V_995 = CTVertex(name = 'V_995',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.W__minus__, P.W__plus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV101 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1960_396})

V_996 = CTVertex(name = 'V_996',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.W__minus__, P.W__plus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV101 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1960_396})

V_997 = CTVertex(name = 'V_997',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV114 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1959_395})

V_998 = CTVertex(name = 'V_998',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVVV114 ],
                 loop_particles = [ [ [P.b, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1959_395})

V_999 = CTVertex(name = 'V_999',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.g, P.g, P.g ],
                 color = [ 'T(3,-1,1)*T(4,-2,-1)*T(5,2,-2)', 'T(3,-1,1)*T(4,2,-2)*T(5,-2,-1)', 'T(3,-2,-1)*T(4,-1,1)*T(5,2,-2)', 'T(3,-2,-1)*T(4,2,-2)*T(5,-1,1)', 'T(3,2,-2)*T(4,-1,1)*T(5,-2,-1)', 'T(3,2,-2)*T(4,-2,-1)*T(5,-1,1)' ],
                 lorentz = [ L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                 loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                 couplings = {(5,5,0):C.UVGC_1789_155,(5,5,1):C.UVGC_1789_156,(5,4,0):C.UVGC_1790_157,(5,4,1):C.UVGC_1790_158,(5,3,0):C.UVGC_1789_155,(5,3,1):C.UVGC_1789_156,(4,5,0):C.UVGC_1789_155,(4,5,1):C.UVGC_1789_156,(4,4,0):C.UVGC_1789_155,(4,4,1):C.UVGC_1789_156,(4,3,0):C.UVGC_1790_157,(4,3,1):C.UVGC_1790_158,(3,5,0):C.UVGC_1790_157,(3,5,1):C.UVGC_1790_158,(3,4,0):C.UVGC_1789_155,(3,4,1):C.UVGC_1789_156,(3,3,0):C.UVGC_1789_155,(3,3,1):C.UVGC_1789_156,(1,5,0):C.UVGC_1789_155,(1,5,1):C.UVGC_1789_156,(1,4,0):C.UVGC_1789_155,(1,4,1):C.UVGC_1789_156,(1,3,0):C.UVGC_1790_157,(1,3,1):C.UVGC_1790_158,(2,5,0):C.UVGC_1790_157,(2,5,1):C.UVGC_1790_158,(2,4,0):C.UVGC_1789_155,(2,4,1):C.UVGC_1789_156,(2,3,0):C.UVGC_1789_155,(2,3,1):C.UVGC_1789_156,(0,5,0):C.UVGC_1789_155,(0,5,1):C.UVGC_1789_156,(0,4,0):C.UVGC_1790_157,(0,4,1):C.UVGC_1790_158,(0,3,0):C.UVGC_1789_155,(0,3,1):C.UVGC_1789_156,(5,2,0):C.UVGC_1786_149,(5,2,1):C.UVGC_1786_150,(5,1,0):C.UVGC_1787_151,(5,1,1):C.UVGC_1787_152,(5,0,0):C.UVGC_1786_149,(5,0,1):C.UVGC_1786_150,(4,2,0):C.UVGC_1786_149,(4,2,1):C.UVGC_1786_150,(4,1,0):C.UVGC_1786_149,(4,1,1):C.UVGC_1786_150,(4,0,0):C.UVGC_1787_151,(4,0,1):C.UVGC_1787_152,(3,2,0):C.UVGC_1787_151,(3,2,1):C.UVGC_1787_152,(3,1,0):C.UVGC_1786_149,(3,1,1):C.UVGC_1786_150,(3,0,0):C.UVGC_1786_149,(3,0,1):C.UVGC_1786_150,(1,2,0):C.UVGC_1786_149,(1,2,1):C.UVGC_1786_150,(1,1,0):C.UVGC_1786_149,(1,1,1):C.UVGC_1786_150,(1,0,0):C.UVGC_1787_151,(1,0,1):C.UVGC_1787_152,(2,2,0):C.UVGC_1787_151,(2,2,1):C.UVGC_1787_152,(2,1,0):C.UVGC_1786_149,(2,1,1):C.UVGC_1786_150,(2,0,0):C.UVGC_1786_149,(2,0,1):C.UVGC_1786_150,(0,2,0):C.UVGC_1786_149,(0,2,1):C.UVGC_1786_150,(0,1,0):C.UVGC_1787_151,(0,1,1):C.UVGC_1787_152,(0,0,0):C.UVGC_1786_149,(0,0,1):C.UVGC_1786_150})

V_1000 = CTVertex(name = 'V_1000',
                  type = 'UV',
                  particles = [ P.s__tilde__, P.s, P.g, P.g, P.g ],
                  color = [ 'T(3,-1,1)*T(4,-2,-1)*T(5,2,-2)', 'T(3,-1,1)*T(4,2,-2)*T(5,-2,-1)', 'T(3,-2,-1)*T(4,-1,1)*T(5,2,-2)', 'T(3,-2,-1)*T(4,2,-2)*T(5,-1,1)', 'T(3,2,-2)*T(4,-1,1)*T(5,-2,-1)', 'T(3,2,-2)*T(4,-2,-1)*T(5,-1,1)' ],
                  lorentz = [ L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                  loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                  couplings = {(5,5,0):C.UVGC_1789_155,(5,5,1):C.UVGC_1789_156,(5,4,0):C.UVGC_1790_157,(5,4,1):C.UVGC_1790_158,(5,3,0):C.UVGC_1789_155,(5,3,1):C.UVGC_1789_156,(4,5,0):C.UVGC_1789_155,(4,5,1):C.UVGC_1789_156,(4,4,0):C.UVGC_1789_155,(4,4,1):C.UVGC_1789_156,(4,3,0):C.UVGC_1790_157,(4,3,1):C.UVGC_1790_158,(3,5,0):C.UVGC_1790_157,(3,5,1):C.UVGC_1790_158,(3,4,0):C.UVGC_1789_155,(3,4,1):C.UVGC_1789_156,(3,3,0):C.UVGC_1789_155,(3,3,1):C.UVGC_1789_156,(1,5,0):C.UVGC_1789_155,(1,5,1):C.UVGC_1789_156,(1,4,0):C.UVGC_1789_155,(1,4,1):C.UVGC_1789_156,(1,3,0):C.UVGC_1790_157,(1,3,1):C.UVGC_1790_158,(2,5,0):C.UVGC_1790_157,(2,5,1):C.UVGC_1790_158,(2,4,0):C.UVGC_1789_155,(2,4,1):C.UVGC_1789_156,(2,3,0):C.UVGC_1789_155,(2,3,1):C.UVGC_1789_156,(0,5,0):C.UVGC_1789_155,(0,5,1):C.UVGC_1789_156,(0,4,0):C.UVGC_1790_157,(0,4,1):C.UVGC_1790_158,(0,3,0):C.UVGC_1789_155,(0,3,1):C.UVGC_1789_156,(5,2,0):C.UVGC_1786_149,(5,2,1):C.UVGC_1786_150,(5,1,0):C.UVGC_1787_151,(5,1,1):C.UVGC_1787_152,(5,0,0):C.UVGC_1786_149,(5,0,1):C.UVGC_1786_150,(4,2,0):C.UVGC_1786_149,(4,2,1):C.UVGC_1786_150,(4,1,0):C.UVGC_1786_149,(4,1,1):C.UVGC_1786_150,(4,0,0):C.UVGC_1787_151,(4,0,1):C.UVGC_1787_152,(3,2,0):C.UVGC_1787_151,(3,2,1):C.UVGC_1787_152,(3,1,0):C.UVGC_1786_149,(3,1,1):C.UVGC_1786_150,(3,0,0):C.UVGC_1786_149,(3,0,1):C.UVGC_1786_150,(1,2,0):C.UVGC_1786_149,(1,2,1):C.UVGC_1786_150,(1,1,0):C.UVGC_1786_149,(1,1,1):C.UVGC_1786_150,(1,0,0):C.UVGC_1787_151,(1,0,1):C.UVGC_1787_152,(2,2,0):C.UVGC_1787_151,(2,2,1):C.UVGC_1787_152,(2,1,0):C.UVGC_1786_149,(2,1,1):C.UVGC_1786_150,(2,0,0):C.UVGC_1786_149,(2,0,1):C.UVGC_1786_150,(0,2,0):C.UVGC_1786_149,(0,2,1):C.UVGC_1786_150,(0,1,0):C.UVGC_1787_151,(0,1,1):C.UVGC_1787_152,(0,0,0):C.UVGC_1786_149,(0,0,1):C.UVGC_1786_150})

V_1001 = CTVertex(name = 'V_1001',
                  type = 'UV',
                  particles = [ P.u__tilde__, P.u, P.g, P.g, P.g ],
                  color = [ 'T(3,-1,1)*T(4,-2,-1)*T(5,2,-2)', 'T(3,-1,1)*T(4,2,-2)*T(5,-2,-1)', 'T(3,-2,-1)*T(4,-1,1)*T(5,2,-2)', 'T(3,-2,-1)*T(4,2,-2)*T(5,-1,1)', 'T(3,2,-2)*T(4,-1,1)*T(5,-2,-1)', 'T(3,2,-2)*T(4,-2,-1)*T(5,-1,1)' ],
                  lorentz = [ L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                  loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                  couplings = {(5,5,0):C.UVGC_1788_153,(5,5,1):C.UVGC_1788_154,(5,4,0):C.UVGC_1791_159,(5,4,1):C.UVGC_1791_160,(5,3,0):C.UVGC_1788_153,(5,3,1):C.UVGC_1788_154,(4,5,0):C.UVGC_1788_153,(4,5,1):C.UVGC_1788_154,(4,4,0):C.UVGC_1788_153,(4,4,1):C.UVGC_1788_154,(4,3,0):C.UVGC_1791_159,(4,3,1):C.UVGC_1791_160,(3,5,0):C.UVGC_1791_159,(3,5,1):C.UVGC_1791_160,(3,4,0):C.UVGC_1788_153,(3,4,1):C.UVGC_1788_154,(3,3,0):C.UVGC_1788_153,(3,3,1):C.UVGC_1788_154,(1,5,0):C.UVGC_1788_153,(1,5,1):C.UVGC_1788_154,(1,4,0):C.UVGC_1788_153,(1,4,1):C.UVGC_1788_154,(1,3,0):C.UVGC_1791_159,(1,3,1):C.UVGC_1791_160,(2,5,0):C.UVGC_1791_159,(2,5,1):C.UVGC_1791_160,(2,4,0):C.UVGC_1788_153,(2,4,1):C.UVGC_1788_154,(2,3,0):C.UVGC_1788_153,(2,3,1):C.UVGC_1788_154,(0,5,0):C.UVGC_1788_153,(0,5,1):C.UVGC_1788_154,(0,4,0):C.UVGC_1791_159,(0,4,1):C.UVGC_1791_160,(0,3,0):C.UVGC_1788_153,(0,3,1):C.UVGC_1788_154,(5,2,0):C.UVGC_1792_161,(5,2,1):C.UVGC_1792_162,(5,1,0):C.UVGC_1793_163,(5,1,1):C.UVGC_1793_164,(5,0,0):C.UVGC_1792_161,(5,0,1):C.UVGC_1792_162,(4,2,0):C.UVGC_1792_161,(4,2,1):C.UVGC_1792_162,(4,1,0):C.UVGC_1792_161,(4,1,1):C.UVGC_1792_162,(4,0,0):C.UVGC_1793_163,(4,0,1):C.UVGC_1793_164,(3,2,0):C.UVGC_1793_163,(3,2,1):C.UVGC_1793_164,(3,1,0):C.UVGC_1792_161,(3,1,1):C.UVGC_1792_162,(3,0,0):C.UVGC_1792_161,(3,0,1):C.UVGC_1792_162,(1,2,0):C.UVGC_1792_161,(1,2,1):C.UVGC_1792_162,(1,1,0):C.UVGC_1792_161,(1,1,1):C.UVGC_1792_162,(1,0,0):C.UVGC_1793_163,(1,0,1):C.UVGC_1793_164,(2,2,0):C.UVGC_1793_163,(2,2,1):C.UVGC_1793_164,(2,1,0):C.UVGC_1792_161,(2,1,1):C.UVGC_1792_162,(2,0,0):C.UVGC_1792_161,(2,0,1):C.UVGC_1792_162,(0,2,0):C.UVGC_1792_161,(0,2,1):C.UVGC_1792_162,(0,1,0):C.UVGC_1793_163,(0,1,1):C.UVGC_1793_164,(0,0,0):C.UVGC_1792_161,(0,0,1):C.UVGC_1792_162})

V_1002 = CTVertex(name = 'V_1002',
                  type = 'UV',
                  particles = [ P.c__tilde__, P.c, P.g, P.g, P.g ],
                  color = [ 'T(3,-1,1)*T(4,-2,-1)*T(5,2,-2)', 'T(3,-1,1)*T(4,2,-2)*T(5,-2,-1)', 'T(3,-2,-1)*T(4,-1,1)*T(5,2,-2)', 'T(3,-2,-1)*T(4,2,-2)*T(5,-1,1)', 'T(3,2,-2)*T(4,-1,1)*T(5,-2,-1)', 'T(3,2,-2)*T(4,-2,-1)*T(5,-1,1)' ],
                  lorentz = [ L.FFVVV147, L.FFVVV148, L.FFVVV149, L.FFVVV85, L.FFVVV86, L.FFVVV87 ],
                  loop_particles = [ [ [P.b] ], [ [P.t] ] ],
                  couplings = {(5,5,0):C.UVGC_1788_153,(5,5,1):C.UVGC_1788_154,(5,4,0):C.UVGC_1791_159,(5,4,1):C.UVGC_1791_160,(5,3,0):C.UVGC_1788_153,(5,3,1):C.UVGC_1788_154,(4,5,0):C.UVGC_1788_153,(4,5,1):C.UVGC_1788_154,(4,4,0):C.UVGC_1788_153,(4,4,1):C.UVGC_1788_154,(4,3,0):C.UVGC_1791_159,(4,3,1):C.UVGC_1791_160,(3,5,0):C.UVGC_1791_159,(3,5,1):C.UVGC_1791_160,(3,4,0):C.UVGC_1788_153,(3,4,1):C.UVGC_1788_154,(3,3,0):C.UVGC_1788_153,(3,3,1):C.UVGC_1788_154,(1,5,0):C.UVGC_1788_153,(1,5,1):C.UVGC_1788_154,(1,4,0):C.UVGC_1788_153,(1,4,1):C.UVGC_1788_154,(1,3,0):C.UVGC_1791_159,(1,3,1):C.UVGC_1791_160,(2,5,0):C.UVGC_1791_159,(2,5,1):C.UVGC_1791_160,(2,4,0):C.UVGC_1788_153,(2,4,1):C.UVGC_1788_154,(2,3,0):C.UVGC_1788_153,(2,3,1):C.UVGC_1788_154,(0,5,0):C.UVGC_1788_153,(0,5,1):C.UVGC_1788_154,(0,4,0):C.UVGC_1791_159,(0,4,1):C.UVGC_1791_160,(0,3,0):C.UVGC_1788_153,(0,3,1):C.UVGC_1788_154,(5,2,0):C.UVGC_1792_161,(5,2,1):C.UVGC_1792_162,(5,1,0):C.UVGC_1793_163,(5,1,1):C.UVGC_1793_164,(5,0,0):C.UVGC_1792_161,(5,0,1):C.UVGC_1792_162,(4,2,0):C.UVGC_1792_161,(4,2,1):C.UVGC_1792_162,(4,1,0):C.UVGC_1792_161,(4,1,1):C.UVGC_1792_162,(4,0,0):C.UVGC_1793_163,(4,0,1):C.UVGC_1793_164,(3,2,0):C.UVGC_1793_163,(3,2,1):C.UVGC_1793_164,(3,1,0):C.UVGC_1792_161,(3,1,1):C.UVGC_1792_162,(3,0,0):C.UVGC_1792_161,(3,0,1):C.UVGC_1792_162,(1,2,0):C.UVGC_1792_161,(1,2,1):C.UVGC_1792_162,(1,1,0):C.UVGC_1792_161,(1,1,1):C.UVGC_1792_162,(1,0,0):C.UVGC_1793_163,(1,0,1):C.UVGC_1793_164,(2,2,0):C.UVGC_1793_163,(2,2,1):C.UVGC_1793_164,(2,1,0):C.UVGC_1792_161,(2,1,1):C.UVGC_1792_162,(2,0,0):C.UVGC_1792_161,(2,0,1):C.UVGC_1792_162,(0,2,0):C.UVGC_1792_161,(0,2,1):C.UVGC_1792_162,(0,1,0):C.UVGC_1793_163,(0,1,1):C.UVGC_1793_164,(0,0,0):C.UVGC_1792_161,(0,0,1):C.UVGC_1792_162})

