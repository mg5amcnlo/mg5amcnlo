import parameters as P
from object_library import all_running_elements, Running


# ordering: {cQQ1, cQQ8, cQt1, cQt8, ctt1, cQq11, cQq13, cQq81, cQq83, ctu1, ctu8, ctd1, ctd8, ctq1, cQu1, cQd1, ctq8, cQu8, cQd8}
#              1     2     3     4     5     6      7       8     9     10     11   12    13    14    15    16    17    18     19   


RGE12 = Running(name        = 'RGE_12',
               run_objects = [
                               [P.cQQ1, P.cQQ8, P.aS]
                           ],
               value       = '-(8./3.)/(4.*cmath.pi)')


RGE21 = Running(name        = 'RGE_21',
               run_objects = [
                               [P.cQQ8, P.cQQ1, P.aS]
                           ],
               value       = '(44./3.)/(4.*cmath.pi)')


RGE22 = Running(name        = 'RGE_22',
               run_objects = [
                               [P.cQQ8, P.cQQ8, P.aS]
                           ],
               value       = '(-16./9.)/(4.*cmath.pi)')



RGE24 = Running(name        = 'RGE_24',
               run_objects = [
                               [P.cQQ8, P.cQt8, P.aS]
                           ],
               value       = '(4./3.)/(4.*cmath.pi)')


RGE28 = Running(name        = 'RGE_28',
               run_objects = [
                               [P.cQQ8, P.cQq81, P.aS]
                           ],
               value       = '(32./3.)/(4.*cmath.pi)')


RGE218 = Running(name        = 'RGE_218',
               run_objects = [
                               [P.cQQ8, P.cQu8, P.aS]
                           ],
               value       = '(8./3.)/(4.*cmath.pi)')


RGE219 = Running(name        = 'RGE_219',
               run_objects = [
                               [P.cQQ8, P.cQd8, P.aS]
                           ],
               value       = '(4.)/(4.*cmath.pi)')


RGE34 = Running(name        = 'RGE_34',
               run_objects = [
                               [P.cQt1, P.cQt8, P.aS]
                           ],
               value       = '(-8./3.)/(4.*cmath.pi)')


RGE41 = Running(name        = 'RGE_41',
               run_objects = [
                               [P.cQt8, P.cQQ1, P.aS]
                           ],
               value       = '(4./3.)/(4.*cmath.pi)')


RGE42 = Running(name        = 'RGE_42',
               run_objects = [
                               [P.cQt8, P.cQQ8, P.aS]
                           ],
               value       = '(10./9.)/(4.*cmath.pi)')


RGE43 = Running(name        = 'RGE_43',
               run_objects = [
                               [P.cQt8, P.cQt1, P.aS]
                           ],
               value       = '(-12.)/(4.*cmath.pi)')


RGE44 = Running(name        = 'RGE_44',
               run_objects = [
                               [P.cQt8, P.cQt8, P.aS]
                           ],
               value       = '(-12.)/(4.*cmath.pi)')


RGE45 = Running(name        = 'RGE_45',
               run_objects = [
                               [P.cQt8, P.ctt1, P.aS]
                           ],
               value       = '(8./3.)/(4.*cmath.pi)')


RGE48 = Running(name        = 'RGE_48',
               run_objects = [
                               [P.cQt8, P.cQq81, P.aS]
                           ],
               value       = '(16./3.)/(4.*cmath.pi)')


RGE410 = Running(name        = 'RGE_410',
               run_objects = [
                               [P.cQt8, P.ctu1, P.aS]
                           ],
               value       = '(-8./3.)/(4.*cmath.pi)')

RGE411 = Running(name        = 'RGE_411',
               run_objects = [
                               [P.cQt8, P.ctu8, P.aS]
                           ],
               value       = '(16./3.)/(4.*cmath.pi)')

RGE413 = Running(name        = 'RGE_413',
               run_objects = [
                               [P.cQt8, P.ctd8, P.aS]
                           ],
               value       = '(2.)/(4.*cmath.pi)')

RGE417 = Running(name        = 'RGE_417',
               run_objects = [
                               [P.cQt8, P.ctq8, P.aS]
                           ],
               value       = '(8./3.)/(4.*cmath.pi)')

RGE418 = Running(name        = 'RGE_418',
               run_objects = [
                               [P.cQt8, P.cQu8, P.aS]
                           ],
               value       = '(4./3.)/(4.*cmath.pi)')

RGE419 = Running(name        = 'RGE_419',
               run_objects = [
                               [P.cQt8, P.cQd8, P.aS]
                           ],
               value       = '(2.)/(4.*cmath.pi)')

RGE54 = Running(name        = 'RGE_54',
               run_objects = [
                               [P.ctt1, P.cQt8, P.aS]
                           ],
               value       = '(4./9.)/(4.*cmath.pi)')

RGE55 = Running(name        = 'RGE_55',
               run_objects = [
                               [P.ctt1, P.ctt1, P.aS]
                           ],
               value       = '(44./9.)/(4.*cmath.pi)')

RGE510 = Running(name        = 'RGE_510',
               run_objects = [
                               [P.ctt1, P.ctu1, P.aS]
                           ],
               value       = '(-8./9.)/(4.*cmath.pi)')

RGE511 = Running(name        = 'RGE_511',
               run_objects = [
                               [P.ctt1, P.ctu8, P.aS]
                           ],
               value       = '(16./27.)/(4.*cmath.pi)')

RGE513 = Running(name        = 'RGE_513',
               run_objects = [
                               [P.ctt1, P.ctd8, P.aS]
                           ],
               value       = '(2./3.)/(4.*cmath.pi)')

RGE517 = Running(name        = 'RGE_517',
               run_objects = [
                               [P.ctt1, P.ctq8, P.aS]
                           ],
               value       = '(8./9.)/(4.*cmath.pi)')

RGE68 = Running(name        = 'RGE_68',
               run_objects = [
                               [P.cQq11, P.cQq81, P.aS]
                           ],
               value       = '(8./3.)/(4.*cmath.pi)')

RGE79 = Running(name        = 'RGE_79',
               run_objects = [
                               [P.cQq13, P.cQq83, P.aS]
                           ],
               value       = '(8./3.)/(4.*cmath.pi)')
   
RGE81 = Running(name        = 'RGE_81',
               run_objects = [
                               [P.cQq81, P.cQQ1, P.aS]
                           ],
               value       = '(2./3.)/(4.*cmath.pi)')
   
RGE82 = Running(name        = 'RGE_82',
               run_objects = [
                               [P.cQq81, P.cQQ8, P.aS]
                           ],
               value       = '(5./9.)/(4.*cmath.pi)')

   
RGE84 = Running(name        = 'RGE_84',
               run_objects = [
                               [P.cQq81, P.cQt8, P.aS]
                           ],
               value       = '(1./3.)/(4.*cmath.pi)')

RGE86 = Running(name        = 'RGE_86',
               run_objects = [
                               [P.cQq81, P.cQq11, P.aS]
                           ],
               value       = '(12.)/(4.*cmath.pi)')

RGE817 = Running(name        = 'RGE_817',
               run_objects = [
                               [P.cQq81, P.ctq8, P.aS]
                           ],
               value       = '(1./3.)/(4.*cmath.pi)')

RGE818 = Running(name        = 'RGE_818',
               run_objects = [
                               [P.cQq81, P.cQu8, P.aS]
                           ],
               value       = '(2./3.)/(4.*cmath.pi)')

RGE819 = Running(name        = 'RGE_819',
               run_objects = [
                               [P.cQq81, P.cQd8, P.aS]
                           ],
               value       = '(1.)/(4.*cmath.pi)')

RGE97 = Running(name        = 'RGE_97',
               run_objects = [
                               [P.cQq83, P.cQq13, P.aS]
                           ],
               value       = '(12.)/(4.*cmath.pi)')

RGE99 = Running(name        = 'RGE_99',
               run_objects = [
                               [P.cQq83, P.cQq83, P.aS]
                           ],
               value       = '(-4)/(4.*cmath.pi)')

RGE1011 = Running(name        = 'RGE_1011',
               run_objects = [
                               [P.ctu1, P.ctu8, P.aS]
                           ],
               value       = '(8./3.)/(4.*cmath.pi)')

RGE114 = Running(name        = 'RGE_114',
               run_objects = [
                               [P.ctu8, P.cQt8, P.aS]
                           ],
               value       = '(2./3.)/(4.*cmath.pi)')

RGE115 = Running(name        = 'RGE_114',
               run_objects = [
                               [P.ctu8, P.ctt1, P.aS]
                           ],
               value       = '(4./3.)/(4.*cmath.pi)')

RGE1110 = Running(name        = 'RGE_1110',
               run_objects = [
                               [P.ctu8, P.ctu1, P.aS]
                           ],
               value       = '(10.)/(4.*cmath.pi)')

RGE1111 = Running(name        = 'RGE_1111',
               run_objects = [
                               [P.ctu8, P.ctu8, P.aS]
                           ],
               value       = '(-8./3.)/(4.*cmath.pi)')

RGE1113 = Running(name        = 'RGE_1113',
               run_objects = [
                               [P.ctu8, P.ctd8, P.aS]
                           ],
               value       = '(1.)/(4.*cmath.pi)')

RGE1117 = Running(name        = 'RGE_1117',
               run_objects = [
                               [P.ctu8, P.ctq8, P.aS]
                           ],
               value       = '(4./3.)/(4.*cmath.pi)')

RGE1118 = Running(name        = 'RGE_1118',
               run_objects = [
                               [P.ctu8, P.cQu8, P.aS]
                           ],
               value       = '(2./3.)/(4.*cmath.pi)')

RGE1213 = Running(name        = 'RGE_1213',
               run_objects = [
                               [P.ctd1, P.ctd8, P.aS]
                           ],
               value       = '(8./3.)/(4.*cmath.pi)')

RGE134 = Running(name        = 'RGE_134',
               run_objects = [
                               [P.ctd8, P.cQt8, P.aS]
                           ],
               value       = '(4./3.)/(4.*cmath.pi)')

RGE135 = Running(name        = 'RGE_135',
               run_objects = [
                               [P.ctd8, P.ctt1, P.aS]
                           ],
               value       = '(8./3.)/(4.*cmath.pi)')

RGE1310 = Running(name        = 'RGE_1310',
               run_objects = [
                               [P.ctd8, P.ctu1, P.aS]
                           ],
               value       = '(-8./3.)/(4.*cmath.pi)')

RGE1311 = Running(name        = 'RGE_1311',
               run_objects = [
                               [P.ctd8, P.ctu8, P.aS]
                           ],
               value       = '(16./9.)/(4.*cmath.pi)')

RGE1312 = Running(name        = 'RGE_1312',
               run_objects = [
                               [P.ctd8, P.ctd1, P.aS]
                           ],
               value       = '(12.)/(4.*cmath.pi)')

RGE1313 = Running(name        = 'RGE_1313',
               run_objects = [
                               [P.ctd8, P.ctd8, P.aS]
                           ],
               value       = '(-4./3.)/(4.*cmath.pi)')

RGE1317 = Running(name        = 'RGE_1317',
               run_objects = [
                               [P.ctd8, P.ctq8, P.aS]
                           ],
               value       = '(8./3.)/(4.*cmath.pi)')

RGE1319 = Running(name        = 'RGE_1319',
               run_objects = [
                               [P.ctd8, P.cQd8, P.aS]
                           ],
               value       = '(4./3.)/(4.*cmath.pi)')

RGE1417 = Running(name        = 'RGE_1417',
               run_objects = [
                               [P.ctq1, P.ctq8, P.aS]
                           ],
               value       = '(-8./3.)/(4.*cmath.pi)')

RGE1518 = Running(name        = 'RGE_1518',
               run_objects = [
                               [P.cQu1, P.cQu8, P.aS]
                           ],
               value       = '(-8./3.)/(4.*cmath.pi)')

RGE1619 = Running(name        = 'RGE_1619',
               run_objects = [
                               [P.cQd1, P.cQd8, P.aS]
                           ],
               value       = '(-8./3.)/(4.*cmath.pi)')

RGE174 = Running(name        = 'RGE_174',
               run_objects = [
                               [P.ctq8, P.cQt8, P.aS]
                           ],
               value       = '(4./3.)/(4.*cmath.pi)')

RGE175 = Running(name        = 'RGE_175',
               run_objects = [
                               [P.ctq8, P.ctt1, P.aS]
                           ],
               value       = '(8./3.)/(4.*cmath.pi)')

RGE178 = Running(name        = 'RGE_178',
               run_objects = [
                               [P.ctq8, P.cQq81, P.aS]
                           ],
               value       = '(8./3.)/(4.*cmath.pi)')

RGE1710 = Running(name        = 'RGE_1710',
               run_objects = [
                               [P.ctq8, P.ctu1, P.aS]
                           ],
               value       = '(-8./3.)/(4.*cmath.pi)')

RGE1711 = Running(name        = 'RGE_1711',
               run_objects = [
                               [P.ctq8, P.ctu8, P.aS]
                           ],
               value       = '(16./9.)/(4.*cmath.pi)')

RGE1713 = Running(name        = 'RGE_1713',
               run_objects = [
                               [P.ctq8, P.ctd8, P.aS]
                           ],
               value       = '(2.)/(4.*cmath.pi)')

RGE1714 = Running(name        = 'RGE_1714',
               run_objects = [
                               [P.ctq8, P.ctq1, P.aS]
                           ],
               value       = '(-12.)/(4.*cmath.pi)')

RGE1717 = Running(name        = 'RGE_1717',
               run_objects = [
                               [P.ctq8, P.ctq8, P.aS]
                           ],
               value       = '(-32./3.)/(4.*cmath.pi)')

RGE181 = Running(name        = 'RGE_181',
               run_objects = [
                               [P.cQu8, P.cQQ1, P.aS]
                           ],
               value       = '(-4./3.)/(4.*cmath.pi)')

RGE182 = Running(name        = 'RGE_182',
               run_objects = [
                               [P.cQu8, P.cQQ8, P.aS]
                           ],
               value       = '(10./9.)/(4.*cmath.pi)')

RGE184 = Running(name        = 'RGE_184',
               run_objects = [
                               [P.cQu8, P.cQt8, P.aS]
                           ],
               value       = '(2./3.)/(4.*cmath.pi)')

RGE188 = Running(name        = 'RGE_188',
               run_objects = [
                               [P.cQu8, P.cQq81, P.aS]
                           ],
               value       = '(16./3.)/(4.*cmath.pi)')

RGE1810 = Running(name        = 'RGE_1810',
               run_objects = [
                               [P.cQu8, P.ctu1, P.aS]
                           ],
               value       = '(-4./3.)/(4.*cmath.pi)')

RGE1811 = Running(name        = 'RGE_1811',
               run_objects = [
                               [P.cQu8, P.ctu8, P.aS]
                           ],
               value       = '(8./9.)/(4.*cmath.pi)')

RGE1815 = Running(name        = 'RGE_1815',
               run_objects = [
                               [P.cQu8, P.cQu1, P.aS]
                           ],
               value       = '(-12.)/(4.*cmath.pi)')

RGE1818 = Running(name        = 'RGE_1818',
               run_objects = [
                               [P.cQu8, P.cQu8, P.aS]
                           ],
               value       = '(-34./3.)/(4.*cmath.pi)')

RGE1819 = Running(name        = 'RGE_1819',
               run_objects = [
                               [P.cQu8, P.cQd8, P.aS]
                           ],
               value       = '(2.)/(4.*cmath.pi)')


RGE191 = Running(name        = 'RGE_191',
               run_objects = [
                               [P.cQd8, P.cQQ1, P.aS]
                           ],
               value       = '(4./3.)/(4.*cmath.pi)')

RGE192 = Running(name        = 'RGE_192',
               run_objects = [
                               [P.cQd8, P.cQQ8, P.aS]
                           ],
               value       = '(10./9.)/(4.*cmath.pi)')

RGE194 = Running(name        = 'RGE_194',
               run_objects = [
                               [P.cQd8, P.cQt8, P.aS]
                           ],
               value       = '(2./3.)/(4.*cmath.pi)')

RGE198 = Running(name        = 'RGE_198',
               run_objects = [
                               [P.cQd8, P.cQq81, P.aS]
                           ],
               value       = '(16./3.)/(4.*cmath.pi)')

RGE1913 = Running(name        = 'RGE_1913',
               run_objects = [
                               [P.cQd8, P.ctd8, P.aS]
                           ],
               value       = '(2./3.)/(4.*cmath.pi)')

RGE1916 = Running(name        = 'RGE_1916',
               run_objects = [
                               [P.cQd8, P.cQd1, P.aS]
                           ],
               value       = '(-12.)/(4.*cmath.pi)')

RGE1918 = Running(name        = 'RGE_1918',
               run_objects = [
                               [P.cQd8, P.cQu8, P.aS]
                           ],
               value       = '(4./3.)/(4.*cmath.pi)')

RGE1919 = Running(name        = 'RGE_1919',
               run_objects = [
                               [P.cQd8, P.cQd8, P.aS]
                           ],
               value       = '(-32./3.)/(4.*cmath.pi)')