"""Discrete source-component mapping checks, without running a shower.

The compact records below preserve saved sidecar prefixes from the five
2026 campaign processes.  They contain only routing/ancestry fields; the
tests are portable and never open or alter a production directory.  Extra
synthetic cases exercise orientations absent from those five Born states.
"""

from copy import deepcopy
from fractions import Fraction as F
import json
import math
from pathlib import Path
import re
import tempfile
import unittest

from madgraph.various.macnlops_components import (
    MappingError, branch_candidates, canonical_colour_topology,
    component_orbit_factors, transport_component_basis)


# Provenance and numerical momenta retained from bounded existing prefixes.
PREFIXES = json.loads(r'''[{"label":"dy","origin":"DY_5FS/Events/delta_1M/macnlops/SubProcesses__P0_uux_emep__GF2.0_9__events/shower.jsonl","sequence":1,"expected_sector":1,"registry":{"outer":{"context":3,"provider":3,"directory":"P0_uux_emep","nexternal":5,"born_pdgs":[[2,-2,11,-11],[4,-4,11,-11]],"sectors":[{"sector":1,"fks":{"i":5,"j":1,"ij":1,"splitting_type":["QCD"]},"processes":[[null,[[2],[-2],[11],[-11],[21]]],[null,[[4],[-4],[11],[-11],[21]]]]},{"sector":2,"fks":{"i":5,"j":2,"ij":2,"splitting_type":["QCD"]},"processes":[[null,[[2],[-2],[11],[-11],[21]]],[null,[[4],[-4],[11],[-11],[21]]]]},{"sector":3,"fks":{"i":5,"j":1,"ij":1,"splitting_type":["QCD"]},"processes":[[null,[[21],[-2],[11],[-11],[-2]]],[null,[[21],[-4],[11],[-11],[-4]]]]},{"sector":4,"fks":{"i":5,"j":2,"ij":2,"splitting_type":["QCD"]},"processes":[[null,[[2],[21],[11],[-11],[2]]],[null,[[4],[21],[11],[-11],[4]]]]}]}},"record":{"schema":"macnlops-one-step-v1","sequence":1,"emitted":true,"emission_type":"ISR","branch_indices":{"radiator_before":3,"radiator_after":7,"emitted":11,"recoilers":[8]},"hard_pythia_indices":[7,8,9,10,11],"source_particles":[{"id":4,"status":-1,"colors":[501,0],"p":[45.31713726615834,0,0,45.31713726615834]},{"id":-4,"status":-1,"colors":[0,501],"p":[45.670694550643674,0,0,-45.670694550643674]},{"id":11,"status":1,"colors":[0,0],"p":[45.35104349133858,17.448423915428528,20.330772844313014,36.59138319568871]},{"id":-11,"status":1,"colors":[0,0],"p":[45.636788325463435,-17.448423915428528,-20.330772844313014,-36.94494048017403]}],"particles":[{"id":4,"status":-1,"colors":[502,0],"p":[58.94530408512554,3.3306690738754696e-16,-8.881784197001252e-16,58.94530408512552]},{"id":-4,"status":-1,"colors":[0,501],"p":[45.67069455064369,-4.440892098500626e-16,8.881784197001252e-16,-45.67069455064367]},{"id":11,"status":1,"colors":[0,0],"p":[56.44121362462997,16.095257378251148,28.190893943889183,46.171709779835524]},{"id":-11,"status":1,"colors":[0,0],"p":[37.12102456513563,-17.591957834620416,-19.497029209846296,-26.236602987062547]},{"id":21,"status":1,"colors":[502,501],"p":[11.053760446003613,1.4967004563692652,-8.693864734042887,-6.660497258291112]}],"pythia_event":[{"index":1,"id":2212,"status":-12,"mothers":[0,0],"daughters":[7,0],"colors":[0,0],"p":[6500,0,0,6499.999932280723]},{"index":2,"id":2212,"status":-12,"mothers":[0,0],"daughters":[8,0],"colors":[0,0],"p":[6500,0,0,-6499.999932280723]},{"index":3,"id":4,"status":-21,"mothers":[7,0],"daughters":[5,6],"colors":[501,0],"p":[45.31713726615834,0,0,45.31713726615834]},{"index":4,"id":-4,"status":-21,"mothers":[8,8],"daughters":[5,6],"colors":[0,501],"p":[45.670694550643674,0,0,-45.670694550643674]},{"index":5,"id":11,"status":-23,"mothers":[3,4],"daughters":[9,9],"colors":[0,0],"p":[45.35104349133858,17.448423915428528,20.330772844313014,36.59138319568871]},{"index":6,"id":-11,"status":-23,"mothers":[3,4],"daughters":[10,10],"colors":[0,0],"p":[45.636788325463435,-17.448423915428528,-20.330772844313014,-36.94494048017403]},{"index":7,"id":4,"status":-41,"mothers":[1,0],"daughters":[11,3],"colors":[502,0],"p":[58.94530408512554,3.3306690738754696e-16,-8.881784197001252e-16,58.94530408512552]},{"index":8,"id":-4,"status":-42,"mothers":[2,0],"daughters":[4,4],"colors":[0,501],"p":[45.67069455064369,-4.440892098500626e-16,8.881784197001252e-16,-45.67069455064367]},{"index":9,"id":11,"status":44,"mothers":[5,5],"daughters":[0,0],"colors":[0,0],"p":[56.44121362462997,16.095257378251148,28.190893943889183,46.171709779835524]},{"index":10,"id":-11,"status":44,"mothers":[6,6],"daughters":[0,0],"colors":[0,0],"p":[37.12102456513563,-17.591957834620416,-19.497029209846296,-26.236602987062547]},{"index":11,"id":21,"status":43,"mothers":[7,0],"daughters":[0,0],"colors":[502,501],"p":[11.053760446003613,1.4967004563692652,-8.693864734042887,-6.660497258291112]}]}},{"label":"zz","origin":"ZZ_5FS/Events/delta_1M/macnlops/SubProcesses__P0_uux_zz__GF1.0_13__events/shower.jsonl","sequence":1,"expected_sector":1,"registry":{"outer":{"context":3,"provider":3,"directory":"P0_uux_zz","nexternal":5,"born_pdgs":[[2,-2,23,23],[4,-4,23,23]],"sectors":[{"sector":1,"fks":{"i":5,"j":1,"ij":1,"splitting_type":["QCD"]},"processes":[[null,[[2],[-2],[23],[23],[21]]],[null,[[4],[-4],[23],[23],[21]]]]},{"sector":2,"fks":{"i":5,"j":2,"ij":2,"splitting_type":["QCD"]},"processes":[[null,[[2],[-2],[23],[23],[21]]],[null,[[4],[-4],[23],[23],[21]]]]},{"sector":3,"fks":{"i":5,"j":1,"ij":1,"splitting_type":["QCD"]},"processes":[[null,[[21],[-2],[23],[23],[-2]]],[null,[[21],[-4],[23],[23],[-4]]]]},{"sector":4,"fks":{"i":5,"j":2,"ij":2,"splitting_type":["QCD"]},"processes":[[null,[[2],[21],[23],[23],[2]]],[null,[[4],[21],[23],[23],[4]]]]}]}},"record":{"schema":"macnlops-one-step-v1","sequence":1,"emitted":true,"emission_type":"ISR","branch_indices":{"radiator_before":3,"radiator_after":7,"emitted":11,"recoilers":[8]},"hard_pythia_indices":[7,8,9,10,11],"source_particles":[{"id":2,"status":-1,"colors":[501,0],"p":[1702.9178697550974,0,0,1702.9178697550974]},{"id":-2,"status":-1,"colors":[0,501],"p":[9.306539225047914,0,0,-9.306539225047999]},{"id":23,"status":1,"colors":[0,0],"p":[893.6428791445046,76.64498328315469,-40.3414993528173,884.7488084587463]},{"id":23,"status":1,"colors":[0,0],"p":[818.5815298356406,-76.64498328315469,40.3414993528173,808.862522071303]}],"particles":[{"id":2,"status":-1,"colors":[502,0],"p":[1730.4056085880572,0,-2.220446049250313e-16,1730.4056085880572]},{"id":-2,"status":-1,"colors":[0,501],"p":[9.306539225048027,0,0,-9.306539225048027]},{"id":23,"status":1,"colors":[0,0],"p":[903.6128642437544,77.36827768242806,-40.9868515175701,894.7263183197072]},{"id":23,"status":1,"colors":[0,0],"p":[827.6097511875572,-75.98308119708422,39.7509238630456,818.0886775530914]},{"id":21,"status":1,"colors":[502,501],"p":[8.489532381801908,-1.3851964853438283,1.2359276545244806,8.28407349022423]}],"pythia_event":[{"index":1,"id":2212,"status":-12,"mothers":[0,0],"daughters":[7,0],"colors":[0,0],"p":[6500,0,0,6499.999932280723]},{"index":2,"id":2212,"status":-12,"mothers":[0,0],"daughters":[8,0],"colors":[0,0],"p":[6500,0,0,-6499.999932280723]},{"index":3,"id":2,"status":-21,"mothers":[7,0],"daughters":[5,6],"colors":[501,0],"p":[1702.9178697550974,0,0,1702.9178697550974]},{"index":4,"id":-2,"status":-21,"mothers":[8,8],"daughters":[5,6],"colors":[0,501],"p":[9.306539225047914,0,0,-9.306539225047914]},{"index":5,"id":23,"status":-23,"mothers":[3,4],"daughters":[9,9],"colors":[0,0],"p":[893.6428791445047,76.64498328315469,-40.3414993528173,884.7488084587463]},{"index":6,"id":23,"status":-23,"mothers":[3,4],"daughters":[10,10],"colors":[0,0],"p":[818.5815298356405,-76.64498328315469,40.3414993528173,808.862522071303]},{"index":7,"id":2,"status":-41,"mothers":[1,0],"daughters":[11,3],"colors":[502,0],"p":[1730.4056085880572,0,-2.220446049250313e-16,1730.4056085880572]},{"index":8,"id":-2,"status":-42,"mothers":[2,0],"daughters":[4,4],"colors":[0,501],"p":[9.306539225048027,0,0,-9.306539225048027]},{"index":9,"id":23,"status":44,"mothers":[5,5],"daughters":[0,0],"colors":[0,0],"p":[903.6128642437544,77.36827768242806,-40.9868515175701,894.7263183197072]},{"index":10,"id":23,"status":44,"mothers":[6,6],"daughters":[0,0],"colors":[0,0],"p":[827.6097511875572,-75.98308119708422,39.7509238630456,818.0886775530914]},{"index":11,"id":21,"status":43,"mothers":[7,0],"daughters":[0,0],"colors":[502,501],"p":[8.489532381801908,-1.3851964853438283,1.2359276545244806,8.28407349022423]}]}},{"label":"ttbar","origin":"TTbar_5FS/Events/delta_1M/macnlops/SubProcesses__P0_gg_ttx__GF3.0_34__events/shower.jsonl","sequence":5,"expected_sector":4,"registry":{"outer":{"context":1,"provider":1,"directory":"P0_gg_ttx","nexternal":5,"born_pdgs":[[21,21,6,-6]],"sectors":[{"sector":1,"fks":{"i":5,"j":1,"ij":1,"splitting_type":["QCD"]},"processes":[[null,[[21],[21],[6],[-6],[21]]]]},{"sector":2,"fks":{"i":5,"j":2,"ij":2,"splitting_type":["QCD"]},"processes":[[null,[[21],[21],[6],[-6],[21]]]]},{"sector":3,"fks":{"i":5,"j":3,"ij":3,"splitting_type":["QCD"]},"processes":[[null,[[21],[21],[6],[-6],[21]]]]},{"sector":4,"fks":{"i":5,"j":4,"ij":4,"splitting_type":["QCD"]},"processes":[[null,[[21],[21],[6],[-6],[21]]]]},{"sector":5,"fks":{"i":5,"j":1,"ij":1,"splitting_type":["QCD"]},"processes":[[null,[[-1],[21],[6],[-6],[-1]]],[null,[[-2],[21],[6],[-6],[-2]]],[null,[[-3],[21],[6],[-6],[-3]]],[null,[[-4],[21],[6],[-6],[-4]]],[null,[[-5],[21],[6],[-6],[-5]]]]},{"sector":6,"fks":{"i":5,"j":1,"ij":1,"splitting_type":["QCD"]},"processes":[[null,[[1],[21],[6],[-6],[1]]],[null,[[2],[21],[6],[-6],[2]]],[null,[[3],[21],[6],[-6],[3]]],[null,[[4],[21],[6],[-6],[4]]],[null,[[5],[21],[6],[-6],[5]]]]},{"sector":7,"fks":{"i":5,"j":2,"ij":2,"splitting_type":["QCD"]},"processes":[[null,[[21],[-1],[6],[-6],[-1]]],[null,[[21],[-2],[6],[-6],[-2]]],[null,[[21],[-3],[6],[-6],[-3]]],[null,[[21],[-4],[6],[-6],[-4]]],[null,[[21],[-5],[6],[-6],[-5]]]]},{"sector":8,"fks":{"i":5,"j":2,"ij":2,"splitting_type":["QCD"]},"processes":[[null,[[21],[1],[6],[-6],[1]]],[null,[[21],[2],[6],[-6],[2]]],[null,[[21],[3],[6],[-6],[3]]],[null,[[21],[4],[6],[-6],[4]]],[null,[[21],[5],[6],[-6],[5]]]]}]}},"record":{"schema":"macnlops-one-step-v1","sequence":5,"emitted":true,"emission_type":"FSR","branch_indices":{"radiator_before":6,"radiator_after":7,"emitted":8,"recoilers":[9]},"hard_pythia_indices":[3,4,9,7,8],"source_particles":[{"id":21,"status":-1,"colors":[503,502],"p":[407.8435527273785,0,0,407.8435527273785]},{"id":21,"status":-1,"colors":[501,503],"p":[473.90747926866726,0,0,-473.90747926866726]},{"id":6,"status":1,"colors":[501,0],"p":[460.849106285201,-65.52492982865658,-297.30172207647473,-299.6182043621193]},{"id":-6,"status":1,"colors":[0,502],"p":[420.90192571084475,65.52492982865658,297.30172207647473,233.55427782083052]}],"particles":[{"id":21,"status":-1,"colors":[503,502],"p":[407.8435527273785,0,0,407.8435527273785]},{"id":21,"status":-1,"colors":[501,503],"p":[473.90747926866726,0,0,-473.90747926866726]},{"id":6,"status":1,"colors":[501,0],"p":[443.56686641569905,-62.63062291054111,-284.1695839235047,-286.614056927469]},{"id":-6,"status":1,"colors":[0,504],"p":[313.7539377592102,-13.514457022748282,208.75894082625507,157.3200481054172]},{"id":21,"status":1,"colors":[504,502],"p":[124.43022782113626,76.1450799332894,75.4106430972497,63.230082280763106]}],"pythia_event":[{"index":1,"id":2212,"status":-12,"mothers":[0,0],"daughters":[3,0],"colors":[0,0],"p":[6500,0,0,6499.999932280723]},{"index":2,"id":2212,"status":-12,"mothers":[0,0],"daughters":[4,0],"colors":[0,0],"p":[6500,0,0,-6499.999932280723]},{"index":3,"id":21,"status":-21,"mothers":[1,0],"daughters":[5,6],"colors":[503,502],"p":[407.8435527273785,0,0,407.8435527273785]},{"index":4,"id":21,"status":-21,"mothers":[2,0],"daughters":[5,6],"colors":[501,503],"p":[473.90747926866726,0,0,-473.90747926866726]},{"index":5,"id":6,"status":-23,"mothers":[3,4],"daughters":[9,9],"colors":[501,0],"p":[460.849106285201,-65.52492982865658,-297.30172207647473,-299.6182043621193]},{"index":6,"id":-6,"status":-23,"mothers":[3,4],"daughters":[7,8],"colors":[0,502],"p":[420.90192571084475,65.52492982865658,297.30172207647473,233.55427782083052]},{"index":7,"id":-6,"status":51,"mothers":[6,0],"daughters":[0,0],"colors":[0,504],"p":[313.7539377592102,-13.514457022748282,208.75894082625507,157.3200481054172]},{"index":8,"id":21,"status":51,"mothers":[6,0],"daughters":[0,0],"colors":[504,502],"p":[124.43022782113626,76.1450799332894,75.4106430972497,63.230082280763106]},{"index":9,"id":6,"status":52,"mothers":[5,5],"daughters":[0,0],"colors":[501,0],"p":[443.56686641569905,-62.63062291054111,-284.1695839235047,-286.614056927469]}]}},{"label":"wbb","origin":"Wbb_4FS/Events/delta_1M/macnlops/SubProcesses__P0_udx_bbxwp__GF2.0_15__events/shower.jsonl","sequence":3,"expected_sector":3,"registry":{"outer":{"context":2,"provider":2,"directory":"P0_udx_bbxwp","nexternal":6,"born_pdgs":[[2,-1,5,-5,24],[4,-3,5,-5,24]],"sectors":[{"sector":1,"fks":{"i":6,"j":1,"ij":1,"splitting_type":["QCD"]},"processes":[[null,[[2],[-1],[5],[-5],[24],[21]]],[null,[[4],[-3],[5],[-5],[24],[21]]]]},{"sector":2,"fks":{"i":6,"j":2,"ij":2,"splitting_type":["QCD"]},"processes":[[null,[[2],[-1],[5],[-5],[24],[21]]],[null,[[4],[-3],[5],[-5],[24],[21]]]]},{"sector":3,"fks":{"i":6,"j":3,"ij":3,"splitting_type":["QCD"]},"processes":[[null,[[2],[-1],[5],[-5],[24],[21]]],[null,[[4],[-3],[5],[-5],[24],[21]]]]},{"sector":4,"fks":{"i":6,"j":4,"ij":4,"splitting_type":["QCD"]},"processes":[[null,[[2],[-1],[5],[-5],[24],[21]]],[null,[[4],[-3],[5],[-5],[24],[21]]]]},{"sector":5,"fks":{"i":6,"j":1,"ij":1,"splitting_type":["QCD"]},"processes":[[null,[[21],[-1],[5],[-5],[24],[-2]]],[null,[[21],[-3],[5],[-5],[24],[-4]]]]},{"sector":6,"fks":{"i":6,"j":2,"ij":2,"splitting_type":["QCD"]},"processes":[[null,[[2],[21],[5],[-5],[24],[1]]],[null,[[4],[21],[5],[-5],[24],[3]]]]}]}},"record":{"schema":"macnlops-one-step-v1","sequence":3,"emitted":true,"emission_type":"FSR","branch_indices":{"radiator_before":5,"radiator_after":8,"emitted":9,"recoilers":[10,11]},"hard_pythia_indices":[3,4,8,10,11,9],"source_particles":[{"id":2,"status":-1,"colors":[502,0],"p":[2045.0606957313296,0,0,2045.0606957313294]},{"id":-1,"status":-1,"colors":[0,501],"p":[18.185587620127308,0,0,-18.185587620127222]},{"id":5,"status":1,"colors":[502,0],"p":[789.4912755191395,16.966274609579287,-8.152836366060338,789.2528497927001]},{"id":-5,"status":1,"colors":[0,501],"p":[1158.1394164334606,5.212630148774785,-1.1664104529550787,1158.1175613340918]},{"id":24,"status":1,"colors":[0,0],"p":[115.61559139885651,-22.178904758354072,9.319246819015417,79.50469698441026]}],"particles":[{"id":2,"status":-1,"colors":[502,0],"p":[2045.0606957313294,0,0,2045.0606957313294]},{"id":-1,"status":-1,"colors":[0,501],"p":[18.185587620127194,0,0,-18.185587620127194]},{"id":5,"status":1,"colors":[503,0],"p":[245.3149971066805,11.367139151182524,8.450251350492595,244.8606526273297]},{"id":-5,"status":1,"colors":[0,501],"p":[1173.745275164151,5.552062493477479,-1.3295185263175657,1173.7219806890903]},{"id":24,"status":1,"colors":[0,0],"p":[116.24694138976984,-21.563903160949597,9.023718925952002,80.62075418474232]},{"id":21,"status":1,"colors":[502,503],"p":[527.9390696908446,4.6447015162895,-16.14445175012687,527.6717206100274]}],"pythia_event":[{"index":1,"id":2212,"status":-12,"mothers":[0,0],"daughters":[3,0],"colors":[0,0],"p":[6500,0,0,6499.999932280723]},{"index":2,"id":2212,"status":-12,"mothers":[0,0],"daughters":[4,0],"colors":[0,0],"p":[6500,0,0,-6499.999932280723]},{"index":3,"id":2,"status":-21,"mothers":[1,0],"daughters":[5,7],"colors":[502,0],"p":[2045.0606957313294,0,0,2045.0606957313294]},{"index":4,"id":-1,"status":-21,"mothers":[2,0],"daughters":[5,7],"colors":[0,501],"p":[18.185587620127194,0,0,-18.185587620127194]},{"index":5,"id":5,"status":-23,"mothers":[3,4],"daughters":[8,9],"colors":[502,0],"p":[789.4912755191395,16.966274609579287,-8.152836366060338,789.2528497927001]},{"index":6,"id":-5,"status":-23,"mothers":[3,4],"daughters":[10,10],"colors":[0,501],"p":[1158.1394164334606,5.212630148774785,-1.1664104529550787,1158.1175613340918]},{"index":7,"id":24,"status":-23,"mothers":[3,4],"daughters":[11,11],"colors":[0,0],"p":[115.61559139885632,-22.178904758354072,9.319246819015417,79.50469698441026]},{"index":8,"id":5,"status":51,"mothers":[5,0],"daughters":[0,0],"colors":[503,0],"p":[245.3149971066805,11.367139151182524,8.450251350492595,244.8606526273297]},{"index":9,"id":21,"status":51,"mothers":[5,0],"daughters":[0,0],"colors":[502,503],"p":[527.9390696908446,4.6447015162895,-16.14445175012687,527.6717206100274]},{"index":10,"id":-5,"status":52,"mothers":[6,6],"daughters":[0,0],"colors":[0,501],"p":[1173.745275164151,5.552062493477479,-1.3295185263175657,1173.7219806890903]},{"index":11,"id":24,"status":52,"mothers":[7,7],"daughters":[0,0],"colors":[0,0],"p":[116.24694138976984,-21.563903160949597,9.023718925952002,80.62075418474232]}]}},{"label":"single_top","origin":"SingleTop_5FS/Events/delta_1M/macnlops/SubProcesses__P0_bu_td__GF1.0_3__events/shower.jsonl","sequence":9,"expected_sector":3,"registry":{"outer":{"context":2,"provider":2,"directory":"P0_bu_td","nexternal":5,"born_pdgs":[[5,2,6,1],[5,4,6,3]],"sectors":[{"sector":1,"fks":{"i":5,"j":1,"ij":1,"splitting_type":["QCD"]},"processes":[[null,[[5],[2],[6],[1],[21]]],[null,[[5],[4],[6],[3],[21]]]]},{"sector":2,"fks":{"i":5,"j":2,"ij":2,"splitting_type":["QCD"]},"processes":[[null,[[5],[2],[6],[1],[21]]],[null,[[5],[4],[6],[3],[21]]]]},{"sector":3,"fks":{"i":5,"j":3,"ij":3,"splitting_type":["QCD"]},"processes":[[null,[[5],[2],[6],[1],[21]]],[null,[[5],[4],[6],[3],[21]]]]},{"sector":4,"fks":{"i":5,"j":4,"ij":4,"splitting_type":["QCD"]},"processes":[[null,[[5],[2],[6],[1],[21]]],[null,[[5],[4],[6],[3],[21]]]]},{"sector":5,"fks":{"i":5,"j":1,"ij":1,"splitting_type":["QCD"]},"processes":[[null,[[21],[2],[6],[1],[-5]]],[null,[[21],[4],[6],[3],[-5]]]]},{"sector":6,"fks":{"i":5,"j":2,"ij":2,"splitting_type":["QCD"]},"processes":[[null,[[5],[21],[6],[1],[-2]]],[null,[[5],[21],[6],[3],[-4]]]]}]}},"record":{"schema":"macnlops-one-step-v1","sequence":9,"emitted":true,"emission_type":"FSR","branch_indices":{"radiator_before":5,"radiator_after":7,"emitted":8,"recoilers":[9]},"hard_pythia_indices":[3,4,7,9,8],"source_particles":[{"id":5,"status":-1,"colors":[501,0],"p":[244.29068518736923,0,0,244.29068518736926]},{"id":2,"status":-1,"colors":[502,0],"p":[292.5501687176328,0,0,-292.55016871763286]},{"id":6,"status":1,"colors":[501,0],"p":[275.25420143915204,41.960467254283515,-1.5583439514280077,209.9351475142658]},{"id":1,"status":1,"colors":[502,0],"p":[261.58665246585,-41.960467254283515,1.5583439514280077,-258.1946310445294]}],"particles":[{"id":5,"status":-1,"colors":[501,0],"p":[244.2906851873693,0,0,244.2906851873693]},{"id":2,"status":-1,"colors":[502,0],"p":[292.5501687176329,0,0,-292.5501687176329]},{"id":6,"status":1,"colors":[503,0],"p":[269.3522432107657,31.625980543161283,-2.891950002142396,203.99280600683]},{"id":1,"status":1,"colors":[502,0],"p":[257.52247298335936,-41.30854228608258,1.5341325115309197,-254.1831522015198]},{"id":21,"status":1,"colors":[501,503],"p":[9.96613771087725,9.682561742921335,1.3578174906114755,1.930862664426277]}],"pythia_event":[{"index":1,"id":2212,"status":-12,"mothers":[0,0],"daughters":[3,0],"colors":[0,0],"p":[6500,0,0,6499.999932280723]},{"index":2,"id":2212,"status":-12,"mothers":[0,0],"daughters":[4,0],"colors":[0,0],"p":[6500,0,0,-6499.999932280723]},{"index":3,"id":5,"status":-21,"mothers":[1,0],"daughters":[5,6],"colors":[501,0],"p":[244.2906851873693,0,0,244.2906851873693]},{"index":4,"id":2,"status":-21,"mothers":[2,0],"daughters":[5,6],"colors":[502,0],"p":[292.5501687176329,0,0,-292.5501687176329]},{"index":5,"id":6,"status":-23,"mothers":[3,4],"daughters":[7,8],"colors":[501,0],"p":[275.25420143915204,41.960467254283515,-1.5583439514280077,209.9351475142658]},{"index":6,"id":1,"status":-23,"mothers":[3,4],"daughters":[9,9],"colors":[502,0],"p":[261.5866524658501,-41.960467254283515,1.5583439514280077,-258.1946310445294]},{"index":7,"id":6,"status":51,"mothers":[5,0],"daughters":[0,0],"colors":[503,0],"p":[269.3522432107657,31.625980543161283,-2.891950002142396,203.99280600683]},{"index":8,"id":21,"status":51,"mothers":[5,0],"daughters":[0,0],"colors":[501,503],"p":[9.96613771087725,9.682561742921335,1.3578174906114755,1.930862664426277]},{"index":9,"id":1,"status":52,"mothers":[6,6],"daughters":[0,0],"colors":[502,0],"p":[257.52247298335936,-41.30854228608258,1.5341325115309197,-254.1831522015198]}]}},{"label":"ttbar_isr","origin":"TTbar_5FS/Events/delta_1M/macnlops/SubProcesses__P0_gg_ttx__GF3.0_34__events/shower.jsonl","sequence":1,"expected_sector":6,"registry":{"outer":{"context":1,"provider":1,"directory":"P0_gg_ttx","nexternal":5,"born_pdgs":[[21,21,6,-6]],"sectors":[{"sector":1,"fks":{"i":5,"j":1,"ij":1,"splitting_type":["QCD"]},"processes":[[null,[[21],[21],[6],[-6],[21]]]]},{"sector":2,"fks":{"i":5,"j":2,"ij":2,"splitting_type":["QCD"]},"processes":[[null,[[21],[21],[6],[-6],[21]]]]},{"sector":3,"fks":{"i":5,"j":3,"ij":3,"splitting_type":["QCD"]},"processes":[[null,[[21],[21],[6],[-6],[21]]]]},{"sector":4,"fks":{"i":5,"j":4,"ij":4,"splitting_type":["QCD"]},"processes":[[null,[[21],[21],[6],[-6],[21]]]]},{"sector":5,"fks":{"i":5,"j":1,"ij":1,"splitting_type":["QCD"]},"processes":[[null,[[-1],[21],[6],[-6],[-1]]],[null,[[-2],[21],[6],[-6],[-2]]],[null,[[-3],[21],[6],[-6],[-3]]],[null,[[-4],[21],[6],[-6],[-4]]],[null,[[-5],[21],[6],[-6],[-5]]]]},{"sector":6,"fks":{"i":5,"j":1,"ij":1,"splitting_type":["QCD"]},"processes":[[null,[[1],[21],[6],[-6],[1]]],[null,[[2],[21],[6],[-6],[2]]],[null,[[3],[21],[6],[-6],[3]]],[null,[[4],[21],[6],[-6],[4]]],[null,[[5],[21],[6],[-6],[5]]]]},{"sector":7,"fks":{"i":5,"j":2,"ij":2,"splitting_type":["QCD"]},"processes":[[null,[[21],[-1],[6],[-6],[-1]]],[null,[[21],[-2],[6],[-6],[-2]]],[null,[[21],[-3],[6],[-6],[-3]]],[null,[[21],[-4],[6],[-6],[-4]]],[null,[[21],[-5],[6],[-6],[-5]]]]},{"sector":8,"fks":{"i":5,"j":2,"ij":2,"splitting_type":["QCD"]},"processes":[[null,[[21],[1],[6],[-6],[1]]],[null,[[21],[2],[6],[-6],[2]]],[null,[[21],[3],[6],[-6],[3]]],[null,[[21],[4],[6],[-6],[4]]],[null,[[21],[5],[6],[-6],[5]]]]}]}},"record":{"schema":"macnlops-one-step-v1","sequence":1,"emitted":true,"emission_type":"ISR","branch_indices":{"radiator_before":3,"radiator_after":7,"emitted":11,"recoilers":[8]},"hard_pythia_indices":[7,8,9,10,11],"source_particles":[{"id":21,"status":-1,"colors":[503,502],"p":[133.70036141974066,0,0,133.7003614197406]},{"id":21,"status":-1,"colors":[501,503],"p":[1007.3226498469451,0,0,-1007.322649846945]},{"id":6,"status":1,"colors":[501,0],"p":[756.6748798736005,-254.25480221455888,125.0692556009813,-679.9559181932161]},{"id":-6,"status":1,"colors":[0,502],"p":[384.34813139308517,254.25480221455888,-125.0692556009813,-193.66637023398823]}],"particles":[{"id":2,"status":-1,"colors":[503,0],"p":[788.1803525467844,1.7053025658242404e-13,-5.684341886080802e-14,788.1803525467853]},{"id":21,"status":-1,"colors":[501,503],"p":[1007.3226498469451,2.842170943040401e-14,1.4210854715202004e-14,-1007.3226498469451]},{"id":6,"status":1,"colors":[501,0],"p":[603.3081839664223,-201.9000893614005,155.3162679212162,-518.8111175329091]},{"id":-6,"status":1,"colors":[0,502],"p":[533.9460171751869,384.3802449103189,-49.89157523753666,-323.93210291702366]},{"id":2,"status":1,"colors":[502,0],"p":[658.2488012521205,-182.4801555489181,-105.42469268367944,623.6009231497726]}],"pythia_event":[{"index":1,"id":2212,"status":-12,"mothers":[0,0],"daughters":[7,0],"colors":[0,0],"p":[6500,0,0,6499.999932280723]},{"index":2,"id":2212,"status":-12,"mothers":[0,0],"daughters":[8,0],"colors":[0,0],"p":[6500,0,0,-6499.999932280723]},{"index":3,"id":21,"status":-21,"mothers":[7,0],"daughters":[5,6],"colors":[503,502],"p":[133.70036141974055,0,0,133.70036141974055]},{"index":4,"id":21,"status":-21,"mothers":[8,8],"daughters":[5,6],"colors":[501,503],"p":[1007.3226498469448,0,0,-1007.3226498469448]},{"index":5,"id":6,"status":-23,"mothers":[3,4],"daughters":[9,9],"colors":[501,0],"p":[756.6748798736004,-254.25480221455888,125.0692556009813,-679.9559181932161]},{"index":6,"id":-6,"status":-23,"mothers":[3,4],"daughters":[10,10],"colors":[0,502],"p":[384.34813139308517,254.25480221455888,-125.0692556009813,-193.66637023398823]},{"index":7,"id":2,"status":-41,"mothers":[1,0],"daughters":[11,3],"colors":[503,0],"p":[788.1803525467844,1.7053025658242404e-13,-5.684341886080802e-14,788.1803525467853]},{"index":8,"id":21,"status":-42,"mothers":[2,0],"daughters":[4,4],"colors":[501,503],"p":[1007.3226498469451,2.842170943040401e-14,1.4210854715202004e-14,-1007.3226498469451]},{"index":9,"id":6,"status":44,"mothers":[5,5],"daughters":[0,0],"colors":[501,0],"p":[603.3081839664223,-201.9000893614005,155.3162679212162,-518.8111175329091]},{"index":10,"id":-6,"status":44,"mothers":[6,6],"daughters":[0,0],"colors":[0,502],"p":[533.9460171751869,384.3802449103189,-49.89157523753666,-323.93210291702366]},{"index":11,"id":2,"status":43,"mothers":[7,0],"daughters":[0,0],"colors":[502,0],"p":[658.2488012521205,-182.4801555489181,-105.42469268367944,623.6009231497726]}]}}]''')


class TestComponentBranchMapping(unittest.TestCase):
    def fixture(self, label):
        case = next(case for case in PREFIXES if case['label'] == label)
        return deepcopy(case['record']), deepcopy(case['registry'])

    def test_five_processes_and_massive_fsr_saved_prefixes(self):
        for case in PREFIXES:
            with self.subTest(process=case['label'], sequence=case['sequence']):
                result = branch_candidates(
                    case['record'], case['registry'],
                    case['registry']['outer']['directory'])
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].sector, case['expected_sector'])
                self.assertEqual(result[0].born_permutation,
                                 tuple(range(len(case['record']['source_particles']))))
                self.assertEqual(result[0].permutation,
                                 tuple(range(len(case['record']['particles']))))
                self.assertFalse(result[0].requires_marking)

    def test_integration_owner_is_not_a_shower_restriction(self):
        record, registry = self.fixture('ttbar_isr')
        expected = branch_candidates(record, registry)
        record['source_partition'] = {'owner': 999, 'native_sector': 999}
        self.assertEqual(branch_candidates(record, registry), expected)
        # Backwards ISR has changed g into an incoming u plus an emitted u.
        self.assertEqual(record['source_particles'][0]['id'], 21)
        self.assertEqual(record['particles'][0]['id'], 2)

    def test_identical_zz_spectators_follow_ancestry_after_reordering(self):
        record, registry = self.fixture('zz')
        record['particles'][2], record['particles'][3] = (
            record['particles'][3], record['particles'][2])
        record['hard_pythia_indices'][2], record['hard_pythia_indices'][3] = (
            record['hard_pythia_indices'][3], record['hard_pythia_indices'][2])
        result = branch_candidates(record, registry)
        self.assertEqual(result[0].permutation, (0, 1, 3, 2, 4))
        self.assertEqual(result[0].born_permutation, (0, 1, 2, 3))

    def test_source_provider_and_born_order_are_required(self):
        record, registry = self.fixture('dy')
        with self.assertRaisesRegex(MappingError, 'source Born directory'):
            branch_candidates(record, registry, 'P_wrong')
        record['source_particles'][2], record['source_particles'][3] = (
            record['source_particles'][3], record['source_particles'][2])
        with self.assertRaisesRegex(MappingError, 'Born ordering'):
            branch_candidates(record, registry)

    def test_wrong_beam_and_corrupt_endpoint_are_rejected(self):
        record, registry = self.fixture('dy')
        record['branch_indices']['radiator_after'] = record['hard_pythia_indices'][1]
        with self.assertRaisesRegex(MappingError, 'beam'):
            branch_candidates(record, registry)
        record, registry = self.fixture('wbb')
        record['particles'][-1]['p'][0] += 1.
        with self.assertRaisesRegex(MappingError, 'snapshot'):
            branch_candidates(record, registry)

    def test_duplicate_flavour_rows_do_not_duplicate_components(self):
        record, registry = self.fixture('single_top')
        for sector in registry['outer']['sectors']:
            sector['processes'] += deepcopy(sector['processes'])
        self.assertEqual(len(branch_candidates(record, registry)), 1)

    def test_missing_source_route_fails_instead_of_using_the_owner(self):
        record, registry = self.fixture('ttbar')
        registry['outer']['sectors'] = []
        with self.assertRaisesRegex(MappingError, 'no certified'):
            branch_candidates(record, registry)

    def gluon_fsr(self, quarks=False):
        record, registry = self.fixture('ttbar')
        before = record['branch_indices']['radiator_before']
        after = record['branch_indices']['radiator_after']
        emitted = record['branch_indices']['emitted']
        record['source_particles'][3]['id'] = 21
        record['source_particles'][3]['colors'] = [505, 502]
        ids = {before: 21, after: 2 if quarks else 21,
               emitted: -2 if quarks else 21}
        colours = {before: [505, 502], after: [505, 504],
                   emitted: [504, 502]}
        for particle in record['pythia_event']:
            if particle['index'] in ids:
                particle['id'] = ids[particle['index']]
                particle['colors'] = colours[particle['index']]
        for position, index in enumerate(record['hard_pythia_indices']):
            if index in ids:
                record['particles'][position]['id'] = ids[index]
                record['particles'][position]['colors'] = colours[index]
        registry['outer']['born_pdgs'] = [[21, 21, 6, 21]]

        def sector(number, pdgs):
            return {'sector': number,
                    'fks': {'i': 5, 'j': 4, 'ij': 4,
                            'splitting_type': ['QCD']},
                    'processes': [[None, [[pid] for pid in pdgs]]]}
        native = [particle['id'] for particle in record['particles']]
        registry['outer']['sectors'] = [sector(7, native)]
        if quarks:
            swapped = native[:]
            swapped[3], swapped[4] = swapped[4], swapped[3]
            registry['outer']['sectors'].append(sector(8, swapped))
        return record, registry

    def test_identical_gluon_daughters_return_both_certified_orientations(self):
        record, registry = self.gluon_fsr()
        result = branch_candidates(record, registry)
        self.assertEqual(len(result), 2)
        self.assertEqual({c.permutation for c in result},
                         {(0, 1, 2, 3, 4), (0, 1, 2, 4, 3)})
        self.assertEqual({c.orientation for c in result}, {'native', 'gg-swapped'})
        self.assertTrue(all(c.requires_marking for c in result))

    def test_gluon_quark_pair_returns_only_exported_orientations(self):
        record, registry = self.gluon_fsr(quarks=True)
        result = branch_candidates(record, registry)
        self.assertEqual({(c.sector, c.orientation) for c in result},
                         {(7, 'native'), (8, 'gqq-swapped')})
        self.assertTrue(all(c.requires_marking for c in result))
        registry['outer']['sectors'].pop()
        result = branch_candidates(record, registry)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].requires_marking)

    def test_ambiguous_original_identical_legs_require_explicit_ancestry(self):
        record, registry = self.fixture('zz')
        originals = [p for p in record['pythia_event']
                     if abs(p['status']) == 23 and p['id'] == 23]
        self.assertEqual(len(originals), 2)
        record['source_particles'][3]['p'] = record['source_particles'][2]['p'][:]
        originals[1]['p'] = originals[0]['p'][:]
        with self.assertRaisesRegex(MappingError, 'Ambiguous'):
            branch_candidates(record, registry)
        record['source_pythia_indices'] = [
            next(p['index'] for p in record['pythia_event']
                 if abs(p['status']) == 21 and p['p'][3] > 0),
            next(p['index'] for p in record['pythia_event']
                 if abs(p['status']) == 21 and p['p'][3] < 0),
            originals[0]['index'], originals[1]['index']]
        self.assertEqual(len(branch_candidates(record, registry)), 1)

    def test_qed_or_missing_snapshot_is_not_silently_relabelled(self):
        record, registry = self.fixture('wbb')
        emitted = record['branch_indices']['emitted']
        record['particles'][-1]['id'] = 22
        next(p for p in record['pythia_event'] if p['index'] == emitted)['id'] = 22
        with self.assertRaisesRegex(MappingError, 'not a QCD'):
            branch_candidates(record, registry)
        record, registry = self.fixture('wbb')
        record['pythia_event'] = []
        with self.assertRaisesRegex(MappingError, 'snapshot'):
            branch_candidates(record, registry)
        self.assertEqual(branch_candidates({'emitted': False}, {}), ())

    def test_colour_topology_preserves_orientation_and_ignores_tag_names(self):
        first = [{'colors': [501, 0]}, {'colors': [0, 502]},
                 {'colors': [502, 501]}]
        second = [{'colors': [42, 0]}, {'colors': [0, 71]},
                  {'colors': [71, 42]}]
        self.assertEqual(canonical_colour_topology(first),
                         canonical_colour_topology(second))
        second[-1]['colors'].reverse()
        self.assertNotEqual(canonical_colour_topology(first),
                            canonical_colour_topology(second))
        with self.assertRaisesRegex(MappingError, 'complete permutation'):
            canonical_colour_topology(first, (0, 0, 2))


class TestBornBasisTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Four external legs and four tree diagrams only; no integration,
        # event generation, compilation or shower is performed.
        from models import import_ufo
        from madgraph.core import base_objects, diagram_generation, helas_objects
        from madgraph.iolibs.born_support import component_basis_metadata
        model = import_ufo.import_model('sm')
        legs = base_objects.LegList([
            base_objects.Leg({'id': 21, 'state': i > 1, 'number': i + 1})
            for i in range(4)])
        process = base_objects.Process({'legs': legs, 'model': model,
                                        'orders': {'QED': 0}})
        born = helas_objects.HelasMatrixElement(diagram_generation.Amplitude(process))
        cls.born = born
        cls.basis = component_basis_metadata(born)

    def test_basis_configurations_match_the_actual_fortran_writer(self):
        from madgraph.iolibs import export_fks, file_writers
        with tempfile.TemporaryDirectory(prefix='macnlops_basis_') as directory:
            path = Path(directory) / 'born_conf.inc'
            writer = file_writers.FortranWriter(str(path))
            try:
                export_fks.ProcessExporterFortranFKS.write_born_conf_file(
                    None, writer, self.born, None)
            finally:
                writer.close()
            configured = [(int(config), int(graph)) for config, graph in re.findall(
                r'(?i)data\s+mapconfig\(\s*(\d+)\s*\)\s*/\s*(\d+)\s*/',
                path.read_text()) if int(config)]
        self.assertEqual(configured, [(item['config'], item['graph'])
                                     for item in self.basis['graphs']])

    def test_actual_gggg_diagram_and_flow_transport_is_an_involution(self):
        registry = {'outer': {'nexternal': 5, 'component_basis': self.basis}}
        self.assertEqual(len(self.basis['graphs']), 3)
        self.assertEqual(len(self.basis['flows']), 6)
        graph_moved, flow_moved = False, False
        for graph in self.basis['graphs']:
            for flow in range(1, 7):
                original = (graph['config'], graph['graph'], flow)
                target = transport_component_basis(registry, (0, 1, 3, 2), *original)
                graph_moved |= target[:2] != original[:2]
                flow_moved |= target[2] != flow
                self.assertEqual(transport_component_basis(
                    registry, (0, 1, 3, 2), *target), original)
        self.assertTrue(graph_moved)
        self.assertTrue(flow_moved)

    def two_born_gluons(self):
        record, registry = TestComponentBranchMapping().gluon_fsr()
        colours = deepcopy(self.basis['flows'][0])
        old_indices = (3, 4, 5, 6)
        for leg, particle in enumerate(record['source_particles']):
            particle['id'], particle['colors'] = 21, colours[leg][:]
        parent_colours = colours[3]
        endpoint_colours = colours[:3] + [[parent_colours[0], 999], [999, parent_colours[1]]]
        by_index = {particle['index']: particle for particle in record['pythia_event']}
        for leg, index in enumerate(old_indices):
            by_index[index]['id'], by_index[index]['colors'] = 21, colours[leg][:]
        for leg, index in enumerate(record['hard_pythia_indices']):
            by_index[index]['id'], by_index[index]['colors'] = 21, endpoint_colours[leg][:]
            record['particles'][leg]['id'] = 21
            record['particles'][leg]['colors'] = endpoint_colours[leg][:]
        registry['outer']['born_pdgs'] = [[21] * 4]
        registry['outer']['component_basis'] = deepcopy(self.basis)
        registry['outer']['sectors'] = [{
            'sector': 7, 'fks': {'i': 5, 'j': 3, 'ij': 3,
                                 'splitting_type': ['QCD']},
            'processes': [[None, [[21] for _ in range(5)]]]}]
        return record, registry

    def test_second_identical_emitter_transports_the_born_basis(self):
        record, registry = self.two_born_gluons()
        candidates = branch_candidates(record, registry)
        self.assertEqual(len(candidates), 2)
        self.assertEqual({candidate.born_permutation for candidate in candidates},
                         {(0, 1, 3, 2)})
        self.assertEqual({candidate.permutation for candidate in candidates},
                         {(0, 1, 3, 2, 4), (0, 1, 4, 2, 3)})
        graph = self.basis['graphs'][1]
        transported = transport_component_basis(
            registry, candidates[0].born_permutation,
            graph['config'], graph['graph'], 1)
        self.assertNotEqual(transported[:2], (graph['config'], graph['graph']))
        self.assertNotEqual(transported[2], 1)

    def test_missing_or_ambiguous_transport_is_not_a_copied_ordinal(self):
        record, registry = self.two_born_gluons()
        del registry['outer']['component_basis']
        with self.assertRaisesRegex(MappingError, 'exported component basis'):
            branch_candidates(record, registry)
        registry = {'outer': {'nexternal': 5, 'component_basis': deepcopy(self.basis)}}
        registry['outer']['component_basis']['flows'] *= 2
        graph = self.basis['graphs'][0]
        with self.assertRaisesRegex(MappingError, 'no unique certified image'):
            transport_component_basis(registry, (0, 1, 3, 2),
                                      graph['config'], graph['graph'], 1)

    def test_distinct_tagged_states_cannot_be_exchanged(self):
        registry = {'outer': {'nexternal': 5, 'component_basis': deepcopy(self.basis)}}
        registry['outer']['component_basis']['external_states'][3] = [21, True, ['tagged']]
        graph = self.basis['graphs'][0]
        with self.assertRaisesRegex(MappingError, 'tagged/helicity'):
            transport_component_basis(registry, (0, 1, 3, 2),
                                      graph['config'], graph['graph'], 1)


class TestComponentOrbitMeasure(unittest.TestCase):
    @staticmethod
    def registry(born_gluons, quarks=0, antiquarks=0, pair=False):
        born_identity = (math.factorial(born_gluons) * math.factorial(quarks)
                         * math.factorial(antiquarks))
        final = ([21] * (born_gluons - 1) + [2] * (quarks + 1)
                 + [-2] * (antiquarks + 1)) if pair else [21] * (born_gluons + 1)
        real_identity = (math.factorial(born_gluons - 1) * math.factorial(quarks + 1)
                         * math.factorial(antiquarks + 1)) if pair else math.factorial(born_gluons + 1)
        i = len(final) + 2
        if pair:
            j = 3 + final.index(2)
        else:
            j = 3
        row = [[21, False], [21, False]] + [[pid, True] for pid in final]
        return {'outer': {'born_identity': born_identity, 'sectors': [{
            'sector': 1, 'real_identity': real_identity,
            'fks': {'i': i, 'j': j}, 'processes': [[None, row]]}]}}

    def test_reduced_orbit_and_actual_emitter_normalizations(self):
        for gluons in (1, 2, 3, 4):
            normalization, emitters, representative = component_orbit_factors(
                self.registry(gluons), 1)
            self.assertEqual(normalization, gluons + 1)
            self.assertEqual(emitters, gluons)
            self.assertEqual(representative, gluons * (gluons + 1))
        for gluons, quarks, antiquarks in ((1, 0, 0), (2, 0, 0), (2, 2, 3)):
            normalization, emitters, representative = component_orbit_factors(
                self.registry(gluons, quarks, antiquarks, pair=True), 1)
            self.assertEqual(normalization, F((quarks + 1) * (antiquarks + 1), gluons))
            self.assertEqual(emitters, gluons)
            self.assertEqual(representative, (quarks + 1) * (antiquarks + 1))
            self.assertEqual(normalization * emitters, representative)

    def test_noninteger_emitter_orbit_is_rejected(self):
        registry = self.registry(1)
        registry['outer']['sectors'][0]['real_identity'] = 3
        with self.assertRaisesRegex(MappingError, 'integer emitter orbit'):
            component_orbit_factors(registry, 1)

    def test_complementary_gg_orientations_match_pythia_unlabelled_kernel(self):
        ca = F(3)
        for gluons in (1, 2, 4):
            normalization, emitters, representative = component_orbit_factors(
                self.registry(gluons), 1)
            for z in (F(1, 7), F(1, 3), F(1, 2)):
                eta = (1-z)**4 / ((1-z)**4 + z**4)
                pgg = 2*ca*(z/(1-z) + (1-z)/z + z*(1-z))
                # Two Pythia colour ends each carry CA/2 and (1+z^3)/(1-z).
                # Forgetting the two identical daughters' shower names sums
                # this directed kernel and its z <-> 1-z counterpart.
                pythia = ca*((1+z**3)/(1-z) + (1+(1-z)**3)/z)
                self.assertEqual(pgg, pythia)
                unit_first = pgg*eta/normalization
                unit_second = pgg*(1-eta)/normalization
                self.assertEqual(normalization*(unit_first+unit_second), pythia)
                self.assertEqual(emitters*normalization*(unit_first+unit_second),
                                 representative*(unit_first+unit_second))
                # A common rescaling must also act on H, not just K.
                negative = unit_first/F(3)
                self.assertEqual(negative/unit_first,
                                 (normalization*negative)/(normalization*unit_first))


if __name__ == '__main__':
    unittest.main()
