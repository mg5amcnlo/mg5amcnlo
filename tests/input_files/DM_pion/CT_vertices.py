# This file was automatically created by FeynRules 2.3.2
# Mathematica version: 9.0 for Linux x86 (64-bit) (November 20, 2012)
# Date: Thu 11 Jun 2015 17:39:55


from object_library import all_vertices, all_CTvertices, Vertex, CTVertex
import particles as P
import CT_couplings as C
import lorentz as L


V_1 = CTVertex(name = 'V_1',
               type = 'R2',
               particles = [ P.b__tilde__, P.b, P.Y0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS4, L.FFS6 ],
               loop_particles = [ [ [P.b, P.g] ] ],
               couplings = {(0,0,0):C.R2GC_192_121,(0,1,0):C.R2GC_193_122})

V_2 = CTVertex(name = 'V_2',
               type = 'R2',
               particles = [ P.c__tilde__, P.c, P.Y0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS4, L.FFS6 ],
               loop_particles = [ [ [P.c, P.g] ] ],
               couplings = {(0,0,0):C.R2GC_204_129,(0,1,0):C.R2GC_205_130})

V_3 = CTVertex(name = 'V_3',
               type = 'R2',
               particles = [ P.d__tilde__, P.d, P.Y0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS4, L.FFS6 ],
               loop_particles = [ [ [P.d, P.g] ] ],
               couplings = {(0,0,0):C.R2GC_219_139,(0,1,0):C.R2GC_220_140})

V_4 = CTVertex(name = 'V_4',
               type = 'R2',
               particles = [ P.s__tilde__, P.s, P.Y0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS4, L.FFS6 ],
               loop_particles = [ [ [P.g, P.s] ] ],
               couplings = {(0,0,0):C.R2GC_237_152,(0,1,0):C.R2GC_238_153})

V_5 = CTVertex(name = 'V_5',
               type = 'R2',
               particles = [ P.t__tilde__, P.t, P.Y0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS4, L.FFS6 ],
               loop_particles = [ [ [P.g, P.t] ] ],
               couplings = {(0,0,0):C.R2GC_257_167,(0,1,0):C.R2GC_258_168})

V_6 = CTVertex(name = 'V_6',
               type = 'R2',
               particles = [ P.u__tilde__, P.u, P.Y0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS4, L.FFS6 ],
               loop_particles = [ [ [P.g, P.u] ] ],
               couplings = {(0,0,0):C.R2GC_283_189,(0,1,0):C.R2GC_284_190})

V_7 = CTVertex(name = 'V_7',
               type = 'R2',
               particles = [ P.g, P.g, P.g ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVV2 ],
               loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.g] ] ],
               couplings = {(0,0,0):C.R2GC_260_169,(0,0,1):C.R2GC_260_170})

V_8 = CTVertex(name = 'V_8',
               type = 'R2',
               particles = [ P.g, P.g, P.g, P.g ],
               color = [ 'd(-1,1,3)*d(-1,2,4)', 'd(-1,1,3)*f(-1,2,4)', 'd(-1,1,4)*d(-1,2,3)', 'd(-1,1,4)*f(-1,2,3)', 'd(-1,2,3)*f(-1,1,4)', 'd(-1,2,4)*f(-1,1,3)', 'f(-1,1,2)*f(-1,3,4)', 'f(-1,1,3)*f(-1,2,4)', 'f(-1,1,4)*f(-1,2,3)', 'Identity(1,2)*Identity(3,4)', 'Identity(1,3)*Identity(2,4)', 'Identity(1,4)*Identity(2,3)' ],
               lorentz = [ L.VVVV10, L.VVVV2, L.VVVV3, L.VVVV5 ],
               loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.g] ] ],
               couplings = {(2,1,0):C.R2GC_165_101,(2,1,1):C.R2GC_165_102,(0,1,0):C.R2GC_165_101,(0,1,1):C.R2GC_165_102,(4,1,0):C.R2GC_163_97,(4,1,1):C.R2GC_163_98,(3,1,0):C.R2GC_163_97,(3,1,1):C.R2GC_163_98,(8,1,0):C.R2GC_164_99,(8,1,1):C.R2GC_164_100,(7,1,0):C.R2GC_169_108,(7,1,1):C.R2GC_265_176,(6,1,0):C.R2GC_168_106,(6,1,1):C.R2GC_266_177,(5,1,0):C.R2GC_163_97,(5,1,1):C.R2GC_163_98,(1,1,0):C.R2GC_163_97,(1,1,1):C.R2GC_163_98,(11,0,0):C.R2GC_167_104,(11,0,1):C.R2GC_167_105,(10,0,0):C.R2GC_167_104,(10,0,1):C.R2GC_167_105,(9,0,1):C.R2GC_166_103,(2,2,0):C.R2GC_165_101,(2,2,1):C.R2GC_165_102,(0,2,0):C.R2GC_165_101,(0,2,1):C.R2GC_165_102,(6,2,0):C.R2GC_262_172,(6,2,1):C.R2GC_262_173,(4,2,0):C.R2GC_163_97,(4,2,1):C.R2GC_163_98,(3,2,0):C.R2GC_163_97,(3,2,1):C.R2GC_163_98,(8,2,0):C.R2GC_164_99,(8,2,1):C.R2GC_264_175,(7,2,0):C.R2GC_169_108,(7,2,1):C.R2GC_169_109,(5,2,0):C.R2GC_163_97,(5,2,1):C.R2GC_163_98,(1,2,0):C.R2GC_163_97,(1,2,1):C.R2GC_163_98,(2,3,0):C.R2GC_165_101,(2,3,1):C.R2GC_165_102,(0,3,0):C.R2GC_165_101,(0,3,1):C.R2GC_165_102,(4,3,0):C.R2GC_163_97,(4,3,1):C.R2GC_163_98,(3,3,0):C.R2GC_163_97,(3,3,1):C.R2GC_163_98,(8,3,0):C.R2GC_164_99,(8,3,1):C.R2GC_261_171,(6,3,0):C.R2GC_168_106,(6,3,1):C.R2GC_168_107,(7,3,0):C.R2GC_263_174,(7,3,1):C.R2GC_165_102,(5,3,0):C.R2GC_163_97,(5,3,1):C.R2GC_163_98,(1,3,0):C.R2GC_163_97,(1,3,1):C.R2GC_163_98})

V_9 = CTVertex(name = 'V_9',
               type = 'R2',
               particles = [ P.u__tilde__, P.d, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3, L.FFS5 ],
               loop_particles = [ [ [P.d, P.g, P.u] ] ],
               couplings = {(0,0,0):C.R2GC_286_192,(0,1,0):C.R2GC_287_193})

V_10 = CTVertex(name = 'V_10',
                type = 'R2',
                particles = [ P.c__tilde__, P.d, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS3, L.FFS5 ],
                loop_particles = [ [ [P.c, P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_223_143,(0,1,0):C.R2GC_222_142})

V_11 = CTVertex(name = 'V_11',
                type = 'R2',
                particles = [ P.u__tilde__, P.s, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS3, L.FFS5 ],
                loop_particles = [ [ [P.g, P.s, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_289_195,(0,1,0):C.R2GC_290_196})

V_12 = CTVertex(name = 'V_12',
                type = 'R2',
                particles = [ P.c__tilde__, P.s, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS3, L.FFS5 ],
                loop_particles = [ [ [P.c, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_241_156,(0,1,0):C.R2GC_240_155})

V_13 = CTVertex(name = 'V_13',
                type = 'R2',
                particles = [ P.t__tilde__, P.b, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS3, L.FFS5 ],
                loop_particles = [ [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_252_162,(0,1,0):C.R2GC_253_163})

V_14 = CTVertex(name = 'V_14',
                type = 'R2',
                particles = [ P.d__tilde__, P.d, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS1 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_217_137})

V_15 = CTVertex(name = 'V_15',
                type = 'R2',
                particles = [ P.s__tilde__, P.s, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS1 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_235_150})

V_16 = CTVertex(name = 'V_16',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS1 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_191_120})

V_17 = CTVertex(name = 'V_17',
                type = 'R2',
                particles = [ P.d__tilde__, P.d, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS2 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_216_136})

V_18 = CTVertex(name = 'V_18',
                type = 'R2',
                particles = [ P.s__tilde__, P.s, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS2 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_234_149})

V_19 = CTVertex(name = 'V_19',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS2 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_190_119})

V_20 = CTVertex(name = 'V_20',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS1 ],
                loop_particles = [ [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_279_185})

V_21 = CTVertex(name = 'V_21',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS1 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_202_127})

V_22 = CTVertex(name = 'V_22',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS1 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_255_165})

V_23 = CTVertex(name = 'V_23',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS2 ],
                loop_particles = [ [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_280_186})

V_24 = CTVertex(name = 'V_24',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS2 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_203_128})

V_25 = CTVertex(name = 'V_25',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS2 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_256_166})

V_26 = CTVertex(name = 'V_26',
                type = 'R2',
                particles = [ P.d__tilde__, P.u, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS3, L.FFS5 ],
                loop_particles = [ [ [P.d, P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_281_187,(0,1,0):C.R2GC_277_183})

V_27 = CTVertex(name = 'V_27',
                type = 'R2',
                particles = [ P.s__tilde__, P.u, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS3, L.FFS5 ],
                loop_particles = [ [ [P.g, P.s, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_282_188,(0,1,0):C.R2GC_278_184})

V_28 = CTVertex(name = 'V_28',
                type = 'R2',
                particles = [ P.d__tilde__, P.c, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS3, L.FFS5 ],
                loop_particles = [ [ [P.c, P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_215_135,(0,1,0):C.R2GC_218_138})

V_29 = CTVertex(name = 'V_29',
                type = 'R2',
                particles = [ P.s__tilde__, P.c, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS3, L.FFS5 ],
                loop_particles = [ [ [P.c, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_233_148,(0,1,0):C.R2GC_236_151})

V_30 = CTVertex(name = 'V_30',
                type = 'R2',
                particles = [ P.b__tilde__, P.t, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS3, L.FFS5 ],
                loop_particles = [ [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_254_164,(0,1,0):C.R2GC_251_161})

V_31 = CTVertex(name = 'V_31',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.Y1 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV5, L.FFV6 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_185_114,(0,1,0):C.R2GC_186_115})

V_32 = CTVertex(name = 'V_32',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.Y1 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV5, L.FFV6 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_197_123,(0,1,0):C.R2GC_198_124})

V_33 = CTVertex(name = 'V_33',
                type = 'R2',
                particles = [ P.d__tilde__, P.d, P.Y1 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV5, L.FFV6 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_209_131,(0,1,0):C.R2GC_210_132})

V_34 = CTVertex(name = 'V_34',
                type = 'R2',
                particles = [ P.s__tilde__, P.s, P.Y1 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV5, L.FFV6 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_227_144,(0,1,0):C.R2GC_228_145})

V_35 = CTVertex(name = 'V_35',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.Y1 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV5, L.FFV6 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_245_157,(0,1,0):C.R2GC_246_158})

V_36 = CTVertex(name = 'V_36',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.Y1 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV5, L.FFV6 ],
                loop_particles = [ [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_270_178,(0,1,0):C.R2GC_271_179})

V_37 = CTVertex(name = 'V_37',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_172_112})

V_38 = CTVertex(name = 'V_38',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_172_112})

V_39 = CTVertex(name = 'V_39',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_172_112})

V_40 = CTVertex(name = 'V_40',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_171_111})

V_41 = CTVertex(name = 'V_41',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_171_111})

V_42 = CTVertex(name = 'V_42',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_171_111})

V_43 = CTVertex(name = 'V_43',
                type = 'R2',
                particles = [ P.d__tilde__, P.u, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV3 ],
                loop_particles = [ [ [P.d, P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_273_181})

V_44 = CTVertex(name = 'V_44',
                type = 'R2',
                particles = [ P.s__tilde__, P.u, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV3 ],
                loop_particles = [ [ [P.g, P.s, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_274_182})

V_45 = CTVertex(name = 'V_45',
                type = 'R2',
                particles = [ P.d__tilde__, P.c, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV3 ],
                loop_particles = [ [ [P.c, P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_212_134})

V_46 = CTVertex(name = 'V_46',
                type = 'R2',
                particles = [ P.s__tilde__, P.c, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV3 ],
                loop_particles = [ [ [P.c, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_230_147})

V_47 = CTVertex(name = 'V_47',
                type = 'R2',
                particles = [ P.b__tilde__, P.t, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV3 ],
                loop_particles = [ [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_248_160})

V_48 = CTVertex(name = 'V_48',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV3, L.FFV8 ],
                loop_particles = [ [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_200_126,(0,1,0):C.R2GC_189_118})

V_49 = CTVertex(name = 'V_49',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV3, L.FFV8 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_200_126,(0,1,0):C.R2GC_189_118})

V_50 = CTVertex(name = 'V_50',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV3, L.FFV8 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_200_126,(0,1,0):C.R2GC_189_118})

V_51 = CTVertex(name = 'V_51',
                type = 'R2',
                particles = [ P.d__tilde__, P.d, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_170_110})

V_52 = CTVertex(name = 'V_52',
                type = 'R2',
                particles = [ P.s__tilde__, P.s, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_170_110})

V_53 = CTVertex(name = 'V_53',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_170_110})

V_54 = CTVertex(name = 'V_54',
                type = 'R2',
                particles = [ P.d__tilde__, P.d, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_171_111})

V_55 = CTVertex(name = 'V_55',
                type = 'R2',
                particles = [ P.s__tilde__, P.s, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_171_111})

V_56 = CTVertex(name = 'V_56',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_171_111})

V_57 = CTVertex(name = 'V_57',
                type = 'R2',
                particles = [ P.u__tilde__, P.d, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV3 ],
                loop_particles = [ [ [P.d, P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_285_191})

V_58 = CTVertex(name = 'V_58',
                type = 'R2',
                particles = [ P.c__tilde__, P.d, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV3 ],
                loop_particles = [ [ [P.c, P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_221_141})

V_59 = CTVertex(name = 'V_59',
                type = 'R2',
                particles = [ P.u__tilde__, P.s, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV3 ],
                loop_particles = [ [ [P.g, P.s, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_288_194})

V_60 = CTVertex(name = 'V_60',
                type = 'R2',
                particles = [ P.c__tilde__, P.s, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV3 ],
                loop_particles = [ [ [P.c, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_239_154})

V_61 = CTVertex(name = 'V_61',
                type = 'R2',
                particles = [ P.t__tilde__, P.b, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV3 ],
                loop_particles = [ [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_248_160})

V_62 = CTVertex(name = 'V_62',
                type = 'R2',
                particles = [ P.d__tilde__, P.d, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV3, L.FFV4 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_188_117,(0,1,0):C.R2GC_189_118})

V_63 = CTVertex(name = 'V_63',
                type = 'R2',
                particles = [ P.s__tilde__, P.s, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV3, L.FFV4 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_188_117,(0,1,0):C.R2GC_189_118})

V_64 = CTVertex(name = 'V_64',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV3, L.FFV4 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_188_117,(0,1,0):C.R2GC_189_118})

V_65 = CTVertex(name = 'V_65',
                type = 'R2',
                particles = [ P.u__tilde__, P.u ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF1, L.FF2 ],
                loop_particles = [ [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_272_180,(0,1,0):C.R2GC_182_113})

V_66 = CTVertex(name = 'V_66',
                type = 'R2',
                particles = [ P.c__tilde__, P.c ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF1, L.FF2 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_199_125,(0,1,0):C.R2GC_182_113})

V_67 = CTVertex(name = 'V_67',
                type = 'R2',
                particles = [ P.t__tilde__, P.t ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF1, L.FF2 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_247_159,(0,1,0):C.R2GC_182_113})

V_68 = CTVertex(name = 'V_68',
                type = 'R2',
                particles = [ P.d__tilde__, P.d ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF1, L.FF2 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_211_133,(0,1,0):C.R2GC_182_113})

V_69 = CTVertex(name = 'V_69',
                type = 'R2',
                particles = [ P.s__tilde__, P.s ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF1, L.FF2 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_229_146,(0,1,0):C.R2GC_182_113})

V_70 = CTVertex(name = 'V_70',
                type = 'R2',
                particles = [ P.b__tilde__, P.b ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FF1, L.FF2 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_187_116,(0,1,0):C.R2GC_182_113})

V_71 = CTVertex(name = 'V_71',
                type = 'R2',
                particles = [ P.g, P.g ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VV2, L.VV3, L.VV4 ],
                loop_particles = [ [ [P.b] ], [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.c] ], [ [P.d] ], [ [P.g] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                couplings = {(0,2,4):C.R2GC_138_1,(0,0,0):C.R2GC_146_31,(0,0,2):C.R2GC_146_32,(0,0,3):C.R2GC_146_33,(0,0,5):C.R2GC_146_34,(0,0,6):C.R2GC_146_35,(0,0,7):C.R2GC_146_36,(0,1,1):C.R2GC_139_2})

V_72 = CTVertex(name = 'V_72',
                type = 'R2',
                particles = [ P.g, P.g, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVS1 ],
                loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_152_51,(0,0,1):C.R2GC_152_52,(0,0,2):C.R2GC_152_53,(0,0,3):C.R2GC_152_54,(0,0,4):C.R2GC_152_55,(0,0,5):C.R2GC_152_56})

V_73 = CTVertex(name = 'V_73',
                type = 'R2',
                particles = [ P.g, P.g, P.Y0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVS1 ],
                loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_153_57,(0,0,1):C.R2GC_153_58,(0,0,2):C.R2GC_153_59,(0,0,3):C.R2GC_153_60,(0,0,4):C.R2GC_153_61,(0,0,5):C.R2GC_153_62})

V_74 = CTVertex(name = 'V_74',
                type = 'R2',
                particles = [ P.g, P.g, P.Y1, P.Y1 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVVV10 ],
                loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_145_25,(0,0,1):C.R2GC_145_26,(0,0,2):C.R2GC_145_27,(0,0,3):C.R2GC_145_28,(0,0,4):C.R2GC_145_29,(0,0,5):C.R2GC_145_30})

V_75 = CTVertex(name = 'V_75',
                type = 'R2',
                particles = [ P.a, P.g, P.g, P.Y1 ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.VVVV10 ],
                loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_143_13,(0,0,1):C.R2GC_143_14,(0,0,2):C.R2GC_143_15,(0,0,3):C.R2GC_143_16,(0,0,4):C.R2GC_143_17,(0,0,5):C.R2GC_143_18})

V_76 = CTVertex(name = 'V_76',
                type = 'R2',
                particles = [ P.g, P.g, P.Y1, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVVV10 ],
                loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_150_43,(0,0,1):C.R2GC_150_44,(0,0,2):C.R2GC_150_45,(0,0,3):C.R2GC_150_46,(0,0,4):C.R2GC_150_47,(0,0,5):C.R2GC_150_48})

V_77 = CTVertex(name = 'V_77',
                type = 'R2',
                particles = [ P.g, P.g, P.g, P.Y1 ],
                color = [ 'd(1,2,3)', 'f(1,2,3)' ],
                lorentz = [ L.VVVV1, L.VVVV10 ],
                loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                couplings = {(1,0,0):C.R2GC_142_7,(1,0,1):C.R2GC_142_8,(1,0,2):C.R2GC_142_9,(1,0,3):C.R2GC_142_10,(1,0,4):C.R2GC_142_11,(1,0,5):C.R2GC_142_12,(0,1,0):C.R2GC_144_19,(0,1,1):C.R2GC_144_20,(0,1,2):C.R2GC_144_21,(0,1,3):C.R2GC_144_22,(0,1,4):C.R2GC_144_23,(0,1,5):C.R2GC_144_24})

V_78 = CTVertex(name = 'V_78',
                type = 'R2',
                particles = [ P.g, P.g, P.W__minus__, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVVV10 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.d] ], [ [P.c, P.s] ], [ [P.d, P.u] ], [ [P.s, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_158_87,(0,0,1):C.R2GC_158_88,(0,0,2):C.R2GC_158_89,(0,0,3):C.R2GC_158_90,(0,0,4):C.R2GC_158_91})

V_79 = CTVertex(name = 'V_79',
                type = 'R2',
                particles = [ P.a, P.g, P.g, P.Z ],
                color = [ 'Identity(2,3)' ],
                lorentz = [ L.VVVV10 ],
                loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_147_37,(0,0,1):C.R2GC_147_38})

V_80 = CTVertex(name = 'V_80',
                type = 'R2',
                particles = [ P.g, P.g, P.Z, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVVV10 ],
                loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_151_49,(0,0,1):C.R2GC_151_50})

V_81 = CTVertex(name = 'V_81',
                type = 'R2',
                particles = [ P.a, P.a, P.g, P.g ],
                color = [ 'Identity(3,4)' ],
                lorentz = [ L.VVVV10 ],
                loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_140_3,(0,0,1):C.R2GC_140_4})

V_82 = CTVertex(name = 'V_82',
                type = 'R2',
                particles = [ P.g, P.g, P.g, P.Z ],
                color = [ 'd(1,2,3)', 'f(1,2,3)' ],
                lorentz = [ L.VVVV1, L.VVVV10 ],
                loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                couplings = {(1,0,0):C.R2GC_149_41,(1,0,1):C.R2GC_149_42,(0,1,0):C.R2GC_148_39,(0,1,1):C.R2GC_148_40})

V_83 = CTVertex(name = 'V_83',
                type = 'R2',
                particles = [ P.a, P.g, P.g, P.g ],
                color = [ 'd(2,3,4)' ],
                lorentz = [ L.VVVV10 ],
                loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c], [P.t], [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_141_5,(0,0,1):C.R2GC_141_6})

V_84 = CTVertex(name = 'V_84',
                type = 'R2',
                particles = [ P.g, P.g, P.H, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVSS1 ],
                loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_154_63,(0,0,1):C.R2GC_154_64,(0,0,2):C.R2GC_154_65,(0,0,3):C.R2GC_154_66,(0,0,4):C.R2GC_154_67,(0,0,5):C.R2GC_154_68})

V_85 = CTVertex(name = 'V_85',
                type = 'R2',
                particles = [ P.g, P.g, P.G0, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVSS1 ],
                loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_154_63,(0,0,1):C.R2GC_154_64,(0,0,2):C.R2GC_154_65,(0,0,3):C.R2GC_154_66,(0,0,4):C.R2GC_154_67,(0,0,5):C.R2GC_154_68})

V_86 = CTVertex(name = 'V_86',
                type = 'R2',
                particles = [ P.g, P.g, P.G__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVSS1 ],
                loop_particles = [ [ [P.b, P.t] ], [ [P.c, P.d] ], [ [P.c, P.s] ], [ [P.d, P.u] ], [ [P.s, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_159_92,(0,0,1):C.R2GC_159_93,(0,0,2):C.R2GC_159_94,(0,0,3):C.R2GC_159_95,(0,0,4):C.R2GC_159_96})

V_87 = CTVertex(name = 'V_87',
                type = 'R2',
                particles = [ P.g, P.g, P.H, P.Y0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVSS1 ],
                loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_156_75,(0,0,1):C.R2GC_156_76,(0,0,2):C.R2GC_156_77,(0,0,3):C.R2GC_156_78,(0,0,4):C.R2GC_156_79,(0,0,5):C.R2GC_156_80})

V_88 = CTVertex(name = 'V_88',
                type = 'R2',
                particles = [ P.g, P.g, P.G0, P.Y0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVSS1 ],
                loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_155_69,(0,0,1):C.R2GC_155_70,(0,0,2):C.R2GC_155_71,(0,0,3):C.R2GC_155_72,(0,0,4):C.R2GC_155_73,(0,0,5):C.R2GC_155_74})

V_89 = CTVertex(name = 'V_89',
                type = 'R2',
                particles = [ P.g, P.g, P.Y0, P.Y0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.VVSS1 ],
                loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                couplings = {(0,0,0):C.R2GC_157_81,(0,0,1):C.R2GC_157_82,(0,0,2):C.R2GC_157_83,(0,0,3):C.R2GC_157_84,(0,0,4):C.R2GC_157_85,(0,0,5):C.R2GC_157_86})

V_90 = CTVertex(name = 'V_90',
                type = 'UV',
                particles = [ P.b__tilde__, P.b, P.Y0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS4, L.FFS6 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.UVGC_192_38,(0,1,0):C.UVGC_193_39})

V_91 = CTVertex(name = 'V_91',
                type = 'UV',
                particles = [ P.c__tilde__, P.c, P.Y0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS4, L.FFS6 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.UVGC_204_50,(0,1,0):C.UVGC_205_51})

V_92 = CTVertex(name = 'V_92',
                type = 'UV',
                particles = [ P.d__tilde__, P.d, P.Y0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS4, L.FFS6 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,0,0):C.UVGC_219_71,(0,1,0):C.UVGC_220_72})

V_93 = CTVertex(name = 'V_93',
                type = 'UV',
                particles = [ P.s__tilde__, P.s, P.Y0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS4, L.FFS6 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,0,0):C.UVGC_237_101,(0,1,0):C.UVGC_238_102})

V_94 = CTVertex(name = 'V_94',
                type = 'UV',
                particles = [ P.t__tilde__, P.t, P.Y0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS4, L.FFS6 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.UVGC_257_137,(0,1,0):C.UVGC_258_138})

V_95 = CTVertex(name = 'V_95',
                type = 'UV',
                particles = [ P.u__tilde__, P.u, P.Y0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS4, L.FFS6 ],
                loop_particles = [ [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.UVGC_283_215,(0,1,0):C.UVGC_284_216})

V_96 = CTVertex(name = 'V_96',
                type = 'UV',
                particles = [ P.g, P.g, P.g ],
                color = [ 'f(1,2,3)' ],
                lorentz = [ L.VVV1, L.VVV2, L.VVV3 ],
                loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.g] ], [ [P.ghG] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                couplings = {(0,1,0):C.UVGC_260_145,(0,1,1):C.UVGC_260_146,(0,1,2):C.UVGC_260_147,(0,1,5):C.UVGC_260_148,(0,1,6):C.UVGC_260_149,(0,1,7):C.UVGC_260_150,(0,2,3):C.UVGC_160_1,(0,0,4):C.UVGC_161_2})

V_97 = CTVertex(name = 'V_97',
                type = 'UV',
                particles = [ P.g, P.g, P.g, P.g ],
                color = [ 'd(-1,1,3)*d(-1,2,4)', 'd(-1,1,3)*f(-1,2,4)', 'd(-1,1,4)*d(-1,2,3)', 'd(-1,1,4)*f(-1,2,3)', 'd(-1,2,3)*f(-1,1,4)', 'd(-1,2,4)*f(-1,1,3)', 'f(-1,1,2)*f(-1,3,4)', 'f(-1,1,3)*f(-1,2,4)', 'f(-1,1,4)*f(-1,2,3)', 'Identity(1,2)*Identity(3,4)', 'Identity(1,3)*Identity(2,4)', 'Identity(1,4)*Identity(2,3)' ],
                lorentz = [ L.VVVV10, L.VVVV2, L.VVVV3, L.VVVV5 ],
                loop_particles = [ [ [P.b] ], [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.c] ], [ [P.d] ], [ [P.g] ], [ [P.ghG] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                couplings = {(2,1,4):C.UVGC_164_8,(2,1,5):C.UVGC_164_7,(0,1,4):C.UVGC_164_8,(0,1,5):C.UVGC_164_7,(4,1,4):C.UVGC_163_5,(4,1,5):C.UVGC_163_6,(3,1,4):C.UVGC_163_5,(3,1,5):C.UVGC_163_6,(8,1,4):C.UVGC_164_7,(8,1,5):C.UVGC_164_8,(7,1,0):C.UVGC_265_177,(7,1,2):C.UVGC_265_178,(7,1,3):C.UVGC_265_179,(7,1,4):C.UVGC_265_180,(7,1,5):C.UVGC_265_181,(7,1,6):C.UVGC_265_182,(7,1,7):C.UVGC_265_183,(7,1,8):C.UVGC_265_184,(6,1,0):C.UVGC_265_177,(6,1,2):C.UVGC_265_178,(6,1,3):C.UVGC_265_179,(6,1,4):C.UVGC_266_185,(6,1,5):C.UVGC_266_186,(6,1,6):C.UVGC_265_182,(6,1,7):C.UVGC_265_183,(6,1,8):C.UVGC_265_184,(5,1,4):C.UVGC_163_5,(5,1,5):C.UVGC_163_6,(1,1,4):C.UVGC_163_5,(1,1,5):C.UVGC_163_6,(11,0,4):C.UVGC_167_11,(11,0,5):C.UVGC_167_12,(10,0,4):C.UVGC_167_11,(10,0,5):C.UVGC_167_12,(9,0,4):C.UVGC_166_9,(9,0,5):C.UVGC_166_10,(2,2,4):C.UVGC_164_8,(2,2,5):C.UVGC_164_7,(0,2,4):C.UVGC_164_8,(0,2,5):C.UVGC_164_7,(6,2,0):C.UVGC_262_159,(6,2,2):C.UVGC_262_160,(6,2,3):C.UVGC_262_161,(6,2,4):C.UVGC_262_162,(6,2,5):C.UVGC_262_163,(6,2,6):C.UVGC_262_164,(6,2,7):C.UVGC_262_165,(6,2,8):C.UVGC_262_166,(4,2,4):C.UVGC_163_5,(4,2,5):C.UVGC_163_6,(3,2,4):C.UVGC_163_5,(3,2,5):C.UVGC_163_6,(8,2,0):C.UVGC_264_169,(8,2,2):C.UVGC_264_170,(8,2,3):C.UVGC_264_171,(8,2,4):C.UVGC_264_172,(8,2,5):C.UVGC_264_173,(8,2,6):C.UVGC_264_174,(8,2,7):C.UVGC_264_175,(8,2,8):C.UVGC_264_176,(7,2,1):C.UVGC_168_13,(7,2,4):C.UVGC_169_15,(7,2,5):C.UVGC_169_16,(5,2,4):C.UVGC_163_5,(5,2,5):C.UVGC_163_6,(1,2,4):C.UVGC_163_5,(1,2,5):C.UVGC_163_6,(2,3,4):C.UVGC_164_8,(2,3,5):C.UVGC_164_7,(0,3,4):C.UVGC_164_8,(0,3,5):C.UVGC_164_7,(4,3,4):C.UVGC_163_5,(4,3,5):C.UVGC_163_6,(3,3,4):C.UVGC_163_5,(3,3,5):C.UVGC_163_6,(8,3,0):C.UVGC_261_151,(8,3,2):C.UVGC_261_152,(8,3,3):C.UVGC_261_153,(8,3,4):C.UVGC_261_154,(8,3,5):C.UVGC_261_155,(8,3,6):C.UVGC_261_156,(8,3,7):C.UVGC_261_157,(8,3,8):C.UVGC_261_158,(6,3,1):C.UVGC_168_13,(6,3,4):C.UVGC_168_14,(6,3,5):C.UVGC_166_9,(7,3,0):C.UVGC_262_159,(7,3,2):C.UVGC_262_160,(7,3,3):C.UVGC_262_161,(7,3,4):C.UVGC_263_167,(7,3,5):C.UVGC_263_168,(7,3,6):C.UVGC_262_164,(7,3,7):C.UVGC_262_165,(7,3,8):C.UVGC_262_166,(5,3,4):C.UVGC_163_5,(5,3,5):C.UVGC_163_6,(1,3,4):C.UVGC_163_5,(1,3,5):C.UVGC_163_6})

V_98 = CTVertex(name = 'V_98',
                type = 'UV',
                particles = [ P.u__tilde__, P.d, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS3, L.FFS5 ],
                loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.UVGC_286_220,(0,0,2):C.UVGC_286_221,(0,0,1):C.UVGC_286_222,(0,1,0):C.UVGC_287_223,(0,1,2):C.UVGC_287_224,(0,1,1):C.UVGC_287_225})

V_99 = CTVertex(name = 'V_99',
                type = 'UV',
                particles = [ P.c__tilde__, P.d, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS3, L.FFS5 ],
                loop_particles = [ [ [P.c, P.d, P.g] ], [ [P.c, P.g] ], [ [P.d, P.g] ] ],
                couplings = {(0,0,1):C.UVGC_223_79,(0,0,2):C.UVGC_223_80,(0,0,0):C.UVGC_223_81,(0,1,1):C.UVGC_222_76,(0,1,2):C.UVGC_222_77,(0,1,0):C.UVGC_222_78})

V_100 = CTVertex(name = 'V_100',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.s, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS3, L.FFS5 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_289_229,(0,0,2):C.UVGC_289_230,(0,0,1):C.UVGC_289_231,(0,1,0):C.UVGC_290_232,(0,1,2):C.UVGC_290_233,(0,1,1):C.UVGC_290_234})

V_101 = CTVertex(name = 'V_101',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS3, L.FFS5 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.s] ], [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_241_109,(0,0,2):C.UVGC_241_110,(0,0,1):C.UVGC_241_111,(0,1,0):C.UVGC_240_106,(0,1,2):C.UVGC_240_107,(0,1,1):C.UVGC_240_108})

V_102 = CTVertex(name = 'V_102',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS3, L.FFS5 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_252_126,(0,0,2):C.UVGC_252_127,(0,0,1):C.UVGC_252_128,(0,1,0):C.UVGC_253_129,(0,1,2):C.UVGC_253_130,(0,1,1):C.UVGC_253_131})

V_103 = CTVertex(name = 'V_103',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS1 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_217_67})

V_104 = CTVertex(name = 'V_104',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS1 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_235_97})

V_105 = CTVertex(name = 'V_105',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS1 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_191_37})

V_106 = CTVertex(name = 'V_106',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS2 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_216_66})

V_107 = CTVertex(name = 'V_107',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS2 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_234_96})

V_108 = CTVertex(name = 'V_108',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS2 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_190_36})

V_109 = CTVertex(name = 'V_109',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS1 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_279_207})

V_110 = CTVertex(name = 'V_110',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS1 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_202_48})

V_111 = CTVertex(name = 'V_111',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS1 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_255_135})

V_112 = CTVertex(name = 'V_112',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS2 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_280_208})

V_113 = CTVertex(name = 'V_113',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS2 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_203_49})

V_114 = CTVertex(name = 'V_114',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS2 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_256_136})

V_115 = CTVertex(name = 'V_115',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS3, L.FFS5 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_281_209,(0,0,2):C.UVGC_281_210,(0,0,1):C.UVGC_281_211,(0,1,0):C.UVGC_277_201,(0,1,2):C.UVGC_277_202,(0,1,1):C.UVGC_277_203})

V_116 = CTVertex(name = 'V_116',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.u, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS3, L.FFS5 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_282_212,(0,0,2):C.UVGC_282_213,(0,0,1):C.UVGC_282_214,(0,1,0):C.UVGC_278_204,(0,1,2):C.UVGC_278_205,(0,1,1):C.UVGC_278_206})

V_117 = CTVertex(name = 'V_117',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.c, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS3, L.FFS5 ],
                 loop_particles = [ [ [P.c, P.d, P.g] ], [ [P.c, P.g] ], [ [P.d, P.g] ] ],
                 couplings = {(0,0,1):C.UVGC_215_63,(0,0,2):C.UVGC_215_64,(0,0,0):C.UVGC_215_65,(0,1,1):C.UVGC_218_68,(0,1,2):C.UVGC_218_69,(0,1,0):C.UVGC_218_70})

V_118 = CTVertex(name = 'V_118',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS3, L.FFS5 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.s] ], [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_233_93,(0,0,2):C.UVGC_233_94,(0,0,1):C.UVGC_233_95,(0,1,0):C.UVGC_236_98,(0,1,2):C.UVGC_236_99,(0,1,1):C.UVGC_236_100})

V_119 = CTVertex(name = 'V_119',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS3, L.FFS5 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_254_132,(0,0,2):C.UVGC_254_133,(0,0,1):C.UVGC_254_134,(0,1,0):C.UVGC_251_123,(0,1,2):C.UVGC_251_124,(0,1,1):C.UVGC_251_125})

V_120 = CTVertex(name = 'V_120',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.Y1 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV5, L.FFV6 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_185_31,(0,1,0):C.UVGC_186_32})

V_121 = CTVertex(name = 'V_121',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.Y1 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV5, L.FFV6 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_197_43,(0,1,0):C.UVGC_198_44})

V_122 = CTVertex(name = 'V_122',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.Y1 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV5, L.FFV6 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_209_55,(0,1,0):C.UVGC_210_56})

V_123 = CTVertex(name = 'V_123',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.Y1 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV5, L.FFV6 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_227_85,(0,1,0):C.UVGC_228_86})

V_124 = CTVertex(name = 'V_124',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.Y1 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV5, L.FFV6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_245_115,(0,1,0):C.UVGC_246_116})

V_125 = CTVertex(name = 'V_125',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.Y1 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV5, L.FFV6 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_270_190,(0,1,0):C.UVGC_271_191})

V_126 = CTVertex(name = 'V_126',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1, L.FFV6 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_172_19,(0,1,0):C.UVGC_268_188})

V_127 = CTVertex(name = 'V_127',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1, L.FFV6 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_172_19,(0,1,0):C.UVGC_195_41})

V_128 = CTVertex(name = 'V_128',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1, L.FFV6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_172_19,(0,1,0):C.UVGC_243_113})

V_129 = CTVertex(name = 'V_129',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV6 ],
                 loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.u] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                 couplings = {(0,0,5):C.UVGC_171_18,(0,1,0):C.UVGC_184_22,(0,1,1):C.UVGC_184_23,(0,1,2):C.UVGC_184_24,(0,1,3):C.UVGC_184_25,(0,1,4):C.UVGC_184_26,(0,1,6):C.UVGC_184_27,(0,1,7):C.UVGC_184_28,(0,1,8):C.UVGC_184_29,(0,1,5):C.UVGC_269_189})

V_130 = CTVertex(name = 'V_130',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV6 ],
                 loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.c, P.g] ], [ [P.d] ], [ [P.g] ], [ [P.ghG] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_171_18,(0,1,0):C.UVGC_184_22,(0,1,1):C.UVGC_184_23,(0,1,3):C.UVGC_184_24,(0,1,4):C.UVGC_184_25,(0,1,5):C.UVGC_184_26,(0,1,6):C.UVGC_184_27,(0,1,7):C.UVGC_184_28,(0,1,8):C.UVGC_184_29,(0,1,2):C.UVGC_196_42})

V_131 = CTVertex(name = 'V_131',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV6 ],
                 loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.t] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                 couplings = {(0,0,5):C.UVGC_171_18,(0,1,0):C.UVGC_184_22,(0,1,1):C.UVGC_184_23,(0,1,2):C.UVGC_184_24,(0,1,3):C.UVGC_184_25,(0,1,4):C.UVGC_184_26,(0,1,6):C.UVGC_184_27,(0,1,7):C.UVGC_184_28,(0,1,8):C.UVGC_184_29,(0,1,5):C.UVGC_244_114})

V_132 = CTVertex(name = 'V_132',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_273_193,(0,0,2):C.UVGC_273_194,(0,0,1):C.UVGC_273_195})

V_133 = CTVertex(name = 'V_133',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.u, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_274_196,(0,0,2):C.UVGC_274_197,(0,0,1):C.UVGC_274_198})

V_134 = CTVertex(name = 'V_134',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.c, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.c, P.d, P.g] ], [ [P.c, P.g] ], [ [P.d, P.g] ] ],
                 couplings = {(0,0,1):C.UVGC_212_58,(0,0,2):C.UVGC_212_59,(0,0,0):C.UVGC_212_60})

V_135 = CTVertex(name = 'V_135',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.s] ], [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_230_88,(0,0,2):C.UVGC_230_89,(0,0,1):C.UVGC_230_90})

V_136 = CTVertex(name = 'V_136',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_248_118,(0,0,2):C.UVGC_248_119,(0,0,1):C.UVGC_248_120})

V_137 = CTVertex(name = 'V_137',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV8 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_275_199,(0,1,0):C.UVGC_276_200})

V_138 = CTVertex(name = 'V_138',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV8 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_200_46,(0,1,0):C.UVGC_201_47})

V_139 = CTVertex(name = 'V_139',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV8 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_249_121,(0,1,0):C.UVGC_250_122})

V_140 = CTVertex(name = 'V_140',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1, L.FFV6 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_170_17,(0,1,0):C.UVGC_207_53})

V_141 = CTVertex(name = 'V_141',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1, L.FFV6 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_170_17,(0,1,0):C.UVGC_225_83})

V_142 = CTVertex(name = 'V_142',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV1, L.FFV6 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_170_17,(0,1,0):C.UVGC_183_21})

V_143 = CTVertex(name = 'V_143',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV6 ],
                 loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.d, P.g] ], [ [P.g] ], [ [P.ghG] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                 couplings = {(0,0,3):C.UVGC_171_18,(0,1,0):C.UVGC_184_22,(0,1,1):C.UVGC_184_23,(0,1,2):C.UVGC_184_24,(0,1,4):C.UVGC_184_25,(0,1,5):C.UVGC_184_26,(0,1,6):C.UVGC_184_27,(0,1,7):C.UVGC_184_28,(0,1,8):C.UVGC_184_29,(0,1,3):C.UVGC_208_54})

V_144 = CTVertex(name = 'V_144',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV6 ],
                 loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.s] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                 couplings = {(0,0,5):C.UVGC_171_18,(0,1,0):C.UVGC_184_22,(0,1,1):C.UVGC_184_23,(0,1,2):C.UVGC_184_24,(0,1,3):C.UVGC_184_25,(0,1,4):C.UVGC_184_26,(0,1,6):C.UVGC_184_27,(0,1,7):C.UVGC_184_28,(0,1,8):C.UVGC_184_29,(0,1,5):C.UVGC_226_84})

V_145 = CTVertex(name = 'V_145',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV1, L.FFV6 ],
                 loop_particles = [ [ [P.b] ], [ [P.b, P.g] ], [ [P.c] ], [ [P.d] ], [ [P.g] ], [ [P.ghG] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                 couplings = {(0,0,1):C.UVGC_171_18,(0,1,0):C.UVGC_184_22,(0,1,2):C.UVGC_184_23,(0,1,3):C.UVGC_184_24,(0,1,4):C.UVGC_184_25,(0,1,5):C.UVGC_184_26,(0,1,6):C.UVGC_184_27,(0,1,7):C.UVGC_184_28,(0,1,8):C.UVGC_184_29,(0,1,1):C.UVGC_184_30})

V_146 = CTVertex(name = 'V_146',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_285_217,(0,0,2):C.UVGC_285_218,(0,0,1):C.UVGC_285_219})

V_147 = CTVertex(name = 'V_147',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.d, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.c, P.d, P.g] ], [ [P.c, P.g] ], [ [P.d, P.g] ] ],
                 couplings = {(0,0,1):C.UVGC_221_73,(0,0,2):C.UVGC_221_74,(0,0,0):C.UVGC_221_75})

V_148 = CTVertex(name = 'V_148',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.s, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_288_226,(0,0,2):C.UVGC_288_227,(0,0,1):C.UVGC_288_228})

V_149 = CTVertex(name = 'V_149',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.s] ], [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_239_103,(0,0,2):C.UVGC_239_104,(0,0,1):C.UVGC_239_105})

V_150 = CTVertex(name = 'V_150',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_248_118,(0,0,2):C.UVGC_248_119,(0,0,1):C.UVGC_248_120})

V_151 = CTVertex(name = 'V_151',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_213_61,(0,1,0):C.UVGC_214_62})

V_152 = CTVertex(name = 'V_152',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_231_91,(0,1,0):C.UVGC_232_92})

V_153 = CTVertex(name = 'V_153',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV3, L.FFV4 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_188_34,(0,1,0):C.UVGC_189_35})

V_154 = CTVertex(name = 'V_154',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF2 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_272_192,(0,1,0):C.UVGC_267_187})

V_155 = CTVertex(name = 'V_155',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF2 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_199_45,(0,1,0):C.UVGC_194_40})

V_156 = CTVertex(name = 'V_156',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF2 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_247_117,(0,1,0):C.UVGC_242_112})

V_157 = CTVertex(name = 'V_157',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF2 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_211_57,(0,1,0):C.UVGC_206_52})

V_158 = CTVertex(name = 'V_158',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF2 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_229_87,(0,1,0):C.UVGC_224_82})

V_159 = CTVertex(name = 'V_159',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF2 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_187_33,(0,1,0):C.UVGC_182_20})

V_160 = CTVertex(name = 'V_160',
                 type = 'UV',
                 particles = [ P.g, P.g ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VV1, L.VV5 ],
                 loop_particles = [ [ [P.b] ], [ [P.c] ], [ [P.d] ], [ [P.g] ], [ [P.ghG] ], [ [P.s] ], [ [P.t] ], [ [P.u] ] ],
                 couplings = {(0,1,0):C.UVGC_259_139,(0,1,1):C.UVGC_259_140,(0,1,2):C.UVGC_259_141,(0,1,5):C.UVGC_259_142,(0,1,6):C.UVGC_259_143,(0,1,7):C.UVGC_259_144,(0,0,3):C.UVGC_162_3,(0,0,4):C.UVGC_162_4})

