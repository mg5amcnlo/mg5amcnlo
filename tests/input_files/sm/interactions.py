interactions = [
{
    'id': 1,
    'particles': [25,25,25,25],
    'color': [1 ],
    'lorentz': ['SSSS1'],
    'couplings': {(0, 0): 'GC_10'},
    'orders': {'QED': 2}
},{
    'id': 2,
    'particles': [25,25,25],
    'color': [1 ],
    'lorentz': ['SSS1'],
    'couplings': {(0, 0): 'GC_31'},
    'orders': {'QED': 1}
},{
    'id': 3,
    'particles': [21,21,21],
    'color': [1 f(0,1,2)],
    'lorentz': ['VVV1'],
    'couplings': {(0, 0): 'GC_4'},
    'orders': {'QCD': 1}
},{
    'id': 4,
    'particles': [21,21,21,21],
    'color': [1 f(-1,0,1) f(2,3,-1), 1 f(-1,0,2) f(1,3,-1), 1 f(-1,0,3) f(1,2,-1)],
    'lorentz': ['VVVV1', 'VVVV3', 'VVVV4'],
    'couplings': {(0, 0): 'GC_6', (1, 1): 'GC_6', (2, 2): 'GC_6'},
    'orders': {'QCD': 2}
},{
    'id': 5,
    'particles': [22,-24,24],
    'color': [1 ],
    'lorentz': ['VVV1'],
    'couplings': {(0, 0): 'GC_26'},
    'orders': {'QED': 1}
},{
    'id': 6,
    'particles': [-24,24,25,25],
    'color': [1 ],
    'lorentz': ['VVSS1'],
    'couplings': {(0, 0): 'GC_11'},
    'orders': {'QED': 2}
},{
    'id': 7,
    'particles': [-24,24,25],
    'color': [1 ],
    'lorentz': ['VVS1'],
    'couplings': {(0, 0): 'GC_32'},
    'orders': {'QED': 1}
},{
    'id': 8,
    'particles': [22,22,-24,24],
    'color': [1 ],
    'lorentz': ['VVVV2'],
    'couplings': {(0, 0): 'GC_28'},
    'orders': {'QED': 2}
},{
    'id': 9,
    'particles': [-24,24,23],
    'color': [1 ],
    'lorentz': ['VVV1'],
    'couplings': {(0, 0): 'GC_7'},
    'orders': {'QED': 1}
},{
    'id': 10,
    'particles': [-24,-24,24,24],
    'color': [1 ],
    'lorentz': ['VVVV2'],
    'couplings': {(0, 0): 'GC_8'},
    'orders': {'QED': 2}
},{
    'id': 11,
    'particles': [22,-24,24,23],
    'color': [1 ],
    'lorentz': ['VVVV5'],
    'couplings': {(0, 0): 'GC_27'},
    'orders': {'QED': 2}
},{
    'id': 12,
    'particles': [23,23,25,25],
    'color': [1 ],
    'lorentz': ['VVSS1'],
    'couplings': {(0, 0): 'GC_30'},
    'orders': {'QED': 2}
},{
    'id': 13,
    'particles': [23,23,25],
    'color': [1 ],
    'lorentz': ['VVS1'],
    'couplings': {(0, 0): 'GC_33'},
    'orders': {'QED': 1}
},{
    'id': 14,
    'particles': [-24,24,23,23],
    'color': [1 ],
    'lorentz': ['VVVV2'],
    'couplings': {(0, 0): 'GC_9'},
    'orders': {'QED': 2}
},{
    'id': 15,
    'particles': [-1,1,22],
    'color': [1 T(1,0)],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_1'},
    'orders': {'QED': 1}
},{
    'id': 16,
    'particles': [-3,3,22],
    'color': [1 T(1,0)],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_1'},
    'orders': {'QED': 1}
},{
    'id': 17,
    'particles': [-5,5,22],
    'color': [1 T(1,0)],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_1'},
    'orders': {'QED': 1}
},{
    'id': 18,
    'particles': [-1,1,21],
    'color': [1 T(2,1,0)],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_5'},
    'orders': {'QCD': 1}
},{
    'id': 19,
    'particles': [-3,3,21],
    'color': [1 T(2,1,0)],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_5'},
    'orders': {'QCD': 1}
},{
    'id': 20,
    'particles': [-5,5,21],
    'color': [1 T(2,1,0)],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_5'},
    'orders': {'QCD': 1}
},{
    'id': 21,
    'particles': [-5,5,25],
    'color': [1 T(1,0)],
    'lorentz': ['FFS1'],
    'couplings': {(0, 0): 'GC_34'},
    'orders': {'QED': 1}
},{
    'id': 22,
    'particles': [-1,1,23],
    'color': [1 T(1,0)],
    'lorentz': ['FFV2', 'FFV3'],
    'couplings': {(0, 1): 'GC_24', (0, 0): 'GC_22'},
    'orders': {'QED': 1}
},{
    'id': 23,
    'particles': [-3,3,23],
    'color': [1 T(1,0)],
    'lorentz': ['FFV2', 'FFV3'],
    'couplings': {(0, 1): 'GC_24', (0, 0): 'GC_22'},
    'orders': {'QED': 1}
},{
    'id': 24,
    'particles': [-5,5,23],
    'color': [1 T(1,0)],
    'lorentz': ['FFV2', 'FFV3'],
    'couplings': {(0, 1): 'GC_24', (0, 0): 'GC_22'},
    'orders': {'QED': 1}
},{
    'id': 25,
    'particles': [-2,1,24],
    'color': [1 T(1,0)],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_17'},
    'orders': {'QED': 1}
},{
    'id': 29,
    'particles': [-4,3,24],
    'color': [1 T(1,0)],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_17'},
    'orders': {'QED': 1}
},{
    'id': 33,
    'particles': [-6,5,24],
    'color': [1 T(1,0)],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_17'},
    'orders': {'QED': 1}
},{
    'id': 34,
    'particles': [-11,11,22],
    'color': [1 ],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_3'},
    'orders': {'QED': 1}
},{
    'id': 35,
    'particles': [-13,13,22],
    'color': [1 ],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_3'},
    'orders': {'QED': 1}
},{
    'id': 36,
    'particles': [-15,15,22],
    'color': [1 ],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_3'},
    'orders': {'QED': 1}
},{
    'id': 39,
    'particles': [-15,15,25],
    'color': [1 ],
    'lorentz': ['FFS1'],
    'couplings': {(0, 0): 'GC_39'},
    'orders': {'QED': 1}
},{
    'id': 40,
    'particles': [-11,11,23],
    'color': [1 ],
    'lorentz': ['FFV2', 'FFV4'],
    'couplings': {(0, 1): 'GC_25', (0, 0): 'GC_22'},
    'orders': {'QED': 1}
},{
    'id': 41,
    'particles': [-13,13,23],
    'color': [1 ],
    'lorentz': ['FFV2', 'FFV4'],
    'couplings': {(0, 1): 'GC_25', (0, 0): 'GC_22'},
    'orders': {'QED': 1}
},{
    'id': 42,
    'particles': [-15,15,23],
    'color': [1 ],
    'lorentz': ['FFV2', 'FFV4'],
    'couplings': {(0, 1): 'GC_25', (0, 0): 'GC_22'},
    'orders': {'QED': 1}
},{
    'id': 43,
    'particles': [-12,11,24],
    'color': [1 ],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_17'},
    'orders': {'QED': 1}
},{
    'id': 44,
    'particles': [-14,13,24],
    'color': [1 ],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_17'},
    'orders': {'QED': 1}
},{
    'id': 45,
    'particles': [-16,15,24],
    'color': [1 ],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_17'},
    'orders': {'QED': 1}
},{
    'id': 46,
    'particles': [-1,2,-24],
    'color': [1 T(1,0)],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_17'},
    'orders': {'QED': 1}
},{
    'id': 50,
    'particles': [-3,4,-24],
    'color': [1 T(1,0)],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_17'},
    'orders': {'QED': 1}
},{
    'id': 54,
    'particles': [-5,6,-24],
    'color': [1 T(1,0)],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_17'},
    'orders': {'QED': 1}
},{
    'id': 55,
    'particles': [-2,2,22],
    'color': [1 T(1,0)],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_2'},
    'orders': {'QED': 1}
},{
    'id': 56,
    'particles': [-4,4,22],
    'color': [1 T(1,0)],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_2'},
    'orders': {'QED': 1}
},{
    'id': 57,
    'particles': [-6,6,22],
    'color': [1 T(1,0)],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_2'},
    'orders': {'QED': 1}
},{
    'id': 58,
    'particles': [-2,2,21],
    'color': [1 T(2,1,0)],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_5'},
    'orders': {'QCD': 1}
},{
    'id': 59,
    'particles': [-4,4,21],
    'color': [1 T(2,1,0)],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_5'},
    'orders': {'QCD': 1}
},{
    'id': 60,
    'particles': [-6,6,21],
    'color': [1 T(2,1,0)],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_5'},
    'orders': {'QCD': 1}
},{
    'id': 62,
    'particles': [-6,6,25],
    'color': [1 T(1,0)],
    'lorentz': ['FFS1'],
    'couplings': {(0, 0): 'GC_38'},
    'orders': {'QED': 1}
},{
    'id': 63,
    'particles': [-2,2,23],
    'color': [1 T(1,0)],
    'lorentz': ['FFV2', 'FFV5'],
    'couplings': {(0, 1): 'GC_24', (0, 0): 'GC_23'},
    'orders': {'QED': 1}
},{
    'id': 64,
    'particles': [-4,4,23],
    'color': [1 T(1,0)],
    'lorentz': ['FFV2', 'FFV5'],
    'couplings': {(0, 1): 'GC_24', (0, 0): 'GC_23'},
    'orders': {'QED': 1}
},{
    'id': 65,
    'particles': [-6,6,23],
    'color': [1 T(1,0)],
    'lorentz': ['FFV2', 'FFV5'],
    'couplings': {(0, 1): 'GC_24', (0, 0): 'GC_23'},
    'orders': {'QED': 1}
},{
    'id': 66,
    'particles': [-11,12,-24],
    'color': [1 ],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_17'},
    'orders': {'QED': 1}
},{
    'id': 67,
    'particles': [-13,14,-24],
    'color': [1 ],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_17'},
    'orders': {'QED': 1}
},{
    'id': 68,
    'particles': [-15,16,-24],
    'color': [1 ],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_17'},
    'orders': {'QED': 1}
},{
    'id': 69,
    'particles': [-12,12,23],
    'color': [1 ],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_29'},
    'orders': {'QED': 1}
},{
    'id': 70,
    'particles': [-14,14,23],
    'color': [1 ],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_29'},
    'orders': {'QED': 1}
},{
    'id': 71,
    'particles': [-16,16,23],
    'color': [1 ],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_29'},
    'orders': {'QED': 1}
}]