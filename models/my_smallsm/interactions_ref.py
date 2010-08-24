#The interaction content of MG5 model, my_smallsm,
#generate from save_model module

interactions = [
{
    'id': 1,
    'particles': [22,-24,24],
    'color': [1 ],
    'lorentz': ['VVV1'],
    'couplings': {(0, 0): 'GC_49'},
    'orders': {'QED': 1}
},{
    'id': 2,
    'particles': [22,22,-24,24],
    'color': [1 ],
    'lorentz': ['VVVV2'],
    'couplings': {(0, 0): 'GC_51'},
    'orders': {'QED': 2}
},{
    'id': 3,
    'particles': [-24,-24,24,24],
    'color': [1 ],
    'lorentz': ['VVVV2'],
    'couplings': {(0, 0): 'GC_13'},
    'orders': {'QED': 2}
},{
    'id': 4,
    'particles': [-5,5,22],
    'color': [1 T(1,0)],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_1'},
    'orders': {'QED': 1}
},{
    'id': 5,
    'particles': [-11,11,22],
    'color': [1 ],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_3'},
    'orders': {'QED': 1}
},{
    'id': 6,
    'particles': [-6,6,22],
    'color': [1 T(1,0)],
    'lorentz': ['FFV1'],
    'couplings': {(0, 0): 'GC_2'},
    'orders': {'QED': 1}
},{
    'id': 7,
    'particles': [-5,6,-24],
    'color': [1 T(1,0)],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_33'},
    'orders': {'QED': 1}
},{
    'id': 8,
    'particles': [-6,5,24],
    'color': [1 T(1,0)],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_134'},
    'orders': {'QED': 1}
},{
    'id': 9,
    'particles': [-11,12,-24],
    'color': [1 ],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_24'},
    'orders': {'QED': 1}
},{
    'id': 10,
    'particles': [-12,11,24],
    'color': [1 ],
    'lorentz': ['FFV2'],
    'couplings': {(0, 0): 'GC_24'},
    'orders': {'QED': 1}
}]
