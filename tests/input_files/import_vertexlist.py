import madgraph.core.base_objects as base_objects
import decay.decay_objects as decay_objects
import copy
import os
from madgraph import  MG5DIR

full_leglist = {}
full_vertexlist = {}

def make_legs(model):
    #Prepare the leglist
    for  part in model['particles']:
        full_leglist[part.get('pdg_code')] = \
            base_objects.Leg({'id':part.get('pdg_code')})
        full_leglist[-part.get('pdg_code')] = \
            base_objects.Leg({'id':-part.get('pdg_code')})
    #print len(full_leglist)

#print len(full_leglist)

def make_vertexlist(model):
    make_legs(model)

    #Prepare the vertexlist
    for inter in model['interactions']:
        temp_legs = base_objects.LegList([copy.copy(full_leglist[part2.get_pdg_code()]) for part2 in inter['particles']])

        for num, part in enumerate(inter['particles']):
            #Set each leg as incoming particle
            temp_legs[num].set('state', False)
            
            #If part is not self-conjugate, change the particle into anti-part
            if not part.get('self_antipart'):
                temp_legs[num].set('id', -part.get_pdg_code())
            
            temp_legs_new = copy.deepcopy(temp_legs)
            #Sort the legs for comparison
            temp_legs_new.sort(decay_objects.legcmp)
            temp_vertex = base_objects.Vertex({'id': inter.get('id'),
                                               'legs':temp_legs_new})
            #Record the vertex with key = (interaction_id, part_id)
            full_vertexlist[(inter.get('id'), part.get_pdg_code())] =temp_vertex
        
            #Reset the leg to normal state and normal id
            temp_legs[num].set('state', True)
            temp_legs[num].set('id', part.get_pdg_code())

    fdata = open(os.path.join(MG5DIR, 'models', model['name'], 'vertices_sort.dat'), 'w')
    fdata.write(str(full_vertexlist))
    fdata.close()
