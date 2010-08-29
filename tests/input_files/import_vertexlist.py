import copy
import os
import math
import madgraph.core.base_objects as base_objects
import decay.decay_objects as decay_objects
from madgraph import  MG5DIR

full_leglist = {}
full_vertexlist = {}
full_vertexlist_newindex = {}

def make_legs(model):
    #Prepare the leglist
    for  pid in model.get('particle_dict').keys():
        full_leglist[pid] = base_objects.Leg({'id' : pid})
        for partnum in [2, 3]:
            for onshell in [True, False]:
                full_vertexlist_newindex[(pid, partnum,onshell)]=\
                    base_objects.VertexList()


#print len(full_leglist)

def make_vertexlist(model):
    make_legs(model)

    #Prepare the vertexlist
    for inter in model['interactions']:
        #Calculate the particle number, total mass
        partnum = len(inter['particles']) - 1
        if partnum > 3:
            continue

        total_mass = math.fsum([eval('decay_objects.' + part.get('mass')).real\
                                for part in inter['particles']])
        #Create the original legs
        temp_legs = base_objects.LegList([copy.copy(full_leglist[part2.get_pdg_code()]) for part2 in inter['particles']])
        

        for num, part in enumerate(inter['particles']):
            #Ignore massless incoming particle
            if part.get('mass') == 'ZERO':
                continue

            pid = part.get_anti_pdg_code()
            radiation = False
            for num2, other in enumerate(inter['particles']):
                if (num2 != num) and (other.get_pdg_code() == pid):
                    radiation = True
                    break

            if radiation:
                continue

            ini_mass = eval('decay_objects.' + part.get('mass')).real

            #Set each leg as incoming particle            
            temp_legs[num].set('state', False)
            #If part is not self-conjugate, change the particle into anti-part
            temp_legs[num].set('id', pid)
            
            temp_legs_new = copy.deepcopy(temp_legs)
            #Sort the legs for comparison
            temp_legs_new.sort(decay_objects.legcmp)
            temp_vertex = base_objects.Vertex({'id': inter.get('id'),
                                               'legs':temp_legs_new})
            #Record the vertex with key = (interaction_id, part_id)
            full_vertexlist[(inter.get('id'), pid)] =temp_vertex
            
            if temp_vertex not in full_vertexlist_newindex[(pid, partnum,
                                      ini_mass > (total_mass - ini_mass))]:
                
                full_vertexlist_newindex[(pid, partnum,
                                          ini_mass > (total_mass - ini_mass))].\
                                          append(temp_vertex)
            
            #Reset the leg to normal state and normal id
            temp_legs[num].set('state', True)
            temp_legs[num].set('id', part.get_pdg_code())

    path_1 = os.path.join(MG5DIR, 'models', model['name'])
    path_2 = os.path.join(MG5DIR, 'tests/input_files', model['name'])

    if os.path.isdir(path_1):
        fdata = open(os.path.join(path_1, 'vertices_decaycondition.dat'), 'w')
        fdata.write(str(full_vertexlist_newindex))
        fdata.close()

        fdata2 = open(os.path.join(path_1, 'vertices_sort.dat'), 'w')
        fdata2.write(str(full_vertexlist))
        fdata2.close()
    elif os.path.isdir(path_2):
        fdata = open(os.path.join(path_2, 'vertices_decaycondition.dat'), 'w')
        fdata.write(str(full_vertexlist_newindex))
        fdata.close()

        fdata2 = open(os.path.join(path_2, 'vertices_sort.dat'), 'w')
        fdata2.write(str(full_vertexlist))
        fdata2.close()
