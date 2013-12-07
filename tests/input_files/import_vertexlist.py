import copy
import os
import math
import madgraph.core.base_objects as base_objects
import mg5decay.decay_objects as decay_objects
import madgraph.iolibs.save_model as save_model
from madgraph import  MG5DIR

full_leglist = {}
full_vertexlist = {}
full_vertexlist_newindex = {}

def make_legs(model):
    # global var
    global full_leglist, full_vertexlist, full_vertexlist_newindex

    #Prepare the leglist
    for  pid in model.get('particle_dict').keys():
        full_leglist[pid] = base_objects.Leg({'id' : pid})


#print len(full_leglist)

def make_vertexlist(model):

    # Initialization
    global full_leglist
    global full_vertexlist
    global full_vertexlist_newindex

    full_leglist = {}
    full_vertexlist = {}
    full_vertexlist_newindex = {}
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
            if part.get('mass') == 'ZERO' or part.get('pdg_code') in [1,2,3,4,5,11,12,13,14,16,21,22]:
                continue

            pid = part.get_anti_pdg_code()
            radiation = False
            for num2, other in enumerate(inter['particles']):
                if (num2 != num) and (other.get('pdg_code') == abs(pid)):
                    radiation = True
                    break

            if radiation:
                continue

            ini_mass = eval('decay_objects.' + part.get('mass')).real

            temp_legs_new = copy.deepcopy(temp_legs)
            #If part is not self-conjugate, change the particle into anti-part
            temp_legs_new[num].set('id', pid)

            # Initial leg should be in the last
            ini_leg = temp_legs_new.pop(num)

            # Sort the other legs
            temp_legs_new.sort(lambda l1, l2: cmp(l1['id'],l2['id']), 
                               reverse=True)
            temp_legs_new.append(ini_leg)
            temp_vertex = base_objects.Vertex({'id': inter.get('id'),
                                               'legs':temp_legs_new})

            # Record the vertex with key = (interaction_id, part_id)
            try:
                full_vertexlist[(inter.get('id'), pid)] =temp_vertex
            except KeyError:
                full_vertexlist[(inter.get('id'), pid)] = \
                    base_objects.VertexList([temp_vertex])

            # new index list made for testing find_vertexlist
            try:                    
                full_vertexlist_newindex[(pid, partnum,
                                          ini_mass > (total_mass - ini_mass))].append(temp_vertex)

            except KeyError:
                full_vertexlist_newindex[(pid, partnum,
                                          ini_mass > (total_mass - ini_mass))]=\
                                          base_objects.VertexList([temp_vertex])
                
            
            #Reset the leg to normal state and normal id
            temp_legs[num].set('state', True)
            temp_legs[num].set('id', part.get_pdg_code())
"""
    path_1 = os.path.join(MG5DIR, 'models', model['name'])
    path_2 = os.path.join(MG5DIR, 'tests/input_files', model['name'])

    fdata = open(os.path.join(path_2, 'vertices_decaycondition.dat'), 'w')
    fdata.write(str(full_vertexlist_newindex))
    fdata.close()
    
    fdata2 = open(os.path.join(path_2, 'vertices_sort.dat'), 'w')
    fdata2.write(str(full_vertexlist))
    fdata2.close()
    
"""
