#!/usr/bin/env python
##### -*- coding: cp1252 -*-

#Extension
import os

try:
    from madgraph.madweight.MW_fct import *
    import madgraph.madweight.diagram_class as diagram_class
    import madgraph.madweight.mod_file as mod_file
    import madgraph.madweight.Cards as Cards
    import madgraph.various.misc as misc
except ImportError:
    from internal.madweight.MW_fct import *
    import internal.madweight.diagram_class as diagram_class
    import internal.madweight.mod_file as mod_file
    import internal.madweight.Cards as Cards
    import internal.misc as misc   


pjoin = os.path.join 
def create_all_fortran_code(MW_info, i=1):
    """goes  in each subprocess and creates the fortran code in each of them"""
    # load template for file    
    template = mod_file.Mod_file(rule_file='./Source/MadWeight/mod_file/mod_main_code')
    # load MadWeight option
    for MW_dir in MW_info.MW_listdir:
        print 'treating', MW_dir, 'directory'
        diag = MG_diagram('./SubProcesses/' + MW_dir, 'param_card_1.dat', './Source/MadWeight/transfer_function/ordering_file.inc', i, MW_info)
        diag.create_all_fortran_code()
        diag.write_code(template)




#################################################################################################################
##
##                                   MG_diagram : write code part
##
#################################################################################################################

class MG_diagram(diagram_class.MG_diagram):
    """ add the write routine for fortran code in this file """

    def __init__(self, dir_file, param_card, tf_file, config, opt='default'):
        """ update object to add write content """
        diagram_class.MG_diagram.__init__(self, dir_file, config, opt)
        self.organize_particle_content(param_card, tf_file)
        self.code = []
        self.ParticlesFile=Cards.Particles_file('./Source/MODEL/particles.dat')
        self.dict_Fmass,self.dict_Fwidth=self.ParticlesFile.give_mass_width_dict()
        self.MWparam = opt

    def create_all_fortran_code(self):
        """go  in each subprocesses and create the fortran code in each of them"""

        Mlist_unaligned = self.detect_non_resonant_area()
        all_pos = Mlist_unaligned.give_combinaison()
        num_sol = 0
        if not all_pos: #no BW ambiguity
            self.clear_solution()
            self.define_Constraint_sector()
            self.solve_blob_sector()
            print self
            num_sol = self.load_fortran_code(num_sol)
            
        for unaligned in all_pos:
            self.clear_solution()
            self.tag_unaligned(unaligned)

            self.define_Constraint_sector()
            self.solve_blob_sector()
            print self
            num_sol = self.load_fortran_code(num_sol)
            
        self.create_permutation_weight_functions_caller()


    def load_fortran_code(self, num_sol=0):
        """ create the code """

        #introduction for d_choices...
        self.init_d_choices_file()
        
##        self.create_pmass2()
        self.create_output_type_info()
        
        self.collect_unaligned_peaks() #defined self.full_sol
        
        self.already_existing_solution = []
        num_sol = len(self.code)
        if num_sol == 0:
            self.sol_nb=[]
            self.num_test_sol=0
        for i in range(0, len(self.full_sol)):
            solution=self.full_sol[i]
            self.num_test_sol+=1
            num_sol += 1
            code=[self.create_MadWeight_main(solution, num_sol)]
            code.append(self.create_MadWeight_data(solution, num_sol))
            if self.is_new_sol(code):
                self.sol_nb.append(self.num_test_sol)
#                code.append(self.create_multi_channel_weight(i, num_sol))
                self.code.append(code)
            else:
                self.already_existing_solution.append(i)
                num_sol -= 1
                
        return num_sol

    def create_permutation_weight_functions_caller(self):
        """Define the function to associate a initial weight to each permutation.
        Those weight can be use both to remove permutation and/or to initialize the grid
        on the permutation."""

        def get_all_child(mother):
            if mother.external:
                return [mother]
            else:
                output = []
                for child in mother.des:
                    output += get_all_child(child)
                return output 
        
        fsock = open(pjoin(self.directory, 'permutation_weight.f'), 'w')
        if self.MWparam['mw_perm']['preselect'].lower() == 'none':
            fsock.write(""" 
            subroutine GET_PERM_WEIGHT()
            return
            end
            """
            )
            return


        template = """      subroutine GET_PERM_WEIGHT()
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'permutation.inc'
      integer perm
      integer perm_id(nexternal-2)
      integer content(nexternal)
      integer i
      double precision WEIGHT
      double precision weight_perm_global, weight_perm_BW
      external weight_perm_global, weight_perm_BW


      do perm = 1, NPERM
         curr_perm = perm
         call get_perm(perm, perm_id)
         call assign_perm(perm_id)

        
        weight = weight_perm_global(perm,perm_id)
        %s
        perm_value(perm, 1) = weight
      enddo
      return
      end
"""
        
        data = []
        
        for part in self.prop_content:
            if part.mass:
                mass = self.dict_Fmass[abs(part.pid)]
                width = self.dict_Fwidth[abs(part.pid)]
                content = [p.MG for p in get_all_child(part)]
                content.append(0)
                initcontent = "\n".join("""          content(%i) = %i""" % (a+1, b) for a,b in enumerate(content))
                if mass.lower() == 'zero' or width.lower() == 'zero':
                    continue
                dico = {'pid': part.pid, 
                             'mass': mass,
                             'width': width,
                             'initcontent': initcontent
                             }
            
                line = """%(initcontent)s
                weight = weight * weight_perm_BW(perm, perm_id, %(mass)s, %(width)s, content, %(pid)i)
                """ % dico
                data.append(line)
                
        text = template % '\n'.join(data)
        fsock.write(text)
        
        if self.MWparam['mw_perm']['preselect'] == 'default':
            fsock.write(open(pjoin(self.directory, '../permutation_weight_default.dat')).read())
        else:
            fsock.write(open(pjoin(self.directory,'../..',self.MWparam['mw_perm']['preselect'])).read())
        




#    def create_MadWeight_code(self, level, num_sol):
#        """ returns the code for the numerator linked to 'level' change of variable 
#            level is the position in self.full_sol
#        """
#        
#        ECS=self.full_sol[level][0]
#        write_text=''
#        if num_sol == 1:
#            write_text += '''       if (config.eq.1) then '''
#        else:
#           write_text += '       elseif (config.eq.' + str(num_sol) + ') then ' 
#        write_text += '\n$B$ S-COMMENT_C $B$\n'
#        write_text += ECS.info()                    # -> write short ECS/BLOB information
#        write_text += '\n$E$ S-COMMENT_C $E$\n\n'  
#        write_text += self.write_channel_weight(self.unaligned_in_sol[level])


    def def_text_for_channel_weight(self,all_peak):
        """ return the text for the definition of external routine and
            local variable needed for the multichannel routine 
        """
            
        text=''
        i=0
        deal_with=[]
        for unaligned,nb in all_peak.items():
            if not nb:
                continue
            i+=1        
            text+=' double precision local_%s \n' % (i)
            if isinstance(unaligned, basestring):
                name = 'tf_E_for_part\n'
            elif unaligned.external:
                name= 'tf_E_for_%s \n'% (unaligned.MG)
            else:
                pid=abs(unaligned.pid)
                name= ' Breit_Wigner_for_part'
            if name not in deal_with:
                deal_with.append(name)
                text += ' double precision %s\n' %name
                text += ' external %s \n' %name 
        
        return text
    
    def write_channel_weight(self,peak_by_channel,all_peak,label):
        """ all peak contains the list of the dictionary {peak:nb_of_appearance} each associated 
            to a specific channel.
            label is the tag for the channel under study
         returns the text defining, in fortran, the weight for this channel
            the sum of the peaks ponderate by the appearance
        """

        def write_call_for_peak(obj,peak):
            """ return the text on how to return the weight associted to this peak """
            
            if isinstance(peak, basestring):
                text = 'tf_E_for_part(%s)' % ( peak )
            elif peak.external:
                text = 'tf_E_for_%s() '% (peak.MG)
            else:
                pid=abs(peak.pid)
                text = 'Breit_Wigner_for_part( %s, %s, %s)' % \
                    (peak.MG,obj.dict_Fmass[pid],obj.dict_Fwidth[pid])            
            return text
        
        def product_of_peak(unaligned_peak,all_peak,peak_to_prov):
            """ Return the product of local_XX associated to this set of
                unaligned peak
            """
            text='1d0'
            for unaligned,nb in all_peak.items():
                if nb == 0: 
                    continue
                if unaligned not in unaligned_peak.keys():                
                    text+=' * local_%s' %(peak_to_prov[unaligned])
            return text 
        
         
        text = ''
        den_text = ' den = 0d0'
        num_text = ' num = 1d0'
        i=0
        peak_to_prov={}
        
        #definition of the local and the definition associated
        for unaligned,nb in all_peak.items():
            if nb == 0: 
                continue
            i+=1
            text += ' local_%s = %s \n' %(i,write_call_for_peak(self,unaligned))
            peak_to_prov[unaligned]=i
        

        if 'restrict_channel' in self.MWparam['mw_gen']:
            if isinstance(self.MWparam['mw_gen']['restrict_channel'], list):
                allowed_channel = [int(i)-1 for i in self.MWparam['mw_gen']['restrict_channel']]
            elif self.MWparam['mw_gen']['restrict_channel'] in ['','0','F']:
                allowed_channel = range(0,len(peak_by_channel))
            else:
                allowed_channel = [int(self.MWparam['mw_gen']['restrict_channel'])-1]
        else:
            allowed_channel = range(0,len(peak_by_channel))
        for j in xrange(0,len(peak_by_channel)):
            if j in allowed_channel:
                den_text += ' + '+product_of_peak(peak_by_channel[j],all_peak,peak_to_prov)
                if j == label:
                    num_text += ' * '+product_of_peak(peak_by_channel[j],all_peak,peak_to_prov)
            elif j ==label:
                if num_text.startswith(' num=0d0'):
                    num_text += ' * '+product_of_peak(peak_by_channel[j],all_peak,peak_to_prov)
                else: 
                    num_text = ' num=0d0 ! '+product_of_peak(peak_by_channel[j],all_peak,peak_to_prov)
                
        #define the return value
        if not num_text.startswith(' num=0d0'):
            text+='\n' #make a break
            text+=num_text+'\n'
            text+=den_text+'\n'
            if i:
                text+=' multi_channel_weight = num/den\n'
            else:
                value=str(1.0/len(peak_by_channel))
                value=value.replace('e','d')
                text=' multi_channel_weight = %s\n' % value
        else:
            text+='\n' #make a break
            text+=num_text+'\n'
            text+=den_text+'\n'
            text = 'c'+text.replace('\n','\nc')
            text += '\n multi_channel_weight = 0d0\n'
        return text
             
    def create_multi_channel_weight(self,label, num_sol):
        """ create the fortran code defining the weighting of each integration channel """
        
        
#        ECS=self.full_sol[label][0]   
#        blob_sol_list=self.full_sol[label][1]       

        write_text=''
        
        if num_sol == 1:
            write_text += '''\n if (config.eq.1) then \n'''
        else:
            write_text += '\n elseif (config.eq.' + str(num_sol) + ') then \n' 
#        write_text += '\n$B$ S-COMMENT_C $B$\n'
#        write_text += ECS.info()                    # -> write short ECS/BLOB information
#        write_text += '\n$E$ S-COMMENT_C $E$\n'
        
        tmp_text = self.write_channel_weight(self.unaligned_in_sol,self.unaligned,label)
        write_text+=tmp_text
        return write_text
        
    def create_MadWeight_main(self, full_sol_obj, num_sol):
        """ create the main_code_$i.inc for all solution
            and the associate d_choices(.f)(.inc)
        """

        
        ECS=full_sol_obj[0]
        blob_sol_list=full_sol_obj[1]
        self.num_fuse=self.ext_part + 3 #+2 for initial particle +1 to be on the next one
        self.fuse_dict={}
        self.use_propa = set()
        #template=self.template
        write_text=''
        #
        #    INTRODUCTION
        #
        #write_text=template.dico['INTRO_FOR_MAIN']           #tag re-interpreted later to insert intro in file
        if num_sol == 1:
            write_text += '''       if (config_pos.eq.1) then '''
        else:
           write_text += '       elseif (config_pos.eq.' + str(num_sol) + ') then ' 
        write_text += '\n$B$ S-COMMENT_C $B$\n'
        write_text += full_sol_obj[0].info()                    # -> write short ECS/BLOB information
        write_text += '\n$E$ S-COMMENT_C $E$\n'        
        #
        #   BLOB
        #
        step=0
        for blob_sol in  blob_sol_list:
            #supress entry for external particle blob
            if len(blob_sol.step) == 1:
                if blob_sol.step[0].chgt_var == '0':
                    continue
            for block in blob_sol.step:
                if blob_sol.step.index(block):
                    write_text += 'C       ++++++++++++           \n'
                step += 1
                if block.chgt_var in ['1', '2', '3']:
                    block_name=' call fuse('
                elif block.chgt_var == '0':                  
                    continue #this is already done by MadWeight
                else:#if block.chgt_var in ['A']:
                    block_name=' call block_' + block.chgt_var.lower() + '(x,'
#                else:
#                    block_name=' call block_' + block.chgt_var.lower() + '(x,n_var,var2random(1,config_pos),'
                line=block_name
                for particle in block.order_content:
                    if particle.MG < 0:
                        self.use_propa.add(particle.MG)
                    if block.chgt_var in ['D']:
                        line += self.write_d_choices(block.order_content)+' '
                        for particle in block.order_content:
                            if particle.MG < 0:
                                self.use_propa.add(particle.MG)
                        break
                    if type(particle.MG) == int:
                        line += str(particle.MG) + ','
                    elif isinstance(particle.MG, basestring):
                        if self.fuse_dict.has_key(particle.MG):
                            line += str(self.fuse_dict[particle.MG]) + ','
                            del self.fuse_dict[particle.MG]
                        else:
                            line += str(self.num_fuse) + ','
                            self.fuse_dict[particle.MG]=self.num_fuse
                            self.num_fuse += 1
                line=line[:-1] + ')\n' #supress last , and add )
                line=put_in_fortran_format(line)
                write_text += line
                if(block.chgt_var not in ['1', '2', '3']):
                    text=' if (jac.le.0d0) return\n'
                else:
                    continue
                write_text += put_in_fortran_format(text)     
       
        #
        #   ECS 
        #
#        write_text+='\n$B$ S-COMMENT_C $B$\n'
#        write_text+=' ENLARGED CONTRAINT SECTOR \t CLASS '+str(ECS.chgt_var.upper())
#        write_text+='\n$E$ S-COMMENT_C $E$\n'         
        for block in ECS.step:
            step += 1           
            if block.chgt_var == '2':
                line=' call fuse('
#            elif block.chgt_var in ['a', 'c', 'e', 'f', 'g']:
#                line=' call class_' + ECS.chgt_var.lower() + '(x,n_var,var2random(1,config_pos),'
            else:
                line=' call class_' + ECS.chgt_var.lower() + '(x,'
            for particle in block.order_content:
                if particle.MG < 0:
                    self.use_propa.add(particle.MG)
                if type(particle.MG) == int:
                    line += str(particle.MG) + ','
                elif isinstance(particle.MG, basestring):
                    if self.fuse_dict.has_key(particle.MG):
                        line += str(self.fuse_dict[particle.MG]) + ','
                        del self.fuse_dict[particle.MG]
                    else:
                        line += str(self.num_fuse) + ','
                        self.fuse_dict[particle.MG]=self.num_fuse
                        self.num_fuse += 1
                            
            line=line[:-1] + ')\n' #supress last , and add )
            line=put_in_fortran_format(line)
            write_text += line
            if block.chgt_var not in ['1','2']:
                text=' if (jac.le.0d0) return\n'
            else:
                text='\n'
            write_text += put_in_fortran_format(text)

        self.nb_block=step
        #
        #   INVISIBLE DECAY
        #
        out=self.check_invisible_decay()
        if out:
            write_text += '\n' + out            
        #
        #    PUT FUSE FOR OTHER PROPAGATOR
        #
        text = ''
        for i in range(1, len(self.prop_content)):
            pos = -1 * i
            if pos not in self.use_propa:
                for particle in self.prop_content:
                    if particle.channel.startswith('T'):
                        continue
                    if particle.MG == pos:
                        text += ' call fuse(%s,%s,%s)\n' %(particle.des[0].MG, particle.des[1].MG, pos) 
                        break

        #add the call for the multichannel weight
        text+='\n jac=jac*multi_channel_weight(%s)\n'%(num_sol)
        write_text += put_in_fortran_format(text)
        return write_text
        
    def create_MadWeight_data(self, full_sol_obj, num_sol):
        """ create the data_$i.inc for all solution """
        #
        # intro write in write_code part, this is a script for one possibility
        #  of generation
        #

        ECS=full_sol_obj[0]
        blob_sol_list=full_sol_obj[1]
        mapping = [0] * ((3*len(self.ext_content))+2)
        m_random, p_random = 0, 0
        var2mrandom = {'a':0,'b':1,'c':2,'d':4,'e':3,'f':2, 'g':2,
                       'A':3,'B':2,'C':1,'D':1,'E':2,
                       '0':0,'1':0,'2':0}
        
        
#        template=self.template
        blob_sol=[]
        for b_sol in blob_sol_list:
            blob_sol += b_sol.step
        write_text=''
        #
        #    INTRODUCTION
        #
        write_text='\n$B$ S-COMMENT_C $B$\n'
        write_text += full_sol_obj[0].info()
        write_text += '\n$E$ S-COMMENT_C $E$\n'                
        num_vis=0
        vis_str=''
        vis_list=[]
        part_treated = set()
        ambiguous_external = set()
        for block in ECS.step + blob_sol:
            m_random += var2mrandom[block.chgt_var]
            if block.chgt_var == '0':
                particle = block.in_part[0]
                if particle.external and not particle.neutrino:
                    ambiguous_external.add(particle.MG)
                    if mapping[3*particle.MG-7] == 0 and particle.tf_level:
                        mapping[3*particle.MG-7] = p_random+1
                        p_random += 1
                continue
            else:
                [part_treated.add(part.MG) for part in block.in_part]
            
            if block.chgt_var == 'e':
                mapping[0] = p_random+1
                p_random += 1 
            elif block.chgt_var in ['f','g']:
                mapping[0] = p_random+1
                mapping[1] = p_random+2            
                p_random += 2
                 
            # treat ONLY NEUTRINO for block B/C
            if block.chgt_var == 'B':
                neut = block.neut_content[0]
                mapping[3*neut.MG-6] = p_random + 1
                p_random +=1 
            elif block.chgt_var == 'C':
                neut = block.neut_content[0]
                mapping[3*neut.MG-6] = p_random + 1
                mapping[3*neut.MG-5] = p_random + 2
                p_random += 2                
                 
            if block.chgt_var not in ['D', 'E', 'a', 'c']:
                for particle in block.in_part:
                    if particle.external and not particle.neutrino:
                        if particle.MG not in vis_list:
                            num_vis += 1
                            vis_str += str(particle.MG) + ','
                            vis_list.append(particle.MG)
                            if mapping[3*particle.MG-7] == 0 and particle.tf_level:
                                mapping[3*particle.MG-7] = p_random+1
                                p_random += 1

            elif block.chgt_var in ['E', 'c']:
                if block.chgt_var == 'E':
                    particle=block.in_part[2] #take the forward particle
                elif block.chgt_var == 'c':
                    particle=block.in_part[1] #take the closest to the neutrino
                if particle.external and not particle.neutrino:
                    if particle.MG not in vis_list:
                        num_vis += 1
                        vis_str += str(particle.MG) + ','
                        vis_list.append(particle.MG)
                        if mapping[3*particle.MG-7] == 0 and particle.tf_level:
                            mapping[3*particle.MG-7] = p_random+1
                            p_random += 1
            elif block.chgt_var in ['D']:
                p1 = block.in_part[0]
                p2 = block.in_part[1]
                if hasattr(p1, 'tf_level') and hasattr(p2,'tf_level'):
                    #one should be generate randomly
                    if mapping[3*p1.MG-7] == 0 and p1.tf_level:
                        mapping[3*p1.MG-7] = p_random+1
                    if mapping[3*p2.MG-7] == 0 and p2.tf_level:
                        mapping[3*p2.MG-7] = p_random+1
                    p_random+=1                   
               
                
        for particle in ambiguous_external:
            if particle not in part_treated:
                #part_treated.append(
                num_vis += 1
                vis_str += str(particle) + ','
                vis_list.append(particle)

        text=' data num_vis(' + str(num_sol) + ') /' + str(num_vis) + '/\n'
        if num_vis:
            vis_list.sort()
            vis_str=','.join([str(MG) for MG in vis_list])
            text += ' data (vis_nb(label,' + str(num_sol) + '),label=1,' + str(num_vis) + ') /' + vis_str + '/\n'
        text += ' data nb_block(' + str(num_sol) + ') / ' + str(self.nb_block) + '/\n'
        # check transfer function
        for particle in self.ext_content:
            if hasattr(particle,'has_theta_tf') and particle.has_theta_tf:
                mapping[3*particle.MG-5] = p_random+1
                p_random += 1
            if hasattr(particle,'has_phi_tf') and particle.has_phi_tf:    
                mapping[3*particle.MG-6] = p_random+1
                p_random += 1

        if ECS.step[0].chgt_var == 'a':
            for particle in ECS.step[0].in_part:
                 mapping[3*particle.MG-7] = p_random+1
                 p_random+=1

        mapping = [str(i+m_random) if i else str(i) for i in mapping]
        text += 'data (var2random(label, %(num_sol)s), label=1,%(size)s) / %(data)s/\n\n\n' % \
             {'num_sol': num_sol, 'size':len(mapping),'data':','.join(mapping)}
        write_text += put_in_fortran_format(text)
        #
        #   PROPAGATOR CONTENT -> LINKED TO SOLUTION
        #
        # 1) collect all generated propagator (in croissant order)
        # 2) write the code
        propa_list=self.collect_generated_propa(ECS, blob_sol_list)
        #text=' integer num_propa\n'
        text=' data num_propa(' + str(num_sol) + ') /' + str(len(propa_list)) + '/ \n'        
        if propa_list:
            text += ' data (propa_cont(label,' + str(num_sol) + '),label=1,' + str(len(propa_list)) + ') /'
            for particle in propa_list:
                text += str(particle.MG) + ','
            text=text[:-1] + '/\n'
        else:
            text += '\n$B$ S-COMMENT_C $B$\n No propagator aligned\n$E$ S-COMMENT_C $E$\n'

        for i in range(0, len(propa_list)):
            text += self.return_propa_generation(propa_list, i, num_sol)
        text=put_in_fortran_format(text)
        write_text += text

        return write_text

##     def create_pmass2(self):
##         """ create the pmass2 for all solution """
##         write_text="" 
##         for particle in self.content.values():
##             text='       pmass('+str(particle.MG)+') = '+str(particle.mass)+'d0\n'
##             if not particle.external:
##                text+='       pwidth('+str(particle.MG)+') = '+str(particle.width)+'d0\n'
##             write_text+=put_in_fortran_format(text)
##
##         ff=open(self.directory+'/pmass2.inc','w')
##         ff.writelines(write_text)
##         ff.close()

    def is_new_sol(self, code):
        """ check if this code is new or already defined """
        #Step 1: supress identical solution
        for i in range(0, len(self.code)):
            if self.code[i][0] == code[0]:
                if self.code[i][1] == code[1]:
                    return 0
        return 1

    def write_code(self, template):
        """ write the data_file and the main_code file """

        self.close_d_choices_file(template)
        self.check_redondant_peak(self.unaligned, self.unaligned_in_sol)
        
        write_main=template.dico['INTRO_FOR_MAIN']
        write_main += template.dico['START_ROUTINE']

        write_data=template.dico['INTRO_FOR_DATA']
        write_data += self.write_f77_parameter()
        write_data += template.dico['COMMON_DEF'] 
        
        write_mchannel=template.dico['INTRO_FOR_MULTICHANNEL']
        write_mchannel+=self.def_text_for_channel_weight(self.unaligned)
        for i in range(0, len(self.code)):
            write_main += self.code[i][0].replace('elseif (config_pos.eq.1)','if (config_pos.eq.1)')
            write_data += self.code[i][1]
            write_mchannel += self.create_multi_channel_weight(i,self.sol_nb[i])
        write_main += '        endif\n'
        write_main += '        return\n'
        write_main += '        end\n'
                
 #       write_mchannel += template.dico['SECONDPART_FOR_MULTICHANNEL'] #contains endif,return+start of following routine
 #       self.unaligned_correct_for_identical_solution()
 #       write_mchannel += self.write_channel_weight(self.unaligned,'+')
        write_mchannel += template.dico['END_FOR_MULTICHANNEL']  
        write_mchannel= put_in_fortran_format(write_mchannel)      

        # Add to data.inc the ordering of the config.
        dico = {'nb_sol_config':len(self.code),
                'values':[]}
        if 'restrict_channel' in self.MWparam['mw_gen']:
            if isinstance(self.MWparam['mw_gen']['restrict_channel'], list):
                allowed_channel = [int(i) for i in self.MWparam['mw_gen']['restrict_channel']]
            elif self.MWparam['mw_gen']['restrict_channel'] in ['','0','F']:
                allowed_channel = range(1,len(self.allowed_channel)+1)
            else:
                allowed_channel = [int(self.MWparam['mw_gen']['restrict_channel'])]
            full = allowed_channel + [0] * (len(self.code) - len(allowed_channel))
            dico['values'] = ','.join(map(str, full))
        else:
            dico['values'] = ','.join(map(str, range(1, len(self.code)+1)))
            
        write_data += put_in_fortran_format("""
C+-----------------------------------------------------------------------+
C|                  ORDERING OF THE CONFIGURATION                        |
C+-----------------------------------------------------------------------+
C|     order from the most important to the least important              |
C|         0 means that the configuration is bypassed                    |
C+-----------------------------------------------------------------------+
        data (config_ordering(label),label=1,%(nb_sol_config)i) /%(values)s/
C+-----------------------------------------------------------------------+
        """ % dico)
        
                    
        mod_file.mod_text(write_main, template.dico, self.directory + '/main_code.f')
        mod_file.mod_text(write_data, template.dico, self.directory + '/data.inc')        
        mod_file.mod_text(write_mchannel, template.dico, self.directory + '/multi_channel.f')

        # create permutations.inc file
        self.write_permutations_file()


    def write_f77_parameter(self):
        """ define the f77 parameter for the data file """
        
#        text=' integer nb_inv_part\n'                    
#        text+=' parameter (nb_inv_part='+str(self.num_neut)+')\n'
        text = ' integer nb_vis_part\n'
        text += ' parameter (nb_vis_part=' + str(len(self.ext_content) - self.num_neut) + ')\n'        
        text += ' integer nb_sol_config\n'                    
        text += ' parameter (nb_sol_config=' + str(len(self.code)) + ')\n'
        text += ' integer dim_phase_space\n parameter (dim_phase_space=%i)\n' % ((3*len(self.ext_content))+2)
        text += ' integer nb_channel\n'
        if self.MWparam['mw_perm']['montecarlo']:
            text += ' parameter (nb_channel=%i)\n' % len(self.code)
        else:
            text += ' parameter (nb_channel=%i)\n' % (len(self.code) * 48)
            
        text+=" integer config_ordering(nb_sol_config)\n "
#        text+=' integer max_branch\n'                    
#        text+=' parameter (max_branch='+str(len(self.ext_content))+')\n'        
        text = put_in_fortran_format(text)
        return text
        
        
    def write_d_choices(self, listpart):
        """ updates/creates the files d_choices.inc, d_choices.f
            return the three particle tag needed to call the block d 
        """
        
        tag1 = listpart[0].MG
        tag2 = listpart[1].MG
        if tag1 > tag2:
            tag1, tag2 = tag2, tag1  #this ensure convention order 
            
        tag3 = listpart[2].MG  #tag for the invariant mass
        if tag1 < 0: #tag2 is larger than tag1, so he cann't be negative 
            return '%s, %s, %s' % (tag2, tag1, tag3)

        
        if 'first_d_' + str(tag1) + '_' + str(tag2) not in self.d_block:
            self.d_block.append('first_d_' + str(tag1) + '_' + str(tag2)) 
            self.d_block.append('second_d_' + str(tag1) + '_' + str(tag2))
        else:
            return 'first_d_' + str(tag1) + '_' + str(tag2) + ', second_d_' + str(tag1) + '_' + str(tag2) + ',' + str(tag3)
         
        #write the definition in the inc file
        inc_text = '\n $B$ S-COMMENT_C $B$\n variable for block d containing:\n ' + \
                 str(tag1) + ' ' + str(tag2) + ' ' + str(tag3) + '\n$E$ S-COMMENT_C $E$\n'
        inc_text += '\n integer first_d_' + str(tag1) + '_' + str(tag2) + '\n'
        inc_text += '\n integer second_d_' + str(tag1) + '_' + str(tag2) + '\n'
        inc_text = put_in_fortran_format(inc_text)
        self.D_inc_text += inc_text      
        
        #write the call in the f file
        f_text = '\n $B$ S-COMMENT_C $B$\n variable for block d containing:\n ' + \
                 str(tag1) + ' ' + str(tag2) + ' ' + str(tag3) + '\n$E$ S-COMMENT_C $E$\n'
        f_text += '\n call init_block_d_alignment(' + str(tag1) + ',' + str(tag2) + ',' + \
               'first_d_' + str(tag1) + '_' + str(tag2) + ', second_d_' + str(tag1) + '_' + str(tag2) + ')\n'
        f_text = put_in_fortran_format(f_text)
        self.D_f_text += f_text  
        
        return 'first_d_' + str(tag1) + '_' + str(tag2) + ', second_d_' + str(tag1) + '_' + str(tag2) + ',' + str(tag3)
 
 
    def write_permutations_file(self):
        """ write the permutations.inc file """

        
        pid_list = []
        for i in range(3, 100):
            if i not in self.content:
                break
            pid_list.append(self.content[i].pid)        

        # assign each particles to a class of identical/equivalent particles
        permutations = get_perms_from_id(pid_list, self.MWparam['mw_perm']['bjet_is_jet'])


        # sanity check ensure that no identical permutation are presnt
        check = set([tuple(i) for i in permutations])
        assert len(check) == len(permutations)
        if not self.MWparam['mw_perm']['permutation']:
            permutations = permutations[0:1]
            
        text = open(self.directory + '/../permutation_template.f', 'r').read()
        
        text += '\n      subroutine get_perm(nb, perm)\n'
        text += '      implicit none\n'
        text += '      integer i,j\n'
        text += '      include \'nexternal.inc\'\n'
        text += '      INTEGER    NB\n'
        text += '      INTEGER    PERM(NEXTERNAL-2)\n'        
        text += '      include \'permutation.inc\'\n'
        text += '      INTEGER PERMS(NPERM, NEXTERNAL-2)\n'
        for i, perm in enumerate(permutations):
            text += "      DATA (PERMS(%s,I),I=1,%s) /%s/\n" % (i+1, len(perm),
                                               ','.join([str(j) for j in perm]))
        text += '        do i=1, NEXTERNAL-2\n'
        text += '            perm(i) = PERMS(nb, i)\n'
        text += '        enddo\n'   
        text += '        return\n' 
        text += '        end\n\n'
        text = put_in_fortran_format(text)
        open(self.directory + '/permutation.f', 'w').write(text)
        
        
        text = '        INTEGER    NPERM\n'
        text += '       PARAMETER (NPERM=%s)\n' % len(permutations)
        text += '       include \'nb_tf.inc\'\n'
        text += ' integer nb_channel2\n'
        text += '         integer min_perm(%i)\n'  % len(self.code)
        if self.MWparam['mw_perm']['montecarlo']:
            text += ' parameter (nb_channel2=%i)\n' % len(self.code)
        else:
            text += ' parameter (nb_channel2=%i)\n' % (len(self.code) * 48)
        
        text += '''        double precision perm_value(NPERM, nb_tf)
        double precision perm_error(NPERM,nb_tf)
        double precision perm_value_it(NPERM, nb_tf)
        double precision perm_error_it(NPERM, nb_tf)
        double precision tf_value_it(nb_tf)
        double precision tf_error_it(nb_tf)
        integer curr_perm, nb_point_by_perm(NPERM), perm_order(NPERM,nb_channel2)
        common/mw_perm_value/ perm_order,perm_value, perm_error, nb_point_by_perm, curr_perm, min_perm 
        common/mc_value_error/perm_value_it, perm_error_it, tf_value_it,tf_error_it
        '''
        text = put_in_fortran_format(text)
        open(self.directory + '/permutation.inc', 'w').write(text)

        #Update main_code.f
        template = """       
C*********************************************************************
        double precision function fct(x,wgt)
        implicit none

        include 'phasespace.inc'
        include 'nexternal.inc'
        include 'run.inc'
        include 'coupl.inc'
        include 'madweight_param.inc'
        integer curr_tf
        common/transfer_fct_curr_tf/curr_tf

c
c       this is the function which is called by the integrator

c
c       parameter
c
        double precision pi
        parameter (pi=3.141592653589793d0)
c
c       arguments
c
        double precision x(20),wgt
c
c       local
c
c        integer i,j ! debug mode
        double precision twgt
        integer new_perm
c
c       global
c
        double precision              S,X1,X2,PSWGT,JAC
        common /PHASESPACE/ S,X1,X2,PSWGT,JAC
        double precision momenta(0:3,-max_branches:2*max_particles)  ! momenta of external/intermediate legs     (MG order)
        double precision mvir2(-max_branches:2*max_particles)        ! squared invariant masses of intermediate particles (MG order)
        common /to_diagram_kin/ momenta, mvir2

        include 'permutation.inc'

        DOUBLE PRECISION Xl(20),XU(20),ACC
        INTEGER NDIM,NCALL,ITMX,NPRN
        COMMON/BVEG1/XL,XU,ACC, NDIM,NCALL,ITMX,NPRN
        integer perm_id(nexternal-2) !permutation of 1,2,...,nexternal-2
C
C     Keep track of whether cuts already calculated for this event 
C     
      LOGICAL CUTSDONE,CUTSPASSED
      COMMON/TO_CUTSDONE/CUTSDONE,CUTSPASSED
c
c       external
c
        double precision dsig
        external dsig
        double precision fct_before_tf, fct2
        double precision alphas
        external alphas
        logical passcuts
        external passcuts
        include 'data.inc'

c       choose the permutation (point by point in the ps)
        curr_tf = 1
        %(perm_init)s
        nb_point_by_perm(curr_perm) = nb_point_by_perm(curr_perm) + 1
        call get_PS_point(x)

         if (jac.gt.0d0) then
         %(use_cuts)s
         %(jac_scaling)s
           xbk(1)=X1
           xbk(2)=X2
           fct_before_tf=jac*dsig(momenta(0,1),wgt)
           
           do curr_tf=1,nb_tf
               call transfer_fct(momenta(0,1),TWGT)
               if (curr_tf.eq.1)then
                   fct = fct_before_tf*twgt
                   fct2 = fct 
               else
                   fct2 = fct_before_tf*twgt
               endif
               
                
         %(histo)s
        perm_value(curr_perm, curr_tf) = perm_value(curr_perm, curr_tf) + fct2*wgt
        perm_error(curr_perm, curr_tf) = perm_error(curr_perm, curr_tf) + fct**2*wgt**2
        enddo
         else
           fct=0d0
         endif
        curr_tf = 1
         
         end
         """
        
        data = {'perm_init': '', 'perm_storing':'',
                'histo':'', 'use_cuts':'', 'jac_scaling': ''}
        
        if self.MWparam['mw_perm']['permutation'] and len(permutations) >1:
            data['perm_init'] = """
        new_perm = perm_order(min_perm(config_pos) + int(((NPERM - min_perm(config_pos) +1) * x(NDIM))), config_pos)
        if (new_perm.ne.curr_perm) then
           call get_perm(new_perm, perm_id)
           call assign_perm(perm_id)
           curr_perm = new_perm
        endif
    """
            data['jac_scaling'] = """    
            jac = jac * (NPERM - min_perm(config_pos) +1)/NPERM
            """
        if not self.MWparam['mw_perm']['montecarlo'] and len(permutations) >1:
            data['perm_init'] = ""
                
        if self.MWparam['mw_run']['histo']:
            data['histo'] = """
        if (histo)  then
           call FILL_plot(fct,wgt,perm_pos*nb_sol_config+config_pos,nexternal)
        endif
            """
            
        if self.MWparam['mw_run']['use_cut']:
            data['use_cuts'] = """
        CUTSPASSED=.FALSE.
        CUTSDONE=.FALSE.
        if (.not.passcuts(momenta(0,1))) then
            fct = 0d0
            return
        endif
            """
        else:
            data['use_cuts'] = """
        if(.not.fixed_ren_scale) then
          call set_ren_scale(momenta(0,1),scale)
          if(scale.gt.0) G = SQRT(4d0*PI*ALPHAS(scale))
          call UPDATE_AS_PARAM()
        endif
        if(.not.fixed_fac_scale) then
          call set_fac_scale(momenta(0,1),q2fact)
        endif
         """
         
        text = put_in_fortran_format(template % data)
        open(self.directory + '/main_code.f', 'a').write(text)
        
 
    def init_d_choices_file(self):
        """ write banner in the fortran/inc file """

        self.d_block = []
        self.D_f_text = '$B$ INTRO_FOR_D_SWITCH_F $E$\n'
        self.D_inc_text = '$B$ INTRO_FOR_D_SWITCH_INC $E$\n'
        self.D_f_text += '\n  subroutine init_d_assignement() \n include \'d_choices.inc\' \n'
       
    def close_d_choices_file(self, template):
        """write the end of the D block related files """   
        #ending f file
        text = '\n return \n end\n'
        text = put_in_fortran_format(text)
        self.D_f_text += text
        
        #endind .inc file (add common)
        text = '\n$B$ S-COMMENT_C $B$\n Definition of the common\n$E$ S-COMMENT_C $E$\n'
        if self.d_block:
            text += '\n common/to_d_block/' + ','.join(self.d_block) + '\n'
        text = put_in_fortran_format(text)
        self.D_inc_text += text
        
        #write text in file
        self.D_f_text = put_in_fortran_format(self.D_f_text)
        self.D_inc_text = put_in_fortran_format(self.D_inc_text)
        mod_file.mod_text(self.D_inc_text, template.dico, self.directory + '/d_choices.inc')
        mod_file.mod_text(self.D_f_text, template.dico, self.directory + '/d_choices.f')       
     
    def collect_generated_propa(self, ECS, blob_sol_list):
        """ return all the propagator that must be generated following BW distibution """

        def id_from_plist(plist):
            return [particle.MG for particle in plist]

        plist = []
        for particle in ECS.step[-1].order_content:
            if not particle.external and type(particle.MG) == int and \
            particle not in plist:
                if not particle in [b.main  for b in ECS.blob_content if hasattr(b, 'main')]:
                    plist.append(particle)
        for blob_sol in blob_sol_list:
            for block in blob_sol.step:
                if block.chgt_var in ['A', 'B', 'C', 'D', 'E']:
                    for particle in block.order_content:
                        if not particle.external and type(particle.MG) == int and \
                            particle not in plist and particle not in block.in_part:
                            plist.append(particle)
#        plist.reverse()

        list2 = []
        list3 = []
        while plist:
            propa = plist.pop()
            if propa.channel == 'S':
                list2.append(propa)
            else:
                list3.append(propa)
                
##                 gen=1
##                 for i in range(0,len(propa.des)):
##                     if propa.des[i] in list:
##                         list.insert(propa,i+1)
##                         gen=0
##                         break
##                 if gen:
##                    list2.append(propa)                 
        return list2 + list3
    
    def collect_unaligned_peaks(self):
        """ first create for each solution a list of all unaligned peaks
            secondly make a full list for the full set of solution
            check if a specific peak is never aligned
        """
        
        def add_peaks(unaligned, peak):
            """ add a peak in obj.unaligned """

            if type(peak) == list:
                for one_peak in peak:
                    add_peaks(unaligned, one_peak)
                return
            
            if unaligned.has_key(peak):
                unaligned[peak] += 1
            else:
                if isinstance(peak, str) or peak.external:
                    unaligned[peak] = 1
                elif isinstance(peak.MG, str):
                    pass
                elif peak.external==0 and peak.channel.startswith('S'):
                    unaligned[peak] = 1
#                else:
#                    print 'rejected'
#            print 'RESULT:'
#            print print_(unaligned)
            return

        def print_(list_local):
            """ return a readable content of unaligned peak"""
            text=''
            if type(list_local)!=list:
                list_local=[list_local]
            for one_sol in list_local:
                for key in one_sol.keys():
                    text+=str(key)+':'+str(one_sol[key])+'\n'
                text+='\n'
            return text
            
    
        if not hasattr(self,'unaligned'):
            self.unaligned = {}
            self.unaligned_in_sol = []
            
        self.full_sol = [] 
                
        for ECS in self.ECS_sol: # ALL ECS SECTOR
            full_solution_tag = [ECS, []]
            full_blob_sol = Multi_list()
            for BLOB in ECS.blob_content:
                full_blob_sol.append(BLOB.solution)
            full_blob_sol = full_blob_sol.give_combinaison()#expanded solution
            for one_full_solution in full_blob_sol:
                self.full_sol.append([ECS,one_full_solution])
                unaligned_in_this_sol = {}
                for block in ECS.step: 
                        add_peaks(unaligned_in_this_sol, block.unaligned)
                        add_peaks(self.unaligned, block.unaligned)
                for blob in one_full_solution:
                    for block in blob.step:
                        add_peaks(unaligned_in_this_sol, block.unaligned)
                        add_peaks(self.unaligned, block.unaligned)
                for particles in self.prop_content:
                    if particles.channel == 'S_flat':
                        add_peaks(unaligned_in_this_sol, particles)
                        add_peaks(self.unaligned, particles)
                self.unaligned_in_sol.append(unaligned_in_this_sol)
            

 
    def unaligned_correct_for_identical_solution(self):
        """ correct self.unaligned from the fact that some solution was take into 
            account more than once 
        """
           
        for i in self.already_existing_solution:
            for peak in self.unaligned_in_sol[i]:
                self.unaligned[peak] -= 1
            
            
    def return_propa_generation(self, list, pos, num_sol):
        """return the line for the definition of how to generate the mass
           typical output are:
           data (propa_???($,label),label=1,$) /$,$,$,$,$,0/ 
        """



        particle = list[pos]
        line1 = ' data (propa_max(' + str(pos + 1) + ',label,' + str(num_sol) + '),label=1,'
        line2 = ' data (propa_min(' + str(pos + 1) + ',label,' + str(num_sol) + '),label=1,'
        generated_mother = []
        generated_twin = []
        generated_son = []
        already_gen = list[:pos]
        
        motherX = list[pos]
        #look for minimal value
        generated_son += self.already_generated_in_decay(motherX, already_gen)
        generated_son.append(0)
        while 1:
            motherXbut1 = motherX
            motherX = motherX.mother
            if motherX == 0:
                break
            #look for maximal value
            generated_twin += self.already_generated_in_decay(motherXbut1.twin, already_gen)                
            if motherX in already_gen:
                generated_mother = [motherX.MG]
                generated_twin.append(0)
                break                               
        if not generated_mother:
             generated_mother = [0]
             generated_twin = []

        gen = generated_mother + generated_twin
        line1 += str(len(gen)) + ') / '
        line2 += str(len(generated_son)) + ') / '
        
        for MG_num in gen:
            line1 += str(MG_num) + ','
        line1 = line1[:-1] + '/\n'
        
        for MG_num in generated_son:
            line2 += str(MG_num) + ','
        line2 = line2[:-1] + '/\n'
        
        return line1 + line2
        
            
        
    def already_generated_in_decay(self, particle, generated_propa):
        """give (recurently) all the first particle already generated in the branchs of desintegration"""

        if particle.external:
            return [particle.MG]
        elif particle in generated_propa:
            return [particle.MG]
        else:
            part1 = self.already_generated_in_decay(particle.des[0], generated_propa)
            part2 = self.already_generated_in_decay(particle.des[1], generated_propa)
            return part1 + part2

    def check_invisible_decay(self):
        """ check if one of the invisible particle decay in 2 invisible particle
            return 0 if not
            return a text with the call of the equivalent subroutine
        """
        decay_num = 0
        for particle in self.neut_content:
            if particle.external:
                continue
            decay_num += 1
            if not decay_num:
                out_text = self.template.comment_text('\t Invisible Propagator', 'C')
            text = ' decay(' + str(particle.MG) + ',' + str(particle.des[0].MG) + ',' + str(particle.des[1].MG) + ')'

            out_text += put_in_fortran_format(text)

        if decay_num:
            return out_text
        else:
            return 0

    def create_output_type_info(self):
        """ create output file containing the number of muon/electron/jet/bjet/invisible_part """


        content = self.output_type_info()  
                    
        ff = open(self.directory + '/info_part.dat', 'w')
        text = ""
        for i in range(0, len(content)):
            text += '\t' + str(content[i])
        ff.writelines(text)
        ff.close()

    def check_redondant_peak(self,dict_all, list_local):
        """  check that in each solution each peaks appears at most one times and 
             remove peaks present in all solution (if any)
             check also conflicts D/E peaks occur  
        """
        list_d=[]
        dict_mg_to_peak={}
        for one_sol in list_local:
            local_mg={}
            for peak, value in one_sol.items():
                if isinstance(peak,basestring):
                    value2=peak.split('_')[-2:]
                    list_d.append(value2)
                else:
                    dict_mg_to_peak[peak.MG]=peak
                        
                if value != 1:
                    dict_all[peak] += 1 - value
                    one_sol[peak] = 1
                        
        nb_sol = len(list_local)
        for peak, value in dict_all.items():
            if value == nb_sol:
                if isinstance(peak,basestring):
                    print 'WARNING a peak associated to a visible particle is never '+ \
                          'aligned. This will slow down the integration'
                elif peak.MG<0 and peak.external == 0 and peak.channel.startswith('S'):
                    print 'WARNING a peak associated to '+str(peak.MG)+' is never '+ \
                           'aligned. This will slow down the integration '
                dict_all[peak] = 0
                for list_peak in list_local:
                    del list_peak[peak]
                
        for peak1_MG,peak2_MG in list_d:
            try:
                peak1,peak2= dict_mg_to_peak[int(peak1_MG)],dict_mg_to_peak[int(peak2_MG)]
            except:
                continue
            for one_sol in list_local:
                if one_sol.has_key(peak1) and one_sol.has_key(peak2):
                    del one_sol[peak1]
                    del one_sol[peak2]
                    name1, name2= 'first_d_%s_%s' % (peak1_MG,peak2_MG),'second_d_%s_%s' % (peak1_MG,peak2_MG)
                    one_sol[name1] = 1
                    one_sol[name2] = 1
                    dict_all[peak1] -=1
                    dict_all[peak2] -=1
                    dict_all[name1] +=1
                    if dict_all.has_key(name2):
                        dict_all[name2] +=1
                    else:
                        dict_all[name2] =1
        return
        
if(__name__ == "__main__"):
    """ launched the generation """
    import MW_param

    MW_param.go_to_main_dir()
    MW_opt = MW_param.MW_info('MadWeight_card.dat')

    create_all_fortran_code(MW_opt)
    


