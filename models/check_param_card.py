from __future__ import division

import itertools
import xml.etree.ElementTree as ET
import math
import StringIO
import os
import re
import shutil
import logging

logger = logging.getLogger('madgraph.models') # -> stdout

try:
    import madgraph.iolibs.file_writers as file_writers
    import madgraph.various.misc as misc    
except:
    import internal.file_writers as file_writers
    import internal.misc as misc

import StringIO
    
class InvalidParamCard(Exception):
    """ a class for invalid param_card """
    pass

class Parameter (object):
    """A class for a param_card parameter"""
    
    def __init__(self, param=None, block=None, lhacode=None, value=None, comment=None):
        """Init the parameter"""

        self.format = 'float'
        if param:
            block = param.lhablock
            lhacode = param.lhacode
            value = param.value
            comment = param.comment
            format = param.format

        self.lhablock = block
        if lhacode:
            self.lhacode = lhacode
        else:
            self.lhacode = []
        self.value = value
        self.comment = comment

    def set_block(self, block):
        """ set the block name """
        
        self.lhablock = block

    def load_str(self, text):
        """ initialize the information from a str"""

        if '#' in text:
            data, self.comment = text.split('#',1)
        else:
            data, self.comment = text, ""


        data = data.split()
        if any(d.startswith('scan') for d in data):
            position = [i for i,d in enumerate(data) if d.startswith('scan')][0]
            data = data[:position] + [' '.join(data[position:])] 
        if not len(data):
            return
        try:
            self.lhacode = tuple([int(d) for d in data[:-1]])
        except Exception:
            self.lhacode = tuple([int(d) for d in data[:-1] if d.isdigit()])
            self.value= ' '.join(data[len(self.lhacode):])
        else:
            self.value = data[-1]
        
        # convert to number when possible
        try:
            self.value = float(self.value)
        except:
            self.format = 'str'
            pass
        else:
            if self.lhablock == 'modsel':
                self.format = 'int'
                self.value = int(self.value)

    def load_decay(self, text):
        """ initialize the decay information from a str"""

        if '#' in text:
            data, self.comment = text.split('#',1)
        else:
            data, self.comment = text, ""


        data = data.split()
        if not len(data):
            return
        self.lhacode = [int(d) for d in data[2:]]
        self.lhacode.sort()
        self.lhacode = tuple([len(self.lhacode)] + self.lhacode)
        
        self.value = float(data[0]) 
        self.format = 'decay_table'

    def __str__(self):
        """ return a SLAH string """

        format = self.format
        if self.format == 'float':
            try:
                value = float(self.value)
            except:
                format = 'str'
        
        if format == 'float':
            if self.lhablock == 'decay' and not isinstance(self.value,basestring):
                return 'DECAY %s %e # %s' % (' '.join([str(d) for d in self.lhacode]), self.value, self.comment)
            elif self.lhablock == 'decay':
                return 'DECAY %s Auto # %s' % (' '.join([str(d) for d in self.lhacode]), self.comment)
            elif self.lhablock and self.lhablock.startswith('qnumbers'):
                return '      %s %i # %s' % (' '.join([str(d) for d in self.lhacode]), int(self.value), self.comment)
            else:
                return '      %s %e # %s' % (' '.join([str(d) for d in self.lhacode]), self.value, self.comment)
        elif format == 'int':
            return '      %s %i # %s' % (' '.join([str(d) for d in self.lhacode]), int(self.value), self.comment)
        elif format == 'str':
            if self.lhablock == 'decay':
                return 'DECAY %s %s # %s' % (' '.join([str(d) for d in self.lhacode]),self.value, self.comment)
            return '      %s %s # %s' % (' '.join([str(d) for d in self.lhacode]), self.value, self.comment)
        elif self.format == 'decay_table':
            return '      %e %s # %s' % ( self.value,' '.join([str(d) for d in self.lhacode]), self.comment)
        elif self.format == 'int':
            return '      %s %i # %s' % (' '.join([str(d) for d in self.lhacode]), int(self.value), self.comment)
        else:
            if self.lhablock == 'decay':
                return 'DECAY %s %d # %s' % (' '.join([str(d) for d in self.lhacode]), self.value, self.comment)
            else:
                return '      %s %d # %s' % (' '.join([str(d) for d in self.lhacode]), self.value, self.comment)


class Block(list):
    """ list of parameter """
    
    def __init__(self, name=None):
        if name:
            self.name = name.lower()
        else:
            self.name = name
        self.scale = None
        self.comment = ''
        self.decay_table = {}
        self.param_dict={}
        list.__init__(self)

    def get(self, lhacode, default=None):
        """return the parameter associate to the lhacode"""
        if not self.param_dict:
            self.create_param_dict()
        
        if isinstance(lhacode, int):
            lhacode = (lhacode,)
            
        try:
            return self.param_dict[tuple(lhacode)]
        except KeyError:
            if default is None:
                raise KeyError, 'id %s is not in %s' % (tuple(lhacode), self.name)
            else:
                return Parameter(block=self, lhacode=lhacode, value=default,
                                                           comment='not define')
        
    def remove(self, lhacode):
        """ remove a parameter """
        list.remove(self, self.get(lhacode))
        # update the dictionary of key
        return self.param_dict.pop(tuple(lhacode))
    
    def __eq__(self, other, prec=1e-4):
        """ """
        if len(self) != len(other):
            return False
        
        return not any(abs(param.value-other.param_dict[key].value)> prec * abs(param.value)
                        for key, param in self.param_dict.items())
        
    def __ne__(self, other, prec=1e-4):
        return not self.__eq__(other, prec)
        
    def append(self, obj):
        
        assert isinstance(obj, Parameter)
        if not hasattr(self, 'name'): #can happen if loeaded from pickle
            self.__init__(obj.lhablock)
        assert not obj.lhablock or obj.lhablock == self.name

        #The following line seems/is stupid but allow to pickle/unpickle this object
        #this is important for madspin (in gridpack mode)
        if not hasattr(self, 'param_dict'):
            self.param_dict = {}
            
        if tuple(obj.lhacode) in self.param_dict:
            if self.param_dict[tuple(obj.lhacode)].value != obj.value:
                raise InvalidParamCard, '%s %s is already define to %s impossible to assign %s' % \
                    (self.name, obj.lhacode, self.param_dict[tuple(obj.lhacode)].value, obj.value)
            return
        list.append(self, obj)
        # update the dictionary of key
        self.param_dict[tuple(obj.lhacode)] = obj

    def create_param_dict(self):
        """create a link between the lhacode and the Parameter"""
        for param in self:
            self.param_dict[tuple(param.lhacode)] = param
        
        return self.param_dict

    def def_scale(self, scale):
        """ """
        self.scale = scale

    def load_str(self, text):
        "set inforamtion from the line"
        
        if '#' in text:
            data, self.comment = text.split('#',1)
        else:
            data, self.commant = text, ""

        data = data.lower()
        data = data.split()
        self.name = data[1] # the first part of data is model
        if len(data) == 3:
            if data[2].startswith('q='):
                #the last part should be of the form Q=
                self.scale = float(data[2][2:])
            elif self.name == 'qnumbers':
                self.name += ' %s' % data[2]
        elif len(data) == 4 and data[2] == 'q=':
            #the last part should be of the form Q=
            self.scale = float(data[3])                
            
        return self
    
    def keys(self):
        """returns the list of id define in this blocks"""
        
        return [p.lhacode for p in self]

    def __str__(self):
        """ return a str in the SLAH format """ 
        
        text = """###################################""" + \
               """\n## INFORMATION FOR %s""" % self.name.upper() +\
               """\n###################################\n"""

        #special case for decay chain
        if self.name == 'decay':
            for param in self:
                pid = param.lhacode[0]
                param.set_block('decay')
                text += str(param)+ '\n'
                if self.decay_table.has_key(pid):
                    text += str(self.decay_table[pid])+'\n'
            return text
        elif self.name.startswith('decay'):
            text = '' # avoid block definition
        #general case 
        elif not self.scale:
            text += 'BLOCK %s # %s\n' % (self.name.upper(), self.comment)
        else:
            text += 'BLOCK %s Q= %e # %s\n' % (self.name.upper(), self.scale, self.comment)
        
        text += '\n'.join([str(param) for param in self])
        return text + '\n'


class ParamCard(dict):
    """ a param Card: list of Block """
    mp_prefix = 'MP__'

    header = \
    """######################################################################\n""" + \
    """## PARAM_CARD AUTOMATICALY GENERATED BY MG5                       ####\n""" + \
    """######################################################################\n"""


    def __init__(self, input_path=None):
        self.order = []
        
        if isinstance(input_path, ParamCard):
            self.read(input_path.write())
            self.input_path = input_path.input_path 
        else:
            self.input_path = input_path
            if input_path:
                self.read(input_path)
        
    def read(self, input_path):
        """ read a card and full this object with the content of the card """

        if isinstance(input_path, str):
            if '\n' in input_path:
                input = StringIO.StringIO(input_path)
            else:
                input = open(input_path)
        else:
            input = input_path #Use for banner loading and test


        cur_block = None
        for line in input:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            line = line.lower()
            if line.startswith('block'):
                cur_block = Block()
                cur_block.load_str(line)
                self.append(cur_block)
                continue
            
            if line.startswith('decay'):
                if not self.has_block('decay'):
                    cur_block = Block('decay')
                    self.append(cur_block)
                else:
                    cur_block = self['decay']
                param = Parameter()
                param.set_block(cur_block.name)
                param.load_str(line[6:])
                cur_block.append(param)
                continue

            if cur_block is None:
                continue            
                    
            if cur_block.name == 'decay':
                # This is a decay table
                id =  cur_block[-1].lhacode[0]
                cur_block = Block('decay_table_%s' % id)
                self['decay'].decay_table[id] = cur_block
            
            

            
            if cur_block.name.startswith('decay_table'):
                param = Parameter()
                param.load_decay(line)
                try:
                    cur_block.append(param)
                except InvalidParamCard:
                    pass
            else:
                param = Parameter()
                param.set_block(cur_block.name)
                param.load_str(line)
                cur_block.append(param)
                  
        return self
    
    def analyze_param_card(self):
        """ Analyzes the comment of the parameter in the param_card and returns
        a dictionary with parameter names in values and the tuple (lhablock, id)
        in value as well as a dictionary for restricted values.
        WARNING: THIS FUNCTION RELIES ON THE FORMATTING OF THE COMMENT IN THE
        CARD TO FETCH THE PARAMETER NAME. This is mostly ok on the *_default.dat
        but typically dangerous on the user-defined card."""
    
        pname2block = {}
        restricted_value = {}

        for bname, block in self.items():
            for lha_id, param in block.param_dict.items():
                all_var = []
                comment = param.comment
                # treat merge parameter
                if comment.strip().startswith('set of param :'):
                    all_var = list(re.findall(r'''[^-]1\*(\w*)\b''', comment))
                # just the variable name as comment
                elif len(comment.split()) == 1:
                    all_var = [comment.strip().lower()]
                # either contraction or not formatted
                else:
                    split = comment.split()
                    if len(split) >2 and split[1] == ':':
                        # NO VAR associated
                        restricted_value[(bname, lha_id)] = ' '.join(split[1:])
                    elif len(split) == 2:
                        if re.search(r'''\[[A-Z]\]eV\^''', split[1]):
                            all_var = [comment.strip().lower()]
                    elif len(split) >=2 and split[1].startswith('('):
                        all_var = [split[0].strip().lower()]
                    else:
                        if not bname.startswith('qnumbers'):
                            logger.debug("not recognize information for %s %s : %s",
                                      bname, lha_id, comment)
                        # not recognized format
                        continue

                for var in all_var:
                    var = var.lower()
                    if var in pname2block:
                        pname2block[var].append((bname, lha_id))
                    else:
                        pname2block[var] = [(bname, lha_id)]
        
        return pname2block, restricted_value
    
    def write(self, outpath=None):
        """schedular for writing a card"""
  
        # order the block in a smart way
        blocks = self.order_block()
        text = self.header
        text += ''.join([str(block) for block in blocks])

        if not outpath:
            return text
        elif isinstance(outpath, str):
            file(outpath,'w').write(text)
        else:
            outpath.write(text) # for test purpose
    
    def create_diff(self, new_card):
        """return a text file allowing to pass from this card to the new one
           via the set command"""
        
        diff = ''
        for blockname, block in self.items():
            for param in block:
                lhacode = param.lhacode
                value = param.value
                new_value = new_card[blockname].get(lhacode).value
                if not misc.equal(value, new_value, 6, zero_limit=False):
                    lhacode = ' '.join([str(i) for i in lhacode])
                    diff += 'set param_card %s %s %s # orig: %s\n' % \
                                       (blockname, lhacode , new_value, value)
        return diff 


    def write_inc_file(self, outpath, identpath, default, need_mp=False):
        """ write a fortran file which hardcode the param value"""
        
        fout = file_writers.FortranWriter(outpath)
        defaultcard = ParamCard(default)
        for line in open(identpath):
            if line.startswith('c  ') or line.startswith('ccccc'):
                continue
            split = line.split()
            if len(split) < 3:
                continue
            block = split[0]
            lhaid = [int(i) for i in split[1:-1]]
            variable = split[-1]
            if block in self:
                try:
                    value = self[block].get(tuple(lhaid)).value
                except KeyError:
                    value =defaultcard[block].get(tuple(lhaid)).value
                    logger.warning('information about \"%s %s" is missing using default value: %s.' %\
                                                          (block, lhaid, value))
            else:
                value =defaultcard[block].get(tuple(lhaid)).value
                logger.warning('information about \"%s %s" is missing (full block missing) using default value: %s.' %\
                                   (block, lhaid, value))
            value = str(value).lower()
            fout.writelines(' %s = %s' % (variable, ('%e'%float(value)).replace('e','d')))
            if need_mp:
                fout.writelines(' mp__%s = %s_16' % (variable, value))
      
    def convert_to_complex_mass_scheme(self):
        """ Convert this param_card to the convention used for the complex mass scheme:
        This includes, removing the Yukawa block if present and making sure the EW input
        scheme is (MZ, MW, aewm1). """
        
        # The yukawa block is irrelevant for the CMS models, we must remove them
        if self.has_block('yukawa'):
            # Notice that the last parameter removed will also remove the block.
            for lhacode in [param.lhacode for param in self['yukawa']]:
                self.remove_param('yukawa', lhacode)
    
        # Now fix the EW input scheme
        EW_input = {('sminputs',(1,)):None,
                    ('sminputs',(2,)):None,
                    ('mass',(23,)):None,
                    ('mass',(24,)):None}
        for block, lhaid in EW_input.keys():
            try:
                EW_input[(block,lhaid)] = self[block].get(lhaid).value
            except:
                pass
            
        # Now specify the missing values. We only support the following EW
        # input scheme:
        # (alpha, GF, MZ) input
        internal_param = [key for key,value in EW_input.items() if value is None]
        if len(internal_param)==0:
            # All parameters are already set, no need for modifications
            return
        
        if len(internal_param)!=1:
            raise InvalidParamCard,' The specified EW inputs has more than one'+\
                ' unknown: [%s]'%(','.join([str(elem) for elem in internal_param]))
        
        
        if not internal_param[0] in [('mass',(24,)), ('sminputs',(2,)),
                                                             ('sminputs',(1,))]:
            raise InvalidParamCard, ' The only EW input scheme currently supported'+\
                        ' are those with either the W mass or GF left internal.'
        
        # Now if the Wmass is internal, then we must change the scheme
        if internal_param[0] == ('mass',(24,)):
            aewm1 = EW_input[('sminputs',(1,))]
            Gf    = EW_input[('sminputs',(2,))]
            Mz    = EW_input[('mass',(23,))]
            try:
                Mw = math.sqrt((Mz**2/2.0)+math.sqrt((Mz**4/4.0)-((
                              (1.0/aewm1)*math.pi*Mz**2)/(Gf*math.sqrt(2.0)))))
            except:
                InvalidParamCard, 'The EW inputs 1/a_ew=%f, Gf=%f, Mz=%f are inconsistent'%\
                                                                   (aewm1,Gf,Mz)
            self.remove_param('sminputs', (2,))
            self.add_param('mass', (24,), Mw, 'MW')
        
    def append(self, obj):
        """add an object to this"""
        
        assert isinstance(obj, Block)
        self[obj.name] = obj
        if not obj.name.startswith('decay_table'): 
            self.order.append(obj)
        
        
        
    def has_block(self, name):
        return self.has_key(name)
    
    def order_block(self):
        """ reorganize the block """
        return self.order
    
    def rename_blocks(self, name_dict):
        """ rename the blocks """
        
        for old_name, new_name in name_dict.items():
            self[new_name] = self.pop(old_name)
            self[new_name].name = new_name
            for param in self[new_name]:
                param.lhablock = new_name
                
    def remove_block(self, name):
        """ remove a blocks """
        assert len(self[name])==0
        [self.order.pop(i) for i,b in enumerate(self.order) if b.name == name]
        self.pop(name)
        
    def remove_param(self, block, lhacode):
        """ remove a parameter """
        if self.has_param(block, lhacode):
            self[block].remove(lhacode)
            if len(self[block]) == 0:
                self.remove_block(block)
    
    def has_param(self, block, lhacode):
        """check if param exists"""
        
        try:
            self[block].get(lhacode)
        except:
            return False
        else:
            return True
        
    def copy_param(self,old_block, old_lha, block=None, lhacode=None):
        """ make a parameter, a symbolic link on another one """
        
        # Find the current block/parameter
        old_block_obj = self[old_block]
        parameter = old_block_obj.get(old_lha)        
        if not block:
            block = old_block
        if not lhacode:
            lhacode = old_lha
            
        self.add_param(block, lhacode, parameter.value, parameter.comment)
        
    def add_param(self,block, lha, value, comment=''):
        
        parameter = Parameter(block=block, lhacode=lha, value=value, 
                              comment=comment)
        try:
            new_block = self[block]
        except KeyError:
            # If the new block didn't exist yet
            new_block = Block(block)
            self.append(new_block)
        new_block.append(parameter)
        
             
    def mod_param(self, old_block, old_lha, block=None, lhacode=None, 
                                              value=None, comment=None):
        """ change a parameter to a new one. This is not a duplication."""

        # Find the current block/parameter
        old_block = self[old_block]
        try:
            parameter = old_block.get(old_lha)
        except:
            if lhacode is not None:
                lhacode=old_lha
            self.add_param(block, lhacode, value, comment)
            return
        

        # Update the parameter
        if block:
            parameter.lhablock = block
        if lhacode:
            parameter.lhacode = lhacode
        if value:
            parameter.value = value
        if comment:
            parameter.comment = comment

        # Change the block of the parameter
        if block:
            old_block.remove(old_lha)
            if not len(old_block):
                self.remove_block(old_block.name)
            try:
                new_block = self[block]
            except KeyError:
                # If the new block didn't exist yet
                new_block = Block(block)
                self.append(new_block)            
            new_block.append(parameter)
        elif lhacode:
            old_block.param_dict[tuple(lhacode)] = \
                                  old_block.param_dict.pop(tuple(old_lha))


    def check_and_remove(self, block, lhacode, value):
        """ check that the value is coherent and remove it"""
        
        if self.has_param(block, lhacode):
            param = self[block].get(lhacode)
            if param.value != value:
                error_msg = 'This card is not suitable to be convert to SLAH1\n'
                error_msg += 'Parameter %s %s should be %s' % (block, lhacode, value)
                raise InvalidParamCard, error_msg   
            self.remove_param(block, lhacode)


class ParamCardMP(ParamCard):
    """ a param Card: list of Block with also MP definition of variables"""
            
    def write_inc_file(self, outpath, identpath, default):
        """ write a fortran file which hardcode the param value"""
        
        fout = file_writers.FortranWriter(outpath)
        defaultcard = ParamCard(default)
        for line in open(identpath):
            if line.startswith('c  ') or line.startswith('ccccc'):
                continue
            split = line.split()
            if len(split) < 3:
                continue
            block = split[0]
            lhaid = [int(i) for i in split[1:-1]]
            variable = split[-1]
            if block in self:
                try:
                    value = self[block].get(tuple(lhaid)).value
                except KeyError:
                    value =defaultcard[block].get(tuple(lhaid)).value
            else:
                value =defaultcard[block].get(tuple(lhaid)).value
            #value = str(value).lower()
            fout.writelines(' %s = %s' % (variable, ('%e' % value).replace('e','d')))
            fout.writelines(' %s%s = %s_16' % (self.mp_prefix, 
                variable, ('%e' % value)))



class ParamCardIterator(ParamCard):
    """A class keeping track of the scan: flag in the param_card and 
       having an __iter__() function to scan over all the points of the scan.
    """

    logging = True
    def __init__(self, input_path=None):
        super(ParamCardIterator, self).__init__(input_path=input_path)
        self.itertag = [] #all the current value use
        self.cross = []   # keep track of all the cross-section computed 
        self.param_order = []
        
    def __iter__(self):
        """generate the next param_card (in a abstract way) related to the scan.
           Technically this generates only the generator."""
        
        if hasattr(self, 'iterator'):
            return self.iterator
        self.iterator = self.iterate()
        return self.iterator
    
    def next(self, autostart=False):
        """call the next iteration value"""
        try:
            iterator = self.iterator
        except:
            if autostart:
                iterator = self.__iter__()
            else:
                raise
        try:
            out = iterator.next()
        except StopIteration:
            del self.iterator
            raise
        return out
    
    def iterate(self):
        """create the actual generator"""
        all_iterators = {} # dictionary of key -> block of object to scan [([param, [values]), ...]
        auto = 'Auto'
        pattern = re.compile(r'''scan\s*(?P<id>\d*)\s*:\s*(?P<value>[^#]*)''', re.I)
        # First determine which parameter to change and in which group
        # so far only explicit value of the scan (no lambda function are allowed)
        for block in self.order:
            for param in block:
                if isinstance(param.value, str) and param.value.strip().lower().startswith('scan'):
                    try:
                        key, def_list = pattern.findall(param.value)[0]
                    except:
                        raise Exception, "Fail to handle scanning tag: Please check that the syntax is valid"
                    if key == '': 
                        key = -1 * len(all_iterators)
                    if key not in all_iterators:
                        all_iterators[key] = []
                    try:
                        all_iterators[key].append( (param, eval(def_list)))
                    except SyntaxError:
                        raise Exception, "Fail to handle your scan definition. Please check your syntax."
                    
        keys = all_iterators.keys() # need to fix an order for the scan
        param_card = ParamCard(self)
        #store the type of parameter
        for key in keys:
            for param, values in all_iterators[key]:
                self.param_order.append("%s#%s" % (param.lhablock, '_'.join(`i` for i in param.lhacode)))
        
        # do the loop
        lengths = [range(len(all_iterators[key][0][1])) for key in keys]
        for positions in itertools.product(*lengths):
            self.itertag = []
            if self.logging:
                logger.info("Create the next param_card in the scan definition", '$MG:color:BLACK')
            for i, pos in enumerate(positions):
                key = keys[i]
                for param, values in all_iterators[key]:
                    # assign the value in the card.
                    param_card[param.lhablock].get(param.lhacode).value = values[pos]
                    self.itertag.append(values[pos])
                    if self.logging:
                        logger.info("change parameter %s with code %s to %s", \
                                   param.lhablock, param.lhacode, values[pos])
            # retrun the current param_card up to next iteration
            yield param_card
        
    
    def store_entry(self, run_name, cross):
        """store the value of the cross-section"""
        self.cross.append({'bench' : self.itertag, 'run_name': run_name, 'cross':cross})
        

    def write_summary(self, path):
        """ """
        
        ff = open(path, 'w')
        ff.write("#%-19s %-20s %-20s\n" % ('run_name',' '.join(self.param_order), 'cross(pb)'))
        for info in self.cross:
            bench = [str(p) for p in info['bench']]
            cross = info['cross']
            name = info['run_name']
            ff.write("%-20s %-20s %-20s \n" % (name,' '.join(bench) ,cross))
            #ff.write("%s %s %s \n" % (name,' '.join(bench) ,cross))
            

class ParamCardRule(object):
    """ A class for storing the linked between the different parameter of
            the param_card.
        Able to write a file 'param_card_rule.dat' 
        Able to read a file 'param_card_rule.dat'
        Able to check the validity of a param_card.dat
    """
        
    
    def __init__(self, inputpath=None):
        """initialize an object """
        
        # constraint due to model restriction
        self.zero = []
        self.one = []    
        self.identical = []
        self.opposite = []

        # constraint due to the model
        self.rule = []
        
        if inputpath:
            self.load_rule(inputpath)
        
    def add_zero(self, lhablock, lhacode, comment=''):
        """add a zero rule"""
        self.zero.append( (lhablock, lhacode, comment) )
        
    def add_one(self, lhablock, lhacode, comment=''):
        """add a one rule"""
        self.one.append( (lhablock, lhacode, comment) )        

    def add_identical(self, lhablock, lhacode, lhacode2, comment=''):
        """add a rule for identical value"""
        self.identical.append( (lhablock, lhacode, lhacode2, comment) )
        
    def add_opposite(self, lhablock, lhacode, lhacode2, comment=''):
        """add a rule for identical value"""
        self.opposite.append( (lhablock, lhacode, lhacode2, comment) )

        
    def add_rule(self, lhablock, lhacode, rule, comment=''):
        """add a rule for constraint value"""
        self.rule.append( (lhablock, lhacode, rule) )
        
    def write_file(self, output=None):
        
        text = """<file>######################################################################
## VALIDITY RULE FOR THE PARAM_CARD   ####
######################################################################\n"""
 
        # ZERO
        text +='<zero>\n'
        for name, id, comment in self.zero:
            text+='     %s %s # %s\n' % (name, '    '.join([str(i) for i in id]), 
                                                                        comment)
        # ONE
        text +='</zero>\n<one>\n'
        for name, id, comment in self.one:
            text+='     %s %s # %s\n' % (name, '    '.join([str(i) for i in id]), 
                                                                        comment)
        # IDENTICAL
        text +='</one>\n<identical>\n'
        for name, id,id2, comment in self.identical:
            text+='     %s %s : %s # %s\n' % (name, '    '.join([str(i) for i in id]), 
                                      '    '.join([str(i) for i in id2]), comment)

        # OPPOSITE
        text +='</identical>\n<opposite>\n'
        for name, id,id2, comment in self.opposite:
            text+='     %s %s : %s # %s\n' % (name, '    '.join([str(i) for i in id]), 
                                      '    '.join([str(i) for i in id2]), comment)
        
        # CONSTRAINT
        text += '</opposite>\n<constraint>\n'
        for name, id, rule, comment in self.rule:
            text += '     %s %s : %s # %s\n' % (name, '    '.join([str(i) for i in id]), 
                                                                  rule, comment)
        text += '</constraint>\n</file>'
    
        if isinstance(output, str):
            output = open(output,'w')
        if hasattr(output, 'write'):
            output.write(text)
        return text
    
    def load_rule(self, inputpath):
        """ import a validity rule file """

        
        try:
            tree = ET.parse(inputpath)
        except IOError:
            if '\n' in inputpath:
                # this is convinient for the tests
                tree = ET.fromstring(inputpath)
            else:
                raise

        #Add zero element
        element = tree.find('zero')
        if element is not None:
            for line in element.text.split('\n'):
                line = line.split('#',1)[0] 
                if not line:
                    continue
                lhacode = line.split()
                blockname = lhacode.pop(0)
                lhacode = [int(code) for code in lhacode ]
                self.add_zero(blockname, lhacode, '')
        
        #Add one element
        element = tree.find('one')
        if element is not None:
            for line in element.text.split('\n'):
                line = line.split('#',1)[0] 
                if not line:
                    continue
                lhacode = line.split()
                blockname = lhacode.pop(0)
                lhacode = [int(code) for code in lhacode ]
                self.add_one(blockname, lhacode, '')

        #Add Identical element
        element = tree.find('identical')
        if element is not None:
            for line in element.text.split('\n'):
                line = line.split('#',1)[0] 
                if not line:
                    continue
                line, lhacode2 = line.split(':')
                lhacode = line.split()
                blockname = lhacode.pop(0)
                lhacode = [int(code) for code in lhacode ]
                lhacode2 = [int(code) for code in lhacode2.split() ]
                self.add_identical(blockname, lhacode, lhacode2, '')        

        #Add Opposite element
        element = tree.find('opposite')
        if element is not None:
            for line in element.text.split('\n'):
                line = line.split('#',1)[0] 
                if not line:
                    continue
                line, lhacode2 = line.split(':')
                lhacode = line.split()
                blockname = lhacode.pop(0)
                lhacode = [int(code) for code in lhacode ]
                lhacode2 = [int(code) for code in lhacode2.split() ]
                self.add_opposite(blockname, lhacode, lhacode2, '') 

        #Add Rule element
        element = tree.find('rule')
        if element is not None:
            for line in element.text.split('\n'):
                line = line.split('#',1)[0] 
                if not line:
                    continue
                line, rule = line.split(':')
                lhacode = line.split()
                blockname = lhacode.pop(0)
                self.add_rule(blockname, lhacode, rule, '')
    
    @staticmethod
    def read_param_card(path):
        """ read a param_card and return a dictionary with the associated value."""
        
        output = ParamCard(path)
        

        
        return output

    @staticmethod
    def write_param_card(path, data):
        """ read a param_card and return a dictionary with the associated value."""
        
        output = {}
        
        if isinstance(path, str):
            output = open(path, 'w')
        else:
            output = path # helpfull for the test
        
        data.write(path)
    
    
    def check_param_card(self, path, modify=False):
        """Check that the restriction card are applied"""
                
        card = self.read_param_card(path)
        
        # check zero 
        for block, id, comment in self.zero:
            try:
                value = float(card[block].get(id).value)
            except KeyError:
                if modify:
                    new_param = Parameter(block=block,lhacode=id, value=0, 
                                    comment='fixed by the model')
                    if block in card:
                        card[block].append(new_param)
                    else:
                        new_block = Block(block)
                        card.append(new_block)
                        new_block.append(new_param)
            else:
                if value != 0:
                    if not modify:
                        raise InvalidParamCard, 'parameter %s: %s is not at zero' % \
                                    (block, ' '.join([str(i) for i in id])) 
                    else:
                        param = card[block].get(id) 
                        param.value = 0.0
                        param.comment += ' fixed by the model'
                        
        # check one 
        for block, id, comment in self.one:
            try:
                value = card[block].get(id).value
            except KeyError:
                if modify:
                    new_param = Parameter(block=block,lhacode=id, value=1, 
                                    comment='fixed by the model')
                    if block in card:
                        card[block].append(new_param)
                    else:
                        new_block = Block(block)
                        card.append(new_block)
                        new_block.append(new_param)
            else:   
                if value != 1:
                    if not modify:
                        raise InvalidParamCard, 'parameter %s: %s is not at one but at %s' % \
                                    (block, ' '.join([str(i) for i in id]), value)         
                    else:
                        param = card[block].get(id) 
                        param.value = 1.0
                        param.comment += ' fixed by the model'

        
        # check identical
        for block, id1, id2, comment in self.identical:
            if block not in card:
                logger.warning('''Param card is not complete: Block %s is simply missing.
                We will use model default for all missing value! Please cross-check that
                this correspond to your expectation.''' % block)
                continue
            value2 = float(card[block].get(id2).value)
            try:
                param = card[block].get(id1)
            except KeyError:
                if modify:
                    new_param = Parameter(block=block,lhacode=id1, value=value2, 
                                    comment='must be identical to %s' %id2)
                    card[block].append(new_param)
            else:
                value1 = float(param.value)

                if value1 != value2:
                    if not modify:
                        raise InvalidParamCard, 'parameter %s: %s is not to identical to parameter  %s' % \
                                    (block, ' '.join([str(i) for i in id1]),
                                            ' '.join([str(i) for i in id2]))         
                    else:
                        param = card[block].get(id1) 
                        param.value = value2
                        param.comment += ' must be identical to %s' % id2

        # check opposite
        for block, id1, id2, comment in self.opposite:
            value2 = float(card[block].get(id2).value)
            try:
                param = card[block].get(id1)
            except KeyError:
                if modify:
                    new_param = Parameter(block=block,lhacode=id1, value=-value2, 
                                    comment='must be opposite to to %s' %id2)
                    card[block].append(new_param)
            else:
                value1 = float(param.value)

                if value1 != -value2:
                    if not modify:
                        raise InvalidParamCard, 'parameter %s: %s is not to opposite to parameter  %s' % \
                                    (block, ' '.join([str(i) for i in id1]),
                                            ' '.join([str(i) for i in id2]))         
                    else:
                        param = card[block].get(id1) 
                        param.value = -value2
                        param.comment += ' must be opposite to %s' % id2

        return card
                        

def convert_to_slha1(path, outputpath=None ):
    """ """
                                                      
    if not outputpath:
        outputpath = path
    card = ParamCard(path)
    if not 'usqmix' in card:
        #already slha1
        card.write(outputpath)
        return
        
    # Mass 
    #card.reorder_mass() # needed?
    card.copy_param('mass', [6], 'sminputs', [6])
    card.copy_param('mass', [15], 'sminputs', [7])
    card.copy_param('mass', [23], 'sminputs', [4])
    # Decay: Nothing to do. 
    
    # MODSEL
    card.add_param('modsel',[1], value=1)
    card['modsel'].get([1]).format = 'int'
    
    # find scale
    scale = card['hmix'].scale
    if not scale:
        scale = 1 # Need to be define (this is dummy value)
    
    # SMINPUTS
    if not card.has_param('sminputs', [2]):
        aem1 = card['sminputs'].get([1]).value
        mz = card['mass'].get([23]).value
        mw = card['mass'].get([24]).value
        gf = math.pi / math.sqrt(2) / aem1 * mz**2/ mw**2 /(mz**2-mw**2)
        card.add_param('sminputs', [2], gf, 'G_F [GeV^-2]')

    # USQMIX
    card.check_and_remove('usqmix', [1,1], 1.0)
    card.check_and_remove('usqmix', [2,2], 1.0)
    card.check_and_remove('usqmix', [4,4], 1.0)
    card.check_and_remove('usqmix', [5,5], 1.0)
    card.mod_param('usqmix', [3,3], 'stopmix', [1,1])
    card.mod_param('usqmix', [3,6], 'stopmix', [1,2])
    card.mod_param('usqmix', [6,3], 'stopmix', [2,1])
    card.mod_param('usqmix', [6,6], 'stopmix', [2,2])

    # DSQMIX
    card.check_and_remove('dsqmix', [1,1], 1.0)
    card.check_and_remove('dsqmix', [2,2], 1.0)
    card.check_and_remove('dsqmix', [4,4], 1.0)
    card.check_and_remove('dsqmix', [5,5], 1.0)
    card.mod_param('dsqmix', [3,3], 'sbotmix', [1,1])
    card.mod_param('dsqmix', [3,6], 'sbotmix', [1,2])
    card.mod_param('dsqmix', [6,3], 'sbotmix', [2,1])
    card.mod_param('dsqmix', [6,6], 'sbotmix', [2,2])     
    
    
    # SELMIX
    card.check_and_remove('selmix', [1,1], 1.0)
    card.check_and_remove('selmix', [2,2], 1.0)
    card.check_and_remove('selmix', [4,4], 1.0)
    card.check_and_remove('selmix', [5,5], 1.0)
    card.mod_param('selmix', [3,3], 'staumix', [1,1])
    card.mod_param('selmix', [3,6], 'staumix', [1,2])
    card.mod_param('selmix', [6,3], 'staumix', [2,1])
    card.mod_param('selmix', [6,6], 'staumix', [2,2])
    
    # FRALPHA
    card.mod_param('fralpha', [1], 'alpha', [' '])
    
    #HMIX
    if not card.has_param('hmix', [3]):
        aem1 = card['sminputs'].get([1]).value
        tanb = card['hmix'].get([2]).value
        mz = card['mass'].get([23]).value
        mw = card['mass'].get([24]).value
        sw = math.sqrt(mz**2 - mw**2)/mz
        ee = 2 * math.sqrt(1/aem1) * math.sqrt(math.pi)
        vu = 2 * mw *sw /ee * math.sin(math.atan(tanb))
        card.add_param('hmix', [3], vu, 'higgs vev(Q) MSSM DRb')
    card['hmix'].scale= scale
    
    # VCKM
    card.check_and_remove('vckm', [1,1], 1.0)
    card.check_and_remove('vckm', [2,2], 1.0)
    card.check_and_remove('vckm', [3,3], 1.0)
    
    #SNUMIX
    card.check_and_remove('snumix', [1,1], 1.0)
    card.check_and_remove('snumix', [2,2], 1.0)
    card.check_and_remove('snumix', [3,3], 1.0)

    #UPMNS
    card.check_and_remove('upmns', [1,1], 1.0)
    card.check_and_remove('upmns', [2,2], 1.0)
    card.check_and_remove('upmns', [3,3], 1.0)

    # Te
    ye = card['ye'].get([3, 3]).value
    te = card['te'].get([3, 3]).value
    card.mod_param('te', [3,3], 'ae', [3,3], value= te/ye, comment='A_tau(Q) DRbar')
    card.add_param('ae', [1,1], 0, 'A_e(Q) DRbar')
    card.add_param('ae', [2,2], 0, 'A_mu(Q) DRbar')
    card['ae'].scale = scale
    card['ye'].scale = scale
            
    # Tu
    yu = card['yu'].get([3, 3]).value
    tu = card['tu'].get([3, 3]).value
    card.mod_param('tu', [3,3], 'au', [3,3], value= tu/yu, comment='A_t(Q) DRbar')
    card.add_param('au', [1,1], 0, 'A_u(Q) DRbar')
    card.add_param('au', [2,2], 0, 'A_c(Q) DRbar')
    card['au'].scale = scale    
    card['yu'].scale = scale
        
    # Td
    yd = card['yd'].get([3, 3]).value
    td = card['td'].get([3, 3]).value
    if td:
        card.mod_param('td', [3,3], 'ad', [3,3], value= td/yd, comment='A_b(Q) DRbar')
    else:
        card.mod_param('td', [3,3], 'ad', [3,3], value= 0., comment='A_b(Q) DRbar')
    card.add_param('ad', [1,1], 0, 'A_d(Q) DRbar')
    card.add_param('ad', [2,2], 0, 'A_s(Q) DRbar')
    card['ad'].scale = scale
    card['yd'].scale = scale    
        
    # MSL2 
    value = card['msl2'].get([1, 1]).value
    card.mod_param('msl2', [1,1], 'msoft', [31], math.sqrt(value))
    value = card['msl2'].get([2, 2]).value
    card.mod_param('msl2', [2,2], 'msoft', [32], math.sqrt(value))
    value = card['msl2'].get([3, 3]).value
    card.mod_param('msl2', [3,3], 'msoft', [33], math.sqrt(value))
    card['msoft'].scale = scale

    # MSE2
    value = card['mse2'].get([1, 1]).value
    card.mod_param('mse2', [1,1], 'msoft', [34], math.sqrt(value))
    value = card['mse2'].get([2, 2]).value
    card.mod_param('mse2', [2,2], 'msoft', [35], math.sqrt(value))
    value = card['mse2'].get([3, 3]).value
    card.mod_param('mse2', [3,3], 'msoft', [36], math.sqrt(value))
    
    # MSQ2                
    value = card['msq2'].get([1, 1]).value
    card.mod_param('msq2', [1,1], 'msoft', [41], math.sqrt(value))
    value = card['msq2'].get([2, 2]).value
    card.mod_param('msq2', [2,2], 'msoft', [42], math.sqrt(value))
    value = card['msq2'].get([3, 3]).value
    card.mod_param('msq2', [3,3], 'msoft', [43], math.sqrt(value))    
    
    # MSU2                
    value = card['msu2'].get([1, 1]).value
    card.mod_param('msu2', [1,1], 'msoft', [44], math.sqrt(value))
    value = card['msu2'].get([2, 2]).value
    card.mod_param('msu2', [2,2], 'msoft', [45], math.sqrt(value))
    value = card['msu2'].get([3, 3]).value
    card.mod_param('msu2', [3,3], 'msoft', [46], math.sqrt(value))   
    
    # MSD2                
    value = card['msd2'].get([1, 1]).value
    card.mod_param('msd2', [1,1], 'msoft', [47], math.sqrt(value))
    value = card['msd2'].get([2, 2]).value
    card.mod_param('msd2', [2,2], 'msoft', [48], math.sqrt(value))
    value = card['msd2'].get([3, 3]).value
    card.mod_param('msd2', [3,3], 'msoft', [49], math.sqrt(value))   

    
    
    #################
    # WRITE OUTPUT
    #################
    card.write(outputpath)
        
        

def convert_to_mg5card(path, outputpath=None, writting=True):
    """
    """
                                                      
    if not outputpath:
        outputpath = path
    card = ParamCard(path)
    if 'usqmix' in card:
        #already mg5(slha2) format
        if outputpath != path and writting:
            card.write(outputpath)
        return card

        
    # SMINPUTS
    card.remove_param('sminputs', [2])
    card.remove_param('sminputs', [4])
    card.remove_param('sminputs', [6])
    card.remove_param('sminputs', [7])
    # Decay: Nothing to do. 
    
    # MODSEL
    card.remove_param('modsel',[1])
    
    
    # USQMIX
    card.add_param('usqmix', [1,1], 1.0)
    card.add_param('usqmix', [2,2], 1.0)
    card.add_param('usqmix', [4,4], 1.0)
    card.add_param('usqmix', [5,5], 1.0)
    card.mod_param('stopmix', [1,1], 'usqmix', [3,3])
    card.mod_param('stopmix', [1,2], 'usqmix', [3,6])
    card.mod_param('stopmix', [2,1], 'usqmix', [6,3])
    card.mod_param('stopmix', [2,2], 'usqmix', [6,6])

    # DSQMIX
    card.add_param('dsqmix', [1,1], 1.0)
    card.add_param('dsqmix', [2,2], 1.0)
    card.add_param('dsqmix', [4,4], 1.0)
    card.add_param('dsqmix', [5,5], 1.0)
    card.mod_param('sbotmix', [1,1], 'dsqmix', [3,3])
    card.mod_param('sbotmix', [1,2], 'dsqmix', [3,6])
    card.mod_param('sbotmix', [2,1], 'dsqmix', [6,3])
    card.mod_param('sbotmix', [2,2], 'dsqmix', [6,6])     
    
    
    # SELMIX
    card.add_param('selmix', [1,1], 1.0)
    card.add_param('selmix', [2,2], 1.0)
    card.add_param('selmix', [4,4], 1.0)
    card.add_param('selmix', [5,5], 1.0)
    card.mod_param('staumix', [1,1], 'selmix', [3,3])
    card.mod_param('staumix', [1,2], 'selmix', [3,6])
    card.mod_param('staumix', [2,1], 'selmix', [6,3])
    card.mod_param('staumix', [2,2], 'selmix', [6,6])
    
    # FRALPHA
    card.mod_param('alpha', [], 'fralpha', [1])
    
    #HMIX
    card.remove_param('hmix', [3])
    
    # VCKM
    card.add_param('vckm', [1,1], 1.0)
    card.add_param('vckm', [2,2], 1.0)
    card.add_param('vckm', [3,3], 1.0)
    
    #SNUMIX
    card.add_param('snumix', [1,1], 1.0)
    card.add_param('snumix', [2,2], 1.0)
    card.add_param('snumix', [3,3], 1.0)

    #UPMNS
    card.add_param('upmns', [1,1], 1.0)
    card.add_param('upmns', [2,2], 1.0)
    card.add_param('upmns', [3,3], 1.0)

    # Te
    ye = card['ye'].get([1, 1], default=0).value
    ae = card['ae'].get([1, 1], default=0).value
    card.mod_param('ae', [1,1], 'te', [1,1], value= ae * ye, comment='T_e(Q) DRbar')
    if ae * ye:
        raise InvalidParamCard, '''This card is not suitable to be converted to MSSM UFO model
Parameter ae [1, 1] times ye [1,1] should be 0'''
    card.remove_param('ae', [1,1])
    #2
    ye = card['ye'].get([2, 2], default=0).value
    
    ae = card['ae'].get([2, 2], default=0).value
    card.mod_param('ae', [2,2], 'te', [2,2], value= ae * ye, comment='T_mu(Q) DRbar')
    if ae * ye:
        raise InvalidParamCard, '''This card is not suitable to be converted to MSSM UFO model
Parameter ae [2, 2] times ye [2,2] should be 0'''
    card.remove_param('ae', [2,2])
    #3
    ye = card['ye'].get([3, 3], default=0).value
    ae = card['ae'].get([3, 3], default=0).value
    card.mod_param('ae', [3,3], 'te', [3,3], value= ae * ye, comment='T_tau(Q) DRbar')
    
    # Tu
    yu = card['yu'].get([1, 1], default=0).value
    au = card['au'].get([1, 1], default=0).value
    card.mod_param('au', [1,1], 'tu', [1,1], value= au * yu, comment='T_u(Q) DRbar')
    if au * yu:
        raise InvalidParamCard, '''This card is not suitable to be converted to MSSM UFO model
Parameter au [1, 1] times yu [1,1] should be 0'''
    card.remove_param('au', [1,1])
    #2
    ye = card['yu'].get([2, 2], default=0).value
    
    ae = card['au'].get([2, 2], default=0).value
    card.mod_param('au', [2,2], 'tu', [2,2], value= au * yu, comment='T_c(Q) DRbar')
    if au * yu:
        raise InvalidParamCard, '''This card is not suitable to be converted to MSSM UFO model
Parameter au [2, 2] times yu [2,2] should be 0'''
    card.remove_param('au', [2,2])
    #3
    yu = card['yu'].get([3, 3]).value
    au = card['au'].get([3, 3]).value
    card.mod_param('au', [3,3], 'tu', [3,3], value= au * yu, comment='T_t(Q) DRbar')
    
    # Td
    yd = card['yd'].get([1, 1], default=0).value
    ad = card['ad'].get([1, 1], default=0).value
    card.mod_param('ad', [1,1], 'td', [1,1], value= ad * yd, comment='T_d(Q) DRbar')
    if ad * yd:
        raise InvalidParamCard, '''This card is not suitable to be converted to MSSM UFO model
Parameter ad [1, 1] times yd [1,1] should be 0'''
    card.remove_param('ad', [1,1])
    #2
    ye = card['yd'].get([2, 2], default=0).value
    
    ae = card['ad'].get([2, 2], default=0).value
    card.mod_param('ad', [2,2], 'td', [2,2], value= ad * yd, comment='T_s(Q) DRbar')
    if ad * yd:
        raise InvalidParamCard, '''This card is not suitable to be converted to MSSM UFO model
Parameter ad [2, 2] times yd [2,2] should be 0'''
    card.remove_param('ad', [2,2])
    #3
    yd = card['yd'].get([3, 3]).value
    ad = card['ad'].get([3, 3]).value
    card.mod_param('ad', [3,3], 'td', [3,3], value= ad * yd, comment='T_b(Q) DRbar')

    
    # MSL2 
    value = card['msoft'].get([31]).value
    card.mod_param('msoft', [31], 'msl2', [1,1], value**2)
    value = card['msoft'].get([32]).value
    card.mod_param('msoft', [32], 'msl2', [2,2], value**2)
    value = card['msoft'].get([33]).value
    card.mod_param('msoft', [33], 'msl2', [3,3], value**2)
    
    # MSE2
    value = card['msoft'].get([34]).value
    card.mod_param('msoft', [34], 'mse2', [1,1], value**2)
    value = card['msoft'].get([35]).value
    card.mod_param('msoft', [35], 'mse2', [2,2], value**2)
    value = card['msoft'].get([36]).value
    card.mod_param('msoft', [36], 'mse2', [3,3], value**2)
    
    # MSQ2                
    value = card['msoft'].get([41]).value
    card.mod_param('msoft', [41], 'msq2', [1,1], value**2)
    value = card['msoft'].get([42]).value
    card.mod_param('msoft', [42], 'msq2', [2,2], value**2)
    value = card['msoft'].get([43]).value
    card.mod_param('msoft', [43], 'msq2', [3,3], value**2)    
    
    # MSU2                
    value = card['msoft'].get([44]).value
    card.mod_param('msoft', [44], 'msu2', [1,1], value**2)
    value = card['msoft'].get([45]).value
    card.mod_param('msoft', [45], 'msu2', [2,2], value**2)
    value = card['msoft'].get([46]).value
    card.mod_param('msoft', [46], 'msu2', [3,3], value**2)   
    
    # MSD2
    value = card['msoft'].get([47]).value
    card.mod_param('msoft', [47], 'msd2', [1,1], value**2)
    value = card['msoft'].get([48]).value
    card.mod_param('msoft', [48], 'msd2', [2,2], value**2)
    value = card['msoft'].get([49]).value
    card.mod_param('msoft', [49], 'msd2', [3,3], value**2)   
    
    #################
    # WRITE OUTPUT
    #################
    if writting:
        card.write(outputpath)
    return card
    
                                                      
def make_valid_param_card(path, restrictpath, outputpath=None):
    """ modify the current param_card such that it agrees with the restriction"""
    
    if not outputpath:
        outputpath = path
        
    cardrule = ParamCardRule()
    cardrule.load_rule(restrictpath)
    try :
        cardrule.check_param_card(path, modify=False)
    except InvalidParamCard:
        new_data = cardrule.check_param_card(path, modify=True)
        cardrule.write_param_card(outputpath, new_data)
    else:
        if path != outputpath:
            shutil.copy(path, outputpath)
    return cardrule

def check_valid_param_card(path, restrictpath=None):
    """ check if the current param_card agrees with the restriction"""
    
    if restrictpath is None:
        restrictpath = os.path.dirname(path)
        restrictpath = os.path.join(restrictpath, os.pardir, os.pardir, 'Source', 
                                                 'MODEL', 'param_card_rule.dat')
        if not os.path.exists(restrictpath):
            restrictpath = os.path.dirname(path)
            restrictpath = os.path.join(restrictpath, os.pardir, 'Source', 
                                                 'MODEL', 'param_card_rule.dat')
            if not os.path.exists(restrictpath):
                return True
    
    cardrule = ParamCardRule()
    cardrule.load_rule(restrictpath)
    cardrule.check_param_card(path, modify=False)

if '__main__' == __name__:


    #make_valid_param_card('./Cards/param_card.dat', './Source/MODEL/param_card_rule.dat', 
    #                       outputpath='tmp1.dat')
    import sys    
    args = sys.argv
    sys.path.append(os.path.dirname(__file__))
    convert_to_slha1(args[1] , args[2])

                         
