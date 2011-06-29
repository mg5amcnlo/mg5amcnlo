








import xml.etree.ElementTree as ET

class InvalidParamCard(Exception):
    """ a class for invalid param_card """
    pass

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
        
        # CONSTRAINT
        text += '</identical>\n<constraint>\n'
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

        tree = ET.parse(inputpath)

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
        
        output = {}
        
        if isinstance(path, str):
            input = open(path)
        else:
            input = path # helpfull for the test
        
        block = ""
        decay_particle = ""
        # Go through lines in param_card
        for line in input:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            line = line.lower()
            if line.startswith('block'):
                #change the current block name
                block = line.split()[1]
                output[block] = {}
                continue
            elif line.startswith('decay'):
                line = line[6:] #remove decay -> pass to usual syntax
                block = 'decay'
                if 'decay' not in output:
                    output['decay'] = {}
                decay_particle = line.split()[0]
            elif block == 'decay':
                #means that this is a decay table. Change block name to say to which part
                block = 'decay_%s' % decay_particle
                if block not in output:
                    output[block] = {}
            
            # This is standard line ($1 $2 $3 ... # comment)
            # Treat possible comment
            if '#' in line:
                line, comment = line.split('#',1)
            data = line.split()
            comment = comment.strip()
            output[block][str([int(i) for i in data[:-1]])] = (float(data[-1]), comment)
        
        return output

    @staticmethod
    def write_param_card(path, data):
        """ read a param_card and return a dictionary with the associated value."""
        
        output = {}
        
        if isinstance(path, str):
            output = open(path, 'w')
        else:
            output = path # helpfull for the test
        
        ParamCardWriter(data, path)
    
    
    def check_param_card(self, path, modify=False):
        """Check that the restriction card are applied"""
                
        card = self.read_param_card(path)
        
        # check zero 
        for block, id, comment in self.zero:
            try:
                value = card[block][str(id)][0]
            except KeyError:
                if modify:
                    if block in card:
                        card[block][str(id)] = (0.0,'fixed by the model')
                    else:
                        card[block] = {str(id):(0.0,'fixed by the model')}
            else:
                if value != 0:
                    if not modify:
                        raise InvalidParamCard, 'parameter %s: %s is not at zero' % \
                                    (block, ' '.join([str(i) for i in id])) 
                    else:
                        card[block][str(id)] = (0.0, 'fixed by the model')
                        
        # check one 
        for block, id, comment in self.one:
            try:
                value = card[block][str(id)][0]
            except KeyError:
                if modify:
                    if block in card:
                        card[block][str(id)] = (1.0,'fixed by the model')
                    else:
                        card[block] = {str(id):(1.0,'fixed by the model')}
            else:   
                if value != 0:
                    if not modify:
                        raise InvalidParamCard, 'parameter %s: %s is not at one' % \
                                    (block, ' '.join([str(i) for i in id]))         
                    else:
                        card[block][str(id)] = (1.0,'fixed by the model')
        
        # check identical
        for block, id1, id2, comment in self.identical:
            value2 = card[block][str(id2)][0]
            try:
                value1, comment = card[block][str(id1)]
            except KeyError:
                if modify:
                    card[block][str(id1)] = (value1, comment+' identical to %s' % id2 )
            else:
                if value1 != value2:
                    if not modify:
                        raise InvalidParamCard, 'parameter %s: %s is not to identical to parameter  %s' % \
                                    (block, ' '.join([str(i) for i in id1]),
                                            ' '.join([str(i) for i in id2]))         
                    else:
                        card[block][str(id1)] = (value1, comment+' identical to %s' % id2 )
        return card
                        
                        
class ParamCardWriter(object):
    """ A writer taken the input from a simple dictionary + this is also able 
    to write the decay table"""
    
    header = \
    """######################################################################\n""" + \
    """## PARAM_CARD AUTOMATICALY GENERATED BY MG5 FOR PYTHIA            ####\n""" + \
    """######################################################################\n"""   
    
    
    def __init__(self, data, filepath=None):
        """ model is a valid MG5 model, filepath is the path were to write the
        param_card.dat """

        self.data = data
    
        if filepath:
            self.define_output_file(filepath)
            self.write_card()    

    
    def write_card(self, path=None):
        """schedular for writing a card"""
  
        if path:
            self.define_output_file(path)
  
        # order the block in a smart way
        blocks = self.order_block(self.data.keys())
        
        for block_name in blocks:
            self.write_block(block_name)
            keys = self.order_keys(self.data[block_name].keys())
            for key in self.data[block_name]:
                self.write_param(block_name, key, *self.data[block_name][key])
               
    def define_output_file(self, path, mode='w'):
        """ initialize the file"""
        
        if isinstance(path, str):
            self.fsock = open(path, mode)
        else:
            self.fsock = path # prebuild file/IOstring
        
        self.fsock.write(self.header)
            

    
    def order_block(self, keys):
        
        return [k for k in keys if not k.startswith('decay_')] 


    def order_keys(self, keys):
        
        return keys 

    def write_block(self, name):
        """ write a comment for a block"""
        
        self.fsock.writelines(
        """\n###################################""" + \
        """\n## INFORMATION FOR %s""" % name.upper() +\
        """\n###################################\n"""
         )
        if name.upper()!='DECAY':
            self.fsock.write("""Block %s \n""" % name.lower())
             
    def write_param(self, block_name, lhaid, value, info):
        """ Write a parameter line"""
        
        lhacode = lhaid[1:-1].replace(',',' ') #[1:-1] removes the [] char
        if block_name != 'decay':
            text = """  %s %e # %s \n""" % (lhacode, value.real, info) 
        else:
            text = '''DECAY %s %e # %s \n''' % (lhacode, value.real, info)
        
        self.fsock.write(text)                          
                        
                          
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
    return new_data