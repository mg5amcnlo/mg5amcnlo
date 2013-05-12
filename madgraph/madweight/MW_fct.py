##########################################################################
##                                                                      ##
##                  TOOL FOR MW (fonction and class)                    ##
##                                                                      ##
##########################################################################
##
## Content:
##
##     - Class Multi_list
##     - fct   permutate
##     - fct   put_in_fortran_format
##     - fct   put_line_in_fortran_format
##     - Class BackRead
##
##########################################################################
import sys
import os
import string
import collections
import itertools
import copy

class Multi_list(list):
    """ a list of list """

    def give_combinaison(self):
        """ return all the combinatory possibility in a list """

        if not self: #test void input
            return [[]]
        
        len_vector=[]
        status_vector=[]
        for i in range(0,len(self)):
            status_vector.append(0)
            if type(self[i])==list:
                len_vector.append(len(self[i]))
            else:
                len_vector.append(1)

        solution=[]
        while 1:
            #print 'add solution'
            solution.append(self.extract_vector(status_vector))
            curent_pos=0
            while 1:
                #print 'enter in while'
                if status_vector[curent_pos]==len_vector[curent_pos]-1:
                    #print 'pass to step',curent_pos+1
                    curent_pos+=1
                    if(curent_pos==len(status_vector)):
                        #print 'break'
                        break
                else:
                    #print 'update'
                    status_vector[curent_pos]+=1
                    for i in range(0,curent_pos):
                        status_vector[i]=0
                    break
            if (curent_pos==len(status_vector)):
                break
                
        return solution
        
    def extract_vector(self,pos_list):
        """ return the list of element corresponding at position in pos_list """

        if len(pos_list)!=len(self):
            return

        solution=[]
        #print 'a',pos_list
        for i in range(0,len(pos_list)):
            solution.append(self[i][pos_list[i]])

        return solution
        

    def give_list_possiblity(self,list,opt=''):
        """ return all permutation of list where each element belongs to corresponding list of the multilist

            example: multilist=[ [a,b,c],[a,b,d],[c,d] ]
                     list=[a,b,c]
                        ->return=[ [a,b,c],[b,a,c] ]
                        
                     multilist=[ [a,b,c],[a,b,d],[a,c,d] ]   
                     list=[a,b,d]
                        ->return=[ [a,b,d],[b,a,d],[b,d,a] ]

            option: if the list is made of object, but multilist are data of this object,
                    you can use opt to check if list[i].opt in self[i]
        """

        if len(list)!=len(self):
            return

        perm_pos=permutate(list)
        sol=[]
        for perm in perm_pos:
            find=1
            for i in range(0,len(list)):
                if opt=='':
                    if perm[i] not in self[i]:
                        find=0
                        break
                else:
                    value=eval('perm[i].'+opt)
                    if value not in self[i]:
                        find=0
                        break                    
            if find:
                sol.append(perm)
        return sol
                
        
        


        





#
#   other function
#



def permutate(seq):
    """permutate a sequence and return a list of the permutations"""

    if not seq:
        return [seq] # is an empty sequence
    else:
        temp = []
        for k in range(len(seq)):
            part = seq[:k] + seq[k+1:]
            #print k, part # test
            for m in permutate(part):
                temp.append(seq[k:k+1] + m)
                #print m, seq[k:k+1], temp # test
        return temp

def put_in_fortran_format(text):
    out=''
    if text.count('\n')>1:
        part=text.split('\n')
        for line in part:
    #        print "line",[line]
            if len(line)>0:
                out+=put_line_in_fortran_format(line)+"\n"
            else:
                out+='\n'
    else:
        out=put_line_in_fortran_format(text)
        
    return out
    
def put_line_in_fortran_format(text):
    "take formula and split in 50-90 columns"


    #take car of special begin with fortran
    if text[0]=="c" or text[0]=="C":
        return text
    if(text[:6].count('&')):
       return text
    if(text[:6]!="      "):
        try:
            a=int(text[:6]) #test for tag
        except:                  
            text="       "+text
 
    #delete final space
    while (text[-1]==" "):
        text=text[:-1]
        if len(text)==0:
            break
#    print "pass here: len " ,len(text)
    comment_mode = 0
    out=""
    while(len(text)>90):
        tag=len(text)
        list_split=["\n",",","+","-",'('," "]
        i=0
        while 1:
            if text[50:90].count(list_split[i]):
                out+=text[:50]
                text=text[50:]
                tag=text.index(list_split[i])+1
                break
            i+=1
        if '!' in text[:tag]+out[-50:]:
            comment_mode = 1
        out+=text[:tag]+"\n"
        if(tag<len(text)):
            if comment_mode==0:
                text="     &"+text[tag:]
            else:
                text="     &!"+text[tag:]
        
    out+=text

    return out


## class list(list):
##     "new list"

##     def __str__(self,opt=''):
##         if opt=='':
##             return list.__str__(self)
##         else:
##             text='['
##             for content in list:
##                 text+=str(getattr(A,opt))+','
##             text=text[:-1]+']'

class BackRead(file):
  """read a file returning the lines in reverse order for each call of readline()
This actually just reads blocks (4096 bytes by default) of data from the end of
the file and returns last line in an internal buffer.  I believe all the corner
cases are handled, but never can be sure..."""


  def readline(self):
    while len(self.data) == 1 and ((self.blkcount * self.blksize) < self.size):
      self.blkcount = self.blkcount + 1
      line = self.data[0]
      try:
        self.seek(-self.blksize * self.blkcount, 2) # read from end of file
        self.data = string.split(self.read(self.blksize) + line, '\n')
      except IOError:  # can't seek before the beginning of the file
        self.seek(0)
        self.data = string.split(self.read(self.size - (self.blksize * (self.blkcount-1))) + line, '\n')

    if len(self.data) == 0:
      return ""

    # self.data.pop()
    # make it compatible with python <= 1.5.1
    line = self.data[-1]
    self.data = self.data[:-1]
    return line + '\n'

  def __init__(self, filepos, blksize=4096):
    """initialize the internal structures"""

    # get the file size
    self.size = os.stat(filepos)[6]
    # how big of a block to read from the file...
    self.blksize = blksize
    # how many blocks we've read
    self.blkcount = 1
    file.__init__(self, filepos, 'rb')
    # if the file is smaller than the blocksize, read a block,
    # otherwise, read the whole thing...
    if self.size > self.blksize:
      self.seek(-self.blksize * self.blkcount, 2) # read from end of file
    self.data = string.split(self.read(self.blksize), '\n')
    # strip the last item if it's empty...  a byproduct of the last line having
    # a newline at the end of it
    if not self.data[-1]:
      # self.data.pop()
      self.data = self.data[:-1]

#  def close(self):
#      """ close correctly file """
#      try:
#          self.close()
#      except:
#          pass

def get_perms_from_id(pid_list, bjet_is_jet):
    """ """
    
    assert isinstance(pid_list, list)

    assert isinstance(bjet_is_jet, bool) or bjet_is_jet in [0,1]
    
    list_id = []
    for i,pid in enumerate(pid_list):
        if abs(pid) in [1,2,3,4]:
            list_id.append('j')
        elif abs(pid) == 5:
            if bjet_is_jet:
                list_id.append('j')
            else:
                list_id.append('b')
        elif abs(pid) in [12,14,16,18,1000022,1000023,1000025,1000035]:
            list_id.append('%s_%s' % (i, pid))
        else:
            list_id.append(pid)
    
    #get the id permutations
    return  get_all_permutations(list_id)

def get_all_permutations(cat_list):
    """ """
    
    # create the category names and the id to permute for each category
    nb_cat = collections.defaultdict(list)
    for i,cat in enumerate(cat_list):
        nb_cat[cat].append(i+1) #+1 in order to be in Fortan convention
    cat_names = nb_cat.keys()
    # build an iterator for each category
    iterator = dict([(cat, itertools.permutations(value)) 
                                               for cat,value in nb_cat.items()])
        
    permutations = [] # all possibility
    current = 0       # position of the last modify category
    #initialize all iterator to have starting point value
    current_value = dict([(cat, list(it.next())) for cat,it in iterator.items()])

    while current < len(iterator):
        #store the current value
        perm = []
        curr = copy.deepcopy(current_value)        
        for cat in cat_list:
            perm.append(curr[cat].pop(0))
        permutations.append(perm)
        #update the iterator to have the next value
        while current < len(iterator):
            cat = cat_names[current]
            it  = iterator[cat]
            try:
                new_val = it.next()
            except StopIteration:
                iterator[cat] = itertools.permutations(nb_cat[cat])
                current_value[cat] = list(iterator[cat].next())
                current +=1
            else:
                current_value[cat] = list(new_val)
                current = 0
                break
                
    return permutations