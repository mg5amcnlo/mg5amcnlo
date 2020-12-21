#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from six.moves import input
def giveInfo(class_):
        if type(class_)!=str:
                class_=class_.__class__.__name__
        for info in dir(eval(class_)):
                
                print(class_+'.'+info+' : ',eval(class_+'.'+info+'.__doc__'))



if __name__=='__main__':
    class_=input('enter the name of the class')
    try:
        import class_
    except:
        pass
    giveInfo(class_)
