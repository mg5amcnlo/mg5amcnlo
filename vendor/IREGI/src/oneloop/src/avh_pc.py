import string,re

def prnt(message):
    print message #python2
#    print(message) #python3

def subs(old,new,lines):
    nn = 0
    for line in lines:
        lines[nn] = re.sub(old,new,line)
        nn = nn+1

def clean(lines):
    nn = len(lines)-1
    while (nn >= 0):
        if re.match(r'^![!#{}\[]',lines[nn]):
            del lines[nn]
        elif re.search(r'![\|\]\?]',lines[nn]):
            lines[nn] = re.sub(r'!.*','',lines[nn])
        nn = nn-1  

def case(option,value,lines):
    """
    Switch on lines between lines starting with
    !{option=value  and  !}option=value,
    and switch off lines between lines starting with
    !{option=othervalue  and  !}option=othervalue.
    Switching on-off happens by un-out commenting using !#.
    Also switch on-off lines containing
    !|option=value  and  !|option=othervalue.
    """
    uncom = False
    outcom = False
    nn = 0
    for line in lines:
        if re.match( r'^!{'+option+r'=', line ):
            uncom = re.match( r'^!{'+option+r'='+value, line )
            outcom = not uncom
            lines[nn] = line
        elif re.match( r'^!}'+option+r'=', line ):
            uncom = False
            outcom = False
            lines[nn] = line
        elif re.search( r'!\|'+option+r'=', line ):
            if re.search( r'!\|'+option+r'='+value, line ):
                lines[nn] = re.sub(r'^(!#)*', '', line)
            else:
                lines[nn] = re.sub(r'^(!#)*', '!#', line)
        else:
            if uncom:
                lines[nn] = re.sub(r'^(!#)*', '', line)
            elif outcom:
                lines[nn] = re.sub(r'^(!#)*', '!#', line)
            else:
                lines[nn] = line
        nn = nn+1

def dich(option,value,lines):
    """
    If value='yes', replace lines starting with  ![option  with the same lines
    with  ![option  removed, and  !]option  appendend.
    If value='no', replace lines containing  !]option  with the same lines
    with  !]option  removed, and  ![option  prependend.
    """
    if value=='yes':
        nn = 0
        for line in lines:
            if re.match( r'^!\['+option+'', line ):
                lines[nn] = re.sub(r'!\['+option,'',
                            re.sub(r'\n','!]'+option+'\n',line))
            else:
                lines[nn] = line
            nn = nn+1
    elif value=='no':
        nn = 0
        for line in lines:
            if re.search( r'!\]'+option, line ):
                lines[nn] = re.sub(r'!\]'+option,'',re.sub(r'^','!['+option,line))
            else:
                lines[nn] = line
            nn = nn+1

def incl(srcdir,lines):
    """
    Find include statements in lines, and explicitly include the file, edited
    following the strings of the form  !?option=value  at the end of the line
    with the include statement.
    """
    nn = 0
    for line in lines:
        if re.match( r'^ *include', line ):
            words = string.split(line.rstrip('\n'))
            iofile = open(re.sub(r'[\'"]','',srcdir+words[1]))
            inclines = iofile.readlines()
            for word in words:
                if re.match( r'!\?', word ):
                    option = re.sub(r'!\?','',re.sub(r'=.*','',word))
                    value = re.sub(r'.*=','',word)
                    case(option,value,inclines)
            lines[nn] = '!!'+lines[nn]
            jj = 1
            for ljne in inclines:
               lines.insert(nn+jj,ljne)
               jj = jj+1
        nn = nn+1

