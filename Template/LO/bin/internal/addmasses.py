#!/usr/bin/python
#
# doubleswitch.py
# Author: Stephen Mrenna
# Add masses to charged leptons, fix kinematics
# Insert W/Z when missing

import math,sys,re
from xml.dom import minidom

#tolerance for energy momentum conservation
toler = 1e-4
#lepton masses from PDGLive
massDict = {}
massDict[11]=0.000510998910
massDict[-11]=massDict[11]
massDict[13]=105.6583668e-3
massDict[-13]=massDict[13]
massDict[15]=1.77684
massDict[-15]=massDict[15]

#print out messages if true
verbose=0

#default is NOT to insert missing mothers
motherFlag=1

#useful class to describe 4 momentum
class Momentum:
    def __init__(self,px,py,pz,E,m):
        self.px=px
        self.py=py
        self.pz=pz
        self.E=E
        self.m=m
    def __add__(self,other):
        t=Momentum(self.px+other.px,self.py+other.py,self.pz+other.pz,self.E+other.E,0)
        t.m=t.calcMass()
        return t
    def __sub__(self,other):
        t=Momentum(self.px-other.px,self.py-other.py,self.pz-other.pz,self.E-other.E,0)
        t.m=t.calcMass()
        return t
    def calcMass(self):
        tempMass2=self.E**2-self.px**2-self.py**2-self.pz**2
        if tempMass2 > 0:
            t=math.sqrt(tempMass2)
            if t>toler:
                return t
            else:
                return 0
        else:
            return 0
    def boost(self,ref,rdir):
        pmag=ref.E
        DBX=ref.px*rdir/pmag
        DBY=ref.py*rdir/pmag
        DBZ=ref.pz*rdir/pmag
        DB=math.sqrt(DBX**2+DBY**2+DBZ**2)
        DGA=1.0/math.sqrt(1.0-DB**2)        
        DBP=DBX*self.px+DBY*self.py+DBZ*self.pz
        DGABP=DGA*(DGA*DBP/(1.0+DGA)+self.E)
        self.px = self.px+DGABP*DBX
        self.py = self.py+DGABP*DBY
        self.pz = self.pz+DGABP*DBZ
        self.E  = DGA*(self.E+DBP)
    def reScale(self,pi,po):
        self.px = self.px/pi*po
        self.py = self.py/pi*po
        self.pz = self.pz/pi*po
    def printMe(self):
        li = [self.px,self.py, self.pz, self.E, self.m]
        print "| %18.10E %18.10E %18.10E %18.10E %18.10E |" % tuple(li)    

#useful class to describe a particle
class Particle:
    def __init__(self,i,l):
        self.no = i
        self.id = l[0]
        self.status = l[1]
        self.mo1 = l[2]
        self.mo2 = l[3]
        self.co1 = l[4]
        self.co2 = l[5]
        self.mom = Momentum(l[6],l[7],l[8],l[9],l[10])
        self.life = l[11]
        self.polar = l[12]
    def printMe(self):
        li = [self.no, self.id, self.status,self.mo1, self.mo2, self.co1, self.co2, self.mom.px,self.mom.py, self.mom.pz, self.mom.E, self.mom.m, self.life, self.polar]
        print "%2i | %9i | %4i | %4i %4i | %4i %4i | %18.10E %18.10E %18.10E %18.10E %18.10E | %1.0f. %2.0f" % tuple(li)
    def writeMe(self):
        li = [self.id, self.status,self.mo1, self.mo2, self.co1, self.co2, self.mom.px,self.mom.py, self.mom.pz, self.mom.E, self.mom.m, self.life, self.polar]
        return "%9i %4i %4i %4i %4i %4i %18.10E %18.10E %18.10E %18.10E %18.10E  %1.0f. %2.0f\n" % tuple(li)        

#useful function for converting a string to variables
def parseStringToVars(input):
    if input.find(".")>-1 :
        return float(input)
    else:
        return int(input)

#main part of analysis
if len(sys.argv)!=3:
    print "Usage: addmasses.py <infile> <outfile>    "
    print " Last modified: Fri Nov 21 10:49:14 CST 2008 "
    sys.exit(0)
else:
    print "Running addmasses.py to add masses and correct kinematics of fixed particles"
    
#first print out leading information
try:
    f=open(sys.argv[1],'r')
except IOError:
    print "need a file for reading"
    sys.exit(0)

try:
    g=open(sys.argv[2],'w')
except IOError:
    print "need a file for writing"
    sys.exit(0)

while 1:
    try:
        line=f.readline()
    except IOError:
        print "Problem reading from file ",sys.argv[1]
        sys.exit(0)
    if line.find("<event>")==-1:
        g.write(line)
    else:
        break

f.close()

#let xml find the event tags
try:
    xmldoc = minidom.parse(sys.argv[1])
except IOError:
    print " could not open file for xml parsing ",sys.argv[1]
    sys.exit(0)
    
    
reflist = xmldoc.getElementsByTagName('event')    

for ref in reflist:
    lines = ref.toxml()
    slines = lines.split("\n")
    next = 0
    nup = 0
    nlines = len(slines)
    counter = 0
    event = []
    event_description=""
    event_poundSign=""

    while counter<nlines:
        s=slines[counter]
        if s.find("<event>")>-1:
            next=1
        elif s.find("</event>")>-1:
            pass
        elif s.find("#")>-1:
            event_poundSign=s
        elif next==1:
            event_description=s
            next=0
        elif not s:
            continue
        else:
            t=[]
            for l in s.split(): t.append(parseStringToVars(l))
            nup = nup+1
            event.append(Particle(nup,t))
        counter=counter+1

#default is to skip this
    if motherFlag:

        noMotherList=[]
        for p in event:
            if p.status == -1: continue
            idabs = abs(p.id)
            idmo = p.mo1
            pmo = event[idmo-1]
            if idabs>=11 and idabs<=16:
                if p.mo1==1 or (pmo.co1 !=0 or pmo.co2 !=0):
                    noMotherList.append(p.no)
            elif idabs<=5 and abs(pmo.id)==6:
                if not ( p.co1==pmo.co1 and p.co2==pmo.co2):
                    noMotherList.append(p.no)                

        nAdded=0
        if len(noMotherList)==0:
            pass
        elif len(noMotherList)%2 != 0:
            print "single orphan; do not know how to process"
        else:
            ki=0
            while ki<len(noMotherList)-1:
                l1=event[noMotherList[ki]-1]
                ki=ki+1
                l2=event[noMotherList[ki]-1]
                lq=l1.id + l2.id
                if lq==-1 or lq==1:
                    idMom=24*lq
                elif lq==0:
                    idMom=23
                else:
                    break
                lMom=l1.mom+l2.mom            
                lm=[idMom,2,l1.mo1,l1.mo2,0,0,lMom.px,lMom.py,lMom.pz,lMom.E,lMom.calcMass(),0,0]
                nup=nup+1
                nAdded=nAdded+1
                event.append(Particle(nup,lm))
                l1.mo1=l1.mo2=l2.mo1=l2.mo2=nup
                ki=ki+1

# update number of partons if mothers added
            if nAdded>0:
                s1=event_description.split()[0]
                mySub = re.compile(s1)
                event_description = mySub.sub(str(nup),event_description)

    if nAdded>0:
        for ip in range(len(event)):
            l=event[ip]
            if l.mo1 > ip + 1:
                nmo=l.mo1
                event.insert(ip, event.pop(l.mo1-1))
                event[ip].no = ip + 1
                for l2 in event[ip + 1:]:
                    if l2.no > ip and l2.no < nmo + 1:
                        l2.no += 1
                    if l2.mo1 == nmo:
                        l2.mo1 = l2.mo2 = ip + 1
                    elif l2.mo1 > ip and l2.mo1 < nmo:
                        l2.mo1 = l2.mo2 = l2.mo1 + 1

# identify mothers
    particleDict={}
    for p in event:
        idabs = abs(p.id)
        if idabs>=11 and idabs<=16:
            if p.mo1==1:
                pass
            else:
                if p.mo1 in particleDict:
                    l=particleDict[p.mo1]
                    l.append(p.no)
                else:
                    l=[p.no]
                particleDict[p.mo1]=l

# repair kinematics
    for k in particleDict:
        if len(particleDict[k]) != 2: continue
        t=particleDict[k]
        p1=event[t[0]-1]
        p1.mom.boost(event[k-1].mom,-1)
        p2=event[t[1]-1]
        p2.mom.boost(event[k-1].mom,-1)
        rsh=event[k-1].mom.m
        if p1.id in massDict: p1.mom.m=massDict[p1.id]
        if p2.id in massDict: p2.mom.m=massDict[p2.id]
        p1.mom.E = (rsh*rsh + p1.mom.m**2 - p2.mom.m**2)/(2.0*rsh)
        pmagOld=math.sqrt(p1.mom.px**2+p1.mom.py**2+p1.mom.pz**2)
        pmagNew=math.sqrt(p1.mom.E**2-p1.mom.m**2)
        p1.mom.reScale(pmagOld,pmagNew)
        p2.mom = Momentum(0,0,0,rsh,rsh) - p1.mom
        p1.mom.boost(event[k-1].mom,1)
        p2.mom.boost(event[k-1].mom,1)

    pSum = Momentum(0,0,0,0,0)
    for p in event:
        if p.status== 2 :
            pass
        elif p.status==-1:
            pSum = pSum - p.mom
        elif p.status==1:
            pSum = pSum + p.mom

    if abs(pSum.px)>toler or abs(pSum.py)>toler or abs(pSum.pz)>toler or abs(pSum.E)>toler:
        print "Event does not pass tolerance ",toler
        pSum.printMe()

    if 1:
        g.write("<event>\n")
        g.write(event_description+"\n")
        for p in event:
            g.write(p.writeMe())
        g.write(event_poundSign+"\n")
        g.write("</event>\n")
#at the end
g.write("</LesHouchesEvents>\n")
g.close()

    

