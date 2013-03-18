COPTS = -O -DDROP_CGAL
INCLUDES =  -I../include
OBJECTS = SysCalc.o tinyxml2.o
LIBS = -lLHAPDF

.cc.o:
	$(CXX) $(COPTS) $(INCLUDES) -c $<

.cpp.o:
	$(CXX) $(COPTS) $(INCLUDES) -c $<

all: ../sys_calc

SysCalc.o: ../include/SysCalc.h ../include/tinyxml2.h
sys_calc.o: ../include/SysCalc.h ../include/tinyxml2.h

../sys_calc: $(OBJECTS) sys_calc.o 
	$(CXX) -o $@ $^ $(LIBS) 

clean:
	rm -f ../sys_calc *.o