FILES = Cards HTML SubProcesses bin lib Source Events index.html README \
           TemplateVersion.txt MGMEVersion.txt
  
amcatnlo.tar.gz: amcatnlo.tar
	rm -f amcatnlo.tar.gz
	gzip amcatnlo.tar
  
amcatnlo.tar: 
	tar cf madevent.tar $(FILES)
  
gridpack.tar.gz: gridpack.tar
	rm -f gridpack.tar.gz
	gzip gridpack.tar
  
gridpack.tar: 
	mv Events Events1
	mkdir Events
	cp Events1/banner_header.txt Events/
	mkdir amcatnlo
	cp bin/run.sh ./
	mv -f $(FILES) amcatnlo
	tar -cf gridpack.tar amcatnlo run.sh
	mv -f amcatnlo/* ./ 
	rmdir amcatnlo
	rm run.sh
	rm -rf Events
	mv Events1 Events

clean:
	rm -rf MCatNLO
	rm -rf Events/*
	rm -f *.log
	rm -f SubProcesses/P*/*.o
	rm -f SubProcesses/P*/*.a
	rm -f SubProcesses/P*/madevent_*
	rm -f SubProcesses/P*/gensym
	rm -f SubProcesses/P*/check_poles
	rm -f SubProcesses/P*/test_M?
	rm -f SubProcesses/P*/test_M?
	rm -f SubProcesses/P*/ajob*
	rm -f SubProcesses/P*/reweight_xsec_events
	rm -rf SubProcesses/P*/GV*
	rm -rf SubProcesses/P*/GF*
	rm -rf SubProcesses/P*/GB*
	rm -rf SubProcesses/P*/viSB*
	rm -rf SubProcesses/P*/novB*
	rm -rf SubProcesses/P*/born_G*
	rm -f SubProcesses/P*/V*/*.o
	rm -f SubProcesses/P*/V*/*.a
	rm -rf lib/*
	rm -f Source/*.o
	rm -f Sourde/*/*.o 

