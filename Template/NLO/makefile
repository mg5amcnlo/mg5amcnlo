FILES = Cards HTML SubProcesses bin lib Source Events index.html README README.lhapdf \
           TemplateVersion.txt MGMEVersion.txt
  
amcatnlo.tar.gz: amcatnlo.tar
	rm -f amcatnlo.tar.gz
	gzip amcatnlo.tar
  
amcatnlo.tar: 
	tar cf amcatnlo.tar $(FILES)
  
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
	rm -f MCatNLO/objects/*
	rm -rf MCatNLO/RUN*
	rm -f *.log
	cd Source; make clean
	rm -f SubProcesses/collect_events
	rm -f SubProcesses/combine_jobs
	rm -f SubProcesses/*.o
	for dir in `ls -d SubProcesses/P*`; do cd $$dir; make clean; cd ../../;\
	    done

