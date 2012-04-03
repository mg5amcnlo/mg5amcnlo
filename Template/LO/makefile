FILES = Cards HTML SubProcesses bin lib Source Events index.html README \
        README.lhapdf TemplateVersion.txt MGMEVersion.txt

madevent.tar.gz: madevent.tar
	rm -f madevent.tar.gz
	gzip madevent.tar

madevent.tar: 
	tar cf madevent.tar $(FILES)

gridpack.tar.gz: gridpack.tar
	rm -f gridpack.tar.gz
	gzip gridpack.tar

gridpack.tar: 
	mv Events Events1
	mkdir Events
	mkdir madevent
	cp bin/internal/Gridpack/run.sh ./
	cp bin/internal/Gridpack/* bin/
	mv -f $(FILES) madevent
	tar -cf gridpack.tar madevent run.sh README.gridpack
	mv -f madevent/* ./ 
	rmdir madevent
	rm run.sh
	rm -rf Events
	mv Events1 Events
