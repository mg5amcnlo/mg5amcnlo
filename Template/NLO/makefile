FILES = Cards HTML SubProcesses bin lib Source Events index.html README \
        TemplateVersion.txt MGMEVersion.txt

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
	cp Events1/banner_header.txt Events/
	mkdir madevent
	cp bin/run.sh ./
	mv -f $(FILES) madevent
	tar -cf gridpack.tar madevent run.sh
	mv -f madevent/* ./ 
	rmdir madevent
	rm run.sh
	rm -rf Events
	mv Events1 Events
