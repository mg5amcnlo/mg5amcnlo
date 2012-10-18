################################################################################
#
# Copyright (c) 2012 The MadGraph Development team and Contributors
#
# This file is a part of the MadGraph 5 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph license which should accompany this 
# distribution.
#
# For more information, please visit: http://madgraph.phys.ucl.ac.be
#
################################################################################
""" A convenient way to define a hash.
   The purpose of this function is to have a unique way to define a hash 
   independently of the OPENGSL library include in the system.
   Note that the resulting functions is not suppose to be used for security 
   measure.
"""
done = False

class Factory:

	def test_all(self):
		try:
			return self.test_hashlib()
		except:
			pass
		try:
			return self.test_md5()
		except:
			pass
		try:
			return self.test_zlib()
		except:
			pass
				
	def test_hashlib(self):
		import hashlib
		def digest(text):
			"""using mg5 for the hash"""
			t = hashlib.md5()
			t.update(text)
			return t.hexdigest()
		return digest
	
	def test_md5(self):
		import md5
		def digest(text):
			"""using mg5 for the hash"""
			t = md5.md5()
			t.update(text)
			return t.hexdigest()
		return digest
	
	def test_zlib(self):
		import zlib
		def digest(text):
			return zlib.adler32(text)
		
digest = Factory().test_all()

if '__main__' == __name__:
	print digest('test')
	print digest
	




