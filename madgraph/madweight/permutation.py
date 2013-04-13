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

import collections
import itertools
import copy

def get_all_permutations(cat_list):
	""" """
	
	# create the category names and the id to permute for each category
	nb_cat = collections.defaultdict(list)
	for i,cat in enumerate(cat_list):
		nb_cat[cat].append(i)
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

if __name__ == '__main__':
	print get_all_permutations(['s','s','s'])
	print get_all_permutations(['b','b','j','j'])		
	print get_all_permutations(['b','b','j','j','c','c'])		
print get_all_permutations(['b','b','j','b'])		