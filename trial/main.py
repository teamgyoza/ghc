import fileinput
import re
import numpy as np
from itertools import count

efficiency = 1.0

def allocate_squares(doodle):
	(M,N) = doodle.shape
	painted = np.zeros((M,N), np.int)
	for j in xrange(N): # columns
		for i in xrange(M): # lines
			if doodle[i,j] and not painted[i,j]:
				# we need to paint it and it is not painted yet,
				# we open a new box
				for l in count(): # size of the box
					L = 2*l + 1
					threshold = (1.0 + L*L) / (1.0+efficiency)
					if i+L > M or j+L > N:
						break
					if doodle[i:i+l,j:j+l].sum() <= threshold:
						break
				# the optimal square is of size L
				yield (i,j,L)