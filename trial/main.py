import fileinput
import re
import numpy as np
from itertools import count
from paint import get_grid, write_output, get_instructions

def allocate_squares(doodle, efficiency=0.02):
    (M,N) = doodle.shape
    remaining = doodle
    painted = np.zeros((M,N), np.int)
    blank = 1-doodle
    for j in xrange(N): # columns
        for i in xrange(M): # lines
            if remaining[i,j]:
                # we need to paint it and it is not painted yet,
                # we open a new box
                for l in count(): # size of the box
                    L = 2*l + 1
                    if i+L > M or j+L > N:
                        break
                    b = blank[i:i+L,j:j+L].sum()
                    r = remaining[i:i+L,j:j+L].sum()
                    if l>1 and 1+b > efficiency*r:
                        break
                    #to_paint = remaining[i:i+L,j:j+L].sum()
                    #blank = ((1-doodle)*(1-painted))[i:i+L,j:j+L].sum()
                    #threshold = (1.0 + L*L - already_painted) / (1.0+efficiency)
                    #if l>1 and (blank[i:i+L,j:j+L].sum() - to_paint[i:i+L,j:j+L].sum()) <= threshold:
                    # print doodle[i:i+L,j:j+L].sum(), threshold
                    #    break
                # the optimal square is of size L - 2
                L -= 2
                remaining[i:i+L,j:j+L] = (doodle[i:i+L,j:j+L] - painted[i:i+L,j:j+L] & doodle[i:i+L,j:j+L])
                painted[i:i+L,j:j+L] = 1
                blank[i:i+L,j:j+L] = ((1-doodle[i:i+L,j:j+L]) & (1-painted[i:i+L,j:j+L]))
                #print i,j,L
                yield (i,j,L)

def score(doodle, allocate):
    (M,N) = doodle.shape
    painted = np.zeros((M,N), np.int)
    cost = 0
    for (i,j,L) in allocate(doodle):
        painted[i:i+L,j:j+L] = 1.0
        cost += 1
    cost += (painted - doodle).sum()
    return cost

def get_instructions(doodle,allocate):
    (M,N) = doodle.shape
    painted = np.zeros((M,N), np.int)
    #cost = 0
    for (i,j,L) in allocate(doodle):
        painted[i:i+L,j:j+L] = 1.0
        l = (L - 1) / 2
        yield 'PAINTSQ {} {} {}'.format(i + l, j + l, l)
    diff = painted - doodle
    assert (diff >= 0).all()
    for j in xrange(N): # columns
        for i in xrange(M): # lines
            if diff[i,j]:
                yield 'ERASECELL {} {}'.format(i, j)



def write_output(instructions):
    print len(instructions)
    for instruction in instructions:
        print instruction

#(_,_,doodle) = get_grid()
#print score(doodle, allocate_squares)

if __name__=="__main__":
    (_,_,doodle) = get_grid()
    #squares = allocate_squares(doodle)
    instructions = list(get_instructions(doodle,allocate_squares))
    write_output(instructions)
