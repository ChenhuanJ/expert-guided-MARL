import os
from juliacall import Main

pwd = os.path.dirname(os.path.abspath(__file__))+"/"
Main.include(pwd+"./GLNS.jl")
def GtspGLNS(num_vertices, num_sets, membership, sets,dist):
    membership = [data+1 for data in membership]
    for i in range(len(sets)):
        sets[i] = [data+1 for data in sets[i]]
    cost,tour = Main.GLNS.solver(num_vertices, num_sets, sets, membership, dist)
    return cost,tour