import numpy as np
import sys
from utils import translate_params
from pso import PSO


class DSSO:

    def __init__(self, data=7):
        self.data = data

    def _cal_score(self, order, layer):
        idx_o = int((order - 1)/2) - 1
        idx_p = int(layer-1)
        if self.data == 7:
            score_mat = np.array([[0.89, 0.879, 0.903, 0.884, 0.881, 0.898, 0.893], 
                                [0.907,0.904,	0.921,	0.914,	0.925,	0.92,	0.923], 
                                [0.905,0.905,	0.914,	0.905,	0.913,	0.927,	0.923], 
                                [0.866,	0.863,	0.912,	0.915,	0.911,	0.922,	0.931], 
                                [0.821,	0.895,	0.892,	0.906,	0.915,	0.93,	0.927]])
        elif self.data == 21:
            score_mat = np.array([[0.52, 0.579, 0.661, 0.692, 0.724, 0.716, 0.722], 
                                [0.503, 0.563, 0.672, 0.639, 0.701, 0.723, 0.72], 
                                [0.532, 0.534, 0.634, 0.68, 0.694, 0.722, 0.739], 
                                [0.537, 0.548, 0.601, 0.697, 0.687, 0.704, 0.738], 
                                [0.522, 0.593, 0.622, 0.681, 0.725, 0.728, 0.756]])
        else:
            print('Unexpected dataset')
            sys.exit(-3)
        
        return 1 - score_mat[idx_p, idx_o]

    def get_score(self, params):
        candidate = [[3, 5, 7, 9, 11, 13, 15], [1, 2, 3, 4, 5]]
        order, layer = translate_params(params, candidate)
        score = self._cal_score(order, layer)
        return score


# pm = [0.3, 0.5]   
net = DSSO(data=7)
# s = instance.get_score(pm)
# print(s)
candidate = [[3, 5, 7, 9, 11, 13, 15], [1, 2, 3, 4, 5]]
pso = PSO(net.get_score, 2, 10, [[0,1], [0,1]], candidate=candidate)

pso.run()
     