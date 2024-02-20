from matrix_aca_t import *
from vector_aca_t import *
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

def plot_rel_err(method, ranks=range(5,51, 5), repeat=50, max_approx=None):
    '''
    Plot relative error for different ranks of decomposition.
    :param method: decomposition function
    :param ranks: different ranks to plot
    :param repeat: sample size average for a given rank
    :param max_approx: param only needed for vector ACA-T
    '''

    print(f'Plotting relative error of {method.__name__} with ranks {list(ranks)} averaged out of {repeat} samples.')

    tensor = np.load("saved_tensors/full_tensor.npy")
    t_norm = np.linalg.norm(tensor)

    x_pos = len(ranks)
    avgs = []
    stdevs = []

    for rank in ranks:
        print(f'rank {rank}')
        rel_errs = []
        for i in range(repeat):
            print(f'-', end="")
            if method == matrix_aca_t:
                dp = method(tensor, rank)
            else:
                dp = method(tensor, rank, max_approx)
            rel_errs.append(np.linalg.norm(tensor-dp.full_tensor())/t_norm)

        avgs.append(stat.mean(rel_errs))
        stdevs.append(stat.stdev(rel_errs))

        print(f'rank {rank} with norm = {avgs[-1]} | stdev = {stdevs[-1]}')

    if method == matrix_aca_t:
        c = 'orange'
    else:
        c = 'blue'

    plt.subplots()
    plt.bar(ranks, avgs, width=2, yerr=stdevs, color=c)

    plt.xlabel('Rang')
    plt.ylabel('Relatieve fout')

    if max_approx is not None:
        d = f'({max_approx})'
    else:
        d = ''
    plt.title(f'Relatieve fout van {method.__name__[0:6]}{d} ACA-T per rang')
    plt.xticks(ranks)

    plt.savefig(f'figures/rel_fout_{method.__name__}{d}(rpt{repeat})(rnk{ranks[-1]}).png')
    plt.show()

def plot_rel_err_per_rel_dtw():
    #TODO
    pass

#plot_rel_err(matrix_aca_t, ranks=range(5,16,5), repeat=3)
plot_rel_err(vector_aca_t, ranks=range(5, 51, 5), repeat=3, max_approx=1)