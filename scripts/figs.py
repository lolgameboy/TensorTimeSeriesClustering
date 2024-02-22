from matrix_aca_t import *
from vector_aca_t import *
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

def plot_rel_err(method, ranks=range(5,51, 5), repeat=50, max_approx=None, ptype='bar', color='blue'):
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
    data = []

    for rank in ranks:
        print(f'rank {rank}')
        rel_errs = []
        for i in range(repeat):
            if method == matrix_aca_t:
                dp = method(tensor, rank)
            else:
                dp = method(tensor, rank, max_approx)
            rel_errs.append(np.linalg.norm(tensor-dp.full_tensor())/t_norm)
            print('<' + i*'#' + (repeat-i)*'-' + '>', end='\r')
        data.append(rel_errs)

        print(f' avg norm = {round(stat.mean(rel_errs), 3)} | stdev = {round(stat.stdev(rel_errs), 3)}')

    fig, ax = plt.subplots()

    if ptype == 'bar':
        avgs = map(stat.mean, data)
        stdevs = map(stat.stdev, data)
        plt.bar(ranks, avgs, width=2, yerr=stdevs, color=color)
    elif ptype == 'violin':
        vp = ax.violinplot(data, ranks, widths=3.5, showmeans=True)
        for body in vp['bodies']:
            body.set_alpha(0.5)
    elif ptype == 'box':
        plt.boxplot(data, positions=ranks, widths=3.2)
    elif ptype == 'scatter':
        xs = sum(map(lambda x : repeat*[x], ranks), [])
        ys = sum(data, [])
        plt.scatter(xs, ys, alpha=0.4, color=color)
    elif ptype == 'scatter-poly':
        xs = sum(map(lambda x : repeat*[x], ranks), [])
        ys = sum(data, [])
        plt.scatter(xs, ys, alpha=0.4, color=color)
        coeffs = np.polyfit(xs, ys, 3) # polynomial of degree 3
        poly = np.poly1d(coeffs)
        plt.plot(ranks, poly(ranks), color='red')
    else:
        raise Exception('Plot type not recognized.')

    plt.xlabel('Rang')
    plt.ylabel('Relatieve fout')

    if max_approx is not None:
        d = f'({max_approx})'
    else:
        d = ''
    plt.title(f'Relatieve fout van {method.__name__[0:6]}{d} ACA-T per rang')
    plt.xticks(ranks)
    plt.grid(axis='y', alpha=0.7)
    plt.ylim(bottom=0)

    plt.savefig(f'figures/rel_fout_{method.__name__}{d}(rpt{repeat})(rnk{ranks[-1]}).png')
    plt.show()

def plot_rel_err_per_rel_dtw():
    #TODO
    pass

# plot_rel_err(matrix_aca_t, ranks=range(5,51,5), repeat=5, ptype='bar', color='orange')
#plot_rel_err(vector_aca_t, ranks=range(5, 51, 5), repeat=5, max_approx=1, ptype='bar', color='blue')
plot_rel_err(vector_aca_t, ranks=range(5, 31, 5), repeat=5, max_approx=1, ptype='scatter-poly', color='blue')