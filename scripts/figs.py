from matrix_aca_t import *
from vector_aca_t import *
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

def plot_rel_err(ranks, max_approxs, colors, add_matrix_aca_t=False, repeat=50, ptype='bar'):
    '''
    Plot relative error for different ranks of decomposition for vector_aca_t
    :param ranks: different ranks to plot
    :param max_approxs: list of vector_aca_t types to include
    :param colors: colors of the different types (ordered), if matrix_aca_t is included, its color is always orange
    :param add_matrix_aca_t: include matrix_aca_t in the plot?
    :param repeat: sample size average for a given rank
    :param ptype: type of graph to plot
    '''

    if len(colors) != len(max_approxs):
        raise Exception("Amount of colors not right.")
        
    matrix_str = ' (matrix_aca_t included)' if add_matrix_aca_t else ''
    print(f'Plotting relative error {max_approxs}{matrix_str} with ranks {list(ranks)} averaged out of {repeat} samples.')

    tensor = np.load("saved_tensors/full_tensor.npy")
    t_norm = np.linalg.norm(tensor)

    x_pos = len(ranks)
    data = []

    # calculate data for every type
    for max_approx in max_approxs:
        data_for_type = []
        print(f'type {max_approx}')
        for rank in ranks:
            rel_errs = []
            for i in range(repeat):
                dp = vector_aca_t(tensor, rank, max_approx)
                rel_errs.append(np.linalg.norm(tensor-dp.full_tensor())/t_norm)
                print(f'rank {rank}' + '<' + i*'#' + (repeat-i)*'-' + '>', end='\r')
            data_for_type.append(rel_errs)
            print('', end='\r')
        data.append(data_for_type)

    # calculate data for matrix_aca_t if needed
    if add_matrix_aca_t:
        data_for_type = []
        print(f'maxtrix_aca_t')
        for rank in ranks:
            rel_errs = []
            for i in range(repeat):
                dp = matrix_aca_t(tensor, rank)
                rel_errs.append(np.linalg.norm(tensor-dp.full_tensor())/t_norm)
                print(f'rank {rank}' + '<' + i*'#' + (repeat-i)*'-' + '>', end='\r')
            data_for_type.append(rel_errs)
        data.append(data_for_type)
        colors.append('orange')

    fig, ax = plt.subplots()

    if ptype == 'bar':
        n = len(data)
        width = 3
        for i, d in enumerate(data):
            avgs = list(map(stat.mean, d))
            stdevs = list(map(stat.stdev, d))
            offset = (i - n/2 + 1/2)*width/n # offset from center of bar to the tick on x-axis
            xs = list(map(lambda x: x + offset, ranks))
            plt.bar(xs, avgs, width=width/n, yerr=stdevs, color=colors[i])
    elif ptype == 'box':
        for i, d in enumerate(data):
            bplot = plt.boxplot(d, positions=ranks, widths=3.2, patch_artist=True)
            for box in bplot['boxes']:
                box.set_facecolor(colors[i])
    elif ptype == 'scatter':
        xs = sum(map(lambda x : repeat*[x], ranks), [])
        for i, d in enumerate(data):
            ys = sum(d, [])
            plt.scatter(xs, ys, alpha=0.4, color=colors[i])
    elif ptype == 'scatter-line':
        xs = sum(map(lambda x : repeat*[x], ranks), [])
        for i, d in enumerate(data):
            ys = sum(d, [])
            plt.scatter(xs, ys, alpha=0.4, color=colors[i])
            plt.plot(ranks, list(map(stat.mean, d)), color=colors[i])
    elif ptype == 'line':
        xs = sum(map(lambda x : repeat*[x], ranks), [])
        for i, d in enumerate(data):
            ys = sum(d, [])
            plt.plot(ranks, list(map(stat.mean, d)), color=colors[i], marker='.', markersize=10, markerfacecolor='white')
    else:
        raise Exception('Plot type not recognized.')

    # Styling of the plot

        # x and y axis
    plt.xlabel('Rang')
    plt.xticks(ranks)

    plt.ylabel('Relatieve fout')
    plt.ylim(bottom=0)
    plt.grid(axis='y', alpha=0.7)

    xlabel = ax.xaxis.get_label()
    ylabel = ax.yaxis.get_label()

    xlabel.set_style('italic')
    ylabel.set_style('italic')
    xlabel.set_size(10)
    ylabel.set_size(10)

        # legend
    #lgd = list(map(lambda p: f'type {p}', max_approxs))
    #if add_matrix_aca_t:
    #    lgd.append('matrix ACA-T')
    #plt.legend(lgd)

        # title
    plt.title(f'Relatieve fout van ACA-T methodes per rang')
    ax.title.set_weight('bold')

        # right and top spines to gray
    ax.spines['right'].set_color((.8,.8,.8))
    ax.spines['top'].set_color((.8,.8,.8))

        # sharpness of plot
    plt.rcParams['figure.dpi'] = 360

    # save and show plot
    plt.savefig(f'figures/rel_fout{str(tuple(max_approxs)).replace(" ", "")}(rpt{repeat})(rnk{ranks[-1]}).png', dpi=600)
    plt.show()

def plot_rel_dtw_per_rank(ranks, max_approxs, colors, add_matrix_aca_t=False, ptype='line'):
    '''
    Plot relative DTW operations for different ranks of decomposition for vector_aca_t
    :param ranks: different ranks to plot
    :param max_approxs: list of vector_aca_t types to include
    :param colors: colors of the different types (ordered), if matrix_aca_t is included, its color is always orange
    :param add_matric_aca_t: include matrix_aca_t in the plot?
    :param ptype: type of graph to plot
    '''

    if len(colors) != len(max_approxs):
        raise Exception("Amount of colors not right.")
        
    matrix_str = ' (matrix_aca_t included)' if add_matrix_aca_t else ''
    print(f'Plotting relative error on types {max_approxs}{matrix_str} with DTW operations at ranks {list(ranks)}')

    tensor = np.load("saved_tensors/full_tensor.npy")
    t_norm = np.linalg.norm(tensor)

    x_pos = len(ranks)
    data = []
    total = 75 * (186 * 186 - 186)/2 # symmetrical slices halves the DTW operations

    # calculate data for every approximation
    for max_approx in max_approxs:
        print(f'type {max_approx}')

        percentages = []
        for rank in ranks:
            print(f'rank {rank}', end='\r')
            dp, count = vector_aca_t(tensor, rank, max_approx, count=True)
            percentages.append(count/total)

        data.append(percentages)
    
    # calculate data for matrix_aca_t
    if add_matrix_aca_t:
        print('matrix_aca_t')

        percentages = []
        for rank in ranks:
            print(f'rank {rank}', end='\r')
            dp, count = matrix_aca_t(tensor, rank, count=True)
            percentages.append(count/total)
        
        data.append(percentages)
        colors.append('orange')

    fig, ax = plt.subplots()

    if ptype == 'scatter':
        for i, ys in enumerate(data):
            plt.scatter(ranks, ys, color=colors[i], alpha=0.9)
    elif ptype == "line":
        for i, ys in enumerate(data):
            plt.plot(ranks, ys, color=colors[i], marker='.', markersize=10, markerfacecolor='white')
    else:
        raise Exception('Plot type not recognized.')

    # Styling of the plot

        # x and y axis
    plt.xlabel('Rang')
    plt.xticks(ranks)

    plt.ylabel('DTW operaties (relatief)')
    plt.ylim(bottom=0)
    plt.grid(axis='y', alpha=0.7)

    xlabel = ax.xaxis.get_label()
    ylabel = ax.yaxis.get_label()

    xlabel.set_style('italic')
    ylabel.set_style('italic')
    xlabel.set_size(10)
    ylabel.set_size(10)

        # legend
    lgd = list(map(lambda p: f'type {p}', max_approxs))
    if add_matrix_aca_t:
        lgd.append('matrix ACA-T')
    plt.legend(lgd)

        # title
    plt.title(f'DTW operaties (relatief) van ACA-T methodes per rang')
    ax.title.set_weight('bold')

        # right and top spines to gray
    ax.spines['right'].set_color((.8,.8,.8))
    ax.spines['top'].set_color((.8,.8,.8))

        # sharpness of plot
    plt.rcParams['figure.dpi'] = 360

    # save and show plot
    plt.savefig(f'figures/rel_dtw{str(tuple(max_approxs)).replace(" ", "")}(rnk{ranks[-1]}).png', dpi=600)
    plt.show()

def plot_rel_err_per_rel_dtw():
    #TODO
    pass


#plot_rel_err(range(5, 11, 5), [1, 3, 5], ['lightblue', 'lightgreen', 'pink'], repeat=5, ptype='box')
#plot_rel_err(range(5, 21, 5), [1, 3, 10], ['lightgreen', 'lightblue', 'pink'], add_matrix_aca_t=False, repeat=10, ptype='bar')
plot_rel_dtw_per_rank(range(5, 21, 5), [1, 3, 10], ['lightgreen', 'lightblue', 'pink'], add_matrix_aca_t=False)