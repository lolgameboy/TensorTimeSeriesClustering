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
    :param repeat: sample size to average from for a given rank
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
            plt.plot(ranks, list(map(stat.mean, d)), color=colors[i])
        for i, d in enumerate(data):
            ys = sum(d, [])
            plt.scatter(xs, ys, alpha=0.4, color=colors[i])
    elif ptype == 'line':
        for i, d in enumerate(data):
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
    lgd = list(map(lambda p: f'type {p}', max_approxs))
    if add_matrix_aca_t:
        lgd.append('matrix ACA-T')
    plt.legend(lgd)

        # title
    plt.title(f'Relatieve fout van ACA-T methodes per rang')
    ax.title.set_weight('bold')

        # right and top spines to gray
    ax.spines['right'].set_color((.8,.8,.8))
    ax.spines['top'].set_color((.8,.8,.8))

        # sharpness of plot (not relevant for .svg)
    #plt.rcParams['figure.dpi'] = 360

    fig.patch.set_facecolor('#DEEBF7')
    ax.set_facecolor('#DEEBF7')

    # save and show plot
    plt.savefig(f'figures/rel_fout{str(tuple(max_approxs)).replace(" ", "")}(rpt{repeat})(rnk{ranks[-1]}).svg', transparent=True, bbox_inches=0)
    plt.show()

def plot_rel_dtw(ranks, max_approxs, colors, add_matrix_aca_t=False, ptype='line'):
    '''
    Plot relative DTW operations for different ranks of decomposition for vector_aca_t
    :param ranks: different ranks to plot
    :param max_approxs: list of vector_aca_t types to include
    :param colors: colors of the different types (ordered), if matrix_aca_t is included, its color is always orange
    :param add_matrix_aca_t: include matrix_aca_t in the plot?
    :param ptype: type of graph to plot
    '''

    if len(colors) != len(max_approxs):
        raise Exception("Amount of colors not right.")
        
    matrix_str = ' (matrix_aca_t included)' if add_matrix_aca_t else ''
    print(f'Plotting relative DTW operations on types {max_approxs}{matrix_str} at ranks {list(ranks)}')

    tensor = np.load("saved_tensors/full_tensor.npy")
    t_norm = np.linalg.norm(tensor)

    x_pos = len(ranks)
    data = []
    total = tensor.shape[0] * (tensor.shape[1] * tensor.shape[1] - tensor.shape[1])/2

    # calculate data for every approximation
    for max_approx in max_approxs:
        print(f'type {max_approx}')

        percentages = []
        for rank in ranks:
            print(f'rank {rank}', end='\r')
            dp, count = vector_aca_t(tensor, rank, max_approx, count=True)
            percentages.append(100*count/total)

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

    plt.ylabel('Relatieve % DTW operaties')
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
    plt.title(f'Relatieve % DTW operaties van ACA-T methodes per rang')
    ax.title.set_weight('bold')

        # right and top spines to gray
    ax.spines['right'].set_color((.8,.8,.8))
    ax.spines['top'].set_color((.8,.8,.8))

        # sharpness of plot (not relevant for .svg)
    #plt.rcParams['figure.dpi'] = 360

    # save and show plot
    plt.savefig(f'figures/rel_dtw{str(tuple(max_approxs)).replace(" ", "")}(rnk{ranks[-1]}).svg')
    plt.show()

def plot_rel_err_vs_rel_dtw(ranks, max_approxs, colors, add_matrix_aca_t=False, repeat=50, ptype='line'):
    '''
    Plot relative error in function of relative DTW operations on different ranks of the decomposition.
    :param ranks: different ranks to plot the relative DTW operations for
    :param max_approxs: list of vector_aca_t types to include
    :param colors: colors of the different types (ordered), if matrix_aca_t is included, its color is always orange
    :param add_matrix_aca_t: include matrix_aca_t in the plot?
    :param repeat: sample size to average from for a given rank
    :param ptype: type of graph to plot
    '''

    if len(colors) != len(max_approxs):
        raise Exception("Amount of colors not right.")
        
    matrix_str = ' (matrix_aca_t included)' if add_matrix_aca_t else ''
    print(f'Plotting relative error {max_approxs}{matrix_str} versus DTW operations at ranks {list(ranks)} averaged out of {repeat} samples.')

    tensor = np.load("saved_tensors/full_tensor.npy")
    t_norm = np.linalg.norm(tensor)

    x_pos = len(ranks)
    err_data = []
    count_data = []
    total = tensor.shape[0] * (tensor.shape[1] * tensor.shape[1] - tensor.shape[1])/2 # symmetrical slices halves the DTW operations

    # calculate data for every type
    for max_approx in max_approxs:
        print(f'type {max_approx}')

        data_for_type = []
        percentages = []
        for rank in ranks:
            rel_errs = []
            count = 0
            for i in range(repeat):
                dp, count = vector_aca_t(tensor, rank, max_approx, count=True)
                rel_errs.append(np.linalg.norm(tensor-dp.full_tensor())/t_norm)
                print(f'rank {rank}' + '<' + i*'#' + (repeat-i)*'-' + '>', end='\r')
            data_for_type.append(rel_errs)
            percentages.append(100*count/total) # count is determenistic -> only last repeat count is used
            print('', end='\r')
        
        err_data.append(data_for_type)
        count_data.append(percentages)

    # calculate data for matrix_aca_t if needed
    if add_matrix_aca_t:
        print(f'maxtrix_aca_t')

        data_for_type = []
        percentages = []
        for rank in ranks:
            rel_errs = []
            count = 0
            for i in range(repeat):
                dp, count = matrix_aca_t(tensor, rank, count=True)
                rel_errs.append(np.linalg.norm(tensor-dp.full_tensor())/t_norm)
                print(f'rank {rank}' + '<' + i*'#' + (repeat-i)*'-' + '>', end='\r')
            data_for_type.append(rel_errs)
            percentages.append(100*count/total)
        err_data.append(data_for_type)
        count_data.append(percentages)
        colors.append('orange')

    fig, ax = plt.subplots()

    data = zip(err_data, count_data)
    if ptype == 'scatter':
        for i, (ed, cd) in enumerate(data):
            xs = sum(map(lambda x : repeat*[x], cd), [])
            ys = sum(ed, [])
            plt.scatter(xs, ys, alpha=0.4, color=colors[i])
    elif ptype == 'line':
        for i, (ed, cd) in enumerate(data):
            plt.plot(cd, list(map(stat.mean, ed)), color=colors[i], marker='.', markersize=10, markerfacecolor='white')
    else:
        raise Exception('Plot type not recognized.')

    # Styling of the plot

        # x and y axis
    plt.xlabel('% Relatieve DTW operaties')
    plt.xticks(list(map(lambda x: round(x, 3), count_data[-1])))

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
    lgd = list(map(lambda p: f'type {p}', max_approxs))
    if add_matrix_aca_t:
        lgd.append('matrix ACA-T')
    plt.legend(lgd)

        # title
    plt.title('Relatieve fout van ACA-T methodes versus hun relatieve % DTW operaties')
    ax.title.set_weight('bold')
    fig.set_size_inches(1.4*6.4, 4.8)

        # right and top spines to gray
    ax.spines['right'].set_color((.8,.8,.8))
    ax.spines['top'].set_color((.8,.8,.8))

        # sharpness of plot (not relevant for .svg)
    #plt.rcParams['figure.dpi'] = 360

    # save and show plot
    plt.savefig(f'figures/rel_fout_rel_dtw{str(tuple(max_approxs)).replace(" ", "")}(rpt{repeat})(rnk{ranks[-1]}).svg')
    plt.show()

colors = ['lightgreen', 'pink', 'firebrick', 'sienna', 'violet', 'indigo', 'teal']
#plot_rel_err(range(5, 11, 5), [1, 3, 5], ['lightblue', 'lightgreen', 'pink'], repeat=5, ptype='box')
plot_rel_err(range(5, 21, 5), [1, 3, 5, 10, 20], ['lightgreen', 'pink', 'lightblue', 'violet', 'teal'], add_matrix_aca_t=False, repeat=3, ptype='bar')
#plot_rel_dtw(range(5, 21, 5), [1, 3, 10], ['lightgreen', 'lightblue', 'pink'], add_matrix_aca_t=False)
#plot_rel_err_vs_rel_dtw(range(5, 26, 5), [1, 3, 5], ['lightgreen', 'lightblue', 'pink'], add_matrix_aca_t=False, repeat=3, ptype='scatter-line')
#plot_rel_err_vs_rel_dtw(range(5, 16, 5), [1], ['lightgreen'], add_matrix_aca_t=False, repeat=3, ptype='scatter-line')