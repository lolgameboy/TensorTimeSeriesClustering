from matrix_aca_t import *
from vector_aca_t import *
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

def calculate_data(ranks, max_approxs, add_matrix_aca_t, repeat=1):
    print(f'Calculating data ({ranks},{max_approxs},{add_matrix_aca_t},{repeat})')

    tensor = np.load("saved_tensors/full_tensor.npy")
    t_norm = np.linalg.norm(tensor)

    x_pos = len(ranks)
    err_data = []
    count_data = []
    total = tensor.shape[0] * tensor.shape[1] * tensor.shape[2] # less optimal total (not using the symmetry of slices and that the diagonals are 0)
    # total = tensor.shape[0] * (tensor.shape[1] * tensor.shape[1] - tensor.shape[1])/2 # symmetrical slices halves the DTW operations

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
        print(f'matrix_aca_t')

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

    np.save(f'saved_fig_data/data(err,{ranks},{max_approxs},{add_matrix_aca_t},{repeat}).npy', err_data)
    np.save(f'saved_fig_data/data(count,{ranks},{max_approxs},{add_matrix_aca_t}).npy', count_data)
    print('data saved to files')

    return err_data, count_data

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
    if repeat < 2:
        raise Exception("Repeat needs to be 2 or higher to calculate standard deviations.")

    try:
        data = np.load(f'saved_fig_data/data(err,{ranks},{max_approxs},{add_matrix_aca_t},{repeat}).npy')
        print('data loaded from files')
    except:
        data, _ = calculate_data(ranks, max_approxs, add_matrix_aca_t, repeat)

    if add_matrix_aca_t:
        colors.append('orange')

    fig, ax = plt.subplots()

    if ptype == 'bar':
        if len(ranks) < 2:
            raise Exception("At least 2 ranks needed for a bar plot.")
        n = len(data)

         # width is based on highest rank and inter rank distance (assumes uniform inter-rank distance: e.g. [5, 10, 15, 20] and NOT [5, 9, 17, 20])
        delta = ranks[-1] - ranks[-2]
        total = ranks[-1]
        alpha = 10 * delta/total
        width = (delta - alpha)/n

        # fig size is based on amount of bars (n)
        fig.set_size_inches(3 + 6.4, 4.8, forward=True) # many bars (large n) needs wider plot

        for i, d in enumerate(data):
            avgs = list(map(stat.mean, d))
            stdevs = list(map(stat.stdev, d))
            offset = (i - n/2 + 1/2)*width # offset from center of bar to the tick on x-axis
            xs = list(map(lambda x: x + offset, ranks))
            plt.bar(xs, avgs, width=width, yerr=stdevs, color=colors[i])
    elif ptype == 'box':
        for i, d in enumerate(data):
            bplot = plt.boxplot(list(d), positions=ranks, widths=3.2, patch_artist=True)
            for box in bplot['boxes']:
                box.set_facecolor(colors[i])
    elif ptype == 'scatter':
        xs = np.array(list(map(lambda x : repeat*[x], ranks))).flatten()
        for i, d in enumerate(data):
            ys = d.flatten()
            plt.scatter(xs, ys, alpha=0.4, color=colors[i])
    elif ptype == 'scatter-line':
        xs = np.array(list(map(lambda x : repeat*[x], ranks))).flatten()
        for i, d in enumerate(data):
            plt.plot(ranks, list(map(stat.mean, d)), color=colors[i])
        for i, d in enumerate(data):
            ys = d.flatten()
            plt.scatter(xs, ys, alpha=0.4, color=colors[i])
    elif ptype == 'line':
        for i, d in enumerate(data):
            plt.plot(ranks, list(map(stat.mean, d)), color=colors[i], marker='.', markersize=10, markerfacecolor='white')
    else:
        raise Exception('Plot type not recognized.')

    # Styling of the plot

        # x and y axis
    plt.xlabel('Aantal termen')
    plt.xticks(ranks, fontsize=15)

    plt.ylabel('Relatieve fout')
    plt.yticks(fontsize=15)
    plt.ylim(bottom=0)
    plt.grid(axis='y', alpha=0.7)

    xlabel = ax.xaxis.get_label()
    ylabel = ax.yaxis.get_label()

    xlabel.set_style('italic')
    ylabel.set_style('italic')
    xlabel.set_size(18)
    ylabel.set_size(18)
    plt.subplots_adjust(bottom=0.14, left=0.075)

        # legend
    if len(max_approxs) == 1 and max_approxs[0] == 1:
        lgd = ['Vector ACA-T']
    else:
        lgd = list(map(lambda p: f'Type {p}', max_approxs))
    if add_matrix_aca_t:
        lgd.append('Matrix ACA-T')
    plt.legend(lgd)

        # title
    plt.title(f'Relatieve fout van ACA-T methodes per aantal termen', fontsize=20)
    ax.title.set_weight('bold')

        # right and top spines to gray
    ax.spines['right'].set_color((.8,.8,.8))
    ax.spines['top'].set_color((.8,.8,.8))

        # sharpness of plot (not relevant for .svg)
    #plt.rcParams['figure.dpi'] = 360

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

    try:
        data = np.load(f'saved_fig_data/data(count,{ranks},{max_approxs},{add_matrix_aca_t}).npy')
        print('data loaded from a save file')
    except:   
        _, data = calculate_data(ranks, max_approxs, add_matrix_aca_t)
            
    if add_matrix_aca_t:
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
    plt.xlabel('Aantal termen')
    plt.xticks(ranks, fontsize=15)

    plt.ylabel('Relatieve % DTW operaties')
    plt.yticks(fontsize=15)
    plt.ylim(bottom=0)
    plt.grid(axis='y', alpha=0.7)

    xlabel = ax.xaxis.get_label()
    ylabel = ax.yaxis.get_label()

    xlabel.set_style('italic')
    ylabel.set_style('italic')
    xlabel.set_size(18)
    ylabel.set_size(18)
    plt.subplots_adjust(bottom=0.14, left=0.075)

        # legend
    if len(max_approxs) == 1 and max_approxs[0] == 1:
        lgd = ['Vector ACA-T']
    else:
        lgd = list(map(lambda p: f'Type {p}', max_approxs))
    if add_matrix_aca_t:
        lgd.append('Matrix ACA-T')
    plt.legend(lgd)

        # title
    plt.title(f'Relatieve % DTW operaties van ACA-T methodes per aantal termen', fontsize=20)
    ax.title.set_weight('bold')

        # right and top spines to gray
    ax.spines['right'].set_color((.8,.8,.8))
    ax.spines['top'].set_color((.8,.8,.8))

        # sharpness of plot (not relevant for .svg)
    #plt.rcParams['figure.dpi'] = 360

    # save and show plot
    plt.savefig(f'figures/rel_dtw{str(tuple(max_approxs)).replace(" ", "")}(rnk{ranks[-1]}).svg', transparent=True, bbox_inches=0)
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
    if repeat < 2:
        raise Exception("Repeat needs to be 2 or higher to calculate standard deviations.")

    try:
        err_data = np.load(f'saved_fig_data/data(err,{ranks},{max_approxs},{add_matrix_aca_t},{repeat}).npy')
        count_data = np.load(f'saved_fig_data/data(count,{ranks},{max_approxs},{add_matrix_aca_t}).npy')
        print('data loaded from files')
    except:
        err_data, count_data = calculate_data(ranks, max_approxs, add_matrix_aca_t, repeat)

    if add_matrix_aca_t:
        colors.append('orange')

    fig, ax = plt.subplots()

    data = zip(err_data, count_data)
    if ptype == 'scatter':
        for i, (ed, cd) in enumerate(data):
            xs = np.array(list(map(lambda x : repeat*[x], cd))).flatten()
            ys = ed.flatten()
            plt.scatter(xs, ys, alpha=0.4, color=colors[i])
    elif ptype == 'line':
        for i, (ed, cd) in enumerate(data):
            plt.plot(cd, list(map(stat.mean, ed)), color=colors[i], marker='.', markersize=10, markerfacecolor='white')
    else:
        raise Exception('Plot type not recognized.')

    # Styling of the plot
    fig.set_size_inches(2 + 6.4, 4.8, forward=True)
        # x and y axis
    plt.xlabel('Relatieve % DTW operaties')
    plt.xticks(list(map(lambda x: round(x, 1), count_data[-1])), fontsize=15)

    plt.ylabel('Relatieve fout')
    plt.yticks(fontsize=15)
    plt.ylim(bottom=0)
    plt.grid(axis='y', alpha=0.7)

    xlabel = ax.xaxis.get_label()
    ylabel = ax.yaxis.get_label()

    xlabel.set_style('italic')
    ylabel.set_style('italic')
    xlabel.set_size(18)
    ylabel.set_size(18)
    plt.subplots_adjust(bottom=0.14, left=0.1, top=0.85)

        # legend
    if len(max_approxs) == 1 and max_approxs[0] == 1:
        lgd = ['Vector ACA-T']
    else:
        lgd = list(map(lambda p: f'Type {p}', max_approxs))
    if add_matrix_aca_t:
        lgd.append('Matrix ACA-T')
    plt.legend(lgd)

        # title
    plt.title('Relatieve fout van ACA-T methodes\n versus hun relatieve % DTW operaties', fontsize=20)
    ax.title.set_weight('bold')
    fig.set_size_inches(1.4*6.4, 4.8)

        # right and top spines to gray
    ax.spines['right'].set_color((.8,.8,.8))
    ax.spines['top'].set_color((.8,.8,.8))

        # sharpness of plot (not relevant for .svg)
    #plt.rcParams['figure.dpi'] = 360

    # save and show plot
    plt.savefig(f'figures/rel_fout_rel_dtw{str(tuple(max_approxs)).replace(" ", "")}(rpt{repeat})(rnk{ranks[-1]}).svg', transparent=True, bbox_inches=0)
    plt.show()

# type     1            2         3              5         8       10/20
colors = ['firebrick', 'indigo', 'greenyellow', 'violet', 'teal', 'indigo']

#plot_rel_err(range(5, 51, 5), [2], ['indigo'], add_matrix_aca_t=False, repeat=50)
#plot_rel_err(range(5, 41, 5), [3], ['greenyellow'], add_matrix_aca_t=False, repeat=50)
#plot_rel_err(range(5, 31, 3), [5], ['violet'], add_matrix_aca_t=False, repeat=50)
#plot_rel_err(range(5, 101, 10), [5], ['violet'], add_matrix_aca_t=False, repeat=2)


plot_rel_err(range(2, 15, 2), [1, 2, 3, 5], ['firebrick', 'indigo', 'greenyellow', 'violet'], add_matrix_aca_t=False, repeat=50)
#plot_rel_err(range(2, 15, 2), [1, 3, 5, 8, 10], ['firebrick', 'greenyellow', 'violet', 'teal', 'indigo'], add_matrix_aca_t=False, repeat=20)
#plot_rel_err_vs_rel_dtw(range(5, 51, 10), [1, 8], ['firebrick', 'teal'], add_matrix_aca_t=True, repeat=3, ptype='line')
#plot_rel_err(range(5,51,5), [1], ['firebrick'], add_matrix_aca_t=True)