import math
import numpy as np
import matplotlib.pyplot as plt

def time_series_example():
    xs = list(map(lambda x: 0.1*x, range(30)))
    ys = list(map(lambda x: -0.5*(x-1.5)**2 + 1, xs)) + np.random.normal(0.4, 0.1, 30)

    fig, ax = plt.subplots()
    fig.set_size_inches(3*6.4, 1.1*4.8, forward=True)

    plt.plot(xs, ys, color='firebrick', marker='.', markersize=10, markerfacecolor='white')


    # Styling of the plot
        # x and y axis
    plt.xlabel('Tijd (s)')
    plt.xticks(xs)

    plt.ylabel('Positie (m)')
    plt.ylim(bottom=0, top=2)
    plt.grid(axis='y', alpha=0.7)

    xlabel = ax.xaxis.get_label()
    ylabel = ax.yaxis.get_label()

    xlabel.set_style('italic')
    ylabel.set_style('italic')
    xlabel.set_size(20)
    ylabel.set_size(20)

        # title
    plt.title(f'Positie doorheen de tijd', fontsize=28)
    ax.title.set_weight('bold')

        # right and top spines to gray
    ax.spines['right'].set_color((.8,.8,.8))
    ax.spines['top'].set_color((.8,.8,.8))

        # sharpness of plot (not relevant for .svg)
    #plt.rcParams['figure.dpi'] = 360

    # save and show plot
    plt.savefig(f'figures/time_series_example.svg', transparent=True, bbox_inches=0)
    plt.show()

time_series_example()