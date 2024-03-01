from clustering import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def show_clusters(method, n_clusters, rank, approx):
    df, fvs = cluster(method, n_clusters, 'rows', rank, approx)
    labels = df["Cluster"].to_list()
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(fvs)

    clusterXs = dict()
    clusterYs = dict()
    for i in range(n_clusters):
        clusterXs[i] = []
        clusterYs[i] = []
    for i in range(len(labels)):
        clusterXs[labels[i]].append(pcs[i, 0])
        clusterYs[labels[i]].append(pcs[i, 1])

    colors = ["y", "r", "g", "brown", "b", "c", "m", "purple", "wheat", "aqua"]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")

    xlabel = ax1.xaxis.get_label()
    ylabel = ax1.yaxis.get_label()

    xlabel.set_style('italic')
    ylabel.set_style('italic')
    xlabel.set_size(10)
    ylabel.set_size(10)

    title = "Clustering in " + str(n_clusters) + " clusters met " + method + " met rang " + str(rank)
    ax1.set_title(title)
    ax1.title.set_weight('bold')
    ax1.spines['right'].set_color((.8, .8, .8))
    ax1.spines['top'].set_color((.8, .8, .8))
    for i in range(n_clusters):
        ax1.scatter(clusterXs[i], clusterYs[i], c=colors[i])
    plt.savefig("figures/" + title + ".svg", transparent=True, bbox_inches=0)


def show_table(rows, method, n_clusters, rank, approx):
    table, fvs = cluster(method, n_clusters, 'rows', rank, approx)

    plt.figure()

    # table
    plt.subplot(111)

    cell_text = []
    for row in range(rows):
        cell_text.append(table.iloc[row])
    plt.table(cellText=cell_text, colLabels=table.columns, loc='center')
    plt.axis('off')

    plt.savefig("figures/table_clustering.svg", transparent=True, bbox_inches=0)

show_clusters("vector_aca_t", 3, 25, 3)
show_clusters("vector_aca_t", 7, 25, 3)
# show_clusters("cp", 3, 10, 3)
# show_table(10,"vector_aca_t", 3, 10, 3)