from clustering_helpers import *
from figs import plot_styling
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
import statistics
import tensor as t



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
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    xlabel.set_style('italic')
    ylabel.set_style('italic')
    xlabel.set_size(18)
    ylabel.set_size(18)
    plt.subplots_adjust(bottom=0.2, left=0.2, top=0.85)

    if method == "matrix_aca_t":
        methodStr = "Matrix ACA-T"
    else:
        methodStr = "Vector ACA-T type " + str(approx)

    title = str(n_clusters) + " Clusters van " + methodStr + "\nrang " + str(rank)
    name = str(n_clusters) + " Clusters " + methodStr + " rang " + str(rank)
    ax1.set_title(title, fontsize=20)
    ax1.title.set_weight('bold')
    ax1.spines['right'].set_color((.8, .8, .8))
    ax1.spines['top'].set_color((.8, .8, .8))
    for i in range(n_clusters):
        ax1.scatter(clusterXs[i], clusterYs[i], c=colors[i])
    plt.savefig("figures/" + name + ".svg", transparent=True, bbox_inches=0)


def show_table(direction, rows, method, n_clusters, n_fvs, approx):
    labels = cluster(method, n_clusters, direction, n_fvs, approx)
    people, exercises, sensors = t.get_people_exercises_sensors()
    if direction == 'rows':
        data = {"Person": people, "Exercise": exercises, "Cluster": labels}
    elif direction == 'tubes':
        data = {"Sensor": sensors, "Cluster": labels}
    table = pd.DataFrame(data=data)
    plt.figure()

    # table
    plt.subplot(111)

    cell_text = []
    for row in range(rows):
        cell_text.append(table.iloc[row])
    plt.table(cellText=cell_text, colLabels=table.columns, loc='center')
    plt.axis('off')
    name = f"table_clustering_{n_clusters}_clusters_{method}_type{approx}_rank{rank}.svg"
    plt.savefig("figures/" + name, transparent=True, bbox_inches=0)


def cluster_ari(types, k_clusters, direction, min_feature_vectors, delta_feature_vectors, max_feature_vectors, true_labels, calc_data=True, sample_size = 10, cp=False, bar=True, bar_width=5, colors={1: 'firebrick', 2:'cornflowerblue', 3:'greenyellow', 5:'violet', 8:'teal', 10:'indigo', 20:'indigo'}, fig_size=(6.4, 4.8)):
    """
    Compares clusters from cp and vector_aca_t using ari. Will compare vector_aca_t for every type in types.
    Calculates ari-scores using the true labels and
    plots the results in function of the amount of feature vectors.
    :param types: a list of the types of vector_aca_t to use.
    e.g. [1,3,7] will cluster and calculate the ari-scores for vector_aca_t
    using 1 vector per term, 3 vectors per term and 7 vectors per term.
    :param k_clusters: the amount of clusters.
    :param max_feature_vectors: the maximum amount of feature vectors per cluster.
    :param direction: the direction of the feature vectors. Options: 'rows', 'columns', and 'tubes'.
    :param true_labels: the true labels of the clustering
    """
    vector_aca_scores_per_type, vector_aca_stdev_per_type, vector_aca_fvs_per_type, cp_scores, cp_fvs = get_ari_scores(types, k_clusters, direction, min_feature_vectors, delta_feature_vectors, max_feature_vectors, true_labels, calc_data=calc_data, sample_size=sample_size, cp=cp)
    lgd = []
    fig = plt.figure() 
    fig.set_size_inches(fig_size[0], fig_size[1], forward=True)
    if not bar:
        if cp:
            plt.plot(cp_fvs, cp_scores, color=colors["cp"], marker='.', markersize=10, markerfacecolor='white')
            lgd.append("cp")
        for i in range(0, len(types)):
            plt.plot(vector_aca_fvs_per_type[i], vector_aca_scores_per_type[i], color=colors[types[i]], marker='.', markersize=10, markerfacecolor='white')
            lgd.append("type " + str(types[i]))
        plt.legend(lgd)
        plt.ylabel("ARI score")
        plt.xlabel("Aantal feature vectoren")
        plt.title("Clustering in 3 clusters met " + direction + " als feature vectoren")
        ax = plt.gca()
        ax.set_ylim([0, 1])
        name = f"ari({direction},{k_clusters},{types},{cp},range({min_feature_vectors},{max_feature_vectors},{delta_feature_vectors}),{sample_size})_lineplot.svg"
        plt.savefig("figures/" + name, transparent=True, bbox_inches=0)
        plt.show()
    else:
        n = len(types)
        if cp:
            n = 1 + len(types)
            offset = (len(types) - n / 2 + 1 / 2) * bar_width / min(2, n)
            xs = list(map(lambda x: x + offset, cp_fvs))
            plt.bar(xs, cp_scores, color=colors["cp"], width=bar_width / min(2, n))
            lgd.append("cp")
        for i in range(0, len(types)):
            offset = (i - n / 2 + 1 / 2) * bar_width / min(2, n)
            xs = list(map(lambda x: x + offset, vector_aca_fvs_per_type[i]))
            plt.bar(xs, vector_aca_scores_per_type[i], width=bar_width / min(2, n), color=colors[types[i]], yerr=vector_aca_stdev_per_type[i])
            lgd.append("type " + str(types[i]))
        plt.legend(lgd)
        plt.ylabel("ARI score")
        plt.xlabel("Aantal feature vectoren")
        plt.title("Clustering in " + str(k_clusters) + " clusters met " + direction + " als feature vectoren")
        ax = plt.gca()
        ax.set_ylim([0, 1])
        plt.xticks(range(min_feature_vectors, max_feature_vectors + 1, delta_feature_vectors))
        name = f"ari({direction},{k_clusters},{types},{cp},range({min_feature_vectors},{max_feature_vectors},{delta_feature_vectors}),{sample_size})_barplot.svg"
        plt.savefig("figures/" + name, transparent=True, bbox_inches=0)
        plt.show()

def cluster_ari_single_type(k, 
                            k_clusters, 
                            min_feature_vectors, delta_feature_vectors, max_feature_vectors, 
                            calc_data=False, 
                            sample_size=10):


    # rows (has k feature vectors per term)
    data_rows, _, _, _, _ = get_ari_scores([k], 
                                           k_clusters, 
                                          "rows", 
                                          k*min_feature_vectors, k*delta_feature_vectors, k*max_feature_vectors, 
                                          get_overview()["exercise"], 
                                          calc_data, 
                                          sample_size)

    # tubes (has 1 feature vector per term)
    _, _, sensors = t.get_people_exercises_sensors()
    labels = []
    for i in range(len(sensors)):
        labels.append(sensors[i][-1])

    data_tubes, _, _, _, _ = get_ari_scores([k], 
                                            k_clusters, 
                                            "tubes", 
                                            min_feature_vectors, delta_feature_vectors, max_feature_vectors, 
                                            labels, 
                                            calc_data, 
                                            sample_size)
    ys_rows = data_rows[0]
    ys_tubes = data_tubes[0]
    xs = range(k*min_feature_vectors, k*max_feature_vectors, k*delta_feature_vectors)
    xs_minor = range(min_feature_vectors, max_feature_vectors, delta_feature_vectors)

    fig, ax = plt.subplots()

    plt.plot(xs, ys_rows,  color='firebrick',      marker='.', markersize=10, markerfacecolor='white')
    plt.plot(xs, ys_tubes, color='cornflowerblue', marker='.', markersize=10, markerfacecolor='white')

    plot_styling(fig, ax,
                 xticks=xs,
                 xlabel='Aantal feature vectoren',
                 ylabel='ARI-score',
                 title=f'Clustering van type {k} voor rows en tubes')
    ax.set_xticks(xs_minor, fontsize=10, minor=True)

    plt.legend(['rows', 'tubes'])

    plt.show()


def get_ari_scores(types, k_clusters, direction, min_feature_vectors, delta_feature_vectors, max_feature_vectors, true_labels, calc_data=True, sample_size = 10, cp=True):
    cp_scores = []
    cp_fvs = []
    vector_aca_scores_per_type = []
    vector_aca_fvs_per_type = []
    vector_aca_stdev_per_type = []
    for i in range(0, len(types)):
        vector_aca_scores_per_type.append([])
        vector_aca_fvs_per_type.append([])
        vector_aca_stdev_per_type.append([])
    if calc_data:
        for i in range(min_feature_vectors, max_feature_vectors + 1, delta_feature_vectors):
            if cp:
                labels = cluster("cp", k_clusters, direction, i)
                ari = adjusted_rand_score(true_labels, labels)
                cp_scores.append(ari)
                cp_fvs.append(i)
            for ty in range(len(types)):
                aris = []
                for j in range(sample_size):
                    labels = cluster("vector_aca_t", k_clusters, direction, i, types[ty])
                    ari = adjusted_rand_score(true_labels, labels)
                    aris.append(ari)
                vector_aca_scores_per_type[ty].append(statistics.mean(aris))
                vector_aca_fvs_per_type[ty].append(i)
                vector_aca_stdev_per_type[ty].append(statistics.stdev(aris))
        if cp:
            name = f"saved_fig_data/data(cp,{k_clusters},{direction}_range({min_feature_vectors},{max_feature_vectors},{delta_feature_vectors})"
            np.save(name + "_scores", cp_scores)
            np.save(name + "_fvs", cp_fvs)
        for i in range(0, len(types)):
            name = f"saved_fig_data/data(type={types[i]},{k_clusters},{direction},range({min_feature_vectors}, {max_feature_vectors}, {delta_feature_vectors}),sample_size={sample_size})"
            np.save(name + "_scores", vector_aca_scores_per_type[i])
            np.save(name + "_fvs", vector_aca_fvs_per_type[i])
            np.save(name + "_stdev", vector_aca_stdev_per_type[i])
    else:
        if cp:
            name = f"saved_fig_data/data(cp,{k_clusters},{direction}_range({min_feature_vectors},{max_feature_vectors},{delta_feature_vectors})"
            cp_scores = np.load(name + "_scores.npy")
            cp_fvs = np.load(name + "_fvs.npy")
        for i in range(0, len(types)):
            name = f"saved_fig_data/data(type={types[i]},{k_clusters},{direction},range({min_feature_vectors}, {max_feature_vectors}, {delta_feature_vectors}),sample_size={sample_size})"
            vector_aca_scores_per_type[i] = np.load(name + "_scores.npy")
            vector_aca_fvs_per_type[i] = np.load(name + "_fvs.npy")
            vector_aca_stdev_per_type[i] = np.load(name + "_stdev.npy")

    return vector_aca_scores_per_type, vector_aca_stdev_per_type, vector_aca_fvs_per_type, cp_scores, cp_fvs



# ex = get_overview()["exercise"]
# et = get_overview()["execution_type"]
# tl = list(map(str, list(zip(ex, et))))
# tl = get_overview()["exercise"]
# cluster_ari([1, 5, 10], 3, 'rows', 100, 10, 150, tl, 10, True, True, 4)

# show_clusters("vector_aca_t", 3, 25, 3)
# show_clusters("vector_aca_t", 7, 25, 3)
# show_clusters("cp", 3, 10, 3)
# show_table(10,"vector_aca_t", 3, 10, 3)