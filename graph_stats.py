import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pygsp as pg
from tqdm import tqdm


def basic(adjacency):
    G = nx.from_numpy_matrix(adjacency)

    print("--- GRAPH STATS ---")
    N = len(G)
    L = G.size()
    print("Number of Nodes: ", N)
    print("Number of Edges: ", L)
    print("Sparsity: ",  L/(N * (N-1) / 2 ))
    deg = [len(nbrs) for n, nbrs in G.adj.items()]
    print("Average Degree: ", np.mean(np.array(deg)))
    giant = max(nx.connected_component_subgraphs(G), key=len)
    print("Number of Connected Components: ", nx.number_connected_components(G))
    print("Size of Largest Connected Component: ", len(giant))
    return G, N, L, deg


def advanced(adjacency, dirname, active_plots=False):
    G, N, L, deg = basic(adjacency)
    plot_degree_distribution(deg, N, dirname, show=active_plots)
    log_plot_degree_distribution(deg, N, dirname, show=active_plots)

    return G, N, L, deg


def allstats(adjaceny, dirname, active_plots=False):
    G, N, L, deg = advanced(adjaceny, dirname, active_plots)
    print("Average Clustering Coefficient: ", nx.average_clustering(G))
    print("Diameter of Graph: ", nx.diameter(G))


def plot_degree_distribution(degree_sequence, N, dirname, show=False):
    fig = plt.figure()
    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    plt.hist(degree_sequence, weights=np.ones_like(degree_sequence) / float(N))
    fname = dirname + "/degree_hist.png"
    plt.savefig(fname)
    if show:
        plt.show()
    plt.close(fig)


def log_plot_degree_distribution(degree_sequence, N, dirname, show=False):
    fig = plt.figure()
    n, bins, patches = plt.hist(degree_sequence, weights=np.ones_like(degree_sequence) / float(N), bins=50)
    y_ax = bins[1:]
    plt.close(fig)

    fig = plt.figure()
    plt.ylabel("p_k")
    plt.xlabel("k")
    plt.title("Degree Distribution Log plot")
    plt.semilogy(y_ax, n, "ro")
    plt.xlabel("k")

    fname = dirname + "/degree_log_hist.png"
    plt.savefig(fname)
    if show:
        plt.show()
    plt.close(fig)


def growth_analysis(adjacency, release_dates, gt_labels, dirname, n_from=50, n_to=1000, every=100, active_plots=False):
    print("Initiating Graph Growth Analysis...")
    start = len(release_dates[np.isnat(release_dates)])

    sorted_idx = np.argsort(release_dates)
    sorted_idx = sorted_idx[start:]
    sorted_gt_labels = gt_labels[sorted_idx]
    sorted_dates = release_dates[sorted_idx]

    adj_temp = adjacency[sorted_idx, :]
    adjacency = adj_temp[:, sorted_idx]

    graph_sizes = []
    average_degree = []
    average_clustering = []
    for i in tqdm(range(n_from, n_to, every)):
        if i > adjacency.shape[0]:
            break
        if active_plots:
            pygsp_graph = pg.graphs.Graph(adjacency[:i + 1, :i + 1], lap_type='normalized')
            pygsp_graph.set_coordinates('spring')  # for visualization
            pygsp_graph.plot_signal(sorted_gt_labels[:i+1], plot_name=str(sorted_dates[i]))

        G = nx.from_numpy_matrix(adjacency[:i + 1, :i + 1])
        deg = [len(nbrs) for n, nbrs in G.adj.items()]
        graph_sizes.append(len(G))
        average_degree.append(np.mean(np.array(deg)))
        average_clustering.append(nx.average_clustering(G))

    fig = plt.figure()
    plt.ylabel("k")
    plt.xlabel("size")
    plt.title("average degree over time")
    plt.plot(graph_sizes, average_degree, 'ro')
    fname = dirname + "/degree_over_time.png"
    plt.savefig(fname)
    if active_plots:
        plt.show()

    fig = plt.figure()
    plt.ylabel("C")
    plt.xlabel("size")
    plt.title("average clustering over time")
    plt.plot(graph_sizes, average_clustering, 'ro')
    fname = dirname + "/clustering_over_time.png"
    plt.savefig(fname)
    if active_plots:
        plt.show()
