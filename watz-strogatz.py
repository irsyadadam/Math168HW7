import pandas
import networkx as nx
import numpy as np

from networkx.generators.random_graphs import watts_strogatz_graph
from networkx import average_shortest_path_length
from tqdm import tqdm
from matplotlib import pyplot as plt


    

def single_realization(n: int, k: int, p: [float]) -> None:
    """ 
    :param n: is the node number of the single graph visualization. n is fixed for varying probabilities
    :param k: is the degree of each node (knn) in the ring lattice. k is also fixed
    :param p: is a list of floats for varying p values
    :return: None
    """
    graphs = []

    #creating the graphs
    for datum in tqdm(p, desc = "Creating Graphs"):
        graphs.append(watts_strogatz_graph(n, k, datum))
    

    #normalizing c value (since only 1 graph, no need for average)
    #cmax is when p = 0
    c_vals = []

    #since the first graph is when p = 0
    c_max = nx.average_clustering(graphs[0])
    for g in tqdm(graphs, desc = "Finding C_val"):
        c_vals.append(nx.average_clustering(g) / c_max)
    
    #finding mean geodesic distance
    #Lmax is also when p = 0
    l = []

    #using l_max when p = 0
    l_max = average_shortest_path_length(graphs[0])

    for g in tqdm(graphs, desc = "Finding L"):
        l.append(average_shortest_path_length(g) / l_max)

    #plotting
    
    plt.scatter(x = p, y = c_vals, s=10, c='b', marker="s", label = "C Val")
    plt.scatter(x = p, y = l, s=10, c='r', marker="o", label='L Val')

    plt.xlabel("Probability")
    plt.ylabel("L Value (L/Lmax) or C Value (C/Cmax")
    plt.legend(loc='upper left')
    plt.title("Single Realization: Nodes = " + str(n) + ", Degree = " + str(k))

    plt.savefig("single_realization.png")




def multi_realization(n: int, k: int, p: [float]) -> None:
    """ 
    generates 15 graphs per datum in p
    :param n: is the node number of the single graph visualization. n is fixed for varying probabilities
    :param k: is the degree of each node (knn) in the ring lattice. k is also fixed
    :param p: is a list of floats for varying p values
    :return: None
    """
    #graphs is a list of lists containing graphs
    graphs = []

    #creating the graphs
    for datum in tqdm(p, desc = "Creating Graphs"):
        subgraphs = []
        for i in range(15):
            subgraphs.append(watts_strogatz_graph(n, k, datum))
        graphs.append(subgraphs)

    #normalizing c value (since only 1 graph, no need for average)
    c_vals = []
    c_max = 0
    #cmax is when p = 0
    for i in graphs[0]:
        c_max += nx.average_clustering(i)
    c_max = c_max / len(graphs[0])

    for g in tqdm(graphs, desc = "Finding C_val"):
        c = 0
        for i in g:
            #add up the global cc
            c += nx.average_clustering(i)
        #divide over length to get average
        c = c / len(g)
        #normalize and append it to list
        c_vals.append(c / c_max)
    
    #finding mean geodesic distance
    #Lmax is also when p = 0
    l_vals = []
    l_max = 0
    #lmax is when p = 0
    for i in graphs[0]:
        #sum the total
        l_max += nx.average_shortest_path_length(i)
    #divide to get average
    l_max = l_max / len(graphs[0])

    for g in tqdm(graphs, desc = "Finding L"):
        l = 0
        for i in g:
            #add up the vals
            l += nx.average_shortest_path_length(i)
        #divide over length to get average
        l = l / len(g)
        #normalize and append it to list
        l_vals.append(l / l_max)
        

    #plotting
    
    plt.scatter(x = p, y = c_vals, s=10, c='b', marker="s", label = "C Val")
    plt.scatter(x = p, y = l_vals, s=10, c='r', marker="o", label='L Val')

    plt.xlabel("Probability")
    plt.ylabel("L Value (L/Lmax) or C Value (C/Cmax")
    plt.legend(loc='upper left')
    plt.title("Multiple Realization: Nodes = " + str(n) + ", Degree = " + str(k))

    plt.savefig("multi_realization.png")




if __name__ == "__main__":
    p = np.arange(0, 0.1, 0.0005)
    multi_realization(100, 10, p)
