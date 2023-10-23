import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# Define a bounding box for your area of interest
place_name = "Your City, Your Country"
#graph = ox.graph_from_place(place_name, network_type="all")

# Convert the graph to a networkx MultiDiGraph
G = nx.MultiDiGraph(loc="graph")
G.add_node(1)
G.add_nodes_from([2, 3])
G.add_nodes_from(range(100, 110))
H = nx.path_graph(10)
G.add_nodes_from(H)

intersection_nodes = []

# Iterate through the nodes in the graph and identify intersection nodes
for node in G.nodes(data=True):
    if len(list(G.edges(node[0], keys=True)) > 2):
        intersection_nodes.append(node)

# Create a map with the base network graph
fig, ax = ox.plot_graph(ox.project_graph(graph), show=False, close=False)

# Highlight intersection nodes on the map
for node in intersection_nodes:
    x, y = node[1]["x"], node[1]["y"]
    plt.scatter(x, y, c='red', s=30, marker='o')

# Show the map with intersection nodes
plt.show()
