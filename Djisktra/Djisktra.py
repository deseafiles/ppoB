from tabulate import tabulate
import networkx as nx
import matplotlib.pyplot as plt


class Dijkstra:
    def __init__(self, graph):
        self.graph = graph
        self.distance = {}
        self.path = {}
        self.visited = set()
        self.start = ""
        self.end = ""

    def initiate_distance(self, start, end):
        for node in self.graph:
            self.distance[node] = float("inf")
        self.distance[start] = 0
        self.start = start
        self.end = end

    def find_shortest_path(self, notVisited):
        shortest_node = None
        shortest_distance = float("inf")

        for node in notVisited:
            if self.distance[node] < shortest_distance:
                shortest_node = node
                shortest_distance = self.distance[node]

        return shortest_node

    def visualize_graph(self):
        G = nx.DiGraph()
        G.add_nodes_from(self.graph.keys())

        for node, neighbors in self.graph.items():
            for neighbor, weight in neighbors.items():
                G.add_edge(node, neighbor, weight=weight)

        pos = nx.spring_layout(G)
        labels = nx.get_edge_attributes(G, 'weight')

        edge_colors = ['black' for _ in G.edges()]

        if self.distance[self.end] < float("inf"):
            pathList = [self.end]
            i = 0
            while self.start not in pathList:
                pathList.append(self.path[pathList[i]])
                i += 1

            pathList.reverse()

            for i in range(len(pathList) - 1):
                if G.has_edge(pathList[i], pathList[i+1]):
                    edge_colors[list(G.edges()).index(
                        (pathList[i], pathList[i+1]))] = 'red'

        nx.draw(G, pos, with_labels=True, node_size=500, edge_color=edge_colors,
                node_color='skyblue', font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title("Graph Visualization")
        plt.show()

    def route(self, start, end):
        self.initiate_distance(start, end)
        notVisited = [node for node in self.distance]
        shortestNode = self.find_shortest_path(notVisited)

        while notVisited:
            currentShortestNodeDistance = self.distance[shortestNode]

            if currentShortestNodeDistance == float("inf"):
                break

            for neighbor, weight in self.graph[shortestNode].items():
                distance_to_neighbor = currentShortestNodeDistance + weight
                if distance_to_neighbor < self.distance[neighbor]:
                    self.distance[neighbor] = distance_to_neighbor
                    self.path[neighbor] = shortestNode

            self.visited.add(shortestNode)  # Tandai node yang telah dikunjungi
            notVisited.remove(shortestNode)
            shortestNode = self.find_shortest_path(notVisited)

        if self.distance[self.end] < float("inf"):
            pathList = [self.end]
            i = 0
            while self.start not in pathList:
                pathList.append(self.path[pathList[i]])
                i += 1

            pathList.reverse()
            print("Shortest Path:", " -> ".join(pathList))
        self.display_all_shortest_paths()
        self.visualize_graph()

    def display_result(self):
        # Prepare the headers for the table
        nodes = list(self.graph.keys())
        edges_table = [[""] + nodes]  # Header row

        # Fill the edges table with weights
        for node in nodes:
            row = [node]
            for neighbor in nodes:
                if neighbor in self.graph[node]:
                    row.append(self.graph[node][neighbor])
                else:
                    row.append("-")  # No edge between nodes
            edges_table.append(row)

        print("Edges with Weights:")
        print(tabulate(edges_table, headers="firstrow", tablefmt="grid"))

    def find_all_paths(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if start not in self.graph:
            return []
        paths = []
        for node in self.graph[start]:
            if node not in path:
                new_paths = self.find_all_paths(node, end, path)
                for new_path in new_paths:
                    paths.append(new_path)
        return paths

    def display_all_shortest_paths(self):
        all_paths = self.find_all_paths(self.start, self.end)
        print("All Shortest Paths:")
        for idx, path in enumerate(all_paths):
            if len(path) > 1:
                print(f"Path {idx + 1}: {' -> '.join(path)}")
graph = {
    'V1': {'V2': 6, 'V3': 6, 'V4': 2},
    'V2': {'V3': 3, 'V5': 7},
    'V3': {'V6': 2, 'V7': 1},
    'V4': {'V3': 2, 'V7': 5},
    'V5': {'V6': 2, 'V8': 3},
    'V6': {'V8': 3},
    'V7': {'V8': 3},
    'V8': {},
}

# Buat objek dari kelas Dijkstra
dijkstra = Dijkstra(graph)

# Tentukan titik awal dan titik akhir untuk pencarian lintasan terpendek
start_point = 'V1'
end_point = 'V8'

# Cari lintasan terpendek dari titik awal ke titik akhir
dijkstra.route(start_point, end_point)
