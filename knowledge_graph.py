# knowledge_graph.py

import networkx as nx
import json

class GraphBuilder:
    def __init__(self):
        """
        Initializes an empty Directed Graph (DiGraph).
        A directed graph means the relationship has a specific direction 
        (e.g., 'Sundar Pichai' -> WORKS_AT -> 'Google' is different than Google working at Sundar).
        """
        self.graph = nx.DiGraph()

    def add_entity_node(self, entity_word, entity_type):
        """
        Adds a single entity to the graph as a node.
        NetworkX automatically ignores it if the node already exists, preventing duplicates.
        """
        self.graph.add_node(entity_word, label=entity_type)

    def add_relationship_edge(self, source_entity, relation_type, target_entity):
        """
        Draws a connecting line (edge) between two entities.
        If the entities aren't in the graph yet, NetworkX will add them automatically.
        """
        self.graph.add_edge(source_entity, target_entity, relationship=relation_type)

    def get_graph_summary(self):
        """Returns a quick count of the graph's size."""
        nodes = self.graph.number_of_nodes()
        edges = self.graph.number_of_edges()
        return f"Knowledge Graph contains {nodes} entities and {edges} relationships."

    def export_to_json(self, filepath):
        """
        Exports the graph mathematical structure to a JSON file.
        This allows other programs or web interfaces to read and visualize the graph later.
        """
        # node_link_data converts the complex Python graph object into a standard dictionary
        graph_data = nx.node_link_data(self.graph)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=4)
            
        print(f"Graph successfully exported to {filepath}")

# --- Example of how the class operates ---
if __name__ == "__main__":
    kg = GraphBuilder()
    
    # Simulating data extracted from model.py
    kg.add_entity_node("Linux", "PRODUCT")
    kg.add_entity_node("Linus Torvalds", "PERSON")
    kg.add_entity_node("Helsinki", "LOCATION")
    
    # Defining how they connect
    kg.add_relationship_edge("Linus Torvalds", "CREATED", "Linux")
    kg.add_relationship_edge("Linus Torvalds", "LIVES_IN", "Helsinki")
    
    print(kg.get_graph_summary())
    kg.export_to_json("test_graph.json")
