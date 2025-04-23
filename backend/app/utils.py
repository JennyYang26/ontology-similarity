import networkx as nx

def get_related_nodes(G, iri):
    ancestors = set()
    try:
        ancestors |= set(nx.ancestors(G, iri))
    except:
        pass
    try:
        ancestors |= set(nx.descendants(G, iri))
    except:
        pass
    return ancestors

def log(msg):
    print(f"[LOG] {msg}")
