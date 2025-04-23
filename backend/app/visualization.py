import os
import webbrowser
import networkx as nx
from pyvis.network import Network

# Optional: Adjust browser path if needed
webbrowser.register('chrome', None, webbrowser.GenericBrowser('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'))

def visualize_similarity_subgraph(G, target_iri, hits, output_file="graphs/similarity_group.html"):
    """
    hits: list of tuples (iri, label, score)
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    sub_nodes = {target_iri}
    for iri, _, _ in hits:
        sub_nodes.add(iri)
        sub_nodes.update(nx.ancestors(G, iri))          # ancestors of each hit
    sub_nodes.update(nx.ancestors(G, target_iri))       # ancestors of target

    H = G.subgraph(sub_nodes).copy()

    # --------------------------- #
    #   build the PyVis network   #
    # --------------------------- #
    net = Network(notebook=False, directed=True,
                  height="800px", width="100%")
    net.barnes_hut()

    max_score = max(sc for _, _, sc in hits)  # for node‑size scaling
    # quick lookup for colour/size
    score_dict = {iri: sc for iri, _, sc in hits}
    label_dict = {iri: lbl for iri, lbl, _ in hits}

    for node in H.nodes:
        base_label = label_dict.get(node, node.split("/")[-1])
        if node == target_iri:          # target node
            net.add_node(node, label=base_label, color="orange", size=35)
        elif node in score_dict:        # hit node
            size = 20 + 30 * (score_dict[node] / max_score)
            net.add_node(node, label=base_label, color="deepskyblue", size=size)
        else:                           # shared ancestor
            net.add_node(node, label=base_label, color="#DDDDDD", size=10)

    for src, dst in H.edges:
        net.add_edge(src, dst)

    net.write_html(output_file)
    webbrowser.get('chrome').open(output_file)
    print(f"✅ Combined graph saved to {output_file}")
    print("=" * 80)

