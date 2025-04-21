import os
import json
import math
import webbrowser
import networkx as nx
from tqdm import tqdm
from pyvis.network import Network
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = "data_json"
NUM_WORKERS = 8

# Register and use a specific browser (e.g., Chrome)
webbrowser.register('chrome', None, webbrowser.GenericBrowser('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'))

def process_file(filepath):
    edges = []
    try:
        with open(filepath, 'r') as f:
            items = json.load(f)
        for item in items:
            iri = item.get("iri")
            if not iri:
                continue
            for key in ["hierarchicalParent", "hierarchicalAncestor"]:
                if item.get(f"has{key[0].upper()}{key[1:]}s") and key in item:
                    parents = item[key]
                    if isinstance(parents, str):
                        parents = [parents]
                    for parent in parents:
                        if isinstance(parent, dict):
                            parent_iri = parent.get("value") or parent.get("iri")
                        else:
                            parent_iri = parent
                        if isinstance(parent_iri, str):
                            edges.append((parent_iri, iri))  # parent → child
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return edges

def build_ontology_graph_parallel(data_dir, start_index=0, end_index=None):
    print("\n" + "=" * 80)
    print("📌 Starting build_ontology_graph_parallel")
    print("-" * 60)
    G = nx.DiGraph()
    json_files = sorted(
        [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[1])
    )
    print(f"Loading {len(json_files)} files: {json_files[:3]} ... {json_files[-3:]}\n")

    if end_index is None:
        end_index = len(json_files)
    json_files = json_files[start_index:end_index]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_file, path): path for path in json_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building graph"):
            edges = future.result()
            G.add_edges_from(edges)

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges\n")
    print("Sample nodes:", list(G.nodes)[:5])
    print("=" * 80 + "\n")
    return G

def compute_hss_from_graph(G, iri1, iri2, alpha=0.5):
    print("\n" + "=" * 80)
    print("📌 Starting compute_hss_from_graph")
    print("-" * 60)
    if iri1 not in G:
        print(f"{iri1} not in graph nodes.\n")
        return 0.0
    if iri2 not in G:
        print(f"{iri2} not in graph nodes.\n")
        return 0.0
    
    ancestors_1 = nx.ancestors(G, iri1)
    ancestors_2 = nx.ancestors(G, iri2)
    print(f"🔍 Ancestors of {iri1.split('/')[-1]}: {len(ancestors_1)}")
    print(ancestors_1)
    print('\n')
    print(f"🔍 Ancestors of {iri2.split('/')[-1]}: {len(ancestors_2)}")
    print(ancestors_2)
    print('\n')

    if not ancestors_1 and not ancestors_2:
        print("⚠️ No ancestors found for either node.\n")
        return 0.0
    intersection = ancestors_1 & ancestors_2
    union = ancestors_1 | ancestors_2
    ancestor_sim = len(intersection) / len(union) if union else 0.0
    print(f"🪢 Intersection count: {len(intersection)}")
    print(f"🔗 Union count: {len(union)}")
    print(f"🧮 Ancestor Jaccard Similarity: {ancestor_sim:.4f}\n")

    depth_1 = len(ancestors_1)
    depth_2 = len(ancestors_2)
    depth_penalty = math.exp(-alpha * abs(depth_1 - depth_2))
    print(f"📏 Depth of {iri1.split('/')[-1]}: {depth_1}")
    print(f"📏 Depth of {iri2.split('/')[-1]}: {depth_2}")
    print(f"📉 Depth Penalty (α={alpha}): {depth_penalty:.4f}")

    hss_score = round(depth_penalty * ancestor_sim, 4)
    print("=" * 80 + "\n")
    return hss_score

def visualize_iri_pair(G, iri1, iri2, output_file="pair_subgraph.html"):
    if iri1 not in G or iri2 not in G:
        print("One or both nodes not in graph.")
        return

    # Get ancestors + the two IRIs
    nodes = nx.ancestors(G, iri1) | nx.ancestors(G, iri2) | {iri1, iri2}
    subgraph = G.subgraph(nodes)
    # subgraph = G.subgraph(nodes).copy()

    # Initialize Pyvis network
    net = Network(notebook=False, height="800px", width="100%", directed=True)
    net.barnes_hut()

    for node in subgraph.nodes:
        label = node.split("/")[-1]
        color = "orange" if node in [iri1, iri2] else "lightblue"
        net.add_node(node, label=label, title=node, color=color)

    for source, target in subgraph.edges:
        net.add_edge(source, target)

    # Write to HTML and open in browser
    net.write_html(output_file)
    webbrowser.get('chrome').open(output_file)
    print(f"\nInteractive graph saved to {output_file}\n")

def visualize_two_nodes(G, iri1, iri2, output_html="compare.html"):
    if iri1 not in G or iri2 not in G:
        print("One or both IRIs not in the graph.")
        return

    nodes = nx.ancestors(G, iri1) | nx.ancestors(G, iri2) | {iri1, iri2}
    subgraph = G.subgraph(nodes).copy()

    net = Network(height="800px", width="100%", directed=True)
    net.from_nx(subgraph)

    for node in net.nodes:
        label = node["id"].split("/")[-1]
        node["label"] = label
        if node["id"] == iri1:
            node["color"] = "red"
        elif node["id"] == iri2:
            node["color"] = "blue"

    net.show(output_html)
    print(f"\nComparison graph saved to {output_html}\n")

# ------------------ Run Example ------------------

if __name__ == "__main__":
    # You can change the range below to speed up development
    G = build_ontology_graph_parallel(DATA_DIR, start_index=0, end_index=400)

    iri_a = "http://purl.obolibrary.org/obo/NCBITaxon_1322365"
    iri_b = "http://purl.obolibrary.org/obo/NCBITaxon_1322366"

    score = compute_hss_from_graph(G, iri_a, iri_b)
    print("\n" + "=" * 80)
    print("📌 Final score")
    print("-" * 60)
    print(f"HSS score between:\n - {iri_a}\n - {iri_b}\n → {score}")
    print("=" * 80 + "\n")

    visualize_iri_pair(G, iri_a, iri_b)
    # visualize_two_nodes(G, iri_a, iri_b, output_html="compare.html")
