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
            iri = item["iri"]
            is_obsolete = item.get("isObsolete", True)
            has_hierarchical_parents = item.get("hasHierarchicalParents", False)

            if is_obsolete or not has_hierarchical_parents:
                continue

            # 1. Get hierarchicalParent (should be one or more direct parents)
            h_parents = item.get("hierarchicalParent", [])
            if isinstance(h_parents, str):
                h_parents = [h_parents]
            for parent in h_parents:
                if isinstance(parent, dict):
                    parent_iri = parent.get("value") or parent.get("iri")
                else:
                    parent_iri = parent
                if parent_iri:
                    edges.append((parent_iri, iri))  # parent → child
            
            # 2. Get hierarchicalAncestor (might be higher-level ancestors)
            h_ancestors = item.get("hierarchicalAncestor", [])
            if isinstance(h_ancestors, str):
                h_ancestors = [h_ancestors]
            for ancestor in h_ancestors:
                if isinstance(ancestor, dict):
                    ancestor_iri = ancestor.get("value") or ancestor.get("iri")
                else:
                    ancestor_iri = ancestor
                if ancestor_iri:
                    edges.append((ancestor_iri, iri))  # ancestor → current node
            
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

def get_related_nodes(iri):
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
    
    set1 = get_related_nodes(iri1)
    set2 = get_related_nodes(iri2)
    print(f"🔍 Ancestors of {iri1.split('/')[-1]}: {len(set1)}")
    print(set1)
    print('\n')
    print(f"🔍 Ancestors of {iri2.split('/')[-1]}: {len(set2)}")
    print(set2)
    print('\n')

    if not set1 and not set2:
        return 0.0

    intersection = set1 & set2
    union = set1 | set2
    structure_sim = len(intersection) / len(union) if union else 0.0
    print(f"🪢 Intersection count: {len(intersection)}")
    print(f"🔗 Union count: {len(union)}")
    print(f"🧮 Ancestor Jaccard Similarity: {structure_sim:.4f}\n")

    depth_1 = len(set1)
    depth_2 = len(set2)
    depth_penalty = math.exp(-alpha * abs(depth_1 - depth_2))
    print(f"📏 Depth of {iri1.split('/')[-1]}: {depth_1}")
    print(f"📏 Depth of {iri2.split('/')[-1]}: {depth_2}")
    print(f"📉 Depth Penalty (α={alpha}): {depth_penalty:.4f}")
    print("=" * 80 + "\n")

    return round(depth_penalty * structure_sim, 4)

def find_and_visualize_similar_nodes(
    G,
    target_iri: str,
    threshold: float = 0.5,
    alpha: float = 0.5,
    output_file: str = "graphs/similarity_group.html",
):
    """
    1. scans every node in G
    2. keeps those whose full‑hierarchy HSS ≥ threshold
    3. prints a ranked table
    4. draws ONE interactive PyVis graph containing the target,
       all similar nodes, and their shared ancestors
    """
    if target_iri not in G:
        print("❌ Target IRI not in graph.")
        return []
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # -------------------------------------------------- #
    # 1. find all nodes that satisfy the similarity cut  #
    # -------------------------------------------------- #
    hits = []
    for iri in tqdm(G.nodes, desc="Scanning graph"):
        if iri == target_iri:
            continue
        score = compute_hss_from_graph(G, target_iri, iri, alpha=alpha)
        if score >= threshold:
            hits.append((iri, score))

    hits.sort(key=lambda x: x[1], reverse=True)

    if not hits:
        print(f"No nodes with score ≥ {threshold}")
        return []

    # ------------------------------ #
    # 2. print a nice ranked report  #
    # ------------------------------ #
    print("\n" + "=" * 80)
    print("📌 Starting find_and_visualize_similar_nodes")
    print("-" * 60)
    print(f"📊 Nodes similar to {target_iri} (threshold = {threshold})")
    for rank, (iri, sc) in enumerate(hits, 1):
        print(f"{rank:>3}. {iri:<60} {sc:.4f}")
    print("=" * 70 + "\n")

    # ---------------------------------------- #
    # 3. collect nodes for ONE combined graph  #
    # ---------------------------------------- #
    sub_nodes = {target_iri}
    for iri, _ in hits:
        sub_nodes.add(iri)
        sub_nodes.update(nx.ancestors(G, iri))          # ancestors of each hit
    sub_nodes.update(nx.ancestors(G, target_iri))       # ancestors of target

    H = G.subgraph(sub_nodes).copy()

    # --------------------------- #
    # 4. build the PyVis network  #
    # --------------------------- #
    net = Network(notebook=False, directed=True,
                  height="800px", width="100%")
    net.barnes_hut()

    max_score = max(sc for _, sc in hits)  # for node‑size scaling

    for node in H.nodes:
        label = node.split("/")[-1]
        if node == target_iri:
            net.add_node(node, label=label, color="orange", size=35)
        elif node in dict(hits):
            size = 20 + 30 * (dict(hits)[node] / max_score)
            net.add_node(node, label=label, color="deepskyblue", size=size)
        else:
            net.add_node(node, label=label, color="#DDDDDD", size=10)

    for src, dst in H.edges:
        net.add_edge(src, dst)

    net.write_html(output_file)
    webbrowser.get('chrome').open(output_file)
    print(f"✅ Combined graph saved to {output_file}")
    print("=" * 80)

    return hits

# ------------------ Run Example ------------------

if __name__ == "__main__":
    # You can change the range below to speed up development
    G = build_ontology_graph_parallel(DATA_DIR, start_index=0, end_index=400)

    target   = "http://purl.obolibrary.org/obo/NCBITaxon_1322365"
    hits     = find_and_visualize_similar_nodes(
        G,
        target_iri = target,
        threshold  = 0.75,   # adjust as you like
        alpha      = 0.5,
        output_file="graphs/similarity_group.html",
    )