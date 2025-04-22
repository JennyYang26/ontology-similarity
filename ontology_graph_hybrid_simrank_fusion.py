import os
import json
import math
import webbrowser
import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = "data_json"
NUM_WORKERS = 8

# Register and use a specific browser (e.g., Chrome)
webbrowser.register('chrome', None, webbrowser.GenericBrowser('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'))

def process_file(filepath):
    edges = []
    attrs = {}
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

            # ---------- NEW: save label ----------------------------
            lbl = None
            if isinstance(item.get("label"), dict):
                lbl = item["label"].get("value")
            if lbl:
                attrs[iri] = {"label": lbl}
            
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
    return edges, attrs

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
        for f in tqdm(as_completed(futures), total=len(futures), desc="Building graph"):
            edges, attrs = f.result()
            G.add_edges_from(edges)
            for iri, d in attrs.items():
                G.add_node(iri)              # ensure node exists
                G.nodes[iri].update(d)       # add/merge attributes

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges\n")
    print("Sample nodes:", list(G.nodes)[:5])
    print("=" * 80 + "\n")
    return G

def jaccard_ancestors(G, iri1, iri2):
    try:
        set1 = set(nx.ancestors(G, iri1))
        set2 = set(nx.ancestors(G, iri2))
        return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
    except:
        return 0

def compute_semantic_sim(labels):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(labels)
    return cosine_similarity(tfidf_matrix)

def hybrid_simrank_fusion(G, target_iri, threshold=0.5, alpha=0.6):
    if target_iri not in G:
        print("❌ Target not in graph")
        return []

    nodes = list(G.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}
    labels = [G.nodes[n].get("label", n.split("/")[-1]) for n in nodes]
    target_label = G.nodes[target_iri].get("label", target_iri.split("/")[-1])
    semantic_sim = compute_semantic_sim(labels)

    results = []
    for other in nodes:
        if other == target_iri:
            continue

        jaccard = jaccard_ancestors(G, target_iri, other)
        semantic = semantic_sim[idx_map[target_iri], idx_map[other]]

        sim_score = alpha * jaccard + (1 - alpha) * semantic
        if sim_score >= threshold:
            results.append((other, G.nodes[other].get("label", other.split("/")[-1]), sim_score))
    
    if not results:
        print(f"No nodes with score ≥ {threshold}")
        return []

    print("\n" + "=" * 80)
    print("📌 Starting find_and_visualize_hybrid_simrank_similar_nodes")
    print("-" * 60)
    print(f"📊 Similar to [{target_label}] ({target_iri})  (≥ {threshold})")
    for rank, (iri, lbl, sc) in enumerate(results, 1):
        print(f"{rank:>3}. {lbl:<45}  {sc:.4f}  |  {iri}")
    print("=" * 70 + "\n")

    results.sort(key=lambda x: x[2], reverse=True)
    return results

def visualize_results(G, target_iri, hits, output_file="graphs/hybrid_fusion_vis.html"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    net = Network(notebook=False, directed=True, height="800px", width="100%")
    net.barnes_hut()

    target_label = G.nodes[target_iri].get("label", target_iri.split("/")[-1])
    max_score = max([sc for _, _, sc in hits]) if hits else 1
    score_dict = {iri: sc for iri, _, sc in hits}
    label_dict = {iri: lbl for iri, lbl, _ in hits}

    sub_nodes = {target_iri}
    for iri, _, _ in hits:
        sub_nodes.add(iri)
        sub_nodes.update(nx.ancestors(G, iri))
    sub_nodes.update(nx.ancestors(G, target_iri))
    H = G.subgraph(sub_nodes).copy()

    for node in H.nodes:
        label = label_dict.get(node, G.nodes[node].get("label", node.split("/")[-1]))
        if node == target_iri:
            net.add_node(node, label=label, color="orange", size=35)
        elif node in score_dict:
            net.add_node(node, label=label, color="deepskyblue", size=20 + 30 * (score_dict[node] / max_score))
        else:
            net.add_node(node, label=label, color="#DDDDDD", size=10)

    for u, v in H.edges:
        net.add_edge(u, v)

    net.write_html(output_file)
    webbrowser.get('chrome').open(output_file)
    print(f"✅ Visualization saved to {output_file}")

if __name__ == "__main__":
    G = build_ontology_graph_parallel(DATA_DIR, start_index=0, end_index=400)
    target = "http://purl.obolibrary.org/obo/NCBITaxon_1322365"
    hits = hybrid_simrank_fusion(G, target, threshold=0.4, alpha=0.6)
    visualize_results(G, target, hits)