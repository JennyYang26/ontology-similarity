import os
import json
import math
import webbrowser
import networkx as nx
import numpy as np
from tqdm import tqdm
from node2vec import Node2Vec
from pyvis.network import Network
from grakel import Graph as GkGraph
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = "data_json"
NUM_WORKERS = 8

# Register and use a specific browser (e.g., Chrome)
webbrowser.register('chrome', None, webbrowser.GenericBrowser('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'))

# Optional WL hashing imports
try:
    from grakel import GraphKernel
    USE_WL_HASH = True
except ImportError:
    USE_WL_HASH = False

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

# ------------------------------------------------------------
# 1. utility: NetworkX DiGraph -> GraKeL Graph (with labels)
# ------------------------------------------------------------
def nx_to_grakel_ego(G, center, radius=2):
    """Return an r-hop ego subgraph around <center> in GraKeL format."""
    H = nx.ego_graph(G, center, radius=radius, undirected=False)
    mapping = {n: i for i, n in enumerate(H.nodes())}
    invmap = {i: n for n, i in mapping.items()}

    g_edge_list = [(mapping[u], mapping[v]) for u, v in H.edges()]
    g_labels = {
        mapping[n]: str(G.nodes[n].get("label", n.split("/")[-1]))
        for n in H.nodes()
    }

    print("✅ Edge list format (first 5):", g_edge_list[:5])
    print("✅ Node label format (first 5):", list(g_labels.items())[:5])

    # ❗ Handle empty edge case (GraKeL needs structure)
    if not g_edge_list or len(g_labels) < 2:
        print("⚠️ Skipping: GraKeL cannot process graphs with no edges or only one node.")
        return None, invmap

    try:
        gk_graph = GkGraph(initialization_object=(g_edge_list, g_labels))
        return gk_graph, invmap
    except Exception as e:
        print("❌ GraKeL graph creation failed:", e)
        return None, invmap

# ------------------------------------------------------------
# 2. WL subtree feature extractor
# ------------------------------------------------------------
def wl_feature(G, node, wl_kernel, radius=2):
    gk_graph, _ = nx_to_grakel_ego(G, node, radius)

    if gk_graph is None:
        print(f"⚠️ WL skipped for [{node}] — fallback to zero vector")
        return np.zeros((1, 10))  # Dummy vector, size should match rest of pipeline

    try:
        φ = wl_kernel.fit_transform([gk_graph])[0]   # 1‑D sparse vec
        return φ
    except Exception as e:
        print(f"❌ WL kernel error for [{node}]:", e)
        return np.zeros((1, 10))  # Also fallback here

# ------------------------------------------------------------
# 3. Node2Vec embedding extractor (once per full graph)
# ------------------------------------------------------------
def train_node2vec_embeddings(G, dim=64, walk_len=80, num_walks=10, workers=4):
    node2vec = Node2Vec(G, dimensions=dim, walk_length=walk_len,
                        num_walks=num_walks, workers=workers, quiet=True)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return model

# ------------------------------------------------------------
# 4. Fusion similarity
# ------------------------------------------------------------
class WLEmbedFusion:
    def __init__(self, G, wl_h=2, n2v_dim=64):
        self.G = G
        # build WL kernel object once
        self.wl_kernel = WeisfeilerLehman(n_iter=wl_h, normalize=True)

        # train Node2Vec once
        self.n2v_model = train_node2vec_embeddings(G, dim=n2v_dim)

    def node_vector(self, node):
        # WL sparse → dense numpy
        wl_vec = wl_feature(self.G, node, self.wl_kernel).flatten()
        # Node2Vec dense
        n2v_vec = np.array(self.n2v_model.wv[str(node)]).flatten()
        # concat and L2‑normalise
        fused = np.hstack([wl_vec, n2v_vec])
        norm  = np.linalg.norm(fused)
        return fused / norm if norm else fused

    # pairwise similarity (cosine)
    def similarity(self, node_a, node_b):
        vec_a = self.node_vector(node_a)
        vec_b = self.node_vector(node_b)
        return float(cosine_similarity([vec_a], [vec_b])[0, 0])

# ------------------------------------------------------------
# 5. convenience: bulk search like your HSS version
# ------------------------------------------------------------
def fusion_similarity_search(G, target_iri, threshold=0.7,
                             wl_h=2, n2v_dim=64):
    fusion = WLEmbedFusion(G, wl_h, n2v_dim)
    target_vec = fusion.node_vector(target_iri)

    results = []
    for iri in G.nodes:
        if iri == target_iri:
            continue
        vec = fusion.node_vector(iri)
        score = float(cosine_similarity([target_vec], [vec])[0, 0])
        if score >= threshold:
            lbl = G.nodes[iri].get("label", iri.split("/")[-1])
            results.append((iri, lbl, score))

    results.sort(key=lambda x: x[2], reverse=True)
    return results


def find_and_visualize_similar_nodes_fusion(
        G, 
        target_iri, 
        threshold=0.7, 
        wl_h=2, 
        n2v_dim=64, 
        output_file="graphs/fusion_similarity_group.html"):
    """
    1. Run WL+Node2Vec similarity search
    2. Print results
    3. Visualize results using PyVis
    """
    if target_iri not in G:
        print("❌ Target IRI not in graph.")
        return []

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    target_label = G.nodes[target_iri].get("label", target_iri.split("/")[-1])

    print("\n" + "=" * 80)
    print("📌 Starting fusion similarity search")
    print("-" * 60)
    print(f"📊 Target: [{target_label}] ({target_iri})")

    hits = fusion_similarity_search(G, target_iri, threshold, wl_h, n2v_dim)
    
    if not hits:
        print(f"⚠️ No similar nodes with similarity ≥ {threshold}")
        return []

    print("\nRanked similar nodes:")
    for rank, (iri, lbl, sc) in enumerate(hits, 1):
        print(f"{rank:>3}. {lbl:<45}  {sc:.4f}  |  {iri}")
    print("=" * 80 + "\n")

    # -----------------------------------
    # Collect relevant nodes for subgraph
    # -----------------------------------
    sub_nodes = {target_iri}
    for iri, _, _ in hits:
        sub_nodes.add(iri)
        sub_nodes.update(nx.ancestors(G, iri))
    sub_nodes.update(nx.ancestors(G, target_iri))
    H = G.subgraph(sub_nodes).copy()

    # ---------------------------
    # Draw PyVis interactive HTML
    # ---------------------------
    net = Network(notebook=False, directed=True, height="800px", width="100%")
    net.barnes_hut()

    max_score = max(sc for _, _, sc in hits)
    score_dict = {iri: sc for iri, _, sc in hits}
    label_dict = {iri: lbl for iri, lbl, _ in hits}

    for node in H.nodes:
        label = label_dict.get(node, G.nodes[node].get("label", node.split("/")[-1]))
        if node == target_iri:
            net.add_node(node, label=label, color="orange", size=35)
        elif node in score_dict:
            size = 20 + 30 * (score_dict[node] / max_score)
            net.add_node(node, label=label, color="deepskyblue", size=size)
        else:
            net.add_node(node, label=label, color="#DDDDDD", size=10)

    for src, dst in H.edges:
        net.add_edge(src, dst)

    net.write_html(output_file)
    webbrowser.get('chrome').open(output_file)
    print(f"✅ Graph saved to {output_file}")
    return hits


# ------------------ Run Example ------------------

if __name__ == "__main__":
    # Build the ontology graph
    G = build_ontology_graph_parallel(DATA_DIR, start_index=0, end_index=1000)

    target = "http://purl.obolibrary.org/obo/NCBITaxon_1322365"

    # Run fusion-based similarity search and visualization
    hits = find_and_visualize_similar_nodes_fusion(
        G,
        target_iri=target,
        threshold=0.4,               # adjust the similarity threshold
        wl_h=2,                      # height for WL kernel
        n2v_dim=64,                  # embedding dimension
        output_file="graphs/wl_node2vec_similarity.html"
    )