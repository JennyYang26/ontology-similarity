import math
import networkx as nx
from tqdm import tqdm
from app.utils import get_related_nodes
from app.visualization import visualize_similarity_subgraph

def compute_hss_from_graph(G, iri1, iri2, alpha=0.5):
    if iri1 not in G or iri2 not in G:
        return 0.0

    set1 = get_related_nodes(G, iri1)
    set2 = get_related_nodes(G, iri2)

    if not set1 and not set2:
        return 0.0

    intersection = set1 & set2
    union = set1 | set2
    structure_sim = len(intersection) / len(union) if union else 0.0

    depth_penalty = math.exp(-alpha * abs(len(set1) - len(set2)))
    return round(depth_penalty * structure_sim, 4)

def hss_findSimilarItems(G, target_iri: str, threshold=0.25, alpha=0.5, output_file=None):
    if target_iri not in G:
        return []

    hits = []
    for iri in tqdm(G.nodes, desc="Scanning with HSS"):
        if iri == target_iri:
            continue
        score = compute_hss_from_graph(G, target_iri, iri, alpha)
        if score >= threshold:
            label = G.nodes[iri].get("label", iri.split("/")[-1])
            hits.append((iri, label, score))

    hits.sort(key=lambda x: x[2], reverse=True)

    # Optional: generate HTML visualization
    if output_file and hits:
        visualize_similarity_subgraph(G, target_iri, hits, output_file)

    # Convert for JSON response
    return [{"iri": iri, "label": label, "score": round(score, 4)} for iri, label, score in hits]
