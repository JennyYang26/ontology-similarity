from fastapi import APIRouter, HTTPException
from app.graph_builder import build_ontology_graph_parallel
from app.algorithms import hss, ratio_hss, wl_node2vec, hybrid_simrank
from .schemas import SimilarityRequest

router = APIRouter()

@router.post("/find_similar_items/")
def find_similar_items(request: SimilarityRequest):
    if not request.target_item:
        raise HTTPException(status_code=422, detail="Target item cannot be empty.")

    algo_map = {
        "hss": hss.hss_findSimilarItems,
        "ratio_hss": ratio_hss.ratio_hss_findSimilarItems,
        "wl": wl_node2vec.wl_embed_similarity,
        "hybrid": hybrid_simrank.hybrid_simrank_fusion,
    }

    if request.algorithm not in algo_map:
        raise HTTPException(status_code=400, detail="Invalid algorithm selected. Choose from: hss, ratio_hss, wl, hybrid.")

    try:
        G = build_ontology_graph_parallel("data_json", start_index=request.start_index, end_index=request.end_index)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph build failed: {str(e)}")

    try:
        result = algo_map[request.algorithm](G, request.target_item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Algorithm execution failed: {str(e)}")

    return {
        "algorithm_used": request.algorithm,
        "target": request.target_item,
        "results": result,
    }
