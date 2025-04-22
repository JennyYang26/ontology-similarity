from pydantic import BaseModel
from typing import Optional

class SimilarityRequest(BaseModel):
    target_item: str
    start_index: Optional[int] = 0
    end_index: Optional[int] = None
    algorithm: str
