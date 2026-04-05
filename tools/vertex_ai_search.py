import os
from google.genai import types
from pydantic import BaseModel, Field

class Tools:
    class Valves(BaseModel):
        VERTEX_AI_RAG_STORE: str = Field(
            default=os.getenv("GOOGLE_VERTEX_AI_RAG_STORE", ""),
            description="Vertex AI RAG Store path for grounding (e.g., projects/PROJECT/locations/LOCATION/ragCorpora/DATA_STORE_ID).",
        )

    def __init__(self):
        self.valves = self.Valves()

    def vertex_ai_search(self) -> types.Tool:
        """
        Enable Vertex AI Search grounding for RAG.
        """
        if not self.valves.VERTEX_AI_RAG_STORE:
            raise ValueError("VERTEX_AI_RAG_STORE valve is not set.")

        return types.Tool(
            retrieval=types.Retrieval(
                vertex_ai_search=types.VertexAISearch(
                    datastore=self.valves.VERTEX_AI_RAG_STORE
                )
            )
        )
