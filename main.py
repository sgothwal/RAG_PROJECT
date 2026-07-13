from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from Backend.Main_pipeline import pipeline


app = FastAPI(title="RAG_PROJECT")
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        chunks, answer= pipeline(request.query)
        sources = [
            {
                "source": c[0].metadata['filename'],
                "page": c[0].metadata['pages'],
                "content": c[0].page_content,
                
                "image_base64": c[0].metadata['img_url']
            }
            for c in chunks
        ]
    
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))