from google import genai
from langchain_qdrant import QdrantVectorStore,FastEmbedSparse,RetrievalMode
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import wrappers
from google.genai import types
from langsmith import traceable
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel,Field
from typing import List,Optional
import cohere
import os


load_dotenv()
dense_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={'device':'mps'})
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/minicoil-v1",kwargs={'device': 'mps'})
co=cohere.ClientV2()
client = QdrantClient(url="http://localhost:6333")
collection_name="RAG_PROJECT"

vector_store = QdrantVectorStore.from_existing_collection(
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            url=os.environ.get("QDRANT_URL"),
            collection_name=collection_name,
            retrieval_mode=RetrievalMode.HYBRID,
            api_key=os.environ.get("QDRANT_API_KEY")
            
        )


@traceable(name="Chunks Retrieved", metadata={"search_type": "hybrid"})
def retrieval(user_query)->list:
    results = vector_store.similarity_search_with_score(user_query,k=15)
    results=[result for result in results if result[1]>0.2]
    return(results)    


@traceable(name="Reranking")
def rerank(chunks,user_query)->list:
    text_chunks=[chunk for chunk in chunks if "_txt" in chunk[0].metadata['eid'] ]
    img_chunks=[chunk for chunk in chunks if "_txt" not in chunk[0].metadata['eid']]
    text_passage=[chunk[0].page_content for chunk in text_chunks]
    response = co.rerank(
    model="rerank-english-v3.0",
    query=user_query,
    documents=text_passage,
    top_n=3,
)
    top_text = [text_chunks[r.index] for r in response.results]
    return top_text + img_chunks[:2]
@traceable
def genereate(chunks,user_query)->str:
    
    class RAGResponse(BaseModel):
        answer: str=Field(description="answer to question")
        used_chunks_ids:  Optional [List[str]]=Field(description="List of IDs of chunks used for answer generation.")

    context="\n\n\n".join([f"Page Content:{result[0].page_content}\n Page Number:{result[0].metadata['pages']}\n File Location:{result[0].metadata['filename']}\n Chunk ID:{result[0].metadata['chunk_id']}"
                        for result in chunks])
    SYSTEM_PROMPT=f"""
    You are a helpful AI assistant who answers user query based on available context, retrieved from pdf file.
    You should ONLY ans the user based on the following context and help user navigate to the right page number and to the right file to know more, ALWAYS mention what detail is available on which page. IF u find no chunk matching the query ONLY send sorry but u didn't find anything 
    Context
    {context}
    """
    client = genai.Client()
    wrapped_client = wrappers.wrap_gemini(
            client,
            
            tracing_extra={
                "tags": ["gemini", "python"],
                "metadata": {
                    "integration": "google-genai",
                    
                },
            },
    )    
    interaction = wrapped_client.models.generate_content(
        model="gemini-3.1-flash-lite", 
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RAGResponse
            ),
        contents=user_query
        
    )
    result=interaction.parsed
    chunk_map = {chunk[0].metadata["chunk_id"]: chunk for chunk in chunks}
    send_chunks=[chunk for chunk in chunks if chunk[0].metadata['chunk_id'] in result.used_chunks_ids]
    
    return(send_chunks,result.answer)
@traceable(name="generation with re_ranker")
def pipeline(user_query):
    chunks=retrieval(user_query)
    rerank_chunks=rerank(chunks,user_query)
    ans=genereate(rerank_chunks,user_query)
    return(ans)


