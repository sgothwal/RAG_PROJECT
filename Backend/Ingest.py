from langchain_core.documents import Document
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore,FastEmbedSparse,RetrievalMode
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
dense_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={'device':'mps'})
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/minicoil-v1",kwargs={'device': 'mps'})
client = QdrantClient(url="http://localhost:6333")
collection_name="RAG_PROJECT"


def create_docu()-> list:
    try:
        base_path = Path(__file__).parent

        with open(base_path.parent / 'json_files' / 'final_data.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: The file final_data.json does not exist.")  
    
    docu_list=[]
    for element in data:
        extra_data={'img_url':element['img_url'],'eid':element['id']}
        doc=Document(page_content=element['text'],  ##langchain documents creation
                metadata=element['metadata']|extra_data
                )
        docu_list.append(doc)

    with open("langchain_docu.json", "w") as f:
        json.dump(docu_list, f, indent=2)
    return(docu_list)
def create_embeddings(documents)->object:
    
        vector_store = client.create_collection(
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            url="http://localhost:6333",
            collection_name=collection_name,
            retrieval_mode=RetrievalMode.HYBRID
            
        )
        print("!!!!Created vector store!!!!")
        
if __name__== "__main__":
     docs=create_docu()
     create_embeddings(docs)