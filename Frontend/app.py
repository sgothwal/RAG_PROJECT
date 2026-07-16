# frontend/app.py
import streamlit as st
import requests
import time
import base64
from PIL import Image
import io


st.title("🤖 PaperAI ")

st.markdown("📄 [View source papers on GitHub](https://github.com/sgothwal/RAG_PROJECT/tree/main/Papers)")
st.markdown("**Try asking:**")
st.caption("• What is the BM25 weighting scheme?")
st.caption("• How does the attention mechanism work in transformers?")
st.caption("• In causal self-attention, what range of inputs does the model have access to when processing a specific item?")
st.caption("• What is the trade-off associated with methods that give more weight to middle-probability words?")
st.caption("• What is the role of embeddings in semantic search?")
st.divider()


query = st.text_input("Ask a question",placeholder='What is the BM25 weighting scheme?')

if st.button("Search", type="primary") and query:
    with st.spinner("Searching..."):
        start = time.time()
        response = requests.post(
            "http://localhost:8000/query",
            json={"query": query}
        )
        latency = round(time.time() - start, 2)
        data = response.json()
    
    st.markdown("### Answer")
    st.write(data["answer"])
    for source in data["sources"]:
        if source.get("image_base64"):
                for b64 in source["image_base64"]:
                   
                    
                    img_bytes = base64.b64decode(b64)
                    img = Image.open(io.BytesIO(img_bytes))
                    st.image(img, caption=f"Figure from {source['source']} p.{source['page']}")

                st.caption(f"⏱️ Response time: {latency}s")
    
    
    with st.expander("🔍 Retrieved Sources"):
        for source in data["sources"]:
            st.markdown(f" File -{source['source']} &nbsp;&nbsp;&nbsp;&nbsp;  Pages -{source['page']}")
            st.caption(source["content"] + "...")
            st.divider() 
            