# frontend/app.py
import streamlit as st
import requests
import time
import base64
from PIL import Image
import io


st.title("📚 AI Research Assistant")

query = st.text_input("Ask a question")

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
            