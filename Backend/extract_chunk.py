from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from IPython.display import Image, display
from dotenv import load_dotenv
from google import genai
from google.genai import types
from dotenv import load_dotenv
import base64
import json
import time
load_dotenv()
all_chunks=[]
base_path = Path(__file__).parent
def docs_loder():
   
    print("---------------------- pipeline starting ----------------------\n\n")
    print("!!!STEPS!!!\n\n")
    print(" I-loading files from directory")
    data_folder = base_path.parent / "Papers"
    files=list(data_folder.glob("*.pdf"))
    print("II-!!✈️ sending loaded file for element creation!!\n\n")
    ele_loader(files)

def ele_loader(files)->tuple[list,str]:

    for i,file in enumerate(files):
        print(f"III-!!creating elements from file =>{file.name}- {i+1}/{len(files)}files!!\n\n")
        start = time.perf_counter()
        elements = partition_pdf(
        filename=file,                 
        strategy="hi_res",                                    
        extract_images_in_pdf=True,  
        extract_image_block_types=["Image", "Table"],          
        extract_image_block_to_payload=True, 
       
        )   
        end = time.perf_counter()
        print(f"Took {(end - start)/60:.2f} minutes for elements formation")
        elements = [e for e in elements if e.category not in ("Header", "Footer")]
        print(f"IV-!!sending elements for chunks formation!!")
        chunker(elements,file.name)

def chunker(elements,file_name)->tuple[list,str]:
   """forms chunks"""
   print(f"V-!!starting chunking for file {file_name}!!\n\n")
   chunks = chunk_by_title(elements,
                        include_orig_elements=True,
                        max_characters=1200,
                        combine_text_under_n_chars=500,
                        new_after_n_chars=1000,
                        overlap=150)
   loader(chunks,file_name)

def loader(chunks,file_name)->list:
    """Extracts data from chunks"""
    print(f"VI-!!starting extraction of data from chunks of file {file_name}!!\n\n")
    
    c_len=len(chunks)
    for i,chunk in enumerate(chunks):
        f_chunk=extractor(chunk,i+1,c_len)
        if(f_chunk['table']or f_chunk['img_url']):
            AI_chunk=summarise(f_chunk.copy(),i+1,c_len)
            print(f"AI chunk appended with text:-{AI_chunk['text'][:100]}\n images {AI_chunk['img_url'][0][:50]}\n id:{AI_chunk['id']}")
            all_chunks.append(AI_chunk)
            f_chunk['img_url']=[]
            f_chunk['table']=[]
            time.sleep(0.5)

        all_chunks.append(f_chunk)
        print(f"chunk {i+1}/{c_len} sent 🛩️ to final list \n\n" )    
    


def extractor(chunk,i,c_length)->dict:
    text=chunk.text
    table=[]
    img=[]
    filename=chunk.metadata.filename
    chunk_id=chunk.id
    pages=set()
  
   
    print(f"processing chunk ⏳:{i}/{c_length}\n\n")
    for ele in chunk.metadata.orig_elements:
        pages.add(ele.metadata.page_number)
        ele_type=ele.category
        try:
            if ele_type=='Table':
                print(f"!!!found Table to Process in chunk {i}!!!\n\n")
                
                img64=ele.metadata.image_base64
                imgurl=ele.metadata.image_url
                if imgurl or img64: #some tables exist as img
                    image_url=getattr(ele.metadata,'image_base64','image_url')
                    img.append(image_url) 
                   
                else:    
                    table_data=getattr(ele.metadata,'text_as_html','text')
                    table.append(table_data)

            elif ele_type=='Image':
                print(f"!!!found Image 🏙️ to Process in chunk {i}!!!\n\n")
                image_url=getattr(ele.metadata,'image_base64','image_url')    
                img.append(image_url)
        
        except Exception as e: 
            print("error",e)
     
    print(f"filename is {filename} \n chunk id is{chunk_id}_txt \n pages {pages}")
    return({"text":text,
                "table":table,
                "img_url":img,
                "img_summary":'',
                "metadata":{"filename":filename,"pages":sorted(pages),"chunk_id":chunk_id},
                "id":f"{chunk_id}_txt"})        
def summarise(chunk,i,c_len)->dict:
    
    max_retries=5
    text=chunk['text']
    images=chunk['img_url']
    tables=chunk['table']
    f_id=chunk['metadata']['chunk_id']
    
    
    contents=[]
    for attempt in range(max_retries):
        try:
            print(f"---- sending chunk {i}/{c_len} with tables and/or images to AI 🤖 for summarising ----\n\n")
            SYSTEM_PROMPT=f"""Here You are provided with some data, it includes image and/or table along with it you are being provided with some text data your task is:
            a)Analyze the image and/or table alongside with the text provided
            b)Come up with a comprhensive summary after analyzing everything 
            c)NEVER go beyond the scope of text, image and/or table and DO NOT add uncessary details
            \nTEXT:{text}\n\n
            """
            if(tables):
                SYSTEM_PROMPT+=f"""TABLES\n\n"""
                for i,table in enumerate(tables):
                    SYSTEM_PROMPT+=f'''Table{i+1}:{table} \n\n\n'''
            if(images):  
                SYSTEM_PROMPT+=f"""IMAGES\n\n"""
                for image in images:
                    contents.append(
                        types.Part.from_bytes(
                            data=base64.b64decode(image),
                            mime_type="image/jpeg"
                        )
                    )
                SYSTEM_PROMPT+=f"""{contents}"""   
                
            client = genai.Client()
            response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=SYSTEM_PROMPT
            )

            print(f"LLM has sent =>{response.text[:200]}")
            summary=response.text
            chunk['text']=summary
            chunk['id']=f"{f_id}_img"
            print(f"our image summary text is now=>{chunk['text'][:100]}")

            return(chunk)
        except Exception as e:
            
            if attempt==max_retries-1:
                raise ValueError("not accepting API calls anymore")
            wait=3*2** attempt
            print(f"Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
            

def export(filename=base_path.parents/"json_files"/"final_data.json",data=all_chunks)->json:
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print("!!!!! VII file is ready in your root folder 📁 !!!! \n\n\n")   
    



if __name__== "__main__":
    docs_loder()
    export()


