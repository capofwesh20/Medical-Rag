from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import json

app = FastAPI()
templates = Jinja2Templates(directory="template")
app.mount("/static", StaticFiles(directory="static"), name="static")

llm = LlamaCpp(
    model_path = "./models/BioMistral-7B.Q4_K_M.gguf",
    temperature = 0.2,
    top_p = 0.9,
    n_ctx=2048
)

print("LLM Initialized....")

prompt_template = """You are an autoregressive language model that has been fine-tuned with instruction-tuning and 
RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. If you 
think there might not be a correct answer, you say so. Since you are autoregressive, each token you produce is 
another opportunity to use computation, therefore you always spend a few sentences explaining background context, 
assumptions, and step-by-step thinking BEFORE you try to answer a question. Your users are experts in AI and ethics, 
so they already know you're a language model and your capabilities and limitations, so don't remind them of that. 
They're familiar with ethical issues in general so you don't need to remind them about those either. Don't be verbose 
in your answers, but do provide details and examples where it might help the explanation. Use the following pieces of 
information to answer the user's question. If you don't know the answer, just say that you don't know, don't try to 
make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_database")

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

retriever = db.as_retriever(search_kwargs={"k": 2})



@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/get_response")
async def get_response(query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True,
                                     chain_type_kwargs=chain_type_kwargs, verbose=False)
    response = qa(query)
    print(response)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    response_data = jsonable_encoder(json.dumps({"answer": answer, "source_document": source_document, "doc": doc}))

    res = Response(response_data)
    return res
