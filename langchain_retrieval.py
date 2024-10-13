from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.globals import set_debug
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from pydantic import Field, BaseModel

load_dotenv()
set_debug(True)

class Destino(BaseModel):
    cidade: str = Field("Cidade a visitar")
    motivo: str = Field("Motivo pelo qual é interessante visitar essa cidade")

llm = ChatOpenAI(model="gpt-4o",
                 temperature=0.5,
                 api_key=os.getenv("OPEN_API_KEY")
                 )

carregador = TextLoader("GTP_gold_Nov23.txt", encoding="utf-8")
documentos = carregador.load()

quebrador = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
textos = quebrador.split_documents(documentos)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(textos, embeddings)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever)

pergunta = "Como devo proceder caso tenha um item comprado roubado"
resultado = qa_chain.invoke({ "query" : pergunta})
print(resultado)