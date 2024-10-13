from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

numero_de_dias = 7
numero_de_criancas = 2
atividade =  "praia"

template = PromptTemplate.from_template(
    "Crie um roteiro de viagem de {numero_de_dias} dias, para uma familia com {numero_de_criancas} crianças, que gostam de {atividade}."
)

prompt = template.format(
    numero_de_dias=numero_de_dias,
    numero_de_criancas=numero_de_criancas,
    atividade=atividade
)

llm = ChatOpenAI(model="gpt-4o",
                 temperature=0.5,
                 api_key=os.getenv("OPEN_API_KEY")
                 )

resposta = llm.invoke(prompt)
print(resposta.content)