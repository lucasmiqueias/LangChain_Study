from langchain_openai import ChatOpenAI
from langchain.globals import set_debug
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from pydantic import Field, BaseModel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
set_debug(True)

class Destino(BaseModel):
    cidade: str = Field("Cidade a visitar")
    motivo: str = Field("Motivo pelo qual é interessante visitar essa cidade")

llm = ChatOpenAI(model="gpt-4o",
                 temperature=0.5,
                 api_key=os.getenv("OPEN_API_KEY")
                 )

mensagens = [
        "Quero visitar um lugar no Brasil famoso por suas praias e cultura. Pode me recomendar?",
        "Qual é o melhor período do ano para visitar em termos de clima?",
        "Quais tipos de atividades ao ar livre estão disponíveis?",
        "Alguma sugestão de acomodação eco-friendly por lá?",
        "Cite outras 20 cidades com características semelhantes às que descrevemos até agora. Rankeie por mais interessante, incluindo no meio ai a que você já sugeriu.",
        "Na primeira cidade que você sugeriu lá atrás, quero saber 5 restaurantes para visitar. Responda somente o nome da cidade e o nome dos restaurantes.",
]

longa_conversa = ""
for mensagem in mensagens :
    longa_conversa += f"Usuário: {mensagem}\n"
    longa_conversa += f"AI:"

    modelo = PromptTemplate(template=longa_conversa,
                            input_variables=[""],
                            )
    cadeia = modelo | llm | StrOutputParser()
    resposta = cadeia.invoke(input={})

    longa_conversa += resposta + "\n"
    print(longa_conversa)