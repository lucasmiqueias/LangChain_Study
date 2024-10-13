from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.globals import set_debug
import os
from dotenv import load_dotenv
from pydantic import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

load_dotenv()
set_debug(True)

class Destino(BaseModel):
    cidade: str = Field("Cidade a visitar")
    motivo: str = Field("Motivo pelo qual é interessante visitar essa cidade")

llm = ChatOpenAI(model="gpt-4o",
                 temperature=0.5,
                 api_key=os.getenv("OPEN_API_KEY")
                 )

parseador = JsonOutputParser(pydantic_object=Destino)

modelo_cidade = ChatPromptTemplate.from_template(
    template="""Sugira uma cidade dado meu interesse por {interesse}.
    {formatacao_de_saida}
    """,
    partial_variables={"formatacao_de_saida": parseador.get_format_instructions()}
)

modelo_restaurantes = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}"
)

modelo_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {cidade}"
)

parte1 = modelo_cidade | llm | parseador
parte2 = modelo_restaurantes | llm | StrOutputParser()
parte3 = modelo_cultural | llm | StrOutputParser()

cadeia = (parte1 | {
    "restaurantes": parte2,
    "locais_culturais": parte3
})

resultado = cadeia.invoke({"interesse": "praias"})
print(resultado)