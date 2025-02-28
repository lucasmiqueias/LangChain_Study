from openai import OpenAI

numero_de_dias = 7
numero_de_criancas = 2
atividade =  "praia"

prompt = f"Crie um roteiro de viagem de {numero_de_dias} dias, para uma familia com {numero_de_criancas} crianças, que gostam de {atividade}."
print(prompt)

cliente = OpenAI(api_key="")

resposta = cliente.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistante."},
        {"role": "user", "content": prompt}
    ]
)

print(resposta)

roteiro_viagem = resposta.choices[0].message.content
print(roteiro_viagem)