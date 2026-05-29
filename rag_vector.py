import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from groq import Groq

data=pd.read_csv("sensor_data.csv")
load_dotenv()
client_q=Groq(api_key=os.getenv("GROQ_API_KEY"))

txt = data.apply(lambda row: 
    f"In {row['timestamp']} temperature:{row['temperature']:.1f}\
        humidity:{row['humidity']:.1f}\
            vibration:{row['vibration']:.4f}\
                pressure:{row['pressure']:.1f}\
                    anomaly:{'Yes' if row['is_anomaly'] == 1 else 'No'}", axis=1).tolist()
 
model =SentenceTransformer("all-MiniLM-L6-v2")
client=chromadb.Client()

collection_normal =client.create_collection("building_normal")
collection_anomaly =client.create_collection("building_anomalies")

normal_txt = [t for t in txt if "anomaly:No" in t]
anomaly_txt = [t for t in txt if "anomaly:Yes" in t]

normal_encode = model.encode(normal_txt).tolist()
anomaly_encode = model.encode(anomaly_txt).tolist()

collection_normal.add(
    documents=normal_txt,
    embeddings=normal_encode,
    ids=[str(i)for i in range (len(normal_txt))]
)

collection_anomaly.add(
    documents=anomaly_txt,
    embeddings=anomaly_encode,
    ids=[str(i)for i in range (len(anomaly_txt))]
)
question =input(f"\n>> Hi,\n you can ask any question related to the latest updates captured by the sensors\n and to understand the general situation.\n Please feel free to ask your question.")    
memory =[]
while True:
    if question.lower() == 'exit':
        break
    question_vector=model.encode([question]).tolist()
    last_context = memory[-1]['content'] if memory else ""
    routing_response = client_q.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
        {"role": "system", "content": "You are a query classifier. Reply with only one word: 'anomaly' if the question is about anomalies, errors, or unusual readings. Reply 'normal' for general data questions."},
        {"role": "user", "content": f"Previous context: {last_context}\nQuestion: {question}"}
          ]
    )

    route = routing_response.choices[0].message.content.strip().lower()

    if 'anomaly' in route:
        results = collection_anomaly.query( query_embeddings=question_vector,
                                       n_results=10
                                       )
    else:
        results = collection_normal.query( query_embeddings=question_vector,
                                      n_results=10
                                      )

    context="\n".join(results['documents'][0])
    prompt =f'''
    Based on this building sensor data:{context}
    Answer this question: {question}'''
    messages = [{"role": "system", "content": "You are an expert in building structural health monitoring. Answer based only on the provided data."}]
    messages.extend(memory)
    messages.append({"role": "user", "content": prompt})
    response =client_q.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    memory.append({"role": "user", "content": prompt})
    memory.append({"role": "assistant", "content": response.choices[0].message.content})
    
    print("\n>> Response:\n")
    print(f'>> {response.choices[0].message.content}\n')
    question =input("\n>> You can ask another question or type 'exit' to quit: ")

