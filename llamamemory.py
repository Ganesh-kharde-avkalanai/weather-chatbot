import boto3
import json
import os
import time
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_aws import BedrockEmbeddings
import requests
import re
import streamlit as st

# Load environment variables
load_dotenv()

# AWS credentials and region
aws_access_key_id = os.getenv("AWS_ACCESS_KEY")
aws_secret_access_key = os.getenv("AWS_SECRET_KEY")
aws_region = os.getenv("AWS_REGION")
model_id = "meta.llama3-70b-instruct-v1:0"

# Initialize Bedrock client and embeddings
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_runtime)

# FAISS index path
faiss_index_folder = "faiss_index"

# Weather API
weather_api_key = "open weather map API key"

# Bedrock model invocation
def invoke_bedrock_model(prompt):
    try:
        start_time = time.time()
        body = json.dumps({"prompt": prompt, "temperature": 0.5, "top_p": 0.9})
        response = bedrock_runtime.invoke_model(
            body=body, modelId=model_id, contentType="application/json", accept="application/json"
        )
        response_body = json.loads(response.get("body").read())
        generation = response_body.get("generation")
        latency = time.time() - start_time
        return generation, latency
    except Exception as e:
        return f"Error: {str(e)}", 0

# Check if query is about weather
def is_weather_query(query):
    weather_keywords = ["weather", "temperature", "forecast", "climate"]
    if any(kw in query.lower() for kw in weather_keywords):
        match = re.search(r"(?:in|at)\s+([A-Za-z\s]+)", query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

# Save chat memory to log file
def save_chat_log(memory, log_path="chat_logs.txt"):
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("----- next chat -----\n")
            for msg in memory:
                role = msg["role"].capitalize()
                message = msg["message"]
                f.write(f"{role}: {message}\n")
            f.write("\n")
    except Exception as e:
        st.warning(f"Error saving chat log: {str(e)}")


# Fetch weather info
def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?appid={weather_api_key}&q={city}&units=metric"
        response = requests.get(url).json()
        if response.get("cod") != 200:
            return f"Could not fetch weather info for '{city}'."
        return (
            f"Weather in {city.title()}:\n"
            f"- Temperature: {response['main']['temp']}°C\n"
            f"- Condition: {response['weather'][0]['description']}\n"
            f"- Humidity: {response['main']['humidity']}%\n"
            f"- Wind Speed: {response['wind']['speed']} m/s"
        )
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

# Streamlit page setup
st.set_page_config(page_title="Chatbot with Weather + FAISS", layout="centered")
st.title("🧠 AI Chatbot with Weather & Document Intelligence")

# Chat history session state
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

# Load retriever once
if "retriever" not in st.session_state:
    faiss = FAISS.load_local(faiss_index_folder, embeddings, allow_dangerous_deserialization=True)
    st.session_state.retriever = VectorStoreRetriever(vectorstore=faiss)

# Display chat messages using Streamlit's chat format
for msg in st.session_state.chat_memory:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["message"])

# Input at bottom using chat_input
user_question = st.chat_input("Type your message and press Enter")

# On user message
if user_question:
    # Add user message to history
    st.session_state.chat_memory.append({"role": "user", "message": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Handle weather query
    city = is_weather_query(user_question)
    if city:
        bot_response = get_weather(city)
    else:
        retriever = st.session_state.retriever
        docs = retriever.invoke(user_question)
        context = "\n\n".join([doc.page_content for doc in docs])

        history_string = ""
        for turn in st.session_state.chat_memory:
            history_string += f"{turn['role'].capitalize()}: {turn['message']}\n"
        history_string += f"User: {user_question}\n"

        prompt_template = """
        <s>[INST] 
        you are a chatbot
        Provide a precise and accurate explanation using only the information explicitly stated in the provided context.
        Your response must not exceed 50 words and should directly answer the question based on available details only.
        Do not infer or create an answer that is not clearly supported by the context.

        If the question suggests that the answer should be given in points or a list, respond using a point-wise format.
        Ensure that:

        Each point is written on a new line.

        Do not sequence points in a single line.

        Always remember to add a new line after every point.

        If the required information is not found in the context, respond with:
        "I’m unable to provide an accurate response at the moment. Please visit <web_link> or reach out to support: Phone: <phone_number>, Email: <support_email>."

        Do not provide any additional information or elaboration beyond what is directly relevant to the question. Also try to lookup to history chat.

        <memory>
        {history}
        </memory>

        <context>
        {context}
        </context>

        Question: {question} 
        [/INST]
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["history", "context", "question"])
        final_prompt = PROMPT.format(history=history_string, context=context, question=user_question)
        bot_response, latency = invoke_bedrock_model(final_prompt)
        st.info(f"Model Response Time: {latency:.2f}s")

    # Display and store bot response
    st.session_state.chat_memory.append({"role": "assistant", "message": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)

    # Save logs to file
    save_chat_log(st.session_state.chat_memory)
