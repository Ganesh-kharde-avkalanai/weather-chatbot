# meta.llama3-70b-instruct-v1:0
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
# Load environment variables from .env file
load_dotenv()

# AWS credentials from environment variables
aws_access_key_id = os.getenv("AWS_ACCESS_KEY")  # Corrected variable name
aws_secret_access_key = os.getenv("AWS_SECRET_KEY")  # Corrected variable name
aws_region = os.getenv("AWS_REGION")

# Bedrock model ID (Llama 3.1 70B Instruct)
model_id = "meta.llama3-70b-instruct-v1:0"

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# Bedrock Embeddings
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_runtime)

# Faiss index folder path (replace with your actual path)
faiss_index_folder = "faiss_index"

# Function to invoke Bedrock model with latency measurement
def invoke_bedrock_model(prompt):
    try:
        start_time = time.time()  # Record start time

        body = json.dumps({"prompt": f"{prompt}", "temperature": 0.5, "top_p": 0.9})

        response = bedrock_runtime.invoke_model(
            body=body, modelId=model_id, contentType="application/json", accept="application/json"
        )
        response_body = json.loads(response.get("body").read())
        generation = response_body.get("generation")
        end_time = time.time()  # Record end time
        latency = end_time - start_time
        return generation, latency

    except Exception as e:
        print(f"Error invoking Bedrock model: {e}")
        return None, None


def is_weather_query(query):
    """Returns the city if query is weather-related, otherwise None"""
    weather_keywords = ["weather", "temperature", "forecast", "climate"]
    if any(kw in query.lower() for kw in weather_keywords):
        # Try extracting city name using regex after "in" or "at"
        match = re.search(r"(?:in|at)\s+([A-Za-z\s]+)", query, re.IGNORECASE)
        if match:
            city = match.group(1).strip()
            return city
    return None

weather_api_key = ("bb5385896e0385285079bc301b7f9311")


def get_weather(city):
    """Fetch current weather using OpenWeatherMap API"""
    try:
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        url = base_url + "appid=" + weather_api_key + "&q=" + city
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return f"Could not fetch weather info for '{city}'."
        
        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]
        return f"Weather in {city.title()}:\n- Temperature: {temp}°C\n- Condition: {description}\n- Humidity: {humidity}%\n- Wind Speed: {wind} m/s"
    
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


# Example usage
def main():
    try:
        # Load Faiss index from folder
        vectorstore_faiss = FAISS.load_local(
            faiss_index_folder, embeddings, allow_dangerous_deserialization=True
        )

        # User's question
#         user_question = '''
# upto how many hours
        
#         '''

        # User's question
        user_question = '''
weather in Delhi
        '''.strip()

        # Check if it's a weather-related query
        city = is_weather_query(user_question)
        if city:
            weather_response = get_weather(city)
            print(weather_response)
            return
        else:

            # Retrieve context from the vector store
            retriever = VectorStoreRetriever(vectorstore=vectorstore_faiss)
            start_retrieval = time.time()
            docs = retriever.invoke(user_question)
            end_retrieval = time.time()
            retrieval_latency = end_retrieval - start_retrieval

            retrieved_content = "\n\n".join([doc.page_content for doc in docs])

            print("retrieved content")
            print(retrieved_content)

            # Prompt template for Llama
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

        Do not provide any additional information or elaboration beyond what is directly relevant to the question. 

            <context>
            {context}
            </context>

            Question: {question} 
            [/INST]
            """

            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

            # Format the prompt
            final_prompt = PROMPT.format(context=retrieved_content, question=user_question)

            # Invoke Bedrock model
            response, bedrock_latency = invoke_bedrock_model(final_prompt)
            print("true")

            if response:
                print("Bedrock Response:")
                print(response)
                print(f"Bedrock Latency: {bedrock_latency:.4f} seconds")
                print(f"Retrieval Latency: {retrieval_latency:.4f} seconds")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()