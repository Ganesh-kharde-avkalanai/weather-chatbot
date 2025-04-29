import streamlit as st
import requests

# Set your OpenWeatherMap API key
API_KEY = "bb5385896e0385285079bc301b7f9311"

def get_weather(city):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    # params = {
    #     "q": city,
    #     "appid": API_KEY,
    #     "units": "metric"  # Use 'imperial' for Fahrenheit
    # }
    try:
        complete_url = base_url + "appid=" + API_KEY + "&q=" + city
        response = requests.get(complete_url)
        # response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}

# Streamlit UI
st.set_page_config(page_title="Weather App", layout="centered")
st.title("☀️ Weather Info App")

city = st.text_input("Enter City Name")

if st.button("Get Weather"):
    if city:
        with st.spinner("Fetching weather..."):
            data = get_weather(city)

        if "error" in data:
            st.error(f"Error: {data['error']}")
        elif data.get("cod") != 200:
            st.error(f"Error: {data.get('message', 'Unknown error')}")
        else:
            st.success(f"Weather in {data['name']}, {data['sys']['country']}")
            st.metric("🌡️ Temperature (°C)", f"{data['main']['temp']} °C")
            st.metric("💧 Humidity", f"{data['main']['humidity']}%")
            st.metric("🌬️ Wind Speed", f"{data['wind']['speed']} m/s")
            st.write(f"**Weather condition:** {data['weather'][0]['description'].capitalize()}")
    else:
        st.warning("Please enter a city name.")
