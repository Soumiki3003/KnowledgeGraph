from google import genai

# GOOGLE_API_KEY is set in the environment
client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Are bunnies considered cute? Also what is your token limit? I got pro.",
)
print(response.text)
