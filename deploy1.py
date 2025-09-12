from google import genai
from google.genai import types
import base64
import os

def generate():
  client = genai.Client(
      vertexai=True,
      api_key=os.environ.get("GOOGLE_CLOUD_API_KEY"),
  )

  msg2_text1 = types.Part.from_text(text="""{\"date\": \"{currentDate}\", \"transactionType\": \"Cash\", \"narration\": \"\", \"bankName\": \"\", \"accountName\": \"\", \"amount\": 0.0, \"chequeNumber\": \"\", \"chequeDate\": \"\", \"transferType\": \"\", \"isReceipt\": \"Receipt\"}""")

  model = "projects/738928595068/locations/us-central1/endpoints/7079072574528815104"
  contents = [
    types.Content(
      role="user",
      parts=[
        types.Part.from_text(text="""receive amount from""")
      ]
    ),
    types.Content(
      role="model",
      parts=[
        msg2_text1
      ]
    ),
    types.Content(
      role="user",
      parts=[
        types.Part.from_text(text="""pay amount to""")
      ]
    ),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 65535,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    thinking_config=types.ThinkingConfig(
      thinking_budget=0,
    ),
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    print(chunk.text, end="")

generate()
