import os
import requests
import re
import html
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from dotenv import load_dotenv
import google.generativeai as genai
import PyPDF2
from io import BytesIO
import markdown
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient
import time
from PIL import Image
import io
import wikipediaapi  # For Wikipedia integration

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize Hugging Face client
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
hf_client = InferenceClient(token=HF_API_TOKEN)

# Initialize Wikipedia API with a user agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',  # Language (e.g., 'en' for English)
    user_agent='MyTelegramBot/1.0 (my-email@example.com)'  # User agent string
)

# Conversation history dictionary
conversation_history = {}

# OpenWeather API key
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Abhijith's persona prompt
ABHIJITH_PROMPT = os.getenv("ABHIJITH_PROMPT")

def render_markdown(text):
    def highlight_code(match):
        language = match.group(1) or "text"
        code = match.group(2)
        try:
            lexer = get_lexer_by_name(language, stripall=True)
        except:
            lexer = get_lexer_by_name("text", stripall=True)
        formatter = HtmlFormatter(style="friendly")
        return f"<pre>{highlight(code, lexer, formatter)}</pre>"

    # Convert code blocks
    text = re.sub(r"```(\w*)\n(.*?)```", highlight_code, text, flags=re.DOTALL)

    # Convert markdown to HTML
    html_content = markdown.markdown(text)

    # Parse HTML and remove unsupported tags
    soup = BeautifulSoup(html_content, "html.parser")

    # Replace unsupported tags with plain text
    for tag in soup.find_all(["ol", "ul", "li"]):
        if tag.name == "ul":
            tag.replace_with("\n".join(f"• {li.get_text()}" for li in tag.find_all("li")))
        elif tag.name == "ol":
            tag.replace_with("\n".join(f"{i + 1}. {li.get_text()}" for i, li in enumerate(tag.find_all("li"))))
        elif tag.name == "li":
            tag.replace_with(f"• {tag.get_text()}")

    # Remove remaining unsupported tags
    for tag in soup.find_all(["p", "div", "span"]):
        tag.unwrap()

    return str(soup)

def get_gemini_response(text, history=None):
    try:
        prompt = ABHIJITH_PROMPT
        if history:
            prompt += "\n\nConversation History:\n" + "\n".join(history)
        prompt += f"\n\nUser: {text}\nAbhijith:"

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating Gemini response: {e}")
        return "Sorry, I couldn't generate a response. Please try again later."

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            humidity = data['main']['humidity']
            wind = data['wind']['speed']
            return f"The current weather in {city} is {desc}.\nTemperature: {temp}°C\nHumidity: {humidity}%\nWind Speed: {wind} m/s"
        else:
            return "Sorry, I couldn't fetch the weather data. Please check the city name and try again."
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return "Sorry, I couldn't fetch the weather data. Please try again later."

def extract_city_from_query(query):
    keywords = ["weather", "temperature", "forecast", "humidity", "wind"]
    if any(k in query.lower() for k in keywords):
        match = re.search(r"(?:weather|temperature|forecast).*?(?:in|for|at)\s+([\w\s]+)", query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def extract_image_query(query):
    image_keywords = ["generate", "create", "show", "image", "picture", "photo"]
    if any(k in query.lower() for k in image_keywords):
        match = re.search(r"(?:generate|create|show)\s+(?:an?)?\s*(?:image|picture|photo)\s+of\s+(.*)", query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def generate_image_with_huggingface(prompt, retries=3, delay=5):
    for attempt in range(retries):
        try:
            # Enhance the prompt for professional images
            enhanced_prompt = f"Professional high-quality image: {prompt}, 4k resolution, realistic, detailed, cinematic lighting"
            
            # Generate the image using the specified model
            image = hf_client.text_to_image(prompt=enhanced_prompt, model="runwayml/stable-diffusion-v1-5")
            
            # Convert PIL.Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            return img_byte_arr
        except Exception as e:
            print(f"Hugging Face Image Error (Attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
    return None

def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        return "".join(page.extract_text() for page in reader.pages)
    except Exception as e:
        print(f"PDF extract error: {e}")
        return None

def get_wikipedia_summary(query):
    try:
        # Search Wikipedia for the query
        page = wiki_wiki.page(query)
        if page.exists():
            return page.summary  # Return the full summary
        else:
            return None
    except Exception as e:
        print(f"Wikipedia Error: {e}")
        return None

def is_wikipedia_query(query):
    """
    Determines if the query is suitable for Wikipedia.
    """
    # List of casual phrases that should not trigger Wikipedia search
    casual_phrases = [
        "hi", "hello", "hey", "how are you", "what's up", "good morning", 
        "good afternoon", "good evening", "how's it going", "sup"
    ]

    # Normalize the query for comparison
    normalized_query = query.lower().strip()

    # Skip Wikipedia search for casual phrases
    if any(phrase in normalized_query for phrase in casual_phrases):
        return False

    # Keywords that indicate a factual question
    wikipedia_keywords = ["who", "what", "where", "when", "why", "how", "tell me about", "explain"]
    return any(k in normalized_query for k in wikipedia_keywords)

async def send_chunked_message(update: Update, text, max_length=400):
    """
    Sends a long message in smaller chunks.
    """
    for i in range(0, len(text), max_length):
        chunk = text[i:i + max_length]
        await update.message.reply_text(render_markdown(chunk), parse_mode="HTML")

async def handle_pdf(update: Update, context: CallbackContext):
    if update.message.document and update.message.document.mime_type == 'application/pdf':
        file = await update.message.document.get_file()
        pdf_file = BytesIO()
        await file.download_to_memory(out=pdf_file)
        pdf_file.seek(0)
        text = extract_text_from_pdf(pdf_file)
        if not text:
            await update.message.reply_text(render_markdown("Couldn't extract text from PDF."), parse_mode="HTML")
            return
        
        # Show typing animation
        await update.message.chat.send_action(ChatAction.TYPING)
        
        summary = get_gemini_response(f"Summarize the following text in 100 words or less:\n{text}")
        await send_chunked_message(update, f"Here's the summary:\n\n{summary}")
    else:
        await update.message.reply_text(render_markdown("Please upload a valid PDF."), parse_mode="HTML")

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(render_markdown("Hello! I'm Abhijith, your friendly Gemini-powered bot. 😊"), parse_mode="HTML")

async def image(update: Update, context: CallbackContext):
    query = ' '.join(context.args)
    if not query:
        await update.message.reply_text(render_markdown("Please provide an image prompt."), parse_mode="HTML")
        return
    
    # Show uploading photo animation
    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    
    image_bytes = generate_image_with_huggingface(query)
    if image_bytes:
        await update.message.reply_photo(photo=image_bytes)
    else:
        await update.message.reply_text(render_markdown("Sorry, the image generation service is currently unavailable. Please try again later."), parse_mode="HTML")

async def handle_message(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    user_message = update.message.text

    city = extract_city_from_query(user_message)
    if city:
        weather = get_weather(city)
        await send_chunked_message(update, weather)
        return

    image_query = extract_image_query(user_message)
    if image_query:
        # Show uploading photo animation
        await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
        
        image_bytes = generate_image_with_huggingface(image_query)
        if image_bytes:
            await update.message.reply_photo(photo=image_bytes)
        else:
            await update.message.reply_text(render_markdown("Sorry, the image generation service is currently unavailable. Please try again later."), parse_mode="HTML")
        return

    # Check if the query is suitable for Wikipedia
    if is_wikipedia_query(user_message):
        wikipedia_summary = get_wikipedia_summary(user_message)
        if wikipedia_summary:
            await update.message.reply_text(render_markdown("Let me check Wikipedia for you... 📚"), parse_mode="HTML")
            await send_chunked_message(update, f"Here's what I found on Wikipedia:\n\n{wikipedia_summary}")
            return

    if user_id not in conversation_history:
        conversation_history[user_id] = []
    conversation_history[user_id].append(f"You: {user_message}")
    
    # Show typing animation
    await update.message.chat.send_action(ChatAction.TYPING)
    
    response = get_gemini_response(user_message, conversation_history[user_id])
    conversation_history[user_id].append(f"Abhijith: {response}")
    await send_chunked_message(update, response)

async def history(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    if conversation_history.get(user_id):
        history_text = "\n".join(conversation_history[user_id])
        await send_chunked_message(update, f"Your conversation history:\n\n{history_text}")
    else:
        await update.message.reply_text(render_markdown("No conversation history yet."), parse_mode="HTML")

async def clear_history(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    conversation_history[user_id] = []
    await update.message.reply_text(render_markdown("Conversation history cleared."), parse_mode="HTML")

def main():
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("image", image))
    application.add_handler(CommandHandler("history", history))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_pdf))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()

if __name__ == '__main__':
    main()