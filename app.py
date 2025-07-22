import os, io, asyncio, tempfile, traceback
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from dotenv import load_dotenv
import chainlit as cl
from google.genai import types
from PIL import Image
from io import BytesIO
from google import genai
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib


# Available models
AVAILABLE_MODELS = {
    "Gemini 2.0 Flash Experimental": "gemini-2.0-flash-exp",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.0 Image Generation": "gemini-2.0-flash-preview-image-generation",
    "Gemini 2.0 Flash Lite": "gemini-2.0-flash-lite"
}
DEFAULT_MODEL = "gemini-2.0-flash-lite"
current_model = DEFAULT_MODEL

GEMINI_AVAILABLE = False

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables or .env file")

# Initialize Gemini client
client = genai.Client(api_key=gemini_api_key)
GEMINI_AVAILABLE = True
# Generation configuration
generation_config = types.GenerateContentConfig(
    temperature=0,
    max_output_tokens=8192,
    response_mime_type="text/plain"
)

# Image generation config
image_generation_config = types.GenerateContentConfig(
    response_modalities=["IMAGE", "TEXT"],
    response_mime_type="text/plain"
)

def savefig(fig):
    """Save a matplotlib figure to a file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        fig.savefig(tmpfile.name, bbox_inches='tight', dpi = 150)
        plt.close(fig)
    return tmpfile.name

def df_to_string(df,max_rows=10):
    """Convert a DataFrame to a string representation."""
    buf = io.StringIO()
    df.info(buf = buf)
    schema = buf.getvalue()
    head = df.head(max_rows).to_markdown(index=False)
    
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_info = "No missing values" if missing.empty else f"Missing values:\n{missing.to_string()}"
    
    return f"### Schema:\n{schema}\n\n### Head:\n{head}\n\n### Missing:\n{missing_info}"

async def text_analysis(prompt_type,df_context):
    if not GEMINI_AVAILABLE:
        return "Gemini API is not available."
    prompts = {
        "plan": f"You are a data analyst. Suggest a concise data analysis plan for the following DataFrame:\n{df_context}",
        "final": f"Summarize the analysis results for the following dataset:\n{df_context}",
    }
    
    try:
        # model = genai.GenerativeModel(GEMINI_MODEL)
        contents = [
            genai.types.Content(
                role="user",
                parts=[genai.types.Part.from_text(text=prompts.get(prompt_type, ""))]
            )
        ]
        res = client.models.generate_content(
            model = current_model ,
            contents= contents,
            config={
                'temperature' : 0.0,
                'max_output_tokens' : 1024,
            }    
        )
        if res.candidates and len(res.candidates) > 0:
            candidate = res.candidates[0]
            if candidate.content and candidate.content.parts:
                return candidate.content.parts[0].text
            else:
                return "Gemini response blocked or empty."
        else:
            return "No response generated."
            
    except Exception as e:
        return f"Error during text analysis: {str(e)}\n{traceback.format_exc()}"
     
async def vision_analysis(img_paths):
    if not GEMINI_AVAILABLE:
        return "Gemini API is not available."
    
    result = []
        
    for title, img_path in img_paths:
        try:
            # Read image file
            with open(img_path, "rb") as img_file:
                img_data = img_file.read()
            
            # Detect image MIME type based on file extension
            if img_path.lower().endswith('.png'):
                mime_type = "image/png"
            elif img_path.lower().endswith(('.jpg', '.jpeg')):
                mime_type = "image/jpeg"
            elif img_path.lower().endswith('.webp'):
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"  # default
            
            # Create contents in the correct format
            contents = [
                genai.types.Content(
                    role="user",
                    parts=[
                        genai.types.Part.from_text(text=f"Analyze the image titled '{title}' and provide insights."),
                        genai.types.Part.from_bytes(data=img_data, mime_type=mime_type)
                    ]
                )
            ]
            
            # Generate content using non-streaming API
            response = client.models.generate_content(
                model=current_model,
                contents=contents,
                config={
                    'temperature': 0.0,
                    'max_output_tokens': 1024,
                }
            )
            
            # Extract text from response
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    result.append((title, candidate.content.parts[0].text))
                else:
                    result.append((title, "Gemini response blocked."))
            else:
                result.append((title, "No response generated."))
                
        except Exception as e:
            result.append((title, f"Error: {str(e)}"))
    
    return result

def generate_visuals(df):
    """Generate visualizations for the DataFrame."""
    visuals = []
    saved_images = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df.select_dtypes('object') if 1 < df[col].nunique() < 30]
    
    try:
        if numeric_cols:
            # Histogram for numeric columns
            for col in numeric_cols:
                fig, ax = plt.subplots()
                df[col].hist(ax=ax, bins=30)
                ax.set_title(f"Histogram of {col}")
                img_path = savefig(fig)
                visuals.append(cl.Image(name=f"Histogram of {col}",path=img_path))
                saved_images.append(img_path)

            # Correlation heatmap
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
                ax.set_title("Correlation Heatmap")
                img_path = savefig(fig)
                visuals.append(cl.Image(name="Correlation Heatmap",path=img_path))
                saved_images.append(img_path)

        if categorical_cols:
            # Bar plots for categorical columns
            for col in categorical_cols:
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f"Bar Plot of {col}")
                img_path = savefig(fig)
                visuals.append(cl.Image(name=f"Bar Plot of {col}",path=img_path))
                saved_images.append(img_path)
    except Exception as e:
        print(f"Error generating visuals: {e}")
        plt.close('all')
        
    return visuals, saved_images

async def cleanup_images(saved_images):
    """Clean up temporary image files."""
    for img_path in saved_images:
        try:
            os.remove(img_path)
        except Exception as e:
            pass

async def process_csv_file(file_path):
    """Process uploaded CSV file and perform EDA"""
    processing_msg = cl.Message(content="Processing your CSV file, please wait...")
    await processing_msg.send()

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        df = pd.read_csv(io.StringIO(content))
        
        if df.empty:
            processing_msg.content="The uploaded file is empty or invalid."
            await processing_msg.update()
            return
            
        cl.user_session.set("df", df)
        info = df_to_string(df)
        await cl.Message(content=info).send()
        
        if GEMINI_AVAILABLE:
            plan = await text_analysis("plan", info)
            await cl.Message(content=f"### Analysis Plan: \n{plan}").send()

        visuals, saved_images = generate_visuals(df)
        batch_size = 7
        for i in range(0, len(visuals), batch_size):
            batch = visuals[i:i+batch_size]
            if batch:  # Only send if batch is not empty
                await cl.Message(
                    content=f"**Generated Visualizations (batch {i//batch_size+1}):**",
                    elements=batch
                ).send()

        visuals = [(img.name, img.path) for img in visuals]
        if GEMINI_AVAILABLE:
            insights = await vision_analysis(visuals)
            for title, insight in insights:
                await cl.Message(content=f"**Insights for {title}:**\n{insight}").send()
            
            final = await text_analysis("final", info)
            await cl.Message(content=f"### Final Summary:\n{final}").send()
        
        processing_msg.content="CSV analysis complete! You can now continue chatting or upload another file."
        await processing_msg.update()
        await cleanup_images([path for _, path in visuals])
    
    except Exception as e:
        processing_msg.content=f"An error occurred during CSV processing: {str(e)}"
        await processing_msg.update()
        print(f"Error: {e}\n{traceback.format_exc()}")

@cl.on_chat_start
async def start_chat():
    cl.user_session.set("current_model", DEFAULT_MODEL)
    cl.user_session.set("generation_config", generation_config)
    
    await cl.ChatSettings([
        cl.input_widget.Select(
            id="model_selector",
            label="Select AI Model",
            values=list(AVAILABLE_MODELS.keys()),
            initial_value=[k for k, v in AVAILABLE_MODELS.items() if v == DEFAULT_MODEL][0]
        )
    ]).send()

    welcome = """
# Gemini EDA Assistant

Welcome to the **Gemini EDA Assistant** with Dataframe analysis and image generation support!

## Getting Started
You can start chatting immediately! The assistant is ready to help with various tasks.

### Available Models
- **Gemini 2.0 Flash Experimental**: Lightweight and experimental
- **Gemini 2.5 Pro**: Advanced reasoning capabilities
- **Gemini 2.5 Flash**: Balanced performance
- **Gemini 2.0 Image Generation**: Create images from text prompts

### Features
- **Normal Chat**: Ask questions, get help with coding, writing, analysis, etc.
- **Image Generation**: Start your prompt with "/image" or "generate an image of"
- **CSV Analysis**: Upload a CSV file anytime during our conversation for automated EDA

### Commands
- `/upload` - Upload a CSV file for analysis
- `/image [description]` - Generate an image

---
*Ready to chat! Feel free to ask questions or upload a CSV file for analysis.*
"""
    await cl.Message(content=welcome.strip()).send()
    
@cl.on_settings_update
async def setup_chat_settings(settings):
    selected_model_name = settings["model_selector"]
    selected_model = AVAILABLE_MODELS[selected_model_name]

    cl.user_session.set("current_model", selected_model)
    cl.user_session.set("generation_config", generation_config)

    await cl.Message(
        content=f"**Settings Updated** Now using: `{selected_model_name}` model."
    ).send()

async def handle_image_generation(prompt: str):
    """Handle image generation requests"""
    msg = cl.Message(author="Gemini Image Generator", content="Generating your image...")
    await msg.send()
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    ]
    
    try:
        stream = client.models.generate_content_stream(
            model="gemini-2.0-flash-preview-image-generation",
            contents=contents,
            config=image_generation_config
        )
        
        for chunk in stream:
            if (chunk.candidates and 
                chunk.candidates[0].content and 
                chunk.candidates[0].content.parts):
                
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        # Handle image data
                        image_data = part.inline_data.data
                        image = Image.open(BytesIO(image_data))
                        
                        # Create Chainlit image element
                        image_element = cl.Image(
                            name="generated-image",
                            display="inline",
                            size="large",
                            content=image_data
                        )
                        
                        await msg.remove()
                        await cl.Message(
                            author="Gemini Image Generator",
                            content=f"Here's your generated image:",
                            elements=[image_element]
                        ).send()
                        return
                    elif hasattr(part, "text"):
                        await msg.stream_token(part.text)
        
        await msg.update()
        
    except Exception as e:
        error_msg = f"\n**Error**: Unable to generate image. Details: {str(e)}"
        await msg.stream_token(error_msg)
        print(f"Error: {e}")

@cl.on_message
async def main(message: cl.Message):
    current_model = cl.user_session.get("current_model", DEFAULT_MODEL)
    config = cl.user_session.get("generation_config", generation_config)
    model_display_name = [k for k, v in AVAILABLE_MODELS.items() if v == current_model][0]
    
    # Check if user wants to upload a CSV file
    if message.content.lower().strip() in ["/upload", "upload csv", "upload a csv", "analyze csv"]:
        files = await cl.AskFileMessage(
            content="Please upload a CSV file for analysis.",
            accept=["text/csv"],
            max_files=1
        ).send()
        
        if files and len(files) > 0:
            await process_csv_file(files[0].path)
        else:
            await cl.Message(content="No file uploaded. You can try again anytime by typing `/upload`.").send()
        return

    # Handle file attachments (CSV files)
    if message.elements:
        csv_files = [file for file in message.elements if hasattr(file, 'path') and file.path.lower().endswith('.csv')]
        if csv_files:
            await process_csv_file(csv_files[0].path)
            return

    # Check if this is an image generation request
    if message.content.lower().startswith(("/image", "generate an image of")):
        await handle_image_generation(message.content)
        return

    # Normal chat handling
    msg = cl.Message(author=model_display_name, content="")
    await msg.send()

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=message.content)]
        )
    ]

    full_response = ""
    try:
        stream = client.models.generate_content_stream(
            model=current_model,
            contents=contents,
            config=config
        )

        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                full_response += text
                await msg.stream_token(text)
            elif getattr(chunk, "candidates", None):
                for candidate in chunk.candidates:
                    parts = getattr(candidate.content, "parts", [])
                    for part in parts:
                        if hasattr(part, "text"):
                            full_response += part.text
                            await msg.stream_token(part.text)

    except Exception as e:
        error_msg = f"\n**Error**: Unable to process request with {model_display_name}. Details: {str(e)}"
        await msg.stream_token(error_msg)
        print(f"Error: {e}")

    await msg.stream_token(f"\n\n---\n**Model**: {model_display_name}")
    await msg.update()