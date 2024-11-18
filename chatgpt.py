from openai import OpenAI
import gradio as gr
from dotenv import load_dotenv
import base64
from pdf2image import convert_from_path
import shutil
import os

# Load environment variables
load_dotenv()
client = OpenAI()

# Initialize message history
temp_folder = "temp_images"
messages = [{"role": "system", "content": "You are a health assistant helping a patient with their health concerns."}]

def encode_image(image_path):
    """Encodes an image file in base64 format."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def convert_pdf_to_images(pdf_path, output_folder=temp_folder):
    """Converts a PDF into images and saves them to the specified folder."""
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        pages = convert_from_path(pdf_path, 500)
        image_paths = []
        for count, page in enumerate(pages):
            image_path = os.path.join(output_folder, f"output_{count}.jpg")
            page.save(image_path, "JPEG")
            image_paths.append(image_path)
        return image_paths
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

def handle_file(file):
    """Processes the uploaded file by converting it into images if necessary."""
    
    try:
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == ".pdf":
            return convert_pdf_to_images(file.name, output_folder=temp_folder)
        elif file_extension in [".jpg", ".jpeg", ".png", ".gif"]:
            return [file.name]
        else:
            print(f"Unsupported file format: {file_extension}")
            return []
    except Exception as e:
        print(f"Error handling file: {e}")
        return []

def CustomHealthAssistant(user_input, file, enable_speech):
    """Generates the chatbot's response and optionally prepares TTS output."""
    try:
        if file is not None:
            image_paths = handle_file(file)
            for image_path in image_paths:
                base64_image = encode_image(image_path)
                if base64_image:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_input},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    })
        else:
            messages.append({"role": "user", "content": user_input})

        # Generate chatbot response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        openai_reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": openai_reply})

        # Generate audio response if Speech is enabled
        audio_response = speak_text(openai_reply) if enable_speech else None

        return openai_reply, audio_response
    except Exception as e:
        return f"Error: {str(e)}", None
    finally:
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

def speak_text(output_text):
    """Generates a text-to-speech audio file for the given text."""
    try:
        response = client.audio.speech.create(
            input=output_text,
            model="tts-1",
            voice="nova",
        )
        audio_file = "output.mp3"
        response.stream_to_file(audio_file)
        return audio_file
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

# Gradio Interface
demo = gr.Interface(
    theme=gr.themes.Soft(text_size=gr.themes.sizes.text_lg),
    fn=CustomHealthAssistant,
    inputs=[
        gr.Textbox(placeholder="What do you need help with?", label="My Input"),
        gr.File(label="Upload Documents / Reports"),
        gr.Checkbox(label="Enable Speech Output", value=False)
    ],
    outputs=[
        gr.Textbox(label="Response"),
        gr.Audio(autoplay=True, label="Audio Response", type="filepath")
    ],
    title="Virtual Health Assistant",
    flagging_options=["Incorrect / False Information", "Error Response (No Credits Left)", "Others"]
)

demo.launch(share=True)
