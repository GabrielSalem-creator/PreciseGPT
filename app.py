import os
import json
from flask import Flask, render_template, request, redirect, url_for
from huggingface_hub import InferenceClient
from docx import Document  # For handling .docx files

# Initialize Flask app and Hugging Face client
app = Flask(__name__)
app.secret_key = 'your_secret_key'

client = InferenceClient(api_key="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")
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model data and click tracking
loaded_models = {}
model_clicks = {}  # Track clicks for each model


# Function to preload JSON models at the start of the app
def preload_models():
    for file in os.listdir(UPLOAD_FOLDER):
        if file.endswith('.json'):
            model_name = file.replace('.json', '')
            with open(os.path.join(UPLOAD_FOLDER, file), 'r') as f:
                model_data = json.load(f)

                # Add concise response prompt if not already included
                system_prompt = {
                    "role": "system",
                    "content": "Respond concisely. Keep answers brief and avoid unnecessary details."
                }
                if not model_data or model_data[0].get("role") != "system":
                    model_data.insert(0, system_prompt)

                loaded_models[model_name] = model_data
                model_clicks[model_name] = 0  # Initialize click count


# Function to parse a .docx file into text
def parse_docx(file_path):
    doc = Document(file_path)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)


# Function to get the bot's response from Hugging Face API
def get_bot_response(conversation_history, json_data):
    # Include the system prompt explicitly in the merged messages
    messages = json_data + conversation_history

    try:
        # Create a chat stream request to Hugging Face
        stream = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            messages=messages,
            max_tokens=250,
            stream=True
        )

        # Collect the response from the stream as it arrives
        bot_response = ""
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                new_content = chunk.choices[0].delta.content
                bot_response += new_content
                print(new_content, end="")  # Print the response incrementally
        return bot_response
    except Exception as e:
        print(f"Error during API request: {e}")
        return "An error occurred while generating the response."


# Home page route - Display models and allow creating new ones
# Home page route - Display models and allow creating new ones
@app.route('/', methods=['GET', 'POST'])
def home():
    # Ensure model_clicks is initialized for any model that is present in loaded_models
    for model_name in loaded_models.keys():
        if model_name not in model_clicks:
            model_clicks[model_name] = 0  # Initialize click count for new models

    # Fetch all models for display
    models = [{'model_name': model_name, 'clicks': model_clicks[model_name]} for model_name in loaded_models.keys()]
    models.sort(key=lambda x: x['clicks'], reverse=True)  # Sort by clicks, descending

    if request.method == 'POST':
        # Handle file uploads and model creation
        uploaded_file = request.files.get('file')
        model_name = request.form.get('model_name')
        content = request.form.get('content')

        if uploaded_file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(filename)

            # Handle text files
            if uploaded_file.filename.endswith('.txt'):
                with open(filename, 'r') as f:
                    file_content = f.read()
                    model_data = [{"role": "user", "content": file_content}]
                    loaded_models[model_name] = model_data

            # Handle .docx files
            elif uploaded_file.filename.endswith('.docx'):
                file_content = parse_docx(filename)
                model_data = [{"role": "user", "content": file_content}]
                loaded_models[model_name] = model_data

            # Handle JSON files
            elif uploaded_file.filename.endswith('.json'):
                with open(filename, 'r') as f:
                    loaded_models[model_name] = json.load(f)

        elif content:
            # If no file is uploaded, create a model from the text
            model_data = [{"role": "user", "content": content}]
            loaded_models[model_name] = model_data

        return redirect(url_for('home'))

    # Return all models (not just top 5) for rendering
    return render_template('home.html', models=models)



# Route to handle model clicks and increment the click count
@app.route('/click/<model_name>', methods=['POST'])
def track_click(model_name):
    if model_name in model_clicks:
        model_clicks[model_name] += 1  # Increment click count
    return '', 204  # Respond with no content


# Route to create a new model
@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        # Handle model creation from file or text input
        uploaded_file = request.files.get('file')
        model_name = request.form.get('model_name')
        content = request.form.get('content')

        # Define concise response system message
        system_prompt = {
            "role": "system",
            "content": "Respond concisely. Keep answers brief and avoid unnecessary details."
        }

        if uploaded_file:
            # Save the uploaded file to the server
            filename = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(filename)

            # Handle text files
            if uploaded_file.filename.endswith('.txt'):
                with open(filename, 'r') as f:
                    file_content = f.read()
                    model_data = [system_prompt, {"role": "user", "content": file_content}]
                    loaded_models[model_name] = model_data

            # Handle .docx files
            elif uploaded_file.filename.endswith('.docx'):
                file_content = parse_docx(filename)
                model_data = [system_prompt, {"role": "user", "content": file_content}]
                loaded_models[model_name] = model_data

            # Handle JSON files
            elif uploaded_file.filename.endswith('.json'):
                with open(filename, 'r') as f:
                    model_data = json.load(f)
                    # Add the system message if not already included
                    if not model_data or model_data[0].get("role") != "system":
                        model_data.insert(0, system_prompt)

                    loaded_models[model_name] = model_data

            # Save the uploaded model as a JSON file in the upload folder
            with open(os.path.join(app.config['UPLOAD_FOLDER'], f"{model_name}.json"), 'w') as json_file:
                json.dump(model_data, json_file, indent=4)

        elif content:
            # If no file is uploaded, create a model from the text
            model_data = [system_prompt, {"role": "user", "content": content}]
            loaded_models[model_name] = model_data

            # Save the model data as a JSON file
            with open(os.path.join(app.config['UPLOAD_FOLDER'], f"{model_name}.json"), 'w') as json_file:
                json.dump(model_data, json_file, indent=4)

        # After saving, redirect to the home page
        return redirect(url_for('home'))

    return render_template('create.html')


# Chat page route - Where users can chat with a selected model
@app.route('/chat/<model_name>', methods=['GET', 'POST'])
def chat(model_name):
    # Track click for the selected model
    if request.method == 'GET':
        # Increment the click count when the model is viewed
        track_click(model_name)

    json_data = loaded_models.get(model_name, [])
    conversation_history = []

    if request.method == 'POST':
        user_input = request.json.get('user_input')
        if user_input:
            # Add user input to conversation history
            conversation_history.append({'role': 'user', 'content': user_input})

            # Get the bot's response
            bot_response = get_bot_response(conversation_history, json_data)

            # Append bot's response to the conversation history
            conversation_history.append({'role': 'assistant', 'content': bot_response})

            # Return the bot's response as JSON
            return {"response": bot_response}

    # Render chat.html for GET requests
    return render_template('chat.html', conversation_history=conversation_history, model_name=model_name)


if __name__ == "__main__":
    # Preload models at the start of the application
    preload_models()
    app.run(debug=True)

#git add .
#git commit -m "Updated the app functionality"
#git push heroku main