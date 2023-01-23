import transformers
from transformers import utils, pipeline, set_seed
import torch
from flask import Flask, request, render_template, session, redirect


app = Flask(__name__)

# Set the secret key for the session
app.secret_key = 'your-secret-key'

MODEL_NAME = "facebook/opt-125m" 

# Initialize the chat history
history = ["Human: Can you tell me the weather forecast for tomorrow?\nBot: Try checking a weather app like a normal person.\nHuman: Can you help me find a good restaurant in the area\nBot: Try asking someone with a functioning sense of taste.\n"]
generator = pipeline('text-generation', model=f"{MODEL_NAME}", do_sample=True, torch_dtype=torch.half)


# Define the chatbot logic
def chatbot_response(input_text, history):
    # Concatenate the input text and history list
    input_text = "\n".join(history) + "\nHuman: " + input_text + " Bot: "
    set_seed(32)
    response_text = generator(input_text, max_length=1024, num_beams=1, num_return_sequences=1)[0]['generated_text']
    # Extract the bot's response from the generated text
    response_text = response_text.split("Bot:")[-1]
    # Cut off any "Human:" or "human:" parts from the response
    response_text = response_text.split("Human:")[0]
    response_text = response_text.split("human:")[0]
    return response_text


@app.route('/', methods=['GET', 'POST'])
def index():
    global history  # Make the history variable global
    if request.method == 'POST':
        input_text = request.form['input_text']
        response_text = chatbot_response(input_text, history)
        # Append the input and response to the chat history
        history.append(f"Human: {input_text}")
        history.append(f"Bot: {response_text}")
    else:
        input_text = ''
        response_text = ''
    # Render the template with the updated chat history
    return render_template('index.html', input_text=input_text, response_text=response_text, history=history)


@app.route('/reset', methods=['POST'])
def reset():
    global history  # Make the history variable global
    history = ["Bot: Hello, how can I help you today? I am a chatbot designed to assist with a variety of tasks and answer questions. You can ask me about anything from general knowledge to specific topics, and I will do my best to provide a helpful and accurate response. Please go ahead and ask me your first question.\n"]
    # Redirect to the chat page
    return redirect('/')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
