import torch
import transformers
from flask import Flask, redirect, render_template, request, session
from transformers import (AutoModelForCausalLM, AutoTokenizer, pipeline,
                          set_seed, utils)

app = Flask(__name__)

# Set the secret key for the session
app.secret_key = "your-secret-key"

MODEL_NAME = "facebook/opt-30b"
MAX_NEW_TOKENS = 50

# Initialize the chat history
history = [
    "Human: Can you tell me the weather forecast for tomorrow?\nBot: Try checking a weather app like a normal person.\nHuman: Can you help me find a good restaurant in the area\nBot: Try asking someone with a functioning sense of taste.\n"
]

# max memory to use when loading the model across gpus
# this is only for the model input_text will add to this
# along with model interemt variables
# change to meet your gpu memory etc
# this works for 2 nvidia rtx 3090's at 8bit
memory_map = {0: "15GiB", 1: "20GiB", "cpu": "40GiB"}

# for 8 bit to work the model needs to be able to fit on gpus
# this wont use the cpu as well you can load torch.float16
# with torch_dtype = torch.float16, set load_in_8bit=False,
# you can even use a ssd cache with offload_folder = "./path/to/ssd"
generator = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    max_memory=memory_map,
    load_in_8bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")


# Define the chatbot logic
def chatbot_response(input_text, history):
    # Concatenate the input text and history list
    input_text = "\n".join(history) + "\nHuman: " + input_text + " Bot: "
    set_seed(32)

    # tokenize the input text and accelorate needs the data sent to 0
    input_text = tokenizer.encode(input_text, return_tensors="pt").to(0)
    # get raw response
    response_text = generator.generate(input_text, max_new_tokens=MAX_NEW_TOKENS)
    # decode the response
    response_text = tokenizer.decode(response_text[0].tolist())

    # Extract the bot's response from the generated text
    response_text = response_text.split("Bot:")[-1]
    # Cut off any "Human:" or "human:" parts from the response
    response_text = response_text.split("Human:")[0]
    response_text = response_text.split("human:")[0]
    return response_text


@app.route("/", methods=["GET", "POST"])
def index():
    global history  # Make the history variable global
    if request.method == "POST":
        input_text = request.form["input_text"]
        response_text = chatbot_response(input_text, history)
        # Append the input and response to the chat history
        history.append(f"Human: {input_text}")
        history.append(f"Bot: {response_text}")
    else:
        input_text = ""
        response_text = ""
    # Render the template with the updated chat history
    return render_template(
        "index.html", input_text=input_text, response_text=response_text, history=history
    )


@app.route("/reset", methods=["POST"])
def reset():
    global history  # Make the history variable global
    history = [
        "Bot: Hello, how can I help you today? I am a chatbot designed to assist with a variety of tasks and answer questions. You can ask me about anything from general knowledge to specific topics, and I will do my best to provide a helpful and accurate response. Please go ahead and ask me your first question.\n"
    ]
    # Redirect to the chat page
    return redirect("/")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
