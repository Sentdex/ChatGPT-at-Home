# ChatGPT-at-Home
ChatGPT @ Home: Large Language Model (LLM) chatbot application, written by ChatGPT

I asked ChatGPT to build an LLM-based chatbot app and this was the result. 

<img src="https://pythonprogramming.net/static/images/chatgptathomesocial.png" width="512"/>

### Running float-16 and 8 bit across multiple GPU's
Hugging face accelerate lets you run LLM's over multiple models,
For half precission across multiple gpus accelerate is only needed.

Install hugging face [accelerate](https://huggingface.co/docs/accelerate/v0.3.0/installation.html)

For 8 bit bitsandbytes is needed (needs Turing or Ampere GPUs.)

Install [bitsandbytes](https://pypi.org/project/bitsandbytes/) and see docs

In app.py update the memory_map max memory parameter to use when loading the model across gpus 
this is only for the model, input_text parameters will add to this along with other pipeline variables 
change to meet your gpu memory needs etc below works for 2 nvidia rtx 3090's at 8bit with "facebook/opt-30b"
```
memory_map = {0: "15GiB", 1: "20GiB", "cpu": "40GiB"}
```
For 8 bit to work the model needs to be able to fit on gpus vram, accelerate wont use the 
cpu as well in 8 bit mode. You can load torch.float16 with torch_dtype = torch.float16, 
set load_in_8bit=False, you can even use a ssd cache with offload_folder = "./path/to/ssd" 
for parameter offloading

#### 8 bit
```
generator = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    max_memory=memory_map,
    load_in_8bit=True,
)
```

#### float16
```
generator = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype = torch.float16
    max_memory=memory_map,
    load_in_8bit=False,
    offload_folder = "./path/to/ssd"
```
