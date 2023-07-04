import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "Salesforce/xgen-7b-8k-inst", 
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "Salesforce/xgen-7b-8k-inst", 
    torch_dtype=torch.bfloat16,
    load_in_8bit = True if torch.cuda.is_available() else False,
    device_map = "auto"
)

def summarize_fn(text):

    # Taken from Model card
    header = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "Tha assistant gives helpfull, detailed and polite answers to the human's questions.\n\n"
    )
    
    prompt = f"### Human: Please summarize the following article: \n\n{text} \n###"
    
    inputs = tokenizer(header + prompt, return_tensors="pt")#.to(device)
    
    # https://huggingface.co/docs/transformers/main_classes/text_generation
    generated_ids = model.generate(
        **inputs, 
        max_length=512,
        do_sample=True,
        top_p=0.95,
        top_k=50,   # 100
        temperature=0.7,    
    )

    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True).lstrip()
    
    # summary starts with ### Assistant: and ends with <|endoftext|>
    # Get just the answer
    output = output.split("### Assistant:")[1]
    output = output.split("<|endoftext|>")[0]
    
    return gr.Textbox.update(value=output)

title = """
<h1 style='text-align: center'> XGen-7B summarizer</p>
"""

description = """
`Salesforce/xgen-7b-8k-inst`: a **XGen-7B** model with standard dense attention (same as LLaMa) on up to **8K sequence length** for up to **1.5T tokens**, finetuned on public-domain instructional data.
* HF `model card`: [xgen-7b-8k-inst](https://huggingface.co/Salesforce/xgen-7b-8k-inst)
* Note that this demo doesn't run on a small resource environment, `basic CPU plan` (`2 vCPU, 16GB RAM`). Try to duplicate the space and run on a **GPU** (Spaces PRO)
* Or try running on Colab <a href="https://colab.research.google.com/github/flaviodeoliveira/xgen-7b-summarization/blob/main/notebook/xgen-7b-summarization.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
"""

article = "<p style='text-align: center'>Salesforce's blog post: <a href='https://blog.salesforceairesearch.com/xgen/' target='_blank'>Long Sequence Modeling with a 7B LLM Trained on 8K Input Sequence Length</a></p>"

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    title = gr.HTML(title),
    description = gr.Markdown(description),

    with gr.Row():

        input = gr.Textbox(lines=10, label="Text to summarize")
        output = gr.Textbox(lines=10, label="Summary")

    with gr.Row():
        
        button = gr.Button(value="Summarize")    
        button_clear = gr.ClearButton(value="Clear")

    article = gr.HTML(article)

    button.click(summarize_fn, inputs=[input], outputs=[output])
    button_clear.click(lambda: [None, None], outputs=[input, output])

if __name__ == "__main__":

    demo.launch()