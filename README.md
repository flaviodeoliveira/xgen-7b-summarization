# xgen-7b-summarization
Gradio App to summarize texts using `xgen-7b-8k-inst` model, an **XGen-7B** model with standard dense attention (same as LLaMa) on up to **8K sequence length** for up to **1.5T tokens**, finetuned on public-domain instructional data.

* Salesforce's blog post: [Long Sequence Modeling with a 7B LLM Trained on 8K Input Sequence Lengt]("https://blog.salesforceairesearch.com/xgen/")
* HF `model card`: [xgen-7b-8k-inst](https://huggingface.co/Salesforce/xgen-7b-8k-inst)
* Note that this demo doesn't run on a small resource Spaces' environment as it is. Try upgrading or try running on Colab (<a href="https://colab.research.google.com/github/flaviodeoliveira/xgen-7b-summarization/blob/main/notebook/xgen-7b-summarization.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>).