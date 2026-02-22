# **DocInsight — PDF Analysis with NLP**

A Python project that takes any PDF, pulls out the text, summarizes it, and then automatically generates questions and answers from the content — all saved into a neat report file. Built as a learning project to explore how NLP pipelines work with real documents.



**What This Project Does:**

The idea is straightforward. You point it at a PDF, and it does the following:

1.Extracts all the text from the PDF page by page

2.Saves a preview so you can check the extraction worked correctly

3.Summarizes the document using a pretrained model

4.Splits the document into manageable passages

5.Generates questions from each passage automatically

6.Finds answers to those questions from within the same passage

7.Saves everything — summary, questions, and answers — into a .txt report file



**Tools and Libraries Used:**


**pdfplumber** — used to open and read text from PDF files page by page. It handles most PDFs well and is straightforward to use. Alternatives like Camelot or PyMuPDF could also work here, with PyMuPDF being faster for large files and Camelot being better if your PDF has a lot of tables.

**NLTK (Natural Language Toolkit)** — used to split the extracted text into individual sentences, which are then grouped into passages of around 250 words each. The punkt tokenizer from NLTK handles sentence boundaries reliably for most English text.

**HuggingFace Transformers** — the core of the project. Three different models are used through the pipeline API:

**sshleifer/distilbart-cnn-12-6 for summarization** - This is a distilled (compressed) version of Facebook's BART model, trained on news articles. It gives noticeably better summaries than the lighter t5-small model while still being reasonably fast. If you want even better quality and don't mind waiting longer, facebook/bart-large-cnn is worth trying.

**valhalla/t5-base-qg-hl for question generation** - This is a T5 model fine-tuned specifically on question generation tasks, which is why it produces more natural and relevant questions compared to using a general-purpose model.

**deepset/roberta-base-squad2 for question answering** - RoBERTa fine-tuned on SQuAD2 is one of the more reliable extractive QA models available — it finds the answer directly within the passage rather than generating one from scratch.

**Why transformers==4.35.0 specifically?** - Newer versions of the transformers library removed "summarization" as a direct pipeline task and changed some internal APIs. Version 4.35.0 is the version where everything in this project works together without conflicts. Similarly, **accelerate==0.24.0**, **huggingface_hub==0.17.3**, and **tokenizers==0.14.1** are pinned to match — mixing versions of these packages causes import errors that are frustrating to debug.

**sentencepiece and protobuf** — required by the T5 tokenizer under the hood. They don't show up in your code directly but the pipeline will throw an error without them.

**PyTorch (CPU version)** — installed via the official PyTorch CPU wheel rather than the default pip version. The CPU-only build is significantly smaller and installs much faster, which is fine for inference tasks like this where we are not training any models.


**How to Run It** 

Step 1 — Set up a fresh environment (Always recommended)
```bash
conda create -n docanalysis python=3.10
conda activate docanalysis
```

Step 2 — Install dependencies
```bash 
pip install pdfplumber nltk sentencepiece protobuf
pip install transformers==4.35.0 accelerate==0.24.0 huggingface_hub==0.17.3 tokenizers==0.14.1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install jupyter
jupyter notebook
```

Step 3 — Update the PDF path

In the extraction cell, change this line to point to your own PDF:

```bash
pythonpdf_path = r"C:\your\path\to\yourfile.pdf"
```
Step 4 — Run all cells in order
The output report will be saved in the same directory as your notebook, named after your PDF file.


**A Note on the Accuracy of the Results**

The questions and answers this project generates are not always going to be correct, and that is worth being upfront about.
The question generation model does a decent job on well-structured text but can produce oddly worded or repetitive questions, especially when the passage has formatting issues from the PDF extraction (like reference lists or columns that got jumbled during parsing). The QA model extracts answers directly from the passage, which means it can only find something that is literally written there — it does not reason or infer. Sometimes the answer it returns is technically in the passage but is not the most useful or meaningful response to the question asked.
For a document like an academic paper, the results are more reliable in the body sections and less reliable in the references section, where the text is dense with citations and not written in natural prose.
Take the outputs as a starting point for reading the document, not as a substitute for reading it yourself.


**What Could Be Better**

There are a few obvious things that could be improved with more time:-

Better PDF parsing - pdfplumber does a good job but multi-column academic papers often get their text scrambled during extraction. PyMuPDF handles layout detection better and would likely produce cleaner text for research papers.

Smarter passage splitting - the current approach splits by word count, which sometimes cuts passages mid-topic. Splitting by semantic similarity or by document headings would produce more coherent passages and therefore better questions.

A more capable summarization model - shleifer/distilbart-cnn-12-6 was trained on news articles, so it works best on that kind of writing. For academic papers or legal documents, a model fine-tuned on those domains would produce more relevant summaries.

Filtering the references section - academic papers end with a long references list that produces noisy, low-quality questions. Adding a step to detect and skip that section would clean up the output considerably.

A simple UI - right now everything runs in a notebook. Even a basic Gradio or Streamlit interface where you upload a PDF and get the report back would make this much more usable for someone who is not comfortable with Jupyter.
