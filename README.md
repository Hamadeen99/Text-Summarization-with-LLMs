# Text Summarization with LLMs

## Overview
This project demonstrates the use of Hugging Face pipelines and pre-trained large language model (LLM), such as (Phi-3), for text summarization. The script generates concise summaries for input text.

---

## Requirments
- PyTorch 1.9+
- Hugging Face Transformers
- LangChain Community
- Einops
- Accelerate
- BitsAndBytes

### Install Dependencies
Run the following commands to install required packages:
```bash
pip install torch transformers
pip install -U langchain-community
pip install -u transformers einops accelerate langchain bitsandbytes
```

---

## Usage
 Run the script:
   ```bash
   python LLMTextSummarization.py
   ```

---

## Examples
### Input:
```
<para>The DataSet object stores training and test data...</para>
```

### Output:
```
<summary>The DataSet object efficiently manages data...</summary>
```

---

## Repository Structure
```
.
├── LLMTextSummarization.py   # Main script for text summarization       
├── README.md                 # Documentation for the project
```


---

##
This project was assigned as part of the NLP and LLM Course, instructed by Professor Ausif Mahmood.

---
