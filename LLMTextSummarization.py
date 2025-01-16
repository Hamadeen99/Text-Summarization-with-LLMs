import sys
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline 


warnings.filterwarnings('ignore')

def summary_generator(text, pipeline):
    """Generates a summary using the provided HuggingFace pipeline."""
    llm_pipeline = HuggingFacePipeline(
        pipeline=pipeline,
        model_kwargs={'temperature': 0}
    )
    summary_prompt = text
    try:
        return llm_pipeline.invoke(summary_prompt)
    except Exception as e:
        print(f"Error in generating summary: {e}")
        return None

def setup_pipeline(model_name, token):
    """Sets up the HuggingFace pipeline with the specified model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
        if torch.cuda.is_available():
            model = model.cuda()  
        else:
            model = model.cpu()  
    except RuntimeError as e:
        print(f"Runtime error during model loading: {e}")
        model = model.cpu()  
    
    is_cuda = next(model.parameters()).is_cuda

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16 if is_cuda else torch.float32,  
        device=0 if is_cuda else -1, 
        max_length=500,
        truncation=True,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

def main():
    """Main function to demonstrate summary generation."""
    token = ""  # Replace with your actual Hugging Face access token
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    llm_pipeline = setup_pipeline(model_name, token)

    context = """ 
    Explain the functionality of an image processing pipeline. Structure the output starting with <summary> and ending with <endsummary>:
    <para>The purpose of the DataSet object is to contain the training and test
    data. If the total training data is small, we can read all the image pixels
    and store all images in multi-dimensional tensors in the dataset.
    However, if the number of images is large and images happen to be relatively
    high resolution, then one option is to store the image filenames (with full
    path info) in the dataset. We can also apply a transform to the images.
    Transform usually resizes the image and changes the 0-255 pixel scale to
    either 0 to 1, or -1 to 1.<endpara>
    """

    print(context)
    result = summary_generator(context, llm_pipeline)
    if result:
        print(result)

if __name__ == "__main__":
    sys.exit(main() or 0)
