import torch
import os
import json
import logging
from typing import List, Dict, Union, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel, AutoTokenizer
import time
import re
from datasets import load_dataset, concatenate_datasets
from src.utils.lm_modeling import load_model, load_text2embedding

logger = logging.getLogger(__name__)

model_name = 'sbert'
path = 'dataset/webqsp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

HF_DIR = "hf"
HF_MODELS_DIR = os.path.join(HF_DIR, "models")
HF_DATASETS_DIR = os.path.join(HF_DIR, "datasets")
os.makedirs(HF_DIR, exist_ok=True)
os.makedirs(HF_MODELS_DIR, exist_ok=True)
os.makedirs(HF_DATASETS_DIR, exist_ok=True)
webqsp_dataset_path = os.path.join(HF_DATASETS_DIR, "RoG-webqsp")

class QuestionRewriter:
    """
    A class for rewriting questions using an LLM to improve retrieval performance.
    This can generate multiple variations of the same question to capture different
    aspects and phrasings that might better match the knowledge graph.
    """
    
    def __init__(
        self, 
        model_name: str = "/home/gridsan/mhadjiivanov/meng/G-Retriever/hf/models/Llama-3.2-3B-Instruct", 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_rewrites: int = 3,
        temperature: float = 0.7,
        max_new_tokens: int = 2700
    ):
        """
        Initialize the question rewriter with a specified LLM.
        
        Args:
            model_name: Name of the HuggingFace model to use
            device: Device to run the model on ('cuda', 'cpu', etc.)
            num_rewrites: Number of question rewrites to generate
            temperature: Temperature for generation (higher = more creative)
            max_new_tokens: Maximum number of tokens to generate
        """
        self.model_name = model_name
        self.device = device
        self.num_rewrites = num_rewrites
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        logger.info(f"Loading question rewriter model: {model_name} on {device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
        
        # Set up generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        logger.info(f"Question rewriter initialized successfully")
    
    
    def generate_prompt(self, question: str) -> str:
        """
        Generate a prompt for the LLM to rewrite the question.
        
        Args:
            question: The original question to rewrite
            
        Returns:
            A formatted prompt for the LLM
        """
        return f"""Please break down the process of answering the question into as few subobjectives as possible based on semantic analysis.
  Here is an example:
  Q: Which of the countries in the Caribbean has the smallest country calling code?
  Output: ['Search the countries in the Caribbean', 'Search the country calling code for each Caribbean country', 'Compare the country calling codes to find the smallest one']

  Now you need to directly output subobjectives of the following question in list format without other information or notes. Create output such that applying eval will return a list of strings. Avoid using single or double quotes within the subobjectives.
  Q: {question}
  Output: """
    
    def parse_response(self, response: str) -> List[str]:
        """
        Parse the LLM response to extract the rewritten questions.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            A list of rewritten questions
        """

        try:
            sub_questions = eval(response)
            return sub_questions
        except:
            pattern = r'\[(.*?)\]'
            match = re.search(pattern, response)
            if match:
                # Split by comma and clean up quotes/apostrophes and whitespace
                items = re.findall(r'["\'](.+?)["\']', match.group(1))
                if items:
                    return items
                else:
                    return response[1:-1].split(',')
            else:
                return response[1:-1].split(',')
                



        # Split by newlines and filter out empty lines
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Take only the specified number of rewrites
        rewrites = lines[:self.num_rewrites]
        
        # If we didn't get enough rewrites, log a warning
        if len(rewrites) < self.num_rewrites:
            logger.warning(f"Expected {self.num_rewrites} rewrites but got {len(rewrites)}")
        
        return rewrites
    
    def rewrite_question(self, question: str) -> List[str]:
        """
        Rewrite a question using the LLM to generate variations.
        
        Args:
            question: The original question to rewrite
            
        Returns:
            A list of rewritten questions
        """        
        # Generate prompt
        prompt = self.generate_prompt(question)
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Please output exactly what is requested of you in the requested format."},
            {"role": "user", "content": prompt}]
        
        # Generate rewrites
        start_time = time.time()
        logger.info(f"Generating rewrites for: {question}")
        
        try:
            outputs = self.pipe(messages)
            logger.info(f"Outputs: {outputs}")
            generated_text = outputs[0]['generated_text'][-1]['content']
            
            # Extract the part after the prompt
            response = generated_text.strip()
            
            # Parse the response
            rewrites = self.parse_response(response)
            
            # Add the original question as the first item
            all_questions = [question] + rewrites

            logger.info(f"Question {question} \n Generated response: {response}")
            
            logger.info(f"Generated {len(rewrites)} rewrites in {time.time() - start_time:.2f}s")
            return all_questions
            
        except Exception as e:
            logger.error(f"Error generating rewrites: {str(e)}")
            # Return just the original question if there's an error
            return [question]
    
    def batch_rewrite_questions(self, questions: List[str]) -> Dict[str, List[str]]:
        """
        Rewrite a batch of questions.
        
        Args:
            questions: List of original questions to rewrite
            
        Returns:
            Dictionary mapping original questions to lists of rewrites
        """
        results = {}
        for question in questions:
            results[question] = self.rewrite_question(question)
        return results


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the rewriter
    rewriter = QuestionRewriter()

    #pretrained_repo = 'sentence-transformers/all-roberta-large-v1'
    pretrained_repo = os.path.join(HF_MODELS_DIR, 'all-roberta-large-v1')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_dataset(webqsp_dataset_path)
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    questions = [i['question'] for i in dataset]


    model, tokenizer, device = load_model[model_name]()
    print(device)
    text2embedding = load_text2embedding[model_name]

    # encode questions
    print('Encoding questions...')
    q_embs = text2embedding(model, tokenizer, device, questions)

    rewritten_questions = [[rewriter.rewrite_question(question)] for question in questions]
    rewritten_questions_embs = [text2embedding(model, tokenizer, device, question[0]) for question in rewritten_questions]

    dataset_rewrites = {}
    for i in range(len(dataset)):
        
        dataset_rewrites[i] = {'q': rewritten_questions[i], 'num_q': len(rewritten_questions[i])}
    
    output_dir = "/home/gridsan/mhadjiivanov/meng/G-Retriever/dataset/webqsp"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "webqsp_rewrites.json")
    with open(output_path, "w") as f:
        json.dump(dataset_rewrites, f, indent=2)
    
    # Get max dimension across all embeddings
    max_dim = max(emb.size(0) for emb in rewritten_questions_embs)

    # Pad each embedding tensor with zeros to match max dimension
    padded_embs = []
    for emb in rewritten_questions_embs:
        if emb.size(0) < max_dim:
            padding = torch.zeros((max_dim - emb.size(0), emb.size(1)), device=emb.device)
            padded_emb = torch.cat([emb, padding])
        else:
            padded_emb = emb
        padded_embs.append(padded_emb)

    # Stack all padded embeddings into single tensor
    all_rewritten_embs = torch.stack(padded_embs)
    print(f"Combined tensor shape: {all_rewritten_embs.shape}")

    
    
    # Save dataset rewrites
    

    torch.save(all_rewritten_embs, os.path.join(output_dir, "webqsp_rewrites_embs.pt"))
        
    logger.info(f"Saved rewrites to {output_path}")