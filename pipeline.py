from mlx_lm import load, generate
from IPython.display import Markdown
from typing import List, Dict, Optional

# Dictionary of available models
all_models = {
    "llama3_8b": "mlx-community/Meta-Llama-3-8B-Instruct-8bit",
    "llama3_8b_1048k": "mlx-community/Llama-3-8B-Instruct-1048k-8bit",
    "llama3_70b_1048k": "mlx-community/Llama-3-70B-Instruct-Gradient-1048k-8bit",
}

class LlamaPipeline:
    def __init__(self, model_name: str = "mlx-community/Meta-Llama-3-8B-Instruct-8bit"):
        self.model, self.tokenizer = load(model_name)

    def prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000000,
        top_p: float = 0.9,
        additional_messages: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Prompt the Llama model with a structured input.

        Args:
        system_prompt (str): The system prompt to set the context
        user_prompt (str): The user's prompt/question
        max_tokens (int): Maximum number of tokens to generate
        top_p (float): Controls diversity of generation. Lower values make output more focused.
        additional_messages (List[Dict[str, str]], optional): Additional messages to include in the conversation

        Returns:
        str: The generated response from the model
        """
        # Set up the chat scenario with roles
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Add any additional messages if provided
        if additional_messages:
            messages.extend(additional_messages)

        # Apply the chat template to format the input for the model
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Decode the tokenized input back to text format to be used as a prompt for the model
        prompt = self.tokenizer.decode(input_ids)

        try:
            # Generate a response using the model
            response = generate(
                self.model, 
                self.tokenizer, 
                prompt=prompt,
                max_tokens=max_tokens,
                top_p=top_p
            )
            return response
        except Exception as e:
            return f"An error occurred: {str(e)}"

# Example usage:
# pipeline = LlamaPipeline()
# response = pipeline.prompt(
#     system_prompt="You are a helpful assistant.",
#     user_prompt="What is the capital of France?",
# )
# print(response)
