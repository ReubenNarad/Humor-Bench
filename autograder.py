# This is where the autograder client will be implemented.
import sys
import os
# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MessageChain
import re
import asyncio
from dotenv import load_dotenv
from prompts import autograder_prompt

# Load environment variables from .env file
load_dotenv()

class AutograderClient:
    """
    A client for grading explanations of cartoon jokes.
    Works with different LLM providers (OpenAI, DeepSeek, Gemini, Claude, Together).
    Supports extended thinking for compatible Claude models.
    """
    
    def __init__(self, model, family=None, api_key=None, thinking_budget=None):
        """
        Initialize the autograder client.
        
        Args:
            model: The model name to use (e.g., "gpt-4o", "claude-3.7-sonnet...")
            family: The model family (optional, inferred if not provided)
            api_key: API key for the specified model family (optional)
            thinking_budget: Max tokens for Claude's extended thinking (optional)
        """
        self.model = model
        self.thinking_budget = thinking_budget
        
        # Infer the family from the model name if not provided
        if family is None:
            family = self._infer_family_from_model(model)
        
        self.family = family
        self.api_key = api_key
        
        # Validate family
        if family not in [MessageChain.OPENAI, MessageChain.DEEPSEEK, MessageChain.GEMINI, MessageChain.CLAUDE, MessageChain.TOGETHER]:
            raise ValueError(f"Unsupported model family: {family}")
        
        # Initialize the appropriate client based on the family
        self.client = self._initialize_client()
        
        # Optional: Add a check if budget is provided for non-Claude models
        if self.thinking_budget is not None and self.family != MessageChain.CLAUDE:
             print(f"Warning: thinking_budget provided for non-Claude model '{self.model}' (Family: {self.family}). It will be ignored.")
    
    def _infer_family_from_model(self, model):
        """
        Infer the model family from the model name.
        
        Args:
            model: Model name
            
        Returns:
            Inferred model family
        """
        model_lower = model.lower()
        
        # Check for specific prefixes first for Together API models
        if any(prefix in model_lower for prefix in ['meta-llama/', 'mistralai/', 'qwen/', 'deepseek-ai/', 'togethercomputer/']):
             return MessageChain.TOGETHER
        # General keywords for Together (if prefix doesn't match)
        elif any(name in model_lower for name in ['llama', 'mistral', 'mixtral', 'falcon']):
             return MessageChain.TOGETHER
        # Then check other families
        elif any(name in model_lower for name in ['gpt', 'o1', 'o3', 'davinci', 'curie', 'babbage', 'ada']):
            return MessageChain.OPENAI
        elif 'claude' in model_lower:
            return MessageChain.CLAUDE
        elif 'gemini' in model_lower:
            return MessageChain.GEMINI
        # Keep DeepSeek specific check last in case there are other deepseek models not via Together
        # OR remove it if ALL deepseek models you use are via Together
        # elif 'deepseek' in model_lower:
        #     return MessageChain.DEEPSEEK # Keep or remove based on your usage
        else:
            # Fallback or raise error
            print(f"Warning: Could not definitively infer family for {model}. Defaulting to TOGETHER.")
            # Or be stricter:
            # raise ValueError(f"Could not infer model family from model name: {model}. Please specify the family explicitly.")
            return MessageChain.TOGETHER # Defaulting to Together as a common provider
    
    def _initialize_client(self):
        """
        Initialize the appropriate ASYNCHRONOUS client based on the model family.
        
        Returns:
            Initialized asynchronous client
        """
        # Get API key if not provided
        api_key = self.api_key
        
        if self.family == MessageChain.OPENAI:
            from openai import AsyncOpenAI
            
            if not api_key:
                # Get from environment
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not found in environment variables")
            
            return AsyncOpenAI(api_key=api_key)
            
        elif self.family == MessageChain.DEEPSEEK:
            from openai import OpenAI
            
            if not api_key:
                # Get from environment
                api_key = os.environ.get("DEEPSEEK_API_KEY")
                if not api_key:
                    raise ValueError("DeepSeek API key not found in environment variables")
            
            return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            
        elif self.family == MessageChain.GEMINI:
            from google import genai
            
            if not api_key:
                # Get from environment
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("Gemini API key not found in environment variables")
            
            # Create and return a Client object instead of configuring globally
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.model)
            
        elif self.family == MessageChain.CLAUDE:
            from anthropic import AsyncAnthropic
            
            if not api_key:
                # Get from environment
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("Anthropic API key not found in environment variables")
            
            return AsyncAnthropic(api_key=api_key)
        
        elif self.family == MessageChain.TOGETHER:
            from openai import AsyncOpenAI
            
            if not api_key:
                # Get from environment
                api_key = os.environ.get("TOGETHER_API_KEY")
                if not api_key:
                    raise ValueError("Together API key not found in environment variables")
            
            return AsyncOpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
    
    def format_prompt(self, description, caption, explanation, anticipated_point):
        """
        Format the prompt with the specific row data.
        
        Args:
            description: The cartoon description
            caption: The caption for the cartoon
            explanation: The student's explanation of the joke
            anticipated_point: The specific point that should be addressed
            
        Returns:
            Formatted prompt string
        """
        prompt = autograder_prompt(description, caption, explanation, anticipated_point)
        return prompt
    
    def parse_response(self, response_text):
        """
        Parse the model response to extract reasoning and judgment.
        Handles potential thinking blocks (ignores them for the final judgment/reasoning).
        
        Args:
            response_text: The raw text response from the LLM (can be simple text or structured content list)
            
        Returns:
            Tuple of (judgment, reasoning)
        """
        # Initialize default values
        judgment = "ERROR"
        reasoning = "Parsing error: Could not extract judgment/reasoning"

        # Check if the response is likely a list of content blocks (Claude thinking)
        if isinstance(response_text, list):
            # Find the first 'text' block
            text_content = None
            for block in response_text:
                # Need to handle the actual ContentBlock object if using sdk > 0.27.0
                # block_dict = block.model_dump() if hasattr(block, 'model_dump') else block
                block_type = getattr(block, 'type', None)
                if block_type == "text":
                    text_content = getattr(block, 'text', "")
                    break
            if text_content is None:
                 reasoning = "Parsing error: No text block found in Claude response"
                 return judgment, reasoning

            # Parse XML tags within the text block
            response_to_parse = text_content

        elif isinstance(response_text, str):
            # Original logic for non-Claude or non-thinking responses
            response_to_parse = response_text
        else:
            # Handle unexpected response format
             reasoning = f"Parsing error: Unexpected response format type: {type(response_text)}"
             return judgment, reasoning

        # Extract reasoning and judgment from the relevant text content
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response_to_parse, re.DOTALL)
        temp_reasoning = reasoning_match.group(1).strip() if reasoning_match else None

        judgment_match = re.search(r'<judgement>(.*?)</judgement>', response_to_parse, re.DOTALL) # Note: original code uses <judgement> tag
        temp_judgment = judgment_match.group(1).strip() if judgment_match else None

        # Update if found
        if temp_reasoning is not None:
            reasoning = temp_reasoning
        if temp_judgment is not None:
            # Validate judgment
            if temp_judgment in ["PASS", "FAIL"]:
                judgment = temp_judgment
            else:
                # Keep judgment as ERROR, update reasoning
                reasoning = f"Parsing error: Invalid judgment value '{temp_judgment}'. Reasoning (if any): {reasoning}" # Ensure reasoning is preserved

        return judgment, reasoning
    
    async def grade_explanation(self, description, caption, anticipated_point, explanation):
        """
        Grade an explanation by checking if it addresses the anticipated point.
        
        Args:
            description: The cartoon description
            caption: The caption for the cartoon
            anticipated_point: The specific point that should be addressed
            explanation: The student's explanation to evaluate
            
        Returns:
            Dictionary with judgment, reasoning, and usage info
        """
        # Format the prompt
        prompt = self.format_prompt(description, caption, explanation, anticipated_point)
        
        # Create message chain
        message_chain = MessageChain(family=self.family)
        message_chain.add_user_message(prompt)
        
        # Format messages for the API
        formatted_messages = message_chain.format_messages()
        
        # Make API call based on family
        response = await self._make_api_call(formatted_messages)
        
        # Parse the response - use the content directly
        judgment, reasoning = self.parse_response(response["content"])
        
        # Return results
        return {
            "judgment": judgment,
            "reasoning": reasoning,
            "usage": response.get("usage", {})
        }
    
    async def _make_api_call(self, formatted_messages):
        """
        Make the ASYNCHRONOUS API call to the LLM based on the model family.
        Includes 'thinking' parameter for compatible Claude models if budget is set.
        
        Args:
            formatted_messages: The formatted messages for the specific API
            
        Returns:
            Response dictionary with "content" and "usage" keys
        """
        if self.family == MessageChain.OPENAI:
            response = await self.client.chat.completions.create(
                model=self.model,
                **formatted_messages
            )
            usage = response.usage
            prompt_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0
            completion_tokens = getattr(usage, 'completion_tokens', 0) if usage else 0
            return {
                "content": response.choices[0].message.content, # Returns string
                "usage": {
                    "tokens_in": prompt_tokens,
                    "tokens_out": completion_tokens,
                }
            }
            
        elif self.family == MessageChain.DEEPSEEK:
            response = self.client.chat.completions.create(
                model=self.model,
                **formatted_messages
            )
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "tokens_in": response.usage.prompt_tokens,
                    "tokens_out": response.usage.completion_tokens
                }
            }
            
        elif self.family == MessageChain.GEMINI:
            contents = formatted_messages.get("contents", [])
            content_input = contents
            if not isinstance(contents, list) or not contents:
                 print("Warning: Could not directly extract 'contents' list for Gemini, attempting direct use.")
                 content_input = formatted_messages

            response = await self.client.generate_content_async(
                contents=content_input
            )
            usage_data = {}
            try:
                prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                candidates_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
                usage_data = { "tokens_in": prompt_tokens, "tokens_out": candidates_tokens }
            except AttributeError:
                 print(f"Warning: response.usage_metadata not found or incomplete for Gemini response.")
            return {
                "content": response.text, # Returns string
                "usage": usage_data
            }
            
        elif self.family == MessageChain.CLAUDE:
            messages = formatted_messages.get("messages", [])
            api_params = {
                "model": self.model,
                "max_tokens": 1024, # Consider making this configurable
                "messages": messages
            }

            # Add thinking parameter if budget is specified and > 0
            if self.thinking_budget is not None and self.thinking_budget > 0:
                 if 'claude-3.7' in self.model.lower() or 'claude-3.5' in self.model.lower(): # Add future model names if needed
                     api_params["thinking"] = {"budget_tokens": self.thinking_budget}
                     # Ensure budget <= max_tokens
                     if self.thinking_budget >= api_params["max_tokens"]:
                          print(f"Warning: thinking_budget ({self.thinking_budget}) >= max_tokens ({api_params['max_tokens']}). Adjusting max_tokens.")
                          api_params["max_tokens"] = self.thinking_budget + 100 # Simple adjustment
                 else:
                      print(f"Warning: thinking_budget specified for Claude model '{self.model}' which may not support it. Ignoring budget.")


            response = await self.client.messages.create(**api_params)

            usage_data = {}
            if hasattr(response, "usage"):
                usage_data = {
                    "tokens_in": response.usage.input_tokens if hasattr(response.usage, "input_tokens") else 0,
                    "tokens_out": response.usage.output_tokens if hasattr(response.usage, "output_tokens") else 0
                }

            # Return the raw content list, parsing happens later
            return {
                "content": response.content, # Return list of ContentBlock objects
                "usage": usage_data
            }
        
        elif self.family == MessageChain.TOGETHER:
            response = await self.client.chat.completions.create(
                model=self.model,
                **formatted_messages
            )
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "tokens_in": response.usage.prompt_tokens,
                    "tokens_out": response.usage.completion_tokens
                }
            }
        
        raise ValueError(f"Unsupported model family for async API call: {self.family}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Autograder client")
    parser.add_argument("--model", type=str, default="gemini-1.5-pro", help="The model to use")
    parser.add_argument("--thinking-budget", type=int, default=None, help="Optional thinking budget for Claude models")
    args = parser.parse_args()

    # Test with the specified model
    async def test_model():
        # Sample data for testing
        description = "A man is a pizza"
        caption = "Pizza pizza"
        explanation = "The man is making a profound statement about the nature of pizza."
        anticipated_point = "This is a reference to little ceasar's pizza ads"
        
        # Create client with auto-detected family and optional budget
        client = AutograderClient(model=args.model, thinking_budget=args.thinking_budget)
        
        # Print the inferred family and budget
        print(f"\n=== Testing model: {args.model} (Family: {client.family}, Thinking Budget: {client.thinking_budget}) ===")
        
        # Run grading
        result = await client.grade_explanation(description, caption, anticipated_point, explanation)
        
        # Print results
        print(f"Judgment: {result['judgment']}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Tokens: {result['usage']}")
    
    # Run the async test function
    asyncio.run(test_model())

