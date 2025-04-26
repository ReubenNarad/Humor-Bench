# This is where the explainer client will be implemented.
import sys
import os
# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MessageChain
import re
import asyncio
from dotenv import load_dotenv
from prompts import explainer_prompt

# Load environment variables from .env file
load_dotenv()

class ExplainerClient:
    """
    A client for explaining cartoon jokes.
    Works with different LLM providers (OpenAI, DeepSeek, Gemini, Claude, Together).
    Supports extended thinking for compatible Claude models.
    Supports reasoning effort for compatible OpenAI models.
    """
    
    def __init__(self, model, family=None, api_key=None, thinking_budget=None, reasoning_effort=None):
        """
        Initialize the explainer client.
        
        Args:
            model: The model name to use (e.g., "gpt-4o", "claude-3.7-sonnet...")
            family: The model family (optional, inferred if not provided)
            api_key: API key for the specified model family (optional)
            thinking_budget: Max tokens for Claude's extended thinking (optional)
            reasoning_effort: Effort level for OpenAI reasoning ('low', 'medium', 'high') (optional)
        """
        self.model = model
        self.thinking_budget = thinking_budget
        self.reasoning_effort = reasoning_effort
        
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
        
        # Optional: Add a check if reasoning_effort is provided for non-OpenAI models
        if self.reasoning_effort is not None:
            if self.family != MessageChain.OPENAI:
                print(f"Warning: reasoning_effort provided for non-OpenAI model '{self.model}' (Family: {self.family}). It will be ignored.")
            else:
                # Validate effort level if OpenAI
                valid_efforts = ["low", "medium", "high"]
                if self.reasoning_effort not in valid_efforts:
                     raise ValueError(f"Invalid reasoning_effort '{self.reasoning_effort}'. Must be one of {valid_efforts}.")
                # Also add check for specific models if needed (already done in _make_api_call)
    
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
        elif any(name in model_lower for name in ['gpt', 'o1', 'o3', 'o4', 'davinci', 'curie', 'babbage', 'ada']):
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
            Initialized client
        """
        api_key = self.api_key

        if self.family == MessageChain.OPENAI:
            from openai import AsyncOpenAI
            
            if not api_key:
                # Get from environment
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not found in environment variables")
            
            return AsyncOpenAI(api_key=api_key)
            
        elif self.family == MessageChain.GEMINI:
            # Updated to use the new Google Gen AI SDK
            from google import genai
            
            if not api_key:
                # Get from environment
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("Gemini API key not found in environment variables")
            
            # Create a client with the API key
            return genai.Client(api_key=api_key)
            
        elif self.family == MessageChain.CLAUDE:
            from anthropic import AsyncAnthropic
            
            if not api_key:
                # Get from environment
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("Anthropic API key not found in environment variables")
            
            return AsyncAnthropic(api_key=api_key)
            
        elif self.family == MessageChain.TOGETHER: # This now handles Together-hosted DeepSeek models
            from openai import AsyncOpenAI
            
            if not api_key:
                # Get from environment
                api_key = os.environ.get("TOGETHER_API_KEY")
                if not api_key:
                    raise ValueError("Together API key not found in environment variables")
            
            return AsyncOpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
        
        raise ValueError(f"Unsupported model family for async client: {self.family}")
    
    def format_prompt(self, description, caption):
        """
        Format the prompt with the specific row data.
        
        Args:
            description: The cartoon description
            caption: The caption for the cartoon
            
        Returns:
            Formatted prompt string
        """
        prompt = explainer_prompt(description, caption)
        return prompt
    
    def parse_response(self, response_content):
        """
        Parse the model response to extract explanation.
        Handles potential thinking blocks (ignores them for the final explanation).

        Args:
            response_content: The raw content list from the Claude API response (list of objects like TextBlock, ThinkingBlock)

        Returns:
            Extracted explanation string
        """
        explanation_text = "ERROR: No explanation found in response" # Default error message

        # Check if the response is a list of content blocks (Claude)
        if isinstance(response_content, list):
            # Find the first 'text' block and extract its text content
            for block in response_content:
                # Access attributes directly, not via .get()
                if hasattr(block, 'type') and block.type == "text" and hasattr(block, 'text'):
                    raw_text = block.text
                    # We still might have XML tags inside the text block
                    explanation_match = re.search(r'<explanation>(.*?)</explanation>', raw_text, re.DOTALL)
                    if explanation_match:
                        explanation_text = explanation_match.group(1).strip()
                    else:
                        # Fallback if tags are missing, use the whole text block
                        explanation_text = raw_text.strip()
                    # Found the first text block, break the loop
                    break
            # If loop finishes without finding a text block, explanation_text remains the error message
            return explanation_text
        elif isinstance(response_content, str):
            # Original logic for non-Claude responses (or simple string responses)
            explanation_match = re.search(r'<explanation>(.*?)</explanation>', response_content, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else response_content.strip()
            return explanation
        else:
            # Handle unexpected response format
            print(f"Warning: Unexpected response content format type: {type(response_content)}. Returning as is.")
            return str(response_content)
    
    async def explain_cartoon(self, description, caption):
        """
        Get an explanation for a cartoon joke.
        
        Args:
            description: The cartoon description
            caption: The caption for the cartoon
            
        Returns:
            Dictionary with explanation and usage info
        """
        # Format the prompt
        prompt = self.format_prompt(description, caption)
        
        # Create message chain
        message_chain = MessageChain(family=self.family)
        message_chain.add_user_message(prompt)
        
        # Format messages for the API
        formatted_messages = message_chain.format_messages()
        
        # Make API call based on family
        response = await self._make_api_call(formatted_messages)
        
        # Parse the response content correctly
        explanation = self.parse_response(response["content"])
        
        # Return results
        return {
            "explanation": explanation,
            "usage": response.get("usage", {})
        }
    
    async def _make_api_call(self, formatted_messages):
        """
        Make the ASYNCHRONOUS API call to the LLM based on the model family.
        Includes 'thinking' parameter for compatible Claude models if budget is set.
        Includes 'reasoning' parameter for compatible OpenAI models if effort is set.
        
        Args:
            formatted_messages: The formatted messages for the specific API
            
        Returns:
            Response dictionary with "content" and "usage" keys
        """
        # --- OpenAI API ---
        if self.family == MessageChain.OPENAI:
            # Check if reasoning effort is specified
            if self.reasoning_effort is not None:
                 # Optional: Add a check/warning if the model might not support it (e.g., non 'o' models)
                if not any(prefix in self.model.lower() for prefix in ['o1', 'o4']):
                     print(f"Warning: Using reasoning_effort='{self.reasoning_effort}' with model '{self.model}'. This parameter may only affect 'o' models like o1/o4.")
                
                # Use the responses.create endpoint for reasoning
                try:
                    response = await self.client.responses.create(
                        model=self.model,
                        reasoning={"effort": self.reasoning_effort},
                        input=formatted_messages.get("messages", []) # Use 'input' key here
                    )
                    # The response structure is different for responses.create
                    # It doesn't seem to return usage data directly in the example.
                    # We might need to adjust if usage is needed/available differently.
                    return {
                        "content": response.output_text, # Use output_text
                        "usage": {} # Assuming usage is not returned here
                    }
                except AttributeError:
                     print(f"Error: The current OpenAI client might not support 'responses.create'. Falling back to chat completions.")
                     # Fall through to standard chat completions if responses.create fails
                except Exception as e:
                    print(f"Error calling OpenAI responses.create: {e}")
                    raise # Re-raise other exceptions

            # --- Standard OpenAI Chat Completion (No reasoning effort) ---
            # Prepare base API parameters
            api_params = {
                "model": self.model,
                **formatted_messages # This should contain 'messages' key
            }
            response = await self.client.chat.completions.create(**api_params)
            # Assume standard chat completions response structure
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "tokens_in": response.usage.prompt_tokens,
                    "tokens_out": response.usage.completion_tokens
                }
            }

        # --- Together API (Uses OpenAI compatible structure but NOT reasoning) ---
        elif self.family == MessageChain.TOGETHER:
            api_params = {
                "model": self.model,
                **formatted_messages # This should contain 'messages' key
            }
            response = await self.client.chat.completions.create(**api_params)
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "tokens_in": response.usage.prompt_tokens,
                    "tokens_out": response.usage.completion_tokens
                }
            }

        # --- Gemini API ---
        elif self.family == MessageChain.GEMINI:
            # Updated to use the new Google Gen AI SDK structure
            # Extract content text from formatted messages
            contents = formatted_messages.get("contents", [])
            
            # Convert contents to Gemini's format if needed
            content_text = ""
            for message in contents:
                # For simplicity, just extract the text part
                if message["role"] == "user":
                    parts = message.get("parts", [])
                    for part in parts:
                        if "text" in part:
                            content_text = part["text"]
            
            # If we couldn't extract from the format, use the original prompt
            if not content_text:
                # Get the last user message
                for msg in reversed(contents):
                    if msg["role"] == "user":
                        for part in msg.get("parts", []):
                            if "text" in part:
                                content_text = part["text"]
                                break
                        break
            
            # Use the new SDK structure
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=content_text
            )
            
            # Extract usage metadata if available
            usage_data = {}
            if hasattr(response, 'usage_metadata'):
                try:
                    # Map fields to our standard keys
                    usage_data = {
                        "tokens_in": response.usage_metadata.prompt_token_count,
                        "tokens_out": response.usage_metadata.candidates_token_count
                    }
                except AttributeError as e:
                    print(f"Warning: Could not access usage_metadata attributes for Gemini: {e}")
            else:
                print(f"Warning: usage_metadata not found for Gemini response.")

            # Return response text
            return {
                "content": response.text,
                "usage": usage_data
            }

        # --- Claude API ---
        elif self.family == MessageChain.CLAUDE:
            # formatted_messages is the dict returned by _format_for_claude
            messages = formatted_messages.get("messages", [])
            
            # --- Prepare API parameters ---
            api_params = {
                "model": self.model,
                "max_tokens": 10000,
                "messages": messages,
            }

            # --- Add thinking parameter with proper minimum ---
            if self.thinking_budget is not None and self.thinking_budget > 0:
                # Enforce the minimum budget_tokens value of 1024
                actual_budget = max(1024, self.thinking_budget)
                
                # Check if we had to adjust the budget
                if actual_budget != self.thinking_budget:
                    print(f"Warning: Adjusted thinking budget from {self.thinking_budget} to {actual_budget} (minimum required by Claude API)")
                
                api_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": actual_budget
                }
                
                # Also ensure max_tokens is sufficient
                if actual_budget >= api_params["max_tokens"]:
                    print(f"Warning: thinking_budget ({actual_budget}) >= max_tokens ({api_params['max_tokens']}). Adjusting max_tokens.")
                    api_params["max_tokens"] = actual_budget + 100
            
            # --- Make the API call ---
            response = await self.client.messages.create(**api_params)

            # --- Process response (logic unchanged) ---
            usage_data = {}
            if hasattr(response, "usage"):
                usage_data = {
                    "tokens_in": response.usage.input_tokens if hasattr(response.usage, "input_tokens") else 0,
                    "tokens_out": response.usage.output_tokens if hasattr(response.usage, "output_tokens") else 0
                }
            return {
                "content": response.content,
                "usage": usage_data
            }
        
        raise ValueError(f"Unsupported model family for async API call: {self.family}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explainer client")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", help="The model to use")
    parser.add_argument("--thinking-budget", type=int, default=None, help="Optional thinking budget for Claude models")
    parser.add_argument("--reasoning-effort", type=str, default=None, choices=["low", "medium", "high"], help="Optional reasoning effort for OpenAI 'o' models")
    args = parser.parse_args()

    # Test with the specified model
    async def test_model():
        # Sample data for testing
        description = "Pizza is DESTROYING THE WORLD AHHH"
        caption = "Olive you glad I didn't say banana?"
        
        # Create client with auto-detected family and optional params
        client = ExplainerClient(
            model=args.model, 
            thinking_budget=args.thinking_budget,
            reasoning_effort=args.reasoning_effort
        )
        
        # Print the inferred family and parameters
        print(f"\n=== Testing model: {args.model} (Family: {client.family}, Thinking Budget: {client.thinking_budget}, Reasoning Effort: {client.reasoning_effort}) ===")
        
        # Run explanation
        result = await client.explain_cartoon(description, caption)
        
        # Print results
        print(f"Explanation: {result['explanation']}")
        print(f"Tokens: {result['usage']}")
    
    # Run the test
    asyncio.run(test_model())
