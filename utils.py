class MessageChain:
    """
    A universal message chain class that can be formatted for different LLM providers:
    OpenAI, DeepSeek, Gemini, and Claude/Anthropic.
    """
    
    # Model family constants
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    CLAUDE = "claude"
    TOGETHER = "together"
    XAI = "xai"
    ALIBABA = "alibaba"
    
    def __init__(self, family):
        """
        Initialize a message chain for a specific model family.
        
        Args:
            family: The model family (openai, deepseek, gemini, claude, together, xai, alibaba)
        """
        if family not in [self.OPENAI, self.DEEPSEEK, self.GEMINI, self.CLAUDE, self.TOGETHER, self.XAI, self.ALIBABA]:
            raise ValueError(f"Unsupported model family: {family}")
        self.family = family
        self.messages = []
    
    def add_user_message(self, content):
        """Add a user message to the chain."""
        self.messages.append({"role": "user", "content": content})
        return self
    
    def add_assistant_message(self, content):
        """Add an assistant message to the chain."""
        self.messages.append({"role": "assistant", "content": content})
        return self
    
    def clear(self):
        """Clear all messages."""
        self.messages = []
        return self
    
    def format_messages(self, family=None):
        """
        Format the message chain for the specified model family's API.
        
        Args:
            family: Override the default family set during initialization
            
        Returns:
            Dict or object matching the expected format for the model family
        """
        target_family = family or self.family
        
        if target_family == self.OPENAI:
            return self._format_for_openai()
        elif target_family == self.DEEPSEEK:
            return self._format_for_deepseek()
        elif target_family == self.GEMINI:
            return self._format_for_gemini()
        elif target_family == self.CLAUDE:
            return self._format_for_claude()
        elif target_family == self.TOGETHER:
            return self._format_for_together()
        elif target_family == self.XAI:
            return self._format_for_openai()
        elif target_family == self.ALIBABA:
            return self._format_for_alibaba()
        else:
            raise ValueError(f"Unsupported model family: {target_family}")
    
    def _format_for_openai(self):
        """Format messages for OpenAI API."""
        return {"messages": self.messages}
    
    def _format_for_deepseek(self):
        """Format messages for DeepSeek API."""
        # DeepSeek follows OpenAI's format
        return self._format_for_openai()
    
    def _format_for_gemini(self):
        """Format messages for Gemini API."""
        gemini_messages = []
        
        # Convert regular messages
        for msg in self.messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            gemini_messages.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
            
        return {"contents": gemini_messages}
    
    def _format_for_claude(self):
        """Format messages for Claude/Anthropic API."""
        return {
            "messages": self.messages
        }

    def _format_for_together(self):
        """Format messages for Together API."""
        # Together API uses the same format as OpenAI
        return self._format_for_openai()

    def _format_for_alibaba(self):
        """Format messages for Alibaba API (OpenAI compatible)."""
        return self._format_for_openai()
