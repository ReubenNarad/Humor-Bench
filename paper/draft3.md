# HumorBench: Evaluating Reasoning in LLMs Through Humor Comprehension

## Abstract

- Humor comprehension is a valuable test case for evaluating reasoning in LLMs
- HumorBench assesses LLMs' ability to identify and explain core comedic elements in cartoons from various sources
- Our approach breaks down humor into discrete mental leaps connecting references, implications, and contexts
- LLM-based autograder validated against human judgments (87% accuracy)
- Results show high correlation between HumorBench and STEM benchmarks
- HumorBench-Hard subset identified where even frontier models struggle
- Provides a novel non-STEM reasoning benchmark highlighting conceptual connection-making abilities

## Introduction

- Humor requires sophisticated mental leaps and connecting disparate concepts
- HumorBench uses cartoons and captions from various sources
- For each cartoon-caption pair, models receive:
  - Text description of visual elements
  - Corresponding caption
  - Request to explain the humor

- Innovation: Decomposing humor into elements requiring specific mental leaps
  - Example: Snow White with dwarfs on roller skates captioned "Workplace morale hasn't been this high since we introduced whistling" requires:
    - Recognizing reference to "Whistle While You Work" from Snow White
    - Understanding roller skates were introduced in workplace context
    - Connecting elements through mental leaps

- Evaluation via LLM autograder (validated with 87% accuracy)
  - Higher False Positive Rate than False Negative Rate

- Key findings:
  - HumorBench provides valuable non-STEM reasoning benchmark
  - Performance correlates with STEM benchmarks
  - Mental leaps for humor appear to rely on similar mechanisms as STEM problem-solving
  - Analysis of challenging examples reveals boundaries of LLM reasoning

## Related Work

- Previous humor NLP work focused on:
  - Humor recognition (detecting if content is humorous)
  - Humor generation (creating humorous content)
  - Humor ranking (predicting human preference)

- Notable examples:
  - Hessel et al. (2022): LLM-based system for New Yorker caption contest rankings
  - Yang et al. (2023): Humor explanation tasks
  - Hasan et al. (2021): Computational humor generation

- Our differentiation:
  - Using humor as proxy for reasoning evaluation
  - Decomposing humor into specific elements requiring mental leaps
  - Expert annotations of humor elements
  - Validated LLM autograder system
  - Exploring correlations with STEM reasoning tasks

## HumorBench

### The Mental Leaps Challenge

- New Yorker cartoon style requires mental leaps connecting:
  - Cultural knowledge and references
  - Understanding of social norms and their subversion
  - Wordplay and dual meanings
  - Juxtaposition of mundane with absurd
  - Implicit connections requiring inference

- Example: Snow White cartoon requires connecting:
  - Visual reference to Snow White and dwarfs
  - The song "Whistle While You Work"
  - Implication of roller skating in workplace
  - Juxtaposition of fairy tale with office culture

- These mental leaps mirror reasoning patterns in other domains, including STEM

### Our Task

- Models must:
  1. Receive text description of cartoon image
  2. Receive corresponding caption
  3. Generate concise explanation identifying key comedic elements

- Explainer prompt:
  > "You are a humor expert extraordinaire, judging the New Yorker Cartoon Caption Contest. Your current task is to help us understand the humor in various submitted captions. Given a cartoon description and a caption submission, explain (in less than 200 words) *what* the joke is, focusing on the material substance of the joke."

### Autograder Construction

- LLM-based autograder (GPT-4o):
  - Receives cartoon description, caption, model explanation, and ground truth humor element
  - Determines if explanation covers the anticipated point
  - Outputs PASS/FAIL judgment with reasoning

- Validated against 400 human judgments (87% accuracy)
  - Higher False Positive Rate (more lenient than human judges)
  - Results represent upper bound on model capabilities

- Autograder prompt assesses whether explanation captures key humor element

## Dataset Curation

### Humor Elements as Reasoning Challenges

- Each cartoon-caption pair has identified humor elements:
  - Cultural references
  - Context-specific implications
  - Wordplay or puns
  - Juxtaposition of unexpected elements
  - Subversion of expectations
  - Absurdity or exaggeration

- These elements are ground truth for evaluation
- Explanations succeed when they identify these elements and connecting mental leaps

### Data Sources and Selection

- Dataset contains:
  - Unique identifiers
  - Cartoon descriptions
  - Winning captions
  - Expert-annotated humor elements

- Sources:
  - Nextml Caption Contest Data
  - jmhessel/newyorker_caption_contest Hugging Face dataset
  - CartoonStock

## Experiments

### Experimental Setup

- Models evaluated:
  - GPT-4o
  - Claude 3.7 Sonnet
  - Gemini 2.5 Pro
  - Llama 4 Maverick
  - Others

- For each model:
  - Generated explanations using explainer prompt
  - Evaluated via autograder
  - Calculated PASS rates
  - Tracked token usage and cost

- Implementation:
  - Asynchronous processing
  - Support for various model APIs
  - Response parsing with XML tags

### Main Results

- Key findings:
  - Frontier models achieved highest PASS rates
  - HumorBench-Hard subset challenging even for best models
  - Performance correlates with cost
  - Certain humor elements (cultural knowledge, multi-step reasoning) challenging across all models

### Correlation with STEM Benchmarks

- High correlation between HumorBench and STEM benchmarks:
  - Suggests general reasoning transfers across domains
  - Mental leaps for humor use similar mechanisms as STEM problem-solving

- Models performing well on GPQA, ARC-AGI, LM Arena ELO also perform well on HumorBench

- Implications:
  - Humor comprehension relies on general reasoning, not just domain knowledge
  - STEM reasoning improvements likely transfer to humor understanding
  - HumorBench complements other benchmarks while testing different knowledge domains

### Analysis of HumorBench-Hard

- Common challenges in difficult examples:
  - Multiple cultural references requiring simultaneous recognition
  - Complex wordplay or puns
  - Non-obvious connections between visual elements and caption
  - Multi-step reasoning chains
  - Nuanced understanding of social norm subversion

- These examples require multiple mental leaps to connect different elements

### Test-Time Scaling and Reasoning Ability

- Performance scales with reasoning resources:
  - Larger context windows improve complex example performance
  - Test-time scaling techniques (Claude's thinking budget, OpenAI's reasoning effort) improve results
  - Improvements support humor comprehension's dependence on reasoning ability

## Conclusion and Future Work

- HumorBench evaluates mental leaps between concepts
- Results show correlation between humor comprehension and STEM reasoning
- Benchmark measures progress in language understanding requiring:
  - Cultural/contextual knowledge integration
  - Recognition of implicit connections
  - Understanding social norms and subversion
  - Multi-step reasoning about intentions and meanings

- Future work:
  - More diverse humor sources
  - Metrics for specific reasoning dimensions
  - Comparing autograders across LLMs
  - Few-shot prompting or fine-tuning
  - Reinforcement learning to enhance concept connections

## Limitations

- Current limitations:
  - Focus on specific cartoon humor style
  - Uses text descriptions rather than images
  - LLM-based autograder has inherent limitations
  - Higher False Positive Rate than False Negative Rate
  - STEM reasoning correlation may not capture all humor aspects 