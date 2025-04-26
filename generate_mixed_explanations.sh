#!/bin/bash

# Default values
LIMIT=100
STRATEGY="round_robin"
MODELS="gpt-4o claude-3-7-sonnet-latest gemini-2.5-pro-exp-03-25"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --strategy)
      STRATEGY="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [--limit N] [--strategy random|round_robin]"
      echo ""
      echo "Generate explanations for cartoons using multiple models."
      echo ""
      echo "Options:"
      echo "  --limit N        Limit to first N rows (default: 10, use 0 for all)"
      echo "  --strategy TYPE  Model selection strategy: random or round_robin (default: random)"
      echo "d  --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Use --help for usage information."
      exit 1
      ;;
  esac
done

# Print configuration
echo "Generating explanations with:"
echo "- Limit: $LIMIT rows"
echo "- Strategy: $STRATEGY"
echo "- Models: $MODELS"
echo ""

# Run the generator script
python generate_explanations.py --models $MODELS --limit $LIMIT --strategy $STRATEGY