"""
Utility script to list available Gemini models.
Run this to check which models are available in your region.
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in environment.")
else:
    genai.configure(api_key=api_key)
    
    print("Available Gemini Models:")
    print("-" * 50)
    
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"  - {model.name}")
            print(f"    Display: {model.display_name}")
            print(f"    Methods: {model.supported_generation_methods}")
            print()
