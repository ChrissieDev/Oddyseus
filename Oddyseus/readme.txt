Hey, i'm creating an LLM model / engine as a 'personal' chatbot, that runs both locally, and with features larger models are often found with.
This includes relationships / emotional responses based on said relationships, and an advanced recall and memory functionality with specific weights toward emotional impact, relationship affinity, and material importance baked right into the storage layer, so every recollection feels intentional.
It's a three-stage stack using tiny embedders for fast search for the first, for the second a lightweight appraisal brain that spits out JSON mood + curated memories, and a heavier speech model that actually replies in text third.
Only the newest four dialogue exchanges tag along in context unless a recalled memory already covers them, keeping prompts light on tokens without (hopefully) losing continuity.

This is not meant for commercial use, it's just a fun project where i gain experience working with vectors, models, and APIs.
Cheers, and be more environmentally responsible. :D

## Requirements

- **.NET 9.0+** (for console & MAUI apps)
- **Groq API key** (free tier at console.groq.com)
- **Qwen3 Embedding Model** (1.2GB ONNX file)
- **ONNX Runtime** (NuGet: Microsoft.ML.OnnxRuntime)
- **HuggingFace token** (optional, for model downloads)

## Setup

1. Clone repo
2. Set env vars:
   ```
   setx GROQ_API_KEY "gsk-your-key"
   setx HF_TOKEN "hf_your-token"
   ```
3. Download embedding model to `Oddyseus/Core/Models/Qwen3EmbModel.onnx`
4. `dotnet run` (console) or `dotnet publish -f net9.0-android` (MAUI)

## Architecture

- **MemoryManager**: Local ONNX embeddings (1.2GB fp16)
- **Orchestrator**: Orchestrates memory recall + emotion + dialogue
- **LlmClient**: HTTP calls to Groq API (appraisal + response generation)
- **RelationshipModeler**: Tracks user relationship state

## Android Build

- Requires 12GB+ RAM device
- Embedding model loads on startup (~2-3s)
- Groq API calls require internet