# FlightAI Assistant

A Gradio-based AI assistant for airline ticket booking, based on the ks_lab9.ipynb notebook.

## Features

- **Basic Chat Interface**: Ask about flight prices to predefined cities (London, Paris, Tokyo, Sydney, Mumbai)
- **Image Generation**: Automatically generates vacation images for mentioned cities using DALL-E 3
- **Web Search Integration**: Advanced version can search for real-time flight prices using Serper API
- **Tool Calling**: Uses OpenAI's function calling to search the web for current flight prices

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

## Usage

### Basic Version (without web search)
```bash
python app.py
```

### Advanced Version (with web search)
```bash
python app.py --advanced
```

## Available Cities

The assistant knows about flights to these cities:
- London: $799 (round trip)
- Paris: $899 (round trip)
- Tokyo: $1400 (round trip)
- Sydney: $2999 (round trip)
- Mumbai: $2599 (round trip)

One-way prices are half of the round-trip prices.

## API Keys Required

- **OpenAI API Key**: Required for chat functionality and image generation
- **Serper API Key**: Required for web search functionality (advanced version only)

## Ports

- Basic version runs on port 7860
- Advanced version runs on port 7861

## Example Queries

- "What's the price to London?"
- "Show me flights to Tokyo"
- "I want to go to Paris"
- "What cities do you serve?"

The assistant will provide witty, snarky responses and generate vacation images for the mentioned cities.
