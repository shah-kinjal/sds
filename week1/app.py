#!/usr/bin/env python3
"""
FlightAI Assistant - A Gradio-based AI assistant for airline ticket booking
Based on ks_lab9.ipynb notebook
"""

import os
import json
import base64
import requests
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import gradio as gr

# Load environment variables
load_dotenv(override=True)

# Configuration
openai_api_key = os.getenv('OPENAI_API_KEY')
serper_api_key = os.getenv('SERPER_API_KEY')
MODEL = "gpt-4o-mini"  # Fixed model name
serper_url = "https://google.serper.dev/search"

# Initialize OpenAI client
openai = OpenAI()

# Ticket prices for FlightAI destinations
ticket_prices = {
    "london": "$799", 
    "paris": "$899", 
    "tokyo": "$1400", 
    "sydney": "$2999", 
    "mumbai": "$2599"
}

# System message for the AI assistant
system_message = f"""You are a helpful assistant for an Airline called FlightAI. 
If relevant, Round trip fares to the certain cities are as follows: 
{ticket_prices}. If the city name is not listed, let the user know we don't serve that city yet and show them the list of cities we do serve. 
If asked for one way prices to these cities, cut the listed prices in half.
Give short, witty, snarky answers, no more than 1 sentence."""

# Enhanced system message for web search functionality
new_system_message = f"""You are a helpful assistant for an Airline called FlightAI. 
If asked about the ticket prices to any city and if you don't know the latest prices, use the search_the_web tool to search the web and use the response to 
find the cheapest flight from San Francisco. Assume all the flights originate from San Francisco.
If asked for one way prices to these cities, cut the listed prices in half. If price is not found, let the user know that you couldn't find a price to that city. 
Give short, witty, snarky answers, no more than 1 sentence."""


def artist_agent(city):
    """Generate an image for a given city using DALL-E 3"""
    if not city:
        return None
    
    try:
        image_response = openai.images.generate(
            model="dall-e-3",
            prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
        image_base64 = image_response.data[0].b64_json
        image_data = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_data))
    except Exception as e:
        print(f"Error generating image: {e}")
        return None


def search_web(web_query):
    """Search the web using Serper API"""
    print(f"Searching web: {web_query}")
    payload = json.dumps({"q": web_query})
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(serper_url, headers=headers, data=payload)
        print(response.text)
        return response.text
    except Exception as e:
        print(f"Error searching web: {e}")
        return f"Error: {str(e)}"


def search_the_web(to_city, from_city="San Francisco", extra_info=""):
    """Search for flight prices between cities"""
    query = f"Find the cheapest round trip flight prices from {from_city} to {to_city} in the next week."
    return search_web(query)


# Tool definition for function calling
search_the_web_json = {
    "name": "search_the_web",
    "description": "Use this tool to search flight prices",
    "parameters": {
        "type": "object",
        "properties": {
            "to_city": {
                "type": "string",
                "description": "Destination City"
            },
            "from_city": {
                "type": "string",
                "description": "Originating city"
            },
            "extra_info": {
                "type": "string",
                "description": "Any additional information about the search that's worth recording to give context"
            }
        },
        "required": ["to_city", "extra_info"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": search_the_web_json}]


def handle_tool_calls(tool_calls):
    """Handle tool calls from the AI model"""
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append({
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call.id
        })
    return results


def chat(history):
    """Basic chat function without web search"""
    if not history:
        return history, None
    
    message = history[-1]["content"]
    messages = [{"role": "system", "content": system_message}] + history
    
    try:
        response = openai.chat.completions.create(model=MODEL, messages=messages)
        lower_message = message.lower()
        city = next((name for name in ticket_prices if name in lower_message), None)
        image = artist_agent(city.title()) if city else None
        reply = response.choices[0].message.content
        history += [{"role": "assistant", "content": reply}]
        return history, image
    except Exception as e:
        print(f"Error in chat: {e}")
        history += [{"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"}]
        return history, None


def chat_with_web(history):
    """Enhanced chat function with web search capabilities"""
    if not history:
        return history, None
    
    message = history[-1]["content"]
    messages = [{"role": "system", "content": new_system_message}] + history
    lower_message = message.lower()
    
    done = False
    try:
        while not done:
            response = openai.chat.completions.create(
                model=MODEL, 
                messages=messages, 
                tools=tools
            )
            print(response.choices[0])
            finish_reason = response.choices[0].finish_reason
            print(f"Finish reason: {finish_reason}")
            
            if finish_reason == "tool_calls":
                message_obj = response.choices[0].message
                tool_calls = message_obj.tool_calls
                results = handle_tool_calls(tool_calls)
                messages.append(message_obj)
                messages.extend(results)
            else:
                done = True
        
        city = next((name for name in ticket_prices if name in lower_message), None)
        image = artist_agent(city.title()) if city else None
        
        reply = response.choices[0].message.content
        history += [{"role": "assistant", "content": reply}]
        return history, image
        
    except Exception as e:
        print(f"Error in chat_with_web: {e}")
        history += [{"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"}]
        return history, None


def do_entry(message, history):
    """Handle user input entry"""
    history += [{"role": "user", "content": message}]
    return "", history


def create_basic_ui():
    """Create the basic Gradio interface without web search"""
    with gr.Blocks(title="FlightAI Assistant - Basic") as ui:
        gr.Markdown("# FlightAI Assistant - Basic Version")
        gr.Markdown("Ask about flight prices to London, Paris, Tokyo, Sydney, or Mumbai!")
        
        with gr.Row():
            chatbot = gr.Chatbot(height=400, type="messages")
            image_output = gr.Image(height=400, label="City Image")
        
        with gr.Row():
            entry = gr.Textbox(label="Chat with our AI Assistant:", placeholder="Ask about flights to any city...")
        
        entry.submit(
            do_entry, 
            inputs=[entry, chatbot], 
            outputs=[entry, chatbot]
        ).then(
            chat, 
            inputs=chatbot, 
            outputs=[chatbot, image_output]
        )
    
    return ui


def create_advanced_ui():
    """Create the advanced Gradio interface with web search"""
    with gr.Blocks(title="FlightAI Assistant - Advanced") as ui:
        gr.Markdown("# FlightAI Assistant - Advanced Version")
        gr.Markdown("Ask about flight prices to any city! I can search the web for real-time prices.")
        
        with gr.Row():
            chatbot = gr.Chatbot(height=400, type="messages")
            image_output = gr.Image(height=400, label="City Image")
        
        with gr.Row():
            entry = gr.Textbox(label="Chat with our AI Assistant:", placeholder="Ask about flights to any city...")
        
        entry.submit(
            do_entry, 
            inputs=[entry, chatbot], 
            outputs=[entry, chatbot]
        ).then(
            chat_with_web, 
            inputs=chatbot, 
            outputs=[chatbot, image_output]
        )
    
    return ui


def main():
    """Main function to run the application"""
    print("Starting FlightAI Assistant...")
    print("Available cities:", list(ticket_prices.keys()))
    
    # Check if API keys are available
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
    
    if not serper_api_key:
        print("Warning: SERPER_API_KEY not found in environment variables")
        print("Web search functionality will not work without this key")
    
    # Create and launch the basic UI
    print("Launching basic version...")
    basic_ui = create_basic_ui()
    basic_ui.launch(share=False, server_name="0.0.0.0", server_port=7860)


def main_advanced():
    """Main function to run the advanced application with web search"""
    print("Starting FlightAI Assistant (Advanced)...")
    print("Available cities:", list(ticket_prices.keys()))
    
    # Check if API keys are available
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    if not serper_api_key:
        print("Error: SERPER_API_KEY not found in environment variables")
        print("Web search functionality requires this key")
        return
    
    # Create and launch the advanced UI
    print("Launching advanced version...")
    advanced_ui = create_advanced_ui()
    advanced_ui.launch(share=False, server_name="0.0.0.0", server_port=7861)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--advanced":
        main_advanced()
    else:
        main()
