"""
MyTwin.py - AI Digital Twin Application
Based on Lab 3 - OpenAI Agents SDK with streaming

This script creates an AI Digital Twin that represents a person and can chat
with visitors on their website, answering questions about their career,
background, skills and experience.
"""

import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, trace
from openai.types.responses import ResponseTextDeltaEvent
from openai import OpenAI
import gradio as gr
from agents.mcp import MCPServerStdio
from datetime import datetime
from pathlib import Path
import requests
import os
import glob   

# Load environment variables
load_dotenv(override=True)

# Model Provider Configuration
# Change this constant to switch between providers: "openai", "anthropic", "grok", "deepseek", or "gemini"
#MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai").lower()
#MODEL_PROVIDER = "anthropic"
MODEL_PROVIDER = "gemini"
#MODEL_PROVIDER = "grok" #"deepseek"
#MODEL_PROVIDER = "deepseek"

# Model names for each provider
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview") 

# API configurations
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
grok_api_key = os.getenv('GROK_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

# Initialize clients for different providers
openai_client = OpenAI()  # Uses OPENAI_API_KEY from environment
anthropic_client = OpenAI(api_key=anthropic_api_key, base_url="https://api.anthropic.com/v1/") if anthropic_api_key else None
grok_client = OpenAI(api_key=grok_api_key, base_url="https://api.x.ai/v1") if grok_api_key else None
deepseek_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com") if deepseek_api_key else None
google_client = OpenAI(api_key=google_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/") if google_api_key else None

# Select the appropriate client and model based on provider
if MODEL_PROVIDER == "anthropic":
    if not anthropic_client:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    SELECTED_CLIENT = anthropic_client
    MODEL = ANTHROPIC_MODEL
elif MODEL_PROVIDER == "grok":
    if not grok_client:
        raise ValueError("GROK_API_KEY not found in environment variables")
    SELECTED_CLIENT = grok_client
    MODEL = GROK_MODEL
elif MODEL_PROVIDER == "deepseek":
    if not deepseek_client:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
    SELECTED_CLIENT = deepseek_client
    MODEL = DEEPSEEK_MODEL
elif MODEL_PROVIDER == "gemini":
    if not google_client:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    SELECTED_CLIENT = google_client
    MODEL = GEMINI_MODEL
else:  # Default to OpenAI
    SELECTED_CLIENT = openai_client
    MODEL = OPENAI_MODEL

print(f"Using provider: {MODEL_PROVIDER.upper()}")
print(f"url: {SELECTED_CLIENT.base_url}")
print(f"Using model: {MODEL}")



@function_tool
def create_document(filename: str, content: str, document_type: str = "analysis") -> str:
    """Create and save a markdown document to the info/output folder.
    
    Use this tool to save analysis results, coding, themes, insights, or any other documents
    during the interview analysis process.
    
    Args:
        filename: The name for the file (without extension). Use descriptive names like 
                 'interview_coding', 'themes_analysis', 'insights_summary', etc.
        content: The full markdown content to save in the file. Should be well-formatted 
                markdown with headers, lists, tables, etc.
        document_type: Type of document being created (e.g., 'coding', 'themes', 'analysis', 
                      'insights'). This helps organize the output.
    
    Returns:
        A confirmation message with the file path where the document was saved.
    """
    # Create output folder if it doesn't exist
    output_folder = os.path.join("info", "output")
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = filename.replace(" ", "_").replace("/", "_")
    full_filename = f"{timestamp}_{safe_filename}.md"
    filepath = os.path.join(output_folder, full_filename)
    
    # Write the content to the file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Document created: {filepath}")
    return f"Document successfully created and saved to: {filepath}"



class InterviewAnalysis:
    """
    AI analysis of an interview transcript.
    """
    
    def __init__(self, name="Kinjal Shah"):
        """
        Initialize the InterviewAnalysis with interview transcript information.
        
        Args:
            name (str): The name of the person being represented
        """
        self.name = name
        self.mcp_servers = None
        self.agent = None
        self._setup_agent()
    
    def send_push_notification(self, message: str):
        """
        Send a push notification to the {first_name}
        """
        payload = {"user": pushover_user, "token": pushover_token, "message": message}
        requests.post(pushover_url, data=payload)
    
    def _setup_mcp_servers(self):
        """
        Setup the MCP servers.
        """
        try:
            params = {"command": "uv", "args": ["run", "db_tools_mcp_server.py"]}
            file_path = Path("memory") / Path("graph.db")
            url = f"file:{file_path.absolute()}"

            memory_graph_params = {"command": "npx","args": ["-y", "mcp-memory-libsql"],"env": {"LIBSQL_URL": url}}

            long_term_path = Path("memory") / Path("knowledge")


            memory_rag_params = {
            "command": "uvx",
            "args": ["mcp-server-qdrant"],
            "env": {
                "QDRANT_LOCAL_PATH": str(long_term_path.absolute()),
                "COLLECTION_NAME": "knowledge",
                "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
                }
            }

            with MCPServerStdio(params=memory_rag_params, client_session_timeout_seconds=30) as long_term_memory:
                self.long_term_memory = long_term_memory
                self.mcp_servers.append(long_term_memory)
            with MCPServerStdio(params=memory_graph_params, client_session_timeout_seconds=30) as medium_term_memory:
                self.medium_term_memory = medium_term_memory
                self.mcp_servers.append(medium_term_memory)
           
            
        except Exception as e:
            print(f"Warning: Failed to setup MCP server: {e}")
            self.db_tools_mcp_server = None
            self.mcp_servers = []
        
    def _load_documents(self):
        """
        Scan the info/ folder for audio/video files and markdown files.
        - Audio/video files are transcribed using OpenAI Whisper and saved as .md files
        - Markdown files (.md) are read directly as pre-existing transcripts
        Supports audio formats (mp3, wav, m4a, aac, ogg, flac), video formats (mp4, mov, avi, mkv, webm), and markdown (.md).
        
        Returns:
            str: Combined transcripts from all files found, each marked with its filename
        """
        client = openai_client  # Use the global OpenAI client for Whisper transcription
        
        # Define supported file extensions
        audio_extensions = ['mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac']
        video_extensions = ['mp4', 'mov', 'avi', 'mkv', 'webm']
        media_extensions = audio_extensions + video_extensions
        
        # Scan info/ folder for media and markdown files
        info_folder = "info"
        media_files = []
        markdown_files = []
        
        # Find audio/video files
        for ext in media_extensions:
            pattern = os.path.join(info_folder, f"*.{ext}")
            media_files.extend(glob.glob(pattern))
        
        # Find markdown files (excluding those in transcript subfolder)
        md_pattern = os.path.join(info_folder, "*.md")
        all_md_files = glob.glob(md_pattern)
        # Filter out files in the transcript subfolder
        markdown_files = [f for f in all_md_files if "transcript" not in f]
        
        total_files = len(media_files) + len(markdown_files)
        
        if total_files == 0:
            print(f"Warning: No media or markdown files found in {info_folder}/ folder.")
            print(f"Supported formats: {', '.join(media_extensions + ['md'])}")
            return ""
        
        print(f"\nFound {total_files} file(s) to process:")
        if media_files:
            print(f"  Media files to transcribe ({len(media_files)}):")
            for file in media_files:
                print(f"    - {os.path.basename(file)}")
        if markdown_files:
            print(f"  Markdown files to read ({len(markdown_files)}):")
            for file in markdown_files:
                print(f"    - {os.path.basename(file)}")
        
        # Create transcript folder if it doesn't exist
        transcript_folder = os.path.join(info_folder, "transcript")
        os.makedirs(transcript_folder, exist_ok=True)
        
        # Process all files and combine transcripts
        combined_transcript = ""
        
        # Process audio/video files
        for media_file in media_files:
            try:
                file_name = os.path.basename(media_file)
                print(f"\nTranscribing {file_name}...")
                
                with open(media_file, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )
                
                # Save transcript to markdown file
                base_name = os.path.splitext(file_name)[0]  # Remove extension
                transcript_file = os.path.join(transcript_folder, f"{base_name}.md")
                
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(f"# Transcript: {file_name}\n\n")
                    f.write(f"**Source File:** `{file_name}`\n\n")
                    f.write(f"**Transcribed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("---\n\n")
                    f.write(transcript)
                
                print(f"✓ Transcription completed for {file_name}")
                print(f"  Saved to: {transcript_file}")
                
                # Add to combined output with clear title
                combined_transcript += f"\n\n{'='*60}\n"
                combined_transcript += f"TRANSCRIPT: {file_name}\n"
                combined_transcript += f"{'='*60}\n\n"
                combined_transcript += transcript
                
            except Exception as e:
                print(f"✗ Error transcribing {file_name}: {e}")
                combined_transcript += f"\n\n{'='*60}\n"
                combined_transcript += f"TRANSCRIPT: {file_name}\n"
                combined_transcript += f"{'='*60}\n\n"
                combined_transcript += f"[Error: Could not transcribe this file - {str(e)}]\n"
        
        # Process markdown files
        for md_file in markdown_files:
            try:
                file_name = os.path.basename(md_file)
                print(f"\nReading {file_name}...")
                
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                print(f"✓ Successfully read {file_name}")
                
                # Add to combined output with clear title
                combined_transcript += f"\n\n{'='*60}\n"
                combined_transcript += f"TRANSCRIPT: {file_name}\n"
                combined_transcript += f"{'='*60}\n\n"
                combined_transcript += content
                
            except Exception as e:
                print(f"✗ Error reading {file_name}: {e}")
                combined_transcript += f"\n\n{'='*60}\n"
                combined_transcript += f"TRANSCRIPT: {file_name}\n"
                combined_transcript += f"{'='*60}\n\n"
                combined_transcript += f"[Error: Could not read this file - {str(e)}]\n"
        
        print(f"\n{'='*60}")
        print(f"Processing complete! Processed {total_files} file(s).")
        if media_files:
            print(f"Audio/video transcripts saved to: {transcript_folder}/")
        print(f"{'='*60}\n")
        
        return combined_transcript.strip()
    
    def _create_context(self, interview_transcript):
        """
        Create the context for the AI agent.
        
        Args:
            interview_transcript (str): Transcribed interview audio
            
        Returns:
            str: Formatted context for the agent
        """
        
        
        context = f"""

            You are an expert in analyzing interview and survey transcripts. You will be provided with one or more interview and survey transcripts that are 
            closely related to each other.

            ## Your task

           
            Fist thing i would like you to do is do coding of the interview transcript.
            Here is General Steps to Code Qualitative Data:

Prepare Data: Get the text data (transcripts, notes) ready, often by numbering lines for easy reference.
Read & Familiarize: Read through the data multiple times to get a general sense of the content.
Choose a Coding Approach:
Inductive (Emergent): Develop codes as you read, letting themes arise from the data (ground-up).
Deductive (Predefined): Start with a set of predetermined codes based on research questions or existing theory (top-down).
Combined: Mix both approaches.
Apply Initial Codes: Go line-by-line or segment-by-segment, highlighting relevant text and assigning descriptive labels (e.g., "Family Support," "Job Satisfaction," "Hiring Process").
Refine & Group Codes: Review initial codes, merge similar ones, split overly broad ones, and group them into higher-level categories or themes.
Develop Themes & Interpret: Look for recurring patterns across codes and data to form overarching themes, then interpret what these themes mean in relation to your research goals.
Use Memos: Write notes (memos) to yourself about your thinking, potential biases, or future research ideas as you go.  
        
        use the tool create_document to create a document with the coding of the interview transcripts. 
        No need to show the coding outcome to the user. Just let the user know that the file has been created. 
            
            After analyzing the interview and survey transcripts and codeing that was just created, 
            use the codeing to generate the common themes and insights from those interviews and surveys.
            Generate snipts of the interview and survey transcript for each theme.

            Give the details of the themes and insights in markdown format with all the data that supports the themes and insights.

            once the themes and insights are generated, create a table of the contents that support the themes and insights. 
            
            Save all the information in a markdown file using the tool create_document  to save the document.
            
            when responding to the user, let the user know that the file has been created.

            Here is the interview transcript:
            {interview_transcript}

            some general guidelines:
            
            I would want to be able to compare themes by profession, experience, and work environment. 
            Later, for example, I might want to see how early career professionals experience support 
            at their workplaces versus later career professionals. I would also want to capture 
            somehow how workplaces might differ from each other. The constant in these interviews is 
            that they were all conducted with Black, female professionals. Everything else might 
            differ. I want to be able to see (if possible) if any differences in how they have been 
            treated at work might be related to their experience, profession, or work environment.

            
            For reference, here is the current date and time:
            {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

            """

        
        
        return context
    
    def _setup_agent(self):
        """Initialize the AI agent with instructions and model."""
        global _interview_analyst_instance
        _interview_analyst_instance = self  # Set the global instance so push function can access it
        
        interview_transcript = self._load_documents()
        instructions = self._create_context(interview_transcript)
        
        # Setup MCP servers first
        #self._setup_mcp_servers()
        print(f"Model: {MODEL}")
        
        # Create agent with provider-specific configuration
        agent_params = {
            "name": "InterviewAnalyst",
            "instructions": instructions,
            "model": MODEL,
            "tools": [create_document]
        }
        
        # Add reasoning_effort only for OpenAI o1 models
        if MODEL_PROVIDER == "openai" and MODEL.startswith("o1"):
            agent_params["reasoning_effort"] = "medium"
        
        self.agent = Agent(**agent_params)
        
    async def chat(self, message, history):
        """
        Chat function for non-streaming responses.
        
        Args:
            message (str): User's message
            history (list): Chat history in tuple format [(user_msg, bot_response), ...]
            
        Returns:
            str: Agent's response
        """
        messages = []
        for user_msg, bot_response in history:
            messages.append({"role": "user", "content": user_msg})
            if bot_response:
                messages.append({"role": "assistant", "content": bot_response})
        
        messages.append({"role": "user", "content": message})
        with trace("Chatting with MyTwinQQ"):
            response = await Runner.run(self.agent, messages)
        return response.final_output
    
    async def chat_streaming(self, message, history):
        """
        Chat function for streaming responses.
        
        Args:
            message (str): User's message
            history (list): Chat history in tuple format [(user_msg, bot_response), ...]
            
        Yields:
            str: Partial response as it streams
        """
        messages = []
        for user_msg, bot_response in history:
            messages.append({"role": "user", "content": user_msg})
            if bot_response:
                messages.append({"role": "assistant", "content": bot_response})
        
        messages.append({"role": "user", "content": message})
        with trace("Streaming response from MyTwinQQ"):
            response = Runner.run_streamed(self.agent, messages)
        reply = ""
        async for event in response.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                reply += event.data.delta
                yield reply
    
    def launch_interface(self, streaming=True, share=False):
        """
        Launch the Gradio chat interface with question generation button.
        
        Args:
            streaming (bool): Whether to use streaming responses
            share (bool): Whether to create a public link
        """
        chat_func = self.chat_streaming if streaming else self.chat
        
        with gr.Blocks(title=f"{self.name}'s Digital Twin") as demo:
            gr.Markdown(f"# Chat with {self.name}'s Digital Twin")
            gr.Markdown("Ask me anything about my career, experience, or background!")
            
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(placeholder="Type your message here...", label="Message")
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")
            
            # Chat functionality with streaming
            def respond(message, history):
                if not message.strip():
                    return history, ""
                
                # Convert history to messages format for the agent
                messages = []
                for user_msg, bot_response in history:
                    messages.append({"role": "user", "content": user_msg})
                    if bot_response:
                        messages.append({"role": "assistant", "content": bot_response})
                
                # Add current message
                messages.append({"role": "user", "content": message})
                
                # Add empty response to history for streaming
                history.append([message, ""])
                
                # Use streaming
                import asyncio
                
                async def stream_response():
                    with trace("Streaming response from MyTwinQQ"):
                        response = Runner.run_streamed(self.agent, messages)
                    reply = ""
                    
                    async for event in response.stream_events():
                        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                            reply += event.data.delta
                            # Update history with streaming response
                            history[-1][1] = reply
                            yield history, ""
                    
                    
                # Create a generator for streaming
                def stream_generator():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        async_gen = stream_response()
                        while True:
                            try:
                                yield loop.run_until_complete(async_gen.__anext__())
                            except StopAsyncIteration:
                                break
                    finally:
                        loop.close()
                
                # Yield from the generator for Gradio streaming
                for result in stream_generator():
                    yield result
            
            # Event handlers
            msg.submit(respond, [msg, chatbot], [chatbot, msg])
            submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
        
        demo.launch(share=share)
    
    

def main():
    """Main function to run the MyTwin application."""
    print(f"Initializing AI Digital Twin...")
    
    # Create the twin
    twin = InterviewAnalysis()
    
    print(f"Digital Twin created for {twin.name}")
    print("Launching chat interface...")
    
    # Launch the interface
    twin.launch_interface(streaming=True, share=False)


if __name__ == "__main__":
    main()
