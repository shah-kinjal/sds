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
from pypdf import PdfReader
import gradio as gr
from agents.mcp import MCPServerStdio
from datetime import datetime
from pathlib import Path
import requests
import os   

# Load environment variables
load_dotenv(override=True)

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

# Global variable to hold the MyTwin instance for the push function
_twin_instance = None

@function_tool
def push(message: str) -> str:
    """Send a text message as a push notification to Kinjal with this brief message

    Args:
        message: The short text message to push to Kinjal.
    """
    if _twin_instance:
        _twin_instance.send_push_notification(message)
        return "Push notification sent"
    return "Push notification failed - twin instance not available"


class MyTwin:
    """
    AI Digital Twin class that represents a person and can chat with visitors.
    """
    
    def __init__(self, name="Kinjal Shah"):
        """
        Initialize the MyTwin with personal information.
        
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
        # Load LinkedIn profile
        try:
            reader = PdfReader("info/LinkedIn_profile_new.pdf")
            profile = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    profile += text
        except FileNotFoundError:
            print("Warning: LinkedIn profile file not found. Using empty string.")
            profile = ""
        
        # Load Resume profile
        try:
            reader = PdfReader("info/KinjalShah_EngLeader_AI_gold.pdf")
            resume = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    resume += text
        except FileNotFoundError:
            print("Warning: Resume file not found. Using empty string.")
            resume = ""
        
        
        return resume, profile
    
    def _create_context(self, resume, profile):
        """
        Create the context for the AI agent.
        
        Args:
            resume (str): Resume text
            profile (str): LinkedIn profile text
            
        Returns:
            str: Formatted context for the agent
        """
        
        name = "Kinjal Shah"
        first_name = "Kinjal"
        context = f"""

            You represent the AI Digital Twin of a human called {name}.

            ## Your task

            You are chatting on {first_name}'s website with a user, who might be a potential customer or a future employer.

            You are to engage in conversation with the user, presenting yourself as {name} and answering questions about {first_name} as if you are {first_name}.
            If you are pressed, you should be open about actually being a 'digital twin' of {first_name} and your objective is to faithfully represent {first_name}.
            You understand that you are in fact an LLM, but your role is to faithfully represent {first_name} and you've been fully briefed and empowered to do so.
            Keep your answer short and concise.
            only use the information provided to answer a question if you can not answer a question. Do not make one up or hallucinate or guess.    
            clearly from the information provided, do not make one up or hallucinate or guess.
            if you do not have the answer from the information provided, please do not hallucinate or make one up or guess.
            Instead tell the user that you do not have the answer to that question.
            Then use the tool to record a question that you cannot answer in the long term memory.
            Also notify the {first_name} that an answer is needed by sending a push notification. 
           
            ## Information about {first_name}

            {first_name} is the Engineering Leader and a consultant with 20+ years of experience in software development building scalable and resilient systems and teams.
            {first_name} loves coding building scalable and resilient systems. Most recently he has been learin Agentic AI and he is really into building interactive applicationand  
            and experimenting with LLMs.
            
            Here is the LinkedIn profile of {first_name}:
            {profile}
            Here is the resume of {first_name}:
            {resume}
            use only this information to answer professional questions.

           
            ## {first_name}'s style

            {first_name} has an empathetic and friendly and slightly humorous style. {first_name} loves to tell jokes and engage with users.

            ## How you should respond

            Channel {first_name}'s personality, style and knowledge.
            Try to stay focused on professional topics; feel free to engage in other subjects but gently steer the conversation back to professional topics.
             
             

            ## Tools

            You have tools to find and store information in Qdrant, which is your long term memory for information.
            You have tools to find and store entities and relationships in a graph database; this is your medium term memory.
            You should make frequent use of both long and medium term memories.
            Most importantly, use the tools to record a question that you cannot answer in the long term and medium term memory. 
            
            you have tool to send push notifications to {first_name} to notify him of any unanwered questions. 
            use should make user of this tool to notify {first_name} of any unanwered questions.
            
            For reference, here is the current date and time:
            {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

            """

        context += f"\n\n## LinkedIn Profile:\n{profile}\n\n## Resume:\n{resume}\n\n"
        
        
        return context
    
    def _setup_agent(self):
        """Initialize the AI agent with instructions and model."""
        global _twin_instance
        _twin_instance = self  # Set the global instance so push function can access it
        
        resume, profile = self._load_documents()
        instructions = self._create_context(resume, profile)
        
        # Setup MCP servers first
        self._setup_mcp_servers()
        
        
        self.agent = Agent(name="MyTwinQQ", instructions=instructions, model="gpt-4o-mini", tools=[push], mcp_servers=self.mcp_servers)
        
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
    
    async def generate_professional_questions(self):
        """
        Generate 4 professional questions based on the resume and LinkedIn profile.
        
        Returns:
            str: Formatted list of 4 professional questions
        """
        resume, profile = self._load_documents()
        
        prompt = f"""Based on the following resume and LinkedIn profile for {self.name}, 
        generate exactly 4 thoughtful, professional questions that a potential employer, 
        client, or business partner might ask during an interview or business meeting. 
        The questions should be relevant to {self.name}'s experience, skills, and background. Please keep the questions short and concise under 10 words.
        The questions should be in the form of a question, not a statement.
        
        Resume: \n{resume}\n\n
        LinkedIn Profile: \n{profile}\n\n
        
        Please format your response as a numbered list of exactly 4 questions, 
        each on a new line, without any additional commentary or introduction."""
        
        instructions="You are a professional interviewer and business consultant who creates insightful, relevant questions based on someone's professional background."
        
        self.question_agent = Agent(name="Question Generator", instructions=instructions, model="gpt-4o-mini")
        messages = [{"role": "user", "content": prompt}]
        with trace("Generating questions for MyTwinQQ"):
            response = await Runner.run(self.question_agent, messages)
        
        return response.final_output
    
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
                question_btn = gr.Button("Generate Questions", variant="secondary")
            
            # Question buttons container (initially hidden)
            with gr.Row(visible=False) as question_row:
                gr.Markdown("**Click on a question to use it:**")
            
            with gr.Row(visible=False) as question_buttons_row:
                question_btn1 = gr.Button("", variant="outline", visible=False)
                question_btn2 = gr.Button("", variant="outline", visible=False)
                question_btn3 = gr.Button("", variant="outline", visible=False)
                question_btn4 = gr.Button("", variant="outline", visible=False)
            
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
            
            # Question generation functionality
            async def generate_questions():
                try:
                    questions_text = await self.generate_professional_questions()
                    # Parse questions from the text (assuming numbered list format)
                    questions = []
                    lines = questions_text.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                            # Remove numbering/bullets and clean up
                            question = line
                            if line[0].isdigit():
                                # Remove "1. ", "2. ", etc.
                                question = line.split('.', 1)[1].strip() if '.' in line else line
                            elif line.startswith('•') or line.startswith('-'):
                                question = line[1:].strip()
                            
                            if question.endswith('?'):
                                questions.append(question)
                    
                    return questions
                except Exception as e:
                    return []
            
            def generate_and_show_questions():
                """Generate questions and show them as clickable buttons"""
                import asyncio
                questions = asyncio.run(generate_questions())
                
                if not questions:
                    return (
                        gr.update(visible=False),  # question_row
                        gr.update(visible=False),  # question_buttons_row
                        gr.update(visible=False),  # question_btn1
                        gr.update(visible=False),  # question_btn2
                        gr.update(visible=False),  # question_btn3
                        gr.update(visible=False),  # question_btn4
                    )
                
                # Show the question row and buttons
                updates = [
                    gr.update(visible=True),  # question_row
                    gr.update(visible=True),  # question_buttons_row
                ]
                
                # Update each button with a question (max 4)
                for i in range(4):
                    if i < len(questions):
                        updates.append(gr.update(visible=True, value=questions[i]))
                    else:
                        updates.append(gr.update(visible=False))
                
                return tuple(updates)
            
            def select_question(question_text):
                """Handle question selection"""
                return question_text
            
            # Event handlers
            msg.submit(respond, [msg, chatbot], [chatbot, msg])
            submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
            
            # Question generation handler
            question_btn.click(
                generate_and_show_questions,
                outputs=[question_row, question_buttons_row, question_btn1, question_btn2, question_btn3, question_btn4]
            )
            
            # Question button click handlers
            question_btn1.click(select_question, inputs=[question_btn1], outputs=[msg])
            question_btn2.click(select_question, inputs=[question_btn2], outputs=[msg])
            question_btn3.click(select_question, inputs=[question_btn3], outputs=[msg])
            question_btn4.click(select_question, inputs=[question_btn4], outputs=[msg])
        
        demo.launch(share=share)
    
    

def main():
    """Main function to run the MyTwin application."""
    print(f"Initializing AI Digital Twin...")
    
    # Create the twin
    twin = MyTwin()
    
    print(f"Digital Twin created for {twin.name}")
    print("Launching chat interface...")
    
    # Launch the interface
    twin.launch_interface(streaming=True, share=False)


if __name__ == "__main__":
    main()
