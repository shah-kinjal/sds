"""
VirtualRealtor.py - AI Virtual Realtor Application

This script creates an AI Virtual Realtor that represents a person and can chat
with visitors on their website, answering questions about properties available for sale. 
It will use the MCP servers to get the information about the properties and answer questions about them.
It will use the Playwright MCP server to navigate the website and get the information about the properties.
if the information is not available, it will record the question in the database and send a push notification to the realtor.
It will also search for properties o the internet based on chat and search criteria.
"""
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, trace
from openai.types.responses import ResponseTextDeltaEvent
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
import requests
import os   


# Load environment variables
load_dotenv(override=True)

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

# Global variable to hold the MyTwin instance for the push function
_realtor_instance = None

@function_tool
def push(message: str) -> str:
    """Send a text message as a push notification to {self.name} with this brief message

    Args:
        message: The short text message to push to {self.name}.
    """
    if _realtor_instance:
        _realtor_instance.send_push_notification(message)
        return "Push notification sent"
    return "Push notification failed - realtor instance not available"
    
#@function_tool
def get_properties_info(address: str) -> str:
    """Get the information about the properties in the given address

    Args:
        address: The address of the property to get the information about.
    """
    if _realtor_instance:
        _realtor_instance.get_property_info   (address)
        return "Property information sent"
    return "Property information failed - realtor instance not available"


class VirtualRealtor:
    """
    AI Digital Twin class that represents a person and can chat with visitors.
    """
    
    def __init__(self, name="Monika Trivedi", dre_number="CA DRE #01975393"):
        """
        Initialize the MyTwin with personal information.
        
        Args:
            name (str): The name of the person being represented
        """
        self.name = name
        self.dre_number = dre_number
        self.phone_number = "+1 (510) 468-0602"
        self.email = "monika@monikarealty.com"
        self.website = "https://monikarealty.com"
        self.company = "Legacy Real Estate"
        self.team_name = "Monika Realty Team"
        self.mcp_servers = None
        self.agent = None
        self._setup_agent()

    
    def _setup_mcp_servers(self):
        """
        Setup the MCP servers.
        """
        try:
            #params = {"command": "uv", "args": ["run", "db_tools_mcp_server.py"]}
            #with MCPServerStdio(params=params, client_session_timeout_seconds=30) as db_tools_mcp_server:
            #    self.db_tools_mcp_server = db_tools_mcp_server
            #    self.mcp_servers.append(db_tools_mcp_server)

            playwright_params = {   
                "command": "npx",
                "args": [
                    "@playwright/mcp@latest"
                ]
            }
            with MCPServerStdio(params=playwright_params, client_session_timeout_seconds=30) as playwright:
                tools =  playwright.session.list_tools()
                self.playwright_tools = tools.tools
                self.mcp_servers.append(playwright)
        except Exception as e:
            print(f"Warning: Failed to setup MCP server: {e}")
            #self.db_tools_mcp_server = None
            self.mcp_servers = []
            self.playwright_tools = []
        
    ###def _load_documents(self):
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
    ###
    
    def _create_instructions(self):
        """
        Create the instructions for the AI agent.
        
        
            
        Returns:
            str: Formatted instructions for the agent
        """
        instructions = f"""You represent the AI Digital Twin of a Realtor called {self.name}. 
        
        {self.name} is Realtor in the state of California with the DRE number {self.dre_number}.
        Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. 
        You are given the information about {self.name}'s professional background which you can use to answer questions some qustions about {self.name}'s background.
        Always try to bring back the conversation to the properties available for sale.
            Here is about me blurp of information about {self.name}:
            ---
            Having the right real estate agent means having an agent who is committed to helping you buy or sell your home with the highest 
            level of expertise in your local market. This means also to help you in understanding each step of the buying or selling process. 
            This commitment level has helped me build a remarkable track record of delivering results. Nothing is more exciting to me than 
            the gratifying feeling I get from helping people meet their real estate needs. You can count on me to always do what's in your 
            best interest. I pride myself on being honest, trustworthy, and knowledgeable in the real estate market. I know how important 
            it is to find your dream home or get the best offer for your property. Therefore I will make it my responsibility to help you 
            achieve those goals. Whether you are an experienced investor or a first time buyer, I can help you in finding the property of 
            your dreams. 
            {self.name} loves what they do and enjoys helping first time as well as experienced buyers and has a passion 
            for helping people find their dream home. They have a 10+ yars of good track record of delivering results and helping people
             meet their real estate needs. here is their website: {self.website}
             here is their phone number: {self.phone_number}
             here is their email: {self.email}
             here is their company: {self.company}
             here is their team name: {self.team_name}
            ---
            You are answering questions on {self.name}'s website, 
            particularly questions related to properties available for sale. 
            You are able to search the internet using the MCP tool that is
            available to you. Pleaase feel free to visit either  Redfin or Zillow or Realtor.com to get the information about the properties. 
            Please use the MCP tools to get the information about the properties.

            Be professional and engaging, as if talking to a potential or current client. 
            You are friendly and amiable, and you introduce yourself as {self.name}'s Virtual Self.
            Here are the tools that you have available to you:  
            - Tool to search the internet using the MCP tools to find the information about the properties playwright mcp server.
            - Tool to send a push notification to {self.name} to tell him the question you couldn't answer. 
           
            """

        
        
        return instructions
    
    def _setup_agent(self):
        """Initialize the AI agent with instructions and model."""
        global _realtor_instance
        _realtor_instance = self  # Set the global instance so push function can access it
        
        #resume, profile = self._load_documents()
        #instructions = self._create_instructions(resume, profile)
        instructions = self._create_instructions()
        
        # Setup MCP servers first
        self._setup_mcp_servers()
        
        # Create agent with MCP servers if available
        if self.mcp_servers:
            self.agent = Agent(name="VirtualRealtor", instructions=instructions, model="gpt-4o-mini", tools=[push], mcp_servers=self.mcp_servers)
        else:
            self.agent = Agent(name="VirtualRealtor", instructions=instructions, model="gpt-4o-mini", tools=[push])
    
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
        with trace("Chat"):
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
        with trace("Streaming response"):
            response = Runner.run_streamed(self.agent, messages)
        reply = ""
        async for event in response.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                reply += event.data.delta
                yield reply
    
    async def generate_professional_questions(self):
        """
        Generate 4 questions based on the information about the properties available for sale.
        
        Returns:
            str: Formatted list of 4 questions
        """
        
        return "What are the properties available for sale in the area of San Francisco?"
    
    def launch_interface(self, streaming=True, share=False):
        """
        Launch the Gradio chat interface with question generation button.
        
        Args:
            streaming (bool): Whether to use streaming responses
            share (bool): Whether to create a public link
        """
        chat_func = self.chat_streaming if streaming else self.chat
        
        try:
            canned_question = asyncio.run(self.generate_professional_questions())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                canned_question = loop.run_until_complete(self.generate_professional_questions())
            finally:
                loop.close()
        except Exception as exc:
            print(f"Warning: Failed to load canned question: {exc}")
            canned_question = ""
        
        with gr.Blocks(title=f"{self.name}'s Digital Twin") as demo:
            gr.Markdown(f"# Chat with {self.name}'s Digital Twin")
            gr.Markdown("Ask me anything about my career, experience, or background!")
            
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(placeholder="Type your message here...", label="Message")
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")
            
            canned_question_btn = None
            if canned_question:
                with gr.Row():
                    gr.Markdown("**Suggested question:**")
                    canned_question_btn = gr.Button(canned_question, variant="secondary")
            
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
                    with trace("Streaming response"):
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
            
            if canned_question_btn is not None:
                canned_question_btn.click(lambda: canned_question, outputs=[msg])
        
        demo.launch(share=share)
    
    

def main():
    """Main function to run the Virual Realtor application."""
    print(f"Initializing AI virtual realtor...")
    
    # Create the realtor
    v_realtor = VirtualRealtor()
    
    print(f"Digital Twin created for {v_realtor.name}")
    print("Launching virtual realtor interface...")
    
    # Launch the interface
    v_realtor.launch_interface(streaming=True, share=False)


if __name__ == "__main__":
    main()
