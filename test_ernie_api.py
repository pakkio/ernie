import gradio as gr
from gradio_client import Client
import time
import threading

def streaming_ernie_chat(client, message, system_message="", max_tokens=10240):
    """
    Generator function that provides real streaming from Ernie API
    """
    try:
        # Submit job for real streaming
        job = client.submit(
            message,
            system_message,
            max_tokens,
            api_name="/chat"
        )
        
        # Stream the response in real-time
        for partial_response in job:
            global cancel_requested
            if cancel_requested:
                cancel_requested = False
                yield "Request cancelled by user."
                return
                
            if partial_response and len(partial_response) > 0:
                response_text = str(partial_response[0]) if isinstance(partial_response, (list, tuple)) else str(partial_response)
                
                # Basic filtering to remove obvious thinking patterns while preserving markdown
                reasoning_patterns = [
                    "**Thinking**:",
                    "I need to come up with",
                    "Let me think about"
                ]
                
                # Remove thinking patterns but preserve markdown structure
                clean_response = response_text
                for pattern in reasoning_patterns:
                    if pattern in clean_response:
                        # Find the end of the thinking section (usually ends with a paragraph break)
                        start_idx = clean_response.find(pattern)
                        if start_idx != -1:
                            # Look for double newline or end of thinking section
                            end_patterns = ['\n\n', '\n\n---', '\n\n#']
                            end_idx = len(clean_response)
                            for end_pattern in end_patterns:
                                temp_idx = clean_response.find(end_pattern, start_idx)
                                if temp_idx != -1:
                                    end_idx = temp_idx
                                    break
                            clean_response = clean_response[:start_idx] + clean_response[end_idx:]
                
                clean_response = clean_response.strip()
                
                if clean_response:
                    yield clean_response
                    
    except Exception as e:
        yield f"Error calling API: {e}"

# Initialize client globally
try:
    client = Client("baidu/ernie_4.5_turbo_demo")
    print("Loaded as API: https://baidu-ernie-4-5-turbo-demo.hf.space ✔")
except Exception as e:
    print(f"Failed to initialize client: {e}")
    client = None

# Global conversation history
conversation_history = []

# Global variable to track cancellation
cancel_requested = False

# Language options
LANGUAGES = {
    "Italian": "Italian",
    "English": "English", 
    "Spanish": "Spanish",
    "French": "French",
    "German": "German",
    "Portuguese": "Portuguese",
    "Russian": "Russian",
    "Chinese": "Chinese",
    "Japanese": "Japanese",
    "Korean": "Korean"
}

def chat_with_ernie(message, history, language="Italian"):
    """
    Chat function for Gradio interface with streaming support
    """
    if not client:
        yield "Error: Client not initialized"
        return
    
    if not message.strip():
        yield "Please enter a message"
        return
    
    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": message})
    
    # Create context from conversation history
    context = ""
    for msg in conversation_history[-5:]:  # Keep last 5 messages for context
        if msg["role"] == "user":
            context += f"User: {msg['content']}\n"
        else:
            context += f"Assistant: {msg['content']}\n"
    
    # Add current context and language instruction to the message
    language_instruction = f"Answer in {language}."
    full_message = f"{language_instruction}\n\nPrevious conversation:\n{context}\nCurrent question: {message}"
    
    # Get streaming response
    full_response = ""
    for partial_response in streaming_ernie_chat(client, full_message):
        full_response = partial_response
        yield partial_response
    
    # Add assistant response to history
    if full_response:
        conversation_history.append({"role": "assistant", "content": full_response})

def clear_history():
    """
    Clear conversation history
    """
    global conversation_history
    conversation_history.clear()
    return []

def cancel_request():
    """
    Set the cancel flag to stop ongoing requests
    """
    global cancel_requested
    cancel_requested = True
    return "Request cancellation sent..."

def create_gradio_interface():
    """
    Create and configure the Gradio interface
    """
    with gr.Blocks(title="Ernie Chat") as demo:
        gr.Markdown("# Ernie Chat Interface")
        gr.Markdown("Chat with Baidu's Ernie 4.5 Turbo model")
        
        with gr.Row():
            language_dropdown = gr.Dropdown(
                choices=list(LANGUAGES.keys()),
                value="Italian",
                label="Response Language",
                scale=1
            )
        
        chatbot = gr.Chatbot(
            height=500,
            placeholder="Start a conversation with Ernie...",
            render_markdown=True
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your message here...",
                container=False,
                scale=3
            )
            submit = gr.Button("Send", variant="primary", scale=1)
            cancel = gr.Button("Cancel", variant="stop", scale=1)
            clear = gr.Button("Clear", variant="secondary", scale=1)
        
        def respond(message, chat_history, language):
            if not message.strip():
                return "", chat_history
            
            # Add user message to chat history
            chat_history.append([message, ""])
            
            # Stream the response with selected language
            for response in chat_with_ernie(message, chat_history, language):
                chat_history[-1][1] = response
                yield "", chat_history
        
        def clear_chat():
            clear_history()
            return []
        
        # Event handlers
        msg.submit(respond, [msg, chatbot, language_dropdown], [msg, chatbot])
        submit.click(respond, [msg, chatbot, language_dropdown], [msg, chatbot])
        cancel.click(cancel_request, outputs=[])
        clear.click(clear_chat, outputs=[chatbot])
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True, debug=True)
