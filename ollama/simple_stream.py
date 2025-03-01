import streamlit as st
import ollama
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

# Create a thread pool for running ollama calls
executor = ThreadPoolExecutor(max_workers=3)

async def stream_response(prompt: str) -> AsyncGenerator[str, None]:
    """Stream response from Ollama"""
    def _make_request():
        try:
            return ollama.chat(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
        except Exception as e:
            error = str(e).lower()
            if "connection refused" in error:
                print("Could not connect to Ollama server")
                return None
            elif "not found" in error:
                print("Model not found. Please run 'ollama pull llama3.2:3b' first")
                return None
            else:
                print(f"Error making request: {str(e)}")
                return None

    try:
        # Run ollama.chat in thread pool
        response = await asyncio.get_event_loop().run_in_executor(executor, _make_request)
        if not response:
            if "connection refused" in str(e).lower():
                yield "Error: Could not connect to Ollama. Please make sure Ollama is running ('ollama serve')"
            elif "not found" in str(e).lower():
                yield "Error: Model not found. Please run 'ollama pull llama3.2:3b' first"
            else:
                yield f"Error: {str(e)}"
            return

        accumulated = ""
        # Process the response stream
        for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']
                accumulated += content
                yield accumulated

    except Exception as e:
        yield f"Error: {str(e)}"

def main():
    st.title("Simple Ollama Stream")
    
    # Text input for user query
    user_query = st.text_input("Enter your query:", placeholder="e.g., How many 'R's are in the word strawberry?")
    
    if user_query:
        # Create container for output
        output_container = st.empty()
        
        try:
            # Create event loop in a separate thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the streaming response
            async def run_stream():
                async for text in stream_response(user_query):
                    output_container.markdown(text)
            
            loop.run_until_complete(run_stream())
            loop.close()
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()