import streamlit as st
import asyncio
from g1_parallel import generate_response
import json
from typing import AsyncGenerator, Any

# Must be the first Streamlit command
st.set_page_config(page_title="g1 parallel prototype", page_icon="🧠", layout="wide")

async def run_async_response(user_query: str, response_container: Any, time_container: Any, progress_container: Any):
    """Run the async response generator and update the UI with streaming"""
    current_step_container = None
    current_step_title = None
    current_step_content = ""
    
    async for steps, total_thinking_time in generate_response(user_query):
        with response_container.container():
            for i, (title, content, thinking_time) in enumerate(steps):
                # Handle streaming updates
                if title != current_step_title or content != current_step_content:
                    current_step_title = title
                    current_step_content = content
                    
                    if title == "Final Answer":
                        st.markdown("---")
                        st.markdown("### Final Answer")
                        st.success(content)  # Use success box for better visibility
                        st.markdown("---")
                        st.caption(f"Thinking time: {thinking_time:.2f}s")
                    else:
                        with st.expander(title, expanded=True):
                            st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                            st.caption(f"Step thinking time: {thinking_time:.2f}s")
            
            # Show total time when available
            if total_thinking_time is not None:
                time_container.markdown(f"**Total thinking time: {total_thinking_time:.2f} seconds**")
            
            # Update progress
            progress_container.progress(min(100, (len(steps) * 20)))  # Rough progress estimation

def main():
    st.title("g1: Using Llama 3.2 3B with Parallel Processing")
    
    st.markdown("""
    This version uses Ollama with Llama 3.2 3B for faster local inference:
    - Lightweight 3B model for quicker responses
    - Real-time streaming updates
    - Parallel step processing
    - Adaptive reasoning chain length
    - Dynamic content evaluation
    
    Note: Make sure Ollama is running (`ollama serve`) and you have pulled the model (`ollama pull llama3.2:3b`) before using this application.
    """)
    
    # Text input for user query
    user_query = st.text_input("Enter your query:", placeholder="e.g., How many 'R's are in the word strawberry?")
    
    if user_query:
        # Create containers for different parts of the output
        progress_container = st.empty()
        status_container = st.empty()
        response_container = st.empty()
        time_container = st.empty()
        
        status_container.write("Initializing response generation...")
        progress_container.progress(0)
        
        try:
            # Test Ollama connection first
            import subprocess
            try:
                # Try to connect to Ollama using curl
                status_container.write("Testing connection to Ollama...")
                result = subprocess.run(
                    ["curl", "-s", "http://localhost:11434/api/tags"],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    raise ConnectionError("Could not connect to Ollama")
                status_container.write("Connected to Ollama successfully")
            except Exception as e:
                status_container.error("""
                Could not connect to Ollama. Please ensure:
                1. WSL is installed and running
                2. Ollama is installed in WSL (`curl -fsSL https://ollama.ai/install.sh | bash`)
                3. Ollama service is running (`ollama serve`)
                4. You have pulled the model (`ollama pull llama3.2:3b`)
                
                Error details: {str(e)}
                """)
                progress_container.empty()
                return

            # Create event loop in a separate thread
            status_container.write("Initializing async runtime...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async response generator
            status_container.write("Starting response generation...")
            loop.run_until_complete(
                run_async_response(user_query, response_container, time_container, progress_container)
            )
            status_container.empty()  # Clear the status message when done
            loop.close()
        except ConnectionRefusedError as e:
            status_container.error(f"""
            Could not connect to Ollama. Please ensure:
            1. WSL is installed and running
            2. Ollama is installed in WSL (`curl -fsSL https://ollama.ai/install.sh | bash`)
            3. Ollama service is running (`ollama serve`)
            4. You have pulled the model (`ollama pull llama3.2:3b`)
            
            Error details: {str(e)}
            """)
            progress_container.empty()
        except Exception as e:
            import traceback
            status_container.error(f"""
            An error occurred:
            {str(e)}
            
            Stack trace:
            {traceback.format_exc()}
            """)
            progress_container.empty()

if __name__ == "__main__":
    main()