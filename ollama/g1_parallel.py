import ollama
import os
import json
import time
import asyncio
from typing import Dict, List, Tuple, Optional, AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import re

# Create a thread pool for running ollama calls
executor = ThreadPoolExecutor(max_workers=3)

@dataclass
class StepResult:
    title: str
    content: str
    next_action: str
    thinking_time: float

class AdaptiveLogic:
    def __init__(self):
        self.previous_steps: List[StepResult] = []
        
    def evaluate_step(self, step: StepResult) -> bool:
        """
        Evaluate if we should continue generating steps based on:
        - Content quality
        - Repetition detection
        - Step count
        Returns True if we should continue, False if we should stop
        """
        # Check for repetitive content
        for prev_step in self.previous_steps[-3:]:  # Look at last 3 steps
            if self._content_similarity(step.content, prev_step.content) > 0.8:
                return False
                
        # Store step for future comparison
        self.previous_steps.append(step)
        return True
        
    def _content_similarity(self, content1: str, content2: str) -> float:
        """Simple content similarity check"""
        # Convert to sets of words for comparison
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0

class ParallelPipeline:
    def __init__(self):
        self.adaptive_logic = AdaptiveLogic()
        self.message_queue = asyncio.Queue()
        self.final_answer_task = None
        
    async def _stream_response(self, messages: List[Dict], max_tokens: int) -> AsyncIterator[str]:
        """Stream response from Ollama using ThreadPoolExecutor"""
        def _make_request():
            try:
                return ollama.chat(
                    model="llama3.2:3b",
                    messages=messages,
                    options={
                        "temperature": 0.2,
                        "num_predict": max_tokens
                    },
                    stream=True
                )
            except Exception as e:
                error = str(e).lower()
                if "connection refused" in error:
                    raise Exception("Could not connect to Ollama. Please run 'ollama serve'")
                elif "not found" in error:
                    raise Exception("Model not found. Please run 'ollama pull llama3.2:3b'")
                raise
        
        try:
            # Run ollama.chat in thread pool
            response = await asyncio.get_event_loop().run_in_executor(executor, _make_request)
            
            # Process the response stream
            print("\nStarting to process response stream...")
            for chunk in response:
                print("\nRaw chunk:", chunk)
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    print("Extracted content:", content)
                    await asyncio.sleep(0)  # Allow other tasks to run
                    yield content
                else:
                    print("No content in chunk")
                    
        except Exception as e:
            yield f"Error: {str(e)}"
            return

    async def process_step(self, messages: List[Dict], step_count: int) -> AsyncIterator[Optional[StepResult]]:
        """Process a single reasoning step with streaming"""
        start_time = time.time()
        accumulated = ""
        step_complete = False
        
        try:
            print("\nStarting step processing...")
            async for chunk in self._stream_response(messages, 300):
                accumulated += chunk
                print(f"\nAccumulated content ({len(accumulated)} chars):", accumulated)
                
                # Yield None to show progress
                yield None
                
                # Check if we have a complete step
                if "Step " in accumulated and not step_complete:
                    print("\nFound 'Step' marker, looking for next step...")
                    # Find all step markers
                    all_markers = list(re.finditer(r'Step \d+:|Final Answer:', accumulated))
                    if len(all_markers) > 1:  # We have at least two markers (current and next)
                        current_marker = all_markers[-2]  # Get the second-to-last marker
                        next_marker = all_markers[-1]    # Get the last marker
                        print("Found markers:", [m.group() for m in all_markers])
                        
                        # Extract content between current and next marker
                        step_content = accumulated[current_marker.start():next_marker.start()]
                        step_complete = True
                        
                        # Extract step number and content
                        step_match = re.match(r'Step (\d+):(.*)', step_content, re.DOTALL)
                        if step_match:
                            step_num = step_match.group(1)
                            content = step_match.group(2).strip()
                            print(f"\nExtracted Step {step_num} ({len(content)} chars):", content)
                            
                            # Check if this is a new step
                            if int(step_num) == step_count:
                                result = StepResult(
                                    title=f"Step {step_num}",
                                    content=content,
                                    next_action="continue" if "Final Answer:" not in accumulated else "final_answer",
                                    thinking_time=time.time() - start_time
                                )
                                yield result
                                if "Final Answer:" in content:
                                    return
            
            if not step_complete:
                # If we haven't found a complete step, use the whole content
                yield StepResult(
                    title=f"Step {step_count}",
                    content=accumulated.strip(),
                    next_action="continue",
                    thinking_time=time.time() - start_time
                )
                
        except Exception as e:
            yield StepResult(
                title="Error",
                content=f"Failed to process step: {str(e)}",
                next_action="final_answer",
                thinking_time=time.time() - start_time
            )

    async def generate_final_answer(self, messages: List[Dict]) -> AsyncIterator[Tuple[Optional[str], float]]:
        """Generate final answer with streaming"""
        messages = messages.copy()
        messages.append({
            "role": "user",
            "content": "Based on your steps above, what is your final answer?"
        })
        
        start_time = time.time()
        accumulated = ""
        final_answer_found = False
        
        try:
            async for chunk in self._stream_response(messages, 1200):
                accumulated += chunk
                
                # Look for final answer section
                if "Final Answer:" in accumulated and not final_answer_found:
                    final_answer_found = True
                    final_text = accumulated[accumulated.find("Final Answer:"):].split("\n\n")[0]  # Get only the first paragraph
                    final_text = final_text.replace("Final Answer:", "").strip()
                    yield final_text, time.time() - start_time
                    return  # Return instead of break
                elif not final_answer_found:
                    yield None, time.time() - start_time
                
                await asyncio.sleep(0)  # Allow other tasks to run
            
        except Exception as e:
            yield f"Error generating final answer: {str(e)}", time.time() - start_time

async def generate_response(prompt: str) -> AsyncGenerator[Tuple[List[Tuple[str, str, float]], Optional[float]], None]:
    """Main response generator using parallel pipelining"""
    print("\n=== Starting response generation ===")
    print("Prompt:", prompt)
    pipeline = ParallelPipeline()
    
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step.

For each step:
1. Start with "Step X:" where X is the step number
2. Explain your thinking clearly
3. Use at least 3 steps before giving your final answer
4. Consider alternative approaches
5. Be thorough but concise

When you reach your conclusion, write "Final Answer:" followed by your answer.
STOP IMMEDIATELY after giving your final answer - do not continue with more steps."""},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "I'll help you with that. Let me break it down step by step."}
    ]
    
    steps: List[Tuple[str, str, float]] = []
    step_count = 1
    total_thinking_time = 0
    current_final_answer = ""
    
    try:
        while True:
            print(f"\n=== Processing Step {step_count} ===")
            # Process next step with streaming
            current_step = None
            async for step_result in pipeline.process_step(messages, step_count):
                if step_result is None:
                    # Streaming progress update
                    yield steps, None
                    await asyncio.sleep(0)  # Allow other tasks to run
                    continue
                    
                current_step = step_result
                total_thinking_time += step_result.thinking_time
                print(f"\nStep {step_count} completed:", current_step.title)
                
                # Add step to results
                step_title = f"Step {step_count}: {step_result.title}"
                steps.append((step_title, step_result.content, step_result.thinking_time))
                
                # Update messages with plain text
                messages.append({
                    "role": "assistant",
                    "content": f"{step_result.title}:\n{step_result.content}"
                })
                
                # Yield progress
                yield steps, None
            
            if current_step is None:
                print("\nError: Step generation failed")
                break
                
            # Check if current step contains final answer
            if "Final Answer:" in current_step.content:
                # Extract final answer and stop
                final_text = current_step.content[current_step.content.find("Final Answer:"):].split("\n\n")[0]
                final_text = final_text.replace("Final Answer:", "").strip()
                print("\nExtracted final answer:", final_text)
                
                # Add both the step and the final answer
                steps.append((step_title, current_step.content, current_step.thinking_time))
                steps.append(("Final Answer", final_text, current_step.thinking_time))
                
                # Yield one last time to show the final answer
                yield steps, None
                
                # Then yield with total time and return
                yield steps, total_thinking_time
                return
            
            # Check if we should stop
            should_stop = (
                step_count >= 3 and not pipeline.adaptive_logic.evaluate_step(current_step)
                or current_step.next_action == 'final_answer'
                or step_count >= 25
            )
            print(f"\nShould stop? {should_stop} (step_count={step_count}, next_action={current_step.next_action})")
            
            if should_stop:
                break
                
            step_count += 1
            await asyncio.sleep(0)  # Allow other tasks to run
        
        # Yield final result with total time
        yield steps, total_thinking_time
        
    except Exception as e:
        # Handle any unexpected errors
        steps.append(("Error", f"An error occurred: {str(e)}", 0))
        yield steps, total_thinking_time