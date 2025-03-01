import groq
from groq import AsyncGroq
import time
import os
import json
import asyncio
from typing import Dict, List, Tuple, Optional, AsyncGenerator, AsyncIterator
from dataclasses import dataclass

client = AsyncGroq()

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

class DynamicPipeline:
    def __init__(self, custom_client=None):
        self.client = custom_client if custom_client else client
        self.adaptive_logic = AdaptiveLogic()
        self.message_queue = asyncio.Queue()
        self.final_answer_task = None

    async def make_api_call(self, messages: List[Dict], max_tokens: int, is_final_answer: bool = False) -> AsyncIterator[Dict]:
        """Asynchronous streaming API call handler"""
        for attempt in range(3):
            try:
                # Create streaming completion
                stream = await self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.2,
                    response_format={"type": "json_object"} if not is_final_answer else None,
                    stream=True
                )

                accumulated_content = ""
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        accumulated_content += chunk.choices[0].delta.content
                        yield {"chunk": chunk.choices[0].delta.content}

                # After streaming is complete, yield the final parsed content
                if accumulated_content:
                    try:
                        if is_final_answer:
                            yield {"final": accumulated_content}
                        else:
                            yield {"final": json.loads(accumulated_content)}
                    except json.JSONDecodeError:
                        yield {"error": "Failed to parse JSON response"}
                return

            except Exception as e:
                if attempt == 2:
                    if is_final_answer:
                        yield {"error": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
                    else:
                        yield {"error": f"Failed to generate step after 3 attempts. Error: {str(e)}", "next_action": "final_answer"}
                    return
                await asyncio.sleep(1)  # Async wait between retries

    async def process_step(self, messages: List[Dict], step_count: int) -> AsyncIterator[Optional[StepResult]]:
        """Process a single reasoning step with streaming"""
        start_time = time.time()
        accumulated_content = ""
        
        async for chunk_data in self.make_api_call(messages, 300):
            if "chunk" in chunk_data:
                accumulated_content += chunk_data["chunk"]
                # Yield progress for UI updates
                yield None
            elif "final" in chunk_data:
                step_data = chunk_data["final"]
                thinking_time = time.time() - start_time
                
                yield StepResult(
                    title=step_data['title'],
                    content=step_data['content'],
                    next_action=step_data['next_action'],
                    thinking_time=thinking_time
                )
            elif "error" in chunk_data:
                yield StepResult(
                    title="Error",
                    content=chunk_data["error"],
                    next_action="final_answer",
                    thinking_time=time.time() - start_time
                )
    
    async def generate_final_answer(self, messages: List[Dict]) -> AsyncIterator[Tuple[Optional[str], float]]:
        """Generate final answer in the background with streaming"""
        messages = messages.copy()
        messages.append({
            "role": "user",
            "content": "Please provide the final answer based solely on your reasoning above. Do not use JSON formatting. Only provide the text response without any titles or preambles. Retain any formatting as instructed by the original prompt, such as exact formatting for free response or multiple choice."
        })
        
        start_time = time.time()
        accumulated_content = ""
        
        async for chunk_data in self.make_api_call(messages, 1200, is_final_answer=True):
            if "chunk" in chunk_data:
                accumulated_content += chunk_data["chunk"]
                # Yield progress for UI updates
                yield None, time.time() - start_time
            elif "final" in chunk_data:
                yield chunk_data["final"], time.time() - start_time
            elif "error" in chunk_data:
                yield chunk_data["error"], time.time() - start_time

async def generate_response(prompt: str, custom_client=None) -> AsyncGenerator[Tuple[List[Tuple[str, str, float]], Optional[float]], None]:
    """Main response generator using dynamic pipelining with streaming"""
    pipeline = DynamicPipeline(custom_client)
    
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.

Example of a valid JSON response:
```json
{
    "title": "Identifying Key Information",
    "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
    "next_action": "continue"
}```
"""},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]
    
    steps: List[Tuple[str, str, float]] = []
    step_count = 1
    total_thinking_time = 0
    final_answer_task = None
    current_step_content = ""
    
    while True:
        # Process next step with streaming
        current_step = None
        async for step_result in pipeline.process_step(messages, step_count):
            if step_result is None:
                # Streaming progress update
                yield steps, None
                continue
                
            current_step = step_result
            total_thinking_time += step_result.thinking_time
            
            # Add step to results
            step_title = f"Step {step_count}: {step_result.title}"
            steps.append((step_title, step_result.content, step_result.thinking_time))
            
            # Update messages
            messages.append({"role": "assistant", "content": json.dumps({
                "title": step_result.title,
                "content": step_result.content,
                "next_action": step_result.next_action
            })})
            
            # Yield progress
            yield steps, None
        
        if current_step is None:
            # Error occurred during step generation
            break
            
        # Check if we should stop
        should_stop = (
            step_count >= 3 and not pipeline.adaptive_logic.evaluate_step(current_step)
            or current_step.next_action == 'final_answer'
            or step_count >= 25
        )
        
        if should_stop:
            # Generate final answer with streaming
            final_answer_content = ""
            async for final_result, thinking_time in pipeline.generate_final_answer(messages):
                if final_result is not None:
                    final_answer_content = final_result
                    total_thinking_time += thinking_time
                    steps.append(("Final Answer", final_answer_content, thinking_time))
                yield steps, None
            break
            
        step_count += 1
    
    # Yield final result with total time
    yield steps, total_thinking_time
