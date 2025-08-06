# Utility Helper Functions
# 
# This module contains utility functions to support the RAG pipeline,
# particularly for handling synchronous/asynchronous execution compatibility.

import asyncio
import typing


async def run_in_threadpool(
    func: typing.Callable, 
    *args: typing.Any, 
    **kwargs: typing.Any
) -> typing.Any:
    """
    Execute a synchronous function in a separate thread to avoid blocking
    the FastAPI event loop.
    
    This function is essential for integrating synchronous RAG pipeline operations
    with FastAPI's asynchronous architecture. Long-running synchronous operations
    like document processing, embedding generation, and LLM calls can block the
    entire server if run directly in the main event loop.
    
    Args:
        func (typing.Callable): The synchronous function to execute
        *args (typing.Any): Positional arguments to pass to the function
        **kwargs (typing.Any): Keyword arguments to pass to the function
        
    Returns:
        typing.Any: The result returned by the executed function
        
    Example:
        # Instead of blocking the event loop with:
        # result = some_sync_function(arg1, arg2, kwarg1=value1)
        
        # Use this to run in a separate thread:
        # result = await run_in_threadpool(some_sync_function, arg1, arg2, kwarg1=value1)
    """
    return await asyncio.to_thread(func, *args, **kwargs)