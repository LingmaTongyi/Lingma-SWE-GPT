"""
Interfacing with GPT models.
"""

import json
import os
import sys
from typing import List, Tuple
from typing import cast
from typing import Literal

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import (
    Function as OpenaiFunction,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from app import globals
from app.data_structures import FunctionCallIntent
from app.log import log_and_cprint, log_and_print

import requests
import dashscope
from http import HTTPStatus
import time
import random

load_dotenv()

# openai_key = os.getenv("OPENAI_KEY")
# if not openai_key:
#     print("Please set the OPENAI_KEY env var")
#     sys.exit(1)

# client = OpenAI(api_key="temp_key")


def calc_cost(logger, model_name, input_tokens, output_tokens) -> float:
    """
    Calculates the cost of a response from the openai API.

    Args:
        response (openai.ChatCompletion): The response from the API.

    Returns:
        float: The cost of the response.
    """
    cost = (
        globals.MODEL_COST_PER_INPUT[model_name] * input_tokens
        + globals.MODEL_COST_PER_OUTPUT[model_name] * output_tokens
    )
    log_and_cprint(
        logger,
        f"Model API request cost info: "
        f"input_tokens={input_tokens}, output_tokens={output_tokens}, cost={cost:.6f}",
        "yellow",
    )
    return cost


def extract_gpt_content(chat_completion_message: ChatCompletionMessage) -> str:
    """
    Given a chat completion message, extract the content from it.
    """
    content = chat_completion_message.content
    if content is None:
        return ""
    else:
        return content


def extract_gpt_func_calls(
    chat_completion_message: ChatCompletionMessage,
) -> List[FunctionCallIntent]:
    """
    Given a chat completion message, extract the function calls from it.
    Args:
        chat_completion_message (ChatCompletionMessage): The chat completion message.
    Returns:
        List[FunctionCallIntent]: A list of function calls.
    """
    result = []
    tool_calls = chat_completion_message.get('function_call')
    if tool_calls is None:
        return result

    call: ChatCompletionMessageToolCall
    for call in tool_calls:
        called_func: OpenaiFunction = call.function
        func_name = called_func.name
        func_args_str = called_func.arguments
        # maps from arg name to arg value
        if func_args_str == "":
            args_dict = {}
        else:
            try:
                args_dict = json.loads(func_args_str, strict=False)
            except json.decoder.JSONDecodeError:
                args_dict = {}
        func_call_intent = FunctionCallIntent(func_name, args_dict, called_func)
        result.append(func_call_intent)

    return result


def my_retry_error_callback(retry_state):
    """The function called after the retry ends and fails, returning a custom default value"""
    # return "retry error callback", None
    print("retry error callback, retry state: {}".format(retry_state))
    return None

# @retry(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(3))
@retry(
    wait=wait_random_exponential(min=5, max=30),
    stop=stop_after_attempt(5),
    retry_error_callback=my_retry_error_callback
)
def call_gpt(
    logger,
    messages,
    top_p=0.3,
    tools=None,
    response_format: Literal["text", "json_object"] = "text",
    **model_args,
) -> Tuple[
    str, list[ChatCompletionMessageToolCall], List[FunctionCallIntent], float, int, int
]:
    """
    Calls the openai API to generate completions for the given inputs.
    Assumption: we only retrieve one choice from the API response.

    Args:
        messages (List): A list of messages.
                         Each item is a dict (e.g. {"role": "user", "content": "Hello, world!"})
        top_p (float): The top_p to use. We usually do not vary this, so not setting it as a cmd-line argument. (from 0 to 1)
        tools (List, optional): A list of tools.
        **model_args (dict): A dictionary of model arguments.

    Returns:
        Raw response and parsed components.
        The raw response is to be sent back as part of the message history.
    """
    log_and_cprint(
        logger,
        f"model name: {globals.model}",
        "yellow",
    )
    
    if "SWE-GPT" in globals.model:
        try:
            if tools is not None and len(tools) == 1:
                response = None
            else:
                response = chatswegpt(globals.model, messages, globals.model_temperature, tools, response_format)
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            content = response.choices[0].message.content
      
            model_name = globals.model
            cost = calc_cost(logger, model_name, input_tokens, output_tokens)
            raw_tool_calls = []
            func_call_intents = []

            return (
                content,
                raw_tool_calls,
                func_call_intents,
                cost,
                input_tokens,
                output_tokens,
            )
        except BadRequestError as e:
            if e.code == "context_length_exceeded":
                log_and_print(logger, "Context length exceeded")
            else:
                log_and_print(logger, f"Error occurred with code: {e.code}")
            raise e
        except Exception as e:
            log_and_print(logger, f"An unexpected error occurred: {e}")


def merge_continuous_roles(messages):
    if not messages:
        return []

    merged_messages = []
    current_role = messages[0]['role']
    merged_content = messages[0]['content']

    for message in messages[1:]:
        if message['role'] == current_role:
            merged_content += "\n" + message['content']
        else:
            merged_messages.append({
                'role': current_role,
                'content': merged_content
            })
            current_role = message['role']
            merged_content = message['content']

    # 添加最后一个角色的内容
    merged_messages.append({
        'role': current_role,
        'content': merged_content
    })

    guard_merged_messages = []
    for message in merged_messages:
        if len(message['content']) == 0:
            print('[ERROR]: Sorry, please try again.')
        guard_merged_messages.append(message)
        
    return guard_merged_messages

    
@retry(
    wait=wait_random_exponential(min=5, max=10),
    stop=stop_after_attempt(3),
    retry_error_callback=my_retry_error_callback
)
def chatswegpt(model_name, messages, temperature, tools=None,
            response_format: Literal["text", "json_object"] = "text", max_tokens=4096, top_p=1):

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    print(f"[SWE-GPT]:\n{chat_response.choices[0].message.content}")
    
    return chat_response