import pytest
from plunkylib import Prompt, PromptVars, ChatPrompt, CompletionParams, Petition
from typing import Dict, List

def test_prompt_creation():
    prompt = Prompt(name="TestPrompt", text="This is the test prompt.")
    assert prompt.name == "TestPrompt", "Prompt name should be correct."
    assert prompt.text == "This is the test prompt.", "Prompt text should be correct."

def test_prompt_vars_creation():
    prompt_vars = PromptVars(name="TestPromptVars", vars={"var1": "value1", "var2": "value2"})
    assert prompt_vars.name == "TestPromptVars", "PromptVars name should be correct."
    assert prompt_vars.vars == {"var1": "value1", "var2": "value2"}, "PromptVars vars should be correct."

def test_chat_prompt_creation():
    chat_prompt = ChatPrompt(chatprompt_name="TestChatPrompt", messages=[{"user": "Hello"}, {"assistant": "Hi there!"}])
    assert chat_prompt.chatprompt_name == "TestChatPrompt", "ChatPrompt name should be correct."
    assert chat_prompt.messages == [{"user": "Hello"}, {"assistant": "Hi there!"}], "ChatPrompt messages should be correct."

def test_completion_params_creation():
    completion_params = CompletionParams(name="TestCompletionParams")
    assert completion_params.name == "TestCompletionParams", "CompletionParams name should be correct."

def test_petition_creation():
    petition = Petition(name="TestPetition", params_name="TestCompletionParams", prompt_name="TestPrompt")
    assert petition.name == "TestPetition", "Petition name should be correct."
    assert petition.params_name == "TestCompletionParams", "Petition params_name should be correct."
    assert petition.prompt_name == "TestPrompt", "Petition prompt_name should be correct."

def test_petition_load_all():
    petition = Petition(name="TestPetition", params_name="TestCompletionParams", prompt_name="TestPrompt")
    petition.load_all()

    assert isinstance(petition.prompt, Prompt), "Petition prompt should be an instance of Prompt."
    assert isinstance(petition.params, CompletionParams), "Petition params should be an instance of CompletionParams."
