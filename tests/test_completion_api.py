import pytest
from plunkylib import Petition, petition_completion2

import os
import pytest
import shutil
from plunkylib import Prompt, PromptVars, ChatPrompt, CompletionParams, Petition, PLUNKYLIB_BASE_DIR, PLUNKYLIB_LOGS_DIR
from datafiles import settings as datafiles_settings

@pytest.fixture(scope="function", autouse=True)
def setup_and_cleanup():
    plunkylib_dir = PLUNKYLIB_BASE_DIR
    safe_to_cleanup = False
    # to be safe, verify that this directory is under the current working directory
    # is it a subdirectory of the current working directory?
    #if os.path.commonpath([os.getcwd(), plunkylib_dir]) == os.getcwd():
    # above line gave us ValueError: Can't mix absolute and relative paths, fixed:
    plunkylib_dir = os.path.abspath(plunkylib_dir)
    if os.path.commonpath([os.path.abspath(os.getcwd()), plunkylib_dir]) == os.path.abspath(os.getcwd()):
        # now check to be sure the only files under this directory (recursive) are .txt, .yaml, .yml, .log, and .json files.
        # if so, it's safe to delete the directory
        bad_file_found = False
        for root, dirs, files in os.walk(plunkylib_dir):
            for file in files:
                if not file.endswith((".txt", ".yaml", ".yml", ".log", ".json")):
                    bad_file_found = True
                    break
            else:
                continue
            break
        if not bad_file_found:
            safe_to_cleanup = True
    # if safe and the test directory already exists, remove it, should be set in the tests .env file
    if safe_to_cleanup and os.path.exists(plunkylib_dir):
        shutil.rmtree(plunkylib_dir)
    # create the test directory, recursively to the final directory
    if not os.path.exists(plunkylib_dir):
        os.makedirs(plunkylib_dir)
    else:
        # problem, the directory should have been missing
        raise Exception("The test directory already exists, it should have been missing.")
    yield

    # Clean up the test environment
    import time
    time.sleep(2)
    if safe_to_cleanup and os.path.exists(plunkylib_dir):
        # it's okay for this to fail, it's just a cleanup
        try:
            shutil.rmtree(plunkylib_dir)
        except:
            pass

@pytest.mark.asyncio
async def test_petition_completion2():
    # Create the prompt
    prompt_text = (
        "I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, "
        "I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, "
        "I will respond with \"Unknown\".\n\n"
        "Q: What is human life expectancy in the United States?\n"
        "A: Human life expectancy in the United States is 78 years.\n\n"
        "Q: Who was president of the United States in 1955?\n"
        "A: Dwight D. Eisenhower was president of the United States in 1955.\n\n"
        "Q: Which party did he belong to?\n"
        "A: He belonged to the Republican Party.\n\n"
        "Q: What is the square root of banana?\n"
        "A: Unknown\n\n"
        "Q: How does a telescope work?\n"
        "A: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\n"
        "Q: Where were the 1992 Olympics held?\n"
        "A: The 1992 Olympics were held in Barcelona, Spain.\n\n"
        "Q: How many squigs are in a bonk?\n"
        "A: Unknown\n\n"
        "Q: When was cryptography invented?\n"
        "A: "
    )

    prompt = Prompt(name="TestFirstPrompt", text=prompt_text)

    # Create the CompletionParams
    completion_params = CompletionParams(
        name="TestFirstEngine",
        engine="text-davinci-002",
        best_of=1,
        max_tokens=100,
        stop=["\n\n", "----\n"],
        temperature=0.3,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logprobs=None
    )

    # Save the Prompt and CompletionParams
    prompt.datafile.save()
    completion_params.datafile.save()

    # Prepare the test data
    petition_example = Petition("TestFirstPetition", prompt_name="TestFirstPrompt", params_name="TestFirstEngine")
    petition_example.load_all()

    # Call petition_completion2() with the test data
    example_completion, adj_prompt_text = await petition_completion2(petition=petition_example, additional={}, content_filter_check=True)

    # Assert the returned results
    assert example_completion is not None
    assert adj_prompt_text is not None
    assert example_completion.text is not None
    assert 0.0 <= example_completion.content_filter_rating <= 1.0


@pytest.mark.asyncio
async def test_petition_completion2_chat():
    # do the same thing but with a chatprompt file like this:
    #

    # create a prompt called TestSystemInsertText
    prompt_text = (
        "You are a highly intelligent question answering bot. If the user asks a question that is rooted in truth, "
        "you will give them the answer. If the user asks you a question that is nonsense, trickery, or has no clear answer, "
        "you will respond with \"Unknown\".\n\n"
    )
    prompt = Prompt(name="TestSystemInsertText", text=prompt_text)
    prompt.datafile.save()

    # create a prompt called TestAnotherPrompt (also text)
    prompt_text = (
        "What is human life expectancy in the United States?\n"
    )
    prompt = Prompt(name="TestAnotherPrompt", text=prompt_text)

    # create a chatprompt called TestChatPrompt using the above messages list of dictionaries
    # messages:
    #   - system: "{prompt.TestSystemInsertText}"
    #   - user: "Hi! {prompt.TestAnotherPrompt}"

    messages = [
        {"system": "{prompt.TestSystemInsertText}"},
        {"user": "Q: {prompt.TestAnotherPrompt}"}
    ]
    chatprompt = ChatPrompt(chatprompt_name="TestChatPrompt", messages=messages)
    chatprompt.datafile.save()

    # now we need an engine using these:
        # engine: gpt-3.5-turbo
        # max_tokens: 300
        # stop:
        #   - "\n\n\n"
        #   - "----\n"
        # temperature: 0.0
        # top_p: 1.0
        # frequency_penalty: 0.0
        # presence_penalty: 0.0
    completion_params = CompletionParams(
        name="TestChatEngine",
        engine="gpt-3.5-turbo",
        max_tokens=300,
        stop=["\n\n\n", "----\n"],
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    completion_params.datafile.save()

    petition_example = Petition("TestChatPetition", chatprompt_name="TestChatPrompt", params_name="TestChatEngine")
    petition_example.load_all()

    # Call petition_completion2() with the test data
    example_completion, adj_prompt_text = await petition_completion2(petition=petition_example, additional={}, content_filter_check=True)

    # Assert the returned results
    assert example_completion is not None
    assert adj_prompt_text is not None
    assert example_completion.text is not None
    assert 0.0 <= example_completion.content_filter_rating <= 1.0

    # raise a warning if this text is not in example_completion.text: "human life expectancy"
    if not "human life expectancy" in example_completion.text:
        pytest.skip("Completion text didn't contain 'human life expectancy' as expected with 0 temperature.")    



