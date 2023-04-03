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

def test_file_creation_and_modification():
    datafiles_settings.HIDDEN_TRACEBACK = False
    datafiles_settings.WRITE_DELAY = 0.1

    plunkylib_dir = PLUNKYLIB_BASE_DIR
    plunkylib_dir = os.path.abspath(plunkylib_dir)

    # 1. Create instances of the classes with some initial values.
    prompt = Prompt(name="TestPrompt", text="This is the test prompt.")
    #prompt.datafile.save()
    prompt_vars = PromptVars(name="TestPromptVars", vars={"var1": "value1", "var2": "value2"})
    #prompt_vars.datafile.save()
    chat_prompt = ChatPrompt(chatprompt_name="TestChatPrompt", messages=[{"user": "Hello"}, {"assistant": "Hi there!"}])
    #chat_prompt.datafile.save()
    completion_params = CompletionParams(name="TestCompletionParams")
    #completion_params.datafile.save()
    petition = Petition(name="TestPetition", params_name="TestCompletionParams", prompt_name="TestPrompt")
    #petition.datafile.save()

   
    # wait two seconds to be sure the files are created with different timestamps
    import time
    time.sleep(2)

    # test these in PLUNKYLIB_BASE_DIR
    assert os.path.exists(os.path.join(plunkylib_dir, "prompts/TestPrompt.txt")), "Prompt file should be created."
    assert os.path.exists(os.path.join(plunkylib_dir, "prompts/TestPrompt.txt")), "Prompt file should be created."
    assert os.path.exists(os.path.join(plunkylib_dir, "promptvars/TestPromptVars.yml")), "PromptVars file should be created."
    assert os.path.exists(os.path.join(plunkylib_dir, "prompts/TestChatPrompt.yml")), "ChatPrompt file should be created."
    assert os.path.exists(os.path.join(plunkylib_dir, "params/TestCompletionParams.yml")), "CompletionParams file should be created."
    assert os.path.exists(os.path.join(plunkylib_dir, "petition/TestPetition.yml")), "Petition file should be created."


def file_yeah2():
    prompt = Prompt.objects.get("TestPrompt")
    assert prompt.text == "This is the modified test prompt.", "Prompt text should be updated with the modified content."
    del prompt

def test_file_edit():
    datafiles_settings.HIDDEN_TRACEBACK = False

    plunkylib_dir = PLUNKYLIB_BASE_DIR
    plunkylib_dir = os.path.abspath(plunkylib_dir)

    prompt = Prompt(name="TestPrompt", text="This is the test prompt.")
    prompt.datafile.save()


    file_contents = None
    # 3. Read the content of the files and compare it with the initial values of the instances.
    with open(os.path.join(plunkylib_dir, "prompts", "TestPrompt.txt"), "r") as file:
        file_contents = file.read()
        assert "This is the test prompt." in file_contents, "Prompt file content should be correct."

    # close the file
    import time
    time.sleep(2)

    # Modify the content of the files on the disk.
    with open(os.path.join(plunkylib_dir, "prompts", "TestPrompt.txt"), "w") as file:
        # replace "This is the test prompt" with "This is the modified test prompt" but keep all other text the same
        file_contents = file_contents.replace("This is the test prompt.", "This is the modified test prompt.")
        file.write(file_contents)

    prompt = Prompt.objects.get("TestPrompt")
    assert prompt.text == "This is the modified test prompt.", "Prompt text should be updated with the modified content."