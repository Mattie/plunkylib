import pytest
from plunkylib import Prompt
import dataclasses

def test_replace_prompt():
    # Test dataclasses.replace() for Prompt objects
    original_prompt = Prompt(name="OriginalPrompt", text="This is the original prompt.")
    new_prompt = dataclasses.replace(original_prompt, name="NewPrompt")

    assert new_prompt.name == "NewPrompt", "Prompt name should be updated."
    assert new_prompt.text == original_prompt.text, "Prompt text should be unchanged."

def test_modify_prompt():
    # Test modifying Prompt attributes
    prompt = Prompt(name="TestPrompt", text="This is the test prompt.")

    prompt.text = "This is the modified test prompt."

    assert prompt.text == "This is the modified test prompt.", "Prompt text should be updated."

# you can turn test errors into warnings with a decorator



# eventually we should probably prevent these problems
# @pytest.mark.parametrize("name, text, expected_error", [
#     (1, "Test prompt text.", TypeError),
#     (None, "Test prompt text.", TypeError),
#     ("", "Test prompt text.", ValueError),
#     (pytest, None, TypeError),
# ])
# def test_invalid_prompt_creation(name, text, expected_error):
#     # Test creating a Prompt object with invalid values
#     with pytest.raises(expected_error):
#         Prompt(name=name, text=text)

# if __name__ == "__main__":
#     pytest.main(["-v", "-k", "test_prompt.py"])