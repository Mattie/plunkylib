# getting started with gpt3
import nest_asyncio
nest_asyncio.apply()
from datetime import datetime
from chronological import cleaned_completion
from dataclasses import dataclass, field
from typing import IO, ClassVar, List, Dict, Optional, Tuple
from asyncio import run as aiorun
from datafiles import datafile, formats
from datafiles import Missing
from pydoc import locate
import questionary
import random

# Chronological helps us talk to OpenAI's GPT-3 model with a little extra text cleaning

# constants for no (0.0), low (0.4), normal (0.7), high (0.9), and max (1.0) temperature:
@dataclass
class TEMPERATURE:
    """Common choices for GPT-3 model temperature, controls determinism of responses"""    
    MIN: ClassVar[float] = 0.0
    LOW: ClassVar[float] = 0.4
    NORMAL: ClassVar[float] = 0.7
    HIGH: ClassVar[float] = 0.9
    MAX: ClassVar[float] = 1.0


# this format class only works with strings
class TxtFormat(formats.Formatter):
    """Special formatter to use with strings and .txt datafiles for a convenient raw text format for easy document editing on disk"""
    TYPEFORMAT = "{name}|{type}"
    DIVIDER = "#-=-=-=-=-DO-NOT-EDIT-THIS-LINE-PLEASE-=-=-=-=-#"
    @classmethod
    def extensions(cls) -> List[str]:
        return ['.txt']

    @classmethod
    def serialize(cls, data: Dict) -> str:
        # support types:
        _supported_types = [int, float, str, bool]
        # Convert `data` to a string
        output = ""
        for k, v in data.items():
            if type(v) in _supported_types:
                output = output + cls.TYPEFORMAT.format(name=k, type=v.__class__.__name__) + "\n"
                # add the divider then value
                output = output + str(v) + "\n" + cls.DIVIDER + "\n"
            else:
                raise ValueError("Unsupported type: {}".format(type(v)))
        return output

    @classmethod
    def deserialize(cls, file_object: IO) -> Dict:
        # reverse the serialize method
        output = {}
        # read each line of the file_object
        lines = file_object.readlines()
        # loop through the range of lines
        current_key = ''
        current_value = ''
        current_type = None
        current_object = None
        for i in range(len(lines)):
            current_line = lines[i].strip('\n')
            if current_line == cls.DIVIDER:
                # if we have a current object and type, then we can add it to the output
                if current_object is not None and current_type is not None:
                    # apply the type to the current object and save in output
                    output[current_key] = current_type(current_object)
                else:
                    # raise a misformat error
                    raise ValueError("Misformatted file: {}".format(lines[i]))
                # reset the current object and type
                current_object = None
                current_type = None
                # reset the current key
                current_key = ''
            else:
                # line is not the divider, so we need to see if we're starting a new object or finishing an old one
                if current_key == '':
                    # this line needs to be a key and type
                    # get the key and type
                    current_key, current_type = current_line.split("|")
                    # convert the type's string representation to the python type
                    current_type = locate(current_type.strip())
                    current_key = current_key.strip()
                    # move to the next line
                    continue
                else:
                    # this line is part of the current object
                    # if we're not starting a new object, then add the current line to the current object
                    if current_object is None:
                        current_object = current_line
                    else:
                        current_object = current_object + "\n" + current_line
                    continue
        return output
formats.register('.txt', TxtFormat)

########################################################################################################################
# Core datafile classes of Plunkylib
# (1) Prompt, this is the prompt text we save to disk
# (2) PromptVars, optional and somewhat obsolete class for storing variables to be replaces into a prompt when it's 
#     being prepared for use. They're useful if you have values you want to externalize from the main prompt text
# (3) CompletionParams, parameters like engine name and other parameters for the cleaned_completion function. This allows
#     us to externalize/reuse these variables for different prompt types.
# (4) Petition, high-level class that symbolizes a request (petition) to the oracle (GPT-3). It ties params to a prompt
#     as a named entity. Pretty much what plunkylib cares most about
# (5) Completion, stores the results from a cleaned_completion call, more of a log of what happened
# (6) NamedList, a list of named items, used for storing things like random lists you might want to expand inside a prompt


@datafile("prompts/{self.name}.txt")
class Prompt:
    name: str
    text: str

@datafile("promptvars/{self.name}.yml")
class PromptVars:
    name: str
    # need a dictionary
    vars: Dict[str, str]

@datafile("params/{self.name}.yml", defaults=True)
class CompletionParams:
    name: str
    engine: str = "text-davinci-002"
    n: int = 1
    best_of: int = 1
    max_tokens: int = 32
    stop: List[str] = field(default_factory=lambda: ['\n'])
    temperature: float = TEMPERATURE.MIN
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

@datafile("petition/{self.name}.yml")
class Petition:
    name: str
    prompt_name: str
    params_name: str
    promptvars_name: Optional[str] = None
    prompt: ClassVar[Prompt]
    params: ClassVar[CompletionParams]
    promptvars: ClassVar[PromptVars]
    # post init to initialize the prompt and params
    def load_all(self):
        # default to the prompt of the same name
        if self.prompt_name == "":
            self.prompt_name = self.name
        self.prompt = Prompt(self.prompt_name, Missing)
        self.params = CompletionParams(self.params_name)
        if self.promptvars_name is not None:
            self.promptvars = PromptVars(self.promptvars_name, Missing)
        else:
            self.promptvars = None

@datafile("completions/{self.name}.yml")
class Completion:
    name: str
    text: str
    petition_name: str
    parent_name: Optional[str] = None
    petition: ClassVar[Petition]
    post_prompt_text: Optional[str] = None
    rendered_prompt_text: Optional[str] = None
    prompt_shot: Optional[str] = None

    # post init to initialize the prompt and params
    def load_all(self):
       self.petition = Petition(self.petition_name, Missing, Missing) if self.petition_name is not Missing else None
       self.parent = Completion(self.parent_name, Missing, Missing) if self.parent_name is not Missing else None

@datafile("namedlists/{self.list_name}.yml")
class NamedList:
    list_name: str
    items: List[str]

# support for a simple line completion with davinci
async def simple_completion(prompt) -> str:
    response = await cleaned_completion(
        prompt,
        engine="davinci",
        max_tokens=45,
        temperature=TEMPERATURE.MIN,
        stop=["\n"],
    )
    return response

# support for a a line completion with normal temperature
async def normal_completion(prompt) -> str:
    response = await cleaned_completion(
        prompt,
        engine="davinci",
        max_tokens=45,
        temperature=TEMPERATURE.NORMAL,
        stop=["\n"],
    )
    return response

async def petition_completion(petition: Petition) -> str:
    prompt = petition.prompt.text
    params = petition.params
    # call cleaned_completion with the params
    response = await cleaned_completion(
        prompt,
        engine=params.engine,
                n=params.n,
        best_of=params.best_of,
        max_tokens=params.max_tokens,
        stop=params.stop,
        temperature=params.temperature,
        top_p=params.top_p,
        frequency_penalty=params.frequency_penalty,
        presence_penalty=params.presence_penalty,
    )
    return response

# accepts a petition name as a string and calls petition_completion2, returning only the completion text
async def petition_name_completion(petition_name: str, additional: Optional[dict] = None) -> str:
    petition = Petition(petition_name, Missing, Missing)
    completion, prompt1 = await petition_completion2(petition, additional)
    return prompt1 + completion.text

class PromptGetter:
    """Used for variable expansion of prompt text inside other prompts"""
    def __getitem__(self, k):
        prompt = Prompt(k, Missing)
        print(f"{k} produced:\n{prompt.text}")
        return prompt.text
    __getattr__ = __getitem__

# pretend dictionary that maps petition names to execute them dynamically for recursive calls
class PetGetter:
    """Used for performing a nested query of another petition when completing a prompt/petition"""
    def __init__(self, src, addl = None):
        self.src = src
        self.addl = addl

    def __getitem__(self, k):
        async def completer_coro():
            return await self.src(k, self.addl)
        x = aiorun(completer_coro())
        # print x and tell them what we're doing
        print(f"{k} produced:\n{x}")
        return x
    
    __getattr__ = __getitem__

class RandomListGetter:
    """Used for expanding a random list from a named list"""
    def __getitem__(self, k):
        return random.choice(NamedList(k, Missing).items)
    __getattr__ = __getitem__

class QuestionGetter:
    """Used for interactive mode to prompt the user for values when completing a prompt/petition"""
    def __init__(self, src, func):
        self.src = src
        self.func = func

    def __getitem__(self, k):
        tosearch = self.src
        context = ''
        eol = False
        # ask them if they want to complete the petition or type it themselves
        # using questionary.select:
        choice = questionary.select(f"{k}: Do you want to complete the petition {k} with GPT-3?",
            choices=["Yes, autogenerate", "No, I'll type it myself"]).ask()
        if choice == "Yes, autogenerate":
            async def completer_coro():
                return await self.func(k)
            x = aiorun(completer_coro())
            return x
        else:
            # loop through every line looking for k inside of it:
            for line in tosearch.split("\n"):
                if "." + k in line:
                    context = line
                    if line.endswith(k+"}") and not tosearch.endswith(k+"}"):
                        eol = True
                    break

            # ask them if what they want to use to replace "k"?
            x = questionary.text(f"\nFor this line:\n   {context}\nWhat do you want to use to replace 'question.{k}'?").ask()
            
            if eol and len(x) > 1:
                x = x + "\n"
            # print x and tell them what we're doing
            print(f"{k} produced:\n{x}---")
            return x
    
    __getattr__ = __getitem__



def _trimmed_fetch_response(resp, n):
    # resp.json()['completions'][0]['data']['text']
    if n == 1:
        return resp.json()['completions'][0]['data']['text'].strip()
    else:
        texts = []
        for idx in range(0, n):
            texts.append(resp.json()['completions'][idx]['data']['text'].strip())
        return texts

# attempt at making an ai21 jurassic query that mimics the chronological/gpt3 query
async def jurassic_cleaned_completion(prompt, engine="j1-grande", max_tokens=64, temperature=0.7, top_p=1, stop=None, presence_penalty=0, frequency_penalty=0, echo=False, n=1, stream=False, logprobs=None, best_of=1, logit_bias={}, count_penalty=0):
    import requests
    import os
    apikey = os.getenv("AI21_API_KEY")
    resp = requests.post("https://api.ai21.com/studio/v1/" + engine + "/complete",
    headers={"Authorization": "Bearer " + apikey},
    json={
        "prompt": prompt,
        "numResults": n,
        "maxTokens": max_tokens,
        "temperature": temperature,
        "topKReturn": 0,
        "topP":top_p,
        "stopSequences":stop,
        "countPenalty": {
                "scale": count_penalty,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": True
            },
            "frequencyPenalty": {
                "scale": frequency_penalty,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": True
            },
            "presencePenalty": {
                "scale": presence_penalty,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": True
            },
        })
    if resp.status_code != 200:
        return False
    return _trimmed_fetch_response(resp, n)

# method that takes any number of keyword arguments to wrap cleaned_completion, but don't list the named arguments
async def cleaned_completion_wrapper(*args, **kwargs):
    # if the keyword argument "engine" was included
    if "engine" in kwargs:
        # if engine begins with "j1" use the jurassic_cleaned_completion method
        if kwargs["engine"].startswith("j1"):
            return await jurassic_cleaned_completion(*args, **kwargs)
        # otherwise use the cleaned_completion method
        else:
            return await cleaned_completion(*args, **kwargs)


# this returns a Completion object and the string prompt
async def petition_completion2(petition: Petition, additional: Optional[dict] = None) -> Tuple[Completion, str]:
    petition.load_all()
    # remove the prompt_break symbols from the prompt by default
    prompt = petition.prompt.text
    params = petition.params
    
    # print out the prompt name
    #print(f"Prompt name {petition.prompt_name} for petition {petition.name}")

    # replace the prompt text using the dict in promptvars
    if petition.promptvars is not None:
        promptvars = petition.promptvars.vars
        promptvars['prompt_break'] = '{prompt_break}'
        promptvars['prompt_shot'] = '{prompt_shot}'
        prompt = prompt.format(prompt=PromptGetter(), pet=PetGetter(petition_name_completion, additional), randomlist=RandomListGetter(), **promptvars)
    else:
        if additional is None:
            additional = {}
        prompt = prompt.format(pet=PetGetter(petition_name_completion, additional),
                               question=QuestionGetter(prompt, petition_name_completion), 
                               prompt=PromptGetter(),
                               randomlist=RandomListGetter(),
                               prompt_break="{prompt_break}",
                               prompt_shot="{prompt_shot}",
                               **additional)

    # we use "{prompt_break}" automatically as a marker and we need to process it out
    # first lets get all text after "{prompt_break}:
    prompt_split = prompt.split("{prompt_break}")
    prompt_break = prompt_split[-1].strip(" ") if len(prompt_split) > 1 else ""

    # we use "{prompt_shot}" automatically as a marker and we need to process it out
    # first lets get all text after "{prompt_shot}:
    promptshot_split = prompt.split("{prompt_shot}")
    prompt_shot = promptshot_split[-1].strip(" ") if len(promptshot_split) > 1 else prompt
    # remove "{prompt_break}" from the prompt
    prompt_shot = prompt_shot.replace("{prompt_break}", "")

    # remove "{prompt_shot}" from the prompt as well as {prompt_break}
    prompt = prompt.replace("{prompt_shot}", "")
    prompt = prompt.replace("{prompt_break}", "")
    # store the last prompt in a file called last_prompt.log
    with open("last_prompt.log", "w") as f:
        f.write(prompt)

    # call cleaned_completion with the params
    response = await cleaned_completion_wrapper(
        prompt=prompt,
        engine=params.engine,
                n=params.n,
        best_of=params.best_of,
        max_tokens=params.max_tokens,
        stop=params.stop,
        temperature=params.temperature,
        top_p=params.top_p,
        frequency_penalty=params.frequency_penalty,
        presence_penalty=params.presence_penalty,
    )
    # get the current seconds since the epoch from datetime.time
    seconds = int(datetime.now().timestamp())    
    completion = Completion(petition.name + "-" + str(seconds), response, petition.name, None)
    completion.load_all()
    completion.post_prompt_text = prompt_break
    completion.rendered_prompt_text = prompt
    completion.prompt_shot = prompt_shot
    # store the last prompt in a file called last_prompt.log
    with open("last_response.log", "w") as f:
        f.write(response + prompt_break)

    return completion, prompt_break

