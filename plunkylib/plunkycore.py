# getting started with gpt3
import nest_asyncio
nest_asyncio.apply()
from datetime import datetime
from .aiwrapper import cleaned_completion_wrapper, content_classification, search_query
from dataclasses import dataclass, field
from typing import IO, ClassVar, List, Dict, Optional, Tuple
from datafiles import datafile, formats
from datafiles import Missing
from pydoc import locate
import asyncio
import json
import os
import random
import string
from loguru import logger


# if there's no "PLUNKYLIB_DIR" env variable, use that for our default path variable and set it to './datafiles/plunkylib'
# this is the default directory for all plunkylib datafiles
if os.getenv("PLUNKYLIB_BASE_DIR") is None:
    PLUNKYLIB_BASE_DIR = "./datafiles/plunkylib"
else:
    PLUNKYLIB_BASE_DIR = os.getenv("PLUNKYLIB_BASE_DIR")
    PLUNKYLIB_BASE_DIR = PLUNKYLIB_BASE_DIR.rstrip("/")

if os.getenv("PLUNKYLIB_LOGS_DIR") is None:
    PLUNKYLIB_LOGS_DIR = None   # no logging by default
else:
    PLUNKYLIB_LOGS_DIR = os.getenv("PLUNKYLIB_LOGS_DIR")
    PLUNKYLIB_LOGS_DIR = PLUNKYLIB_LOGS_DIR.rstrip("/")



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
        # use an UTF-8 encoding for the file
        file_object = open(file_object.name, 'r', encoding='utf-8')
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


@datafile(PLUNKYLIB_BASE_DIR + "/prompts/{self.name}.txt")
class Prompt:
    name: str
    text: str

@datafile(PLUNKYLIB_BASE_DIR + "/promptvars/{self.name}.yml")
class PromptVars:
    name: str
    # need a dictionary
    vars: Dict[str, str]

@datafile(PLUNKYLIB_BASE_DIR + "/prompts/{self.chatprompt_name}.yml")
class ChatPrompt:
    chatprompt_name: str
    messages: List[Dict[str, str]]

    def add_message(self, role: str, content: str):
        self.messages.append({role: content})
    def get_json(self) -> str:
        new_messages = []
        for message in self.messages:
            for role, content in message.items():
                new_messages.append({"role": role, "content": content})
        return json.dumps(new_messages)
    # async method that gathers will execute an async format method on every message in the chat prompt and gather the results into a final json string
    async def gather_format(self, format_coro, **kwargs) -> str:
        new_messages = []
        for message in self.messages:
            for role, content in message.items():
                new_messages.append({"role": role, "content": content})
        # we now apply the format_coro to the content of each message in each dictionary in the list
        coros = []
        for message in new_messages:
            async def format_key(message):
                logger.trace("formatting key: {role}", role=message['role'])
                message["role"] = await format_coro(message["role"], **kwargs)
                return
            async def format_message(message):
                logger.trace("formatting content: {content}", content=message['content'])
                message["content"] = await format_coro(message["content"], **kwargs)
                return
            coros.append(format_key(message))
            coros.append(format_message(message))
        # gather the results
        await asyncio.gather(*coros)
        logger.trace(new_messages)
        # return the json version of the expanded messages
        return json.dumps(new_messages)

    def format(self, **kwargs) -> str:
        formatted_messages = []
        # loop through all the messages and format each dictionary value
        for message in self.messages:
            for role, content in message.items():
                formatted_messages.append({"role": role, "content": content.format(**kwargs)})
        # return json version of the messages
        return json.dumps(formatted_messages)



@datafile(PLUNKYLIB_BASE_DIR + "/params/{self.name}.yml", defaults=True)
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
    logprobs: Optional[int] = None

@datafile(PLUNKYLIB_BASE_DIR + "/petition/{self.name}.yml")
class Petition:
    name: str
    params_name: str
    prompt_name: Optional[str]  = None
    chatprompt_name: Optional[str] = None
    promptvars_name: Optional[str] = None
    prompt: ClassVar[Prompt]
    chatprompt: ClassVar[ChatPrompt]
    params: ClassVar[CompletionParams]
    promptvars: ClassVar[PromptVars]
    # post init to initialize the prompt and params
    def load_all(self):
        # default to the prompt of the same name
        if (self.prompt_name is None or self.prompt_name == "") and self.chatprompt_name is None:
            self.prompt_name = self.name
        if self.chatprompt_name is not None:
            self.chatprompt = ChatPrompt.objects.get(self.chatprompt_name)
            self.prompt = None
        else:
            self.prompt = Prompt.objects.get(self.prompt_name)
            self.chatprompt = None
        self.params = CompletionParams.objects.get(self.params_name)
        if self.promptvars_name is not None:
            self.promptvars = PromptVars.objects.get(self.promptvars_name)
        else:
            self.promptvars = None

@datafile(PLUNKYLIB_BASE_DIR + "/completions/{self.name}.yml")
class Completion:
    name: str
    text: str
    petition_name: str
    parent_name: Optional[str] = None
    petition: ClassVar[Petition]
    post_prompt_text: Optional[str] = None
    rendered_prompt_text: Optional[str] = None
    prompt_shot: Optional[str] = None
    content_filter_rating: Optional[int] = None
    user_identifier: Optional[str] = None

    # post init to initialize the prompt and params
    def load_all(self):
       self.petition = Petition.objects.get(self.petition_name) if self.petition_name is not Missing and self.petition_name is not None else None
       self.parent = Completion.objects.get(self.parent_name) if self.parent_name is not Missing and self.parent_name is not None else None


@datafile(PLUNKYLIB_BASE_DIR + "/vectorsearchparams/{self.name}.yml", defaults=True)
class VectorSearchParams:
    name: str
    engine: str = "pinecone"
    index: str = "default"
    embedding: str = "ExampleGPT3_Embedding"
    top_k: int = 1
    include_metadata: bool = False
    include_values: bool = False
    result_format: str = "---\n{result}"
    filter: Optional[str] = None
    embedding_params: ClassVar[CompletionParams]

    def load_all(self):
        if self.embedding is not None:
            self.embedding_params = CompletionParams.objects.get(self.embedding)

@datafile(PLUNKYLIB_BASE_DIR + "/vectorsearch/{self.name}.yml")
class VectorSearch:
    name: str
    params_name: str
    prompt_name: Optional[str]  = None
    prompt: ClassVar[Prompt]
    params: ClassVar[VectorSearchParams]
    # post init to initialize the prompt and params
    def load_all(self):
        # default to the prompt of the same name
        if (self.prompt_name is None or self.prompt_name == ""):
            self.prompt_name = self.name
        self.prompt = Prompt.objects.get(self.prompt_name)
        self.params = VectorSearchParams.objects.get(self.params_name)
        self.params.load_all()
        

@datafile(PLUNKYLIB_BASE_DIR + "/namedlists/{self.list_name}.yml")
class NamedList:
    list_name: str
    items: List[str]

# support for a simple line completion with davinci
async def simple_completion(prompt) -> str:
    response = await cleaned_completion_wrapper(
        prompt,
        engine="davinci",
        max_tokens=45,
        temperature=TEMPERATURE.MIN,
        stop=["\n"],
    )
    return response

# support for a a line completion with normal temperature
async def normal_completion(prompt) -> str:
    response = await cleaned_completion_wrapper(
        prompt,
        engine="davinci",
        max_tokens=45,
        temperature=TEMPERATURE.NORMAL,
        stop=["\n"],
    )
    return response

async def petition_completion(petition: Petition) -> str:
    if petition.chatprompt_name is not None:
        # prompt is a chatprompt object now
        prompt = petition.chatprompt.get_json()
    else:
        # prompt is a string now
        prompt = petition.prompt.text    
    params = petition.params
    # call cleaned_completion_wrapper with the params
    response = await cleaned_completion_wrapper(
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
        logprobs=params.logprobs,
    )
    return response

class AsyncFormatter(string.Formatter):
    async def async_expand_field(self, field, args, kwargs):
        if "." in field:
            obj, method = field.split(".", 1)
            if obj in kwargs:
                obj_instance = kwargs[obj]
                if hasattr(obj_instance, method):
                    method_instance = getattr(obj_instance, method)
                    if asyncio.iscoroutinefunction(method_instance):
                        return await method_instance()
                    else:
                        return method_instance() if callable(method_instance) else method_instance
        value, _ = super().get_field(field, args, kwargs)
        return value

    async def async_format(self, format_string, *args, **kwargs):
        coros = []
        parsed_format = list(self.parse(format_string))

        for literal_text, field_name, format_spec, conversion in parsed_format:
            if field_name:
                coro = self.async_expand_field(field_name, args, kwargs)
                coros.append(coro)

        expanded_fields = await asyncio.gather(*coros)
        expanded_iter = iter(expanded_fields)

        return ''.join([
            literal_text + (str(next(expanded_iter)) if field_name else '')
            for literal_text, field_name, format_spec, conversion in parsed_format
        ])
aformatter = AsyncFormatter()

# accepts a petition name as a string and calls petition_completion2, returning only the completion text
async def petition_name_completion(petition_name: str, additional: Optional[dict] = None) -> str:
    petition = Petition.objects.get(petition_name)
    completion, prompt1 = await petition_completion2(petition, additional)
    return prompt1 + completion.text

# need a function that will return the dictionary needed to support prompt formatting
def text_expander(additional: Optional[dict] = None) -> dict:
    if additional is None:
        additional = {}    
    return {
        "pet": AsyncGetter(petition_name_completion, additional),
        "prompt": AsyncGetter(prompt_name_expansion, additional),
        "vsearch": AsyncGetter(vectorsearch_name_query, additional),
        "randomlist": RandomListGetter(),
        "prompt_break": "{prompt_break}",
        "prompt_shot": "{prompt_shot}",
        **additional,
    }


async def prompt_name_expansion(prompt_name: str, additional: Optional[dict] = None) -> str:
    prompt = Prompt.objects.get(prompt_name)
    result = await aformatter.async_format(prompt.text, **text_expander(additional))
    return result

class AsyncGetter:
    """Used for parallel variable expansion"""
    def __init__(self, src, addl = None):
        self.src = src
        self.addl = addl

    def __getitem__(self, k):
        async def completer_coro():
            x = await self.src(k, self.addl)
            logger.trace("Getter lookup for {k} produced:\n{x}", k=k, x=x)
            return x
        return completer_coro
    
    __getattr__ = __getitem__


class RandomListGetter:
    """Used for expanding a random list from a named list"""
    def __getitem__(self, k):
        return random.choice(NamedList(k, Missing).items)
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

# this returns a Completion object and the string prompt
async def petition_completion2(petition: Petition, additional: Optional[dict] = None, content_filter_check: bool = False, user: str = None) -> Tuple[Completion, str]:
    petition.load_all()
    params = petition.params

    
    promptvars = {}
    
    # replace the prompt text using the dict in promptvars
    if petition.promptvars is not None:
        promptvars.update(petition.promptvars.vars)
        promptvars['prompt_break'] = '{prompt_break}'
        promptvars['prompt_shot'] = '{prompt_shot}'

    if additional is not None:
        promptvars.update(additional)

    # remove the prompt_break symbols from the prompt by default
    if petition.chatprompt_name is not None:
        # prompt is a chatprompt object now
        prompt = await petition.chatprompt.gather_format(aformatter.async_format, **text_expander(promptvars))
    else:
        # prompt is a string now
        prompt = petition.prompt.text
        prompt = await aformatter.async_format(prompt, **text_expander(promptvars))


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
    # create logs directory if it doesn't exist
    if PLUNKYLIB_LOGS_DIR is not None:
        # create logs directory if it doesn't exist, recursively
        if not os.path.exists(PLUNKYLIB_LOGS_DIR):
            os.makedirs(PLUNKYLIB_LOGS_DIR)
        with open(PLUNKYLIB_LOGS_DIR + "/last_prompt.log", "w", encoding="utf8") as f:
            f.write(prompt)

    # call cleaned_completion_wrapper with the params
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
        logprobs=params.logprobs,
        user=user,
    )
    # get the current seconds since the epoch from datetime.time
    seconds = int(datetime.now().timestamp())    
    completion = Completion(petition.name + "-" + str(seconds), response, petition.name, None)
    completion.load_all()
    completion.post_prompt_text = prompt_break
    completion.rendered_prompt_text = prompt
    completion.prompt_shot = prompt_shot

    if user:
        completion.user_identifier = user

    # if the content_filter_check is 2, then we need to check if the completion is allowed
    if content_filter_check:
        try:
            level = await content_classification(prompt + response + prompt_break)
        except Exception as e:
            logger.error(f"Error: {e}")
            level = "-1"
        completion.content_filter_rating = int(level)
    

    # store the last prompt in a file called last_prompt.log
    if PLUNKYLIB_LOGS_DIR is not None:
        # create logs directory if it doesn't exist, recursively
        if not os.path.exists(PLUNKYLIB_LOGS_DIR):
            os.makedirs(PLUNKYLIB_LOGS_DIR)
        with open(PLUNKYLIB_LOGS_DIR + "/last_response.log", "w", encoding="utf8") as f:
            f.write(response + prompt_break)
            if user:
                f.write(f"\nUser: {user}")
            if content_filter_check:
                f.write(f"\nContent filter rating: {level}")


    return completion, prompt_break

async def vectorsearch_name_query(vsearch_name: str, additional: Optional[dict] = None) -> str:
    vsearch = VectorSearch.objects.get(vsearch_name)
    vsearch.load_all()
    return await vectorsearch_query(vsearch, additional)

async def vectorsearch_query(vsearch: VectorSearch, additional: Optional[dict] = None) -> str:
    # prompt is a string now
    prompt = vsearch.prompt.text
    # TODO find a cleaner way to do this
    if additional is None:
        additional = {}
    prompt = await aformatter.async_format(prompt, **text_expander(additional))    
    vparams = vsearch.params

    # we are going to take the params and the prompt (as the query string) and use them to call search_query()
    # async def search_query(query, index="default", engine="pinecone", embedding_model="text-embedding-ada-002", top_k=1, include_metadata=True, include_values=False, filter=None, result_format="---\n{result}\n"):

    response = await search_query(
        query=prompt,
        index=vparams.index,
        engine=vparams.engine,
        embedding_model=vparams.embedding_params.engine,
        top_k=vparams.top_k,
        include_metadata=vparams.include_metadata,
        include_values=vparams.include_values,
        filter=vparams.filter,
        result_format=vparams.result_format,
    )
    return response

logger.debug("plunkylib.plunkycore loaded")