# plunkylib - Python LLM User Navigation Kit via YAML

## Your Buddy for OpenAI/GPT4/ChatGPT and more
`plunkylib` aims to be an experimenter's best friend when working with modern text generation engines. It stores prompts in reusable text files and supports nested prompt expansion.

There's a simple CLI interface, but usually after you get going, you'll do most of your query tweaking in text and yaml files that you can version-control separately from your app (or save/restore for research usage).


## Installation

(Coming soon)
```bash
pip install plunkylib
```

## Configure the API keys and other settings

Copy `.env.template` to `.env` and add your API keys there:
```
OPENAI_API_KEY = "ADD_YOUR_OPENAI_KEY"
AI21_API_KEY = "ADD_YOUR_AI21_API_KEY"
PINECONE_API_KEY = "ADD_YOUR_PINECONE_API_KEY"
PINECONE_PROJ_ENVIRONMENT = "ADD_YOUR_PINECONE_PROJECT_ENVIRONMENT"
```

## CLI Interface

For the most part, the expected workflow with `plunkylib` is to create some objects, modify the generated files on disk to do what you want, and then submit a named _petition_ object to the LLM engine of choice to get it to generate the text you want.

The CLI uses the `typer` library for its help/commands. So you can see the commands by running:

```bash
> python -m plunkylib --help
```
and get an output like:
```
Usage: cli.py [OPTIONS] COMMAND [ARGS]...
  --help                          Show this message and exit.

Commands:
  complete    (commands to complete texts from prompts)
  interact    (interactive petition usage)
  params      (records for settings for engine, temperature, etc)
  pet         (records for petititions, reusable combined prompt/params)
  prompt      (records for text used for completions)
  promptvars  (records for replacing text in prompts)
```

Each command shows more of its help by providing its name with no parameters.

You'll typically need a couple of things with easy setup before you begin, the `params` (i.e. _CompletionParams_), the `prompt`, and a `pet` (i.e. _petition_) to wrap those. Then you can use `complete` to generate a _Completion_.

### 1. Create _CompletionParams_ for your engine of choice
There's some samples of a few engines in `datafiles/plunkylib/params/Example*.yml`, let's use GPT3 to start:
```bash
> python -m plunkylib params copyfrom ExampleGPT3 MyFirstEngine
```
```
Copied CompletionParams ExampleGPT3 to MyFirstEngine:
CompletionParams(name='MyFirstEngine', engine='text-davinci-002', n=1, best_of=1, max_tokens=300, stop=['\n\n', '----\n'], temperature=0.3, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0)
```
This creates `datafiles/plunkylib/params/MyFirstEngine.yml` which you can edit to customize your parameters. 

### 2. Create a _Prompt_ to use
Let's use a simple sample first:
```bash
> python -m plunkylib prompt copyfrom ExampleSimple MyFirstPrompt 
```
```
Copied Prompt ExampleSimple to MyFirstPrompt:
Prompt(name='MyFirstPrompt', text='Describe the color of of the sky in flowery language.\n----\n')
```

### 3. Create a _Petition_ to use
Petitions just combine params and prompts into a reusable form since they often pair together. 
```bash
> python -m plunkylib pet create MyFirstPetition MyFirstPrompt MyFirstEngine
```
```
Created Petition MyFirstPetition, Petition(name='MyFirstPetition', prompt_name='MyFirstPrompt', params_name='MyFirstEngine', promptvars_name=None)
```

### 4. Get a completion output!
It'll pay off later for reusability and convenience, but finally we can now execute (and re-execute) the query we wanted to use.
```bash
> python -m plunkylib complete use-pet MyFirstPetition
```
Output:
```
The sky is a deep, rich blue, like the petals of a flower. The blue is so deep that it seems to go on forever. The sky is dotted with white clouds, like cotton balls, which add to the beauty of the sky.
```
You should now see the following files (in ./datafiles/plunkylib) directory, and they can be helpful for a file-based workflow.
* prompts/MyFirstPrompt.txt
* params/MyFirstParams.yml
* petition/MyFirstPetition.yml

As well as
* completions/MyFirstPetition-NNNN.yml

All of these are files you can edit it in your favorite text editor and then rerun the `complete use-pet` command on the petition you made. This makes it very easy to iterate on prompts and parameter combinations until you get what you want. Then you can reuse that again and again to generate the relevant text.

The completions output is a log of the output 

## Advanced Prompt Query Nesting
TODO: Describe the general `{prompt.name}` way to inject other text as well as `{pet.name}` to run another petition expansion recursively inside a prompt you're trying to complete. This makes it very easy to chain prompts together for complex cases.