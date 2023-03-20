# use typer wrapper around plunkylib methods
# to make it more like a command line interface
import nest_asyncio
nest_asyncio.apply()
import dataclasses
from datafiles import Missing
import typer
from .plunkycore import Prompt, CompletionParams, Petition, PromptVars, simple_completion, normal_completion, petition_completion, petition_completion2
from asyncio import run as aiorun
from typing import Optional



app = typer.Typer()
prompt_app = typer.Typer()
app.add_typer(prompt_app, name="prompt", short_help="(records for text used for completions)")
promptvars_app = typer.Typer()
app.add_typer(promptvars_app, name="promptvars", short_help="(records for replacing text in prompts)")
params_app = typer.Typer()
app.add_typer(params_app, name="params", short_help="(records for settings for engine, temperature, etc)")
petitions_app = typer.Typer()
app.add_typer(petitions_app, name="pet", short_help="(records for petititions, reusable combined prompt/params)")
complete_app = typer.Typer()
app.add_typer(complete_app, name="complete", short_help="(commands to complete texts from prompts)")
interact_app = typer.Typer()
app.add_typer(interact_app, name="interact", short_help="(interactive petition usage)")

@complete_app.command()
def simple(prompt: str):
    """
    Ex: cli.py complete simple "prompt"
    """
    async def completer_coro():
        typer.echo(await simple_completion(prompt))
    aiorun(completer_coro())

@complete_app.command()
def normal(prompt: str):
    """
    >python cli.py complete normal "prompt"
    """
    async def completer_coro():
        typer.echo(await normal_completion(prompt))
    aiorun(completer_coro())

@complete_app.command()
def use_pet(petition_name: str, extra_text: Optional[str] = typer.Argument(None)):
    """
    >python cli.py complete use_pet "petition_name"
    """

    petition = Petition.objects.get(petition_name)
    petition.load_all()
    async def completer_coro():
        if extra_text is None:
            completion, prompt_text = await petition_completion2(petition, {})
        else:
            supplemental_vars = {'arg1': extra_text}
            completion, prompt_text = await petition_completion2(petition, supplemental_vars)
        # output the completion name and its text from the same formatted echo
        header = "" #typer.style(f"Completion {completion.name}:\n\n", bold=True)

        initial = prompt_text
        # if initial doesn't end in a newline, add a space
        if not initial.endswith("\n"):
            initial += " "
        # use typer to print text out in a different color
        text = typer.style(f"{completion.text}\n", fg="yellow")
        # echo
        typer.echo(header + initial + text)
    aiorun(completer_coro())


@complete_app.command()
def new_pet(prompt_name: str, params_name: str):
    """
    Example of this command line:
    >python cli.py complete new_pet "prompt_name" "params_name"
    """

    petition = Petition(f"{prompt_name}-{params_name}", prompt_name, params_name)
    petition.load_all()
    async def completer_coro():
        typer.echo(await petition_completion(petition))
    aiorun(completer_coro())

# use typer to make a command "prompt" (that has an "update" subcommand)
# and then use the "update" subcommand to update the prompt
@prompt_app.command()
def update(name: str, text: str):
    """
    Example of this command line:
    >python cli.py prompt update "prompt" "text"
    """
    prompt = Prompt(name, Missing)
    origtext = prompt.text
    prompt.text = text
    typer.echo(f"Prompt {name} updated from \n\t{origtext}\nto\n\t{text}")

# do the same thing for params
@prompt_app.command()
def create(name: str):
    """
    Example of this command line:
    >python cli.py prompt create "name"
    """
    prompt = Prompt(name, "Enter Your Text Here\nEnter Your Text Here")
    typer.echo(f"Prompt {name} created\n{prompt.text}")

@prompt_app.command()
def print(name: str):
    """
    Example of this command line:
    >python cli.py prompt print "name"
    """
    prompt = Prompt(name, Missing)
    typer.echo(f"Prompt {name} is:\n{prompt.text}")

@prompt_app.command()
def copyfrom(source_name: str, dest_name: str):
    obj = dataclasses.replace(Prompt(source_name, Missing), name=dest_name)
    typer.echo(f"Copied Prompt {source_name} to {dest_name}:\n{obj}")

@promptvars_app.command(short_help="(create a new promptvars)")
def create(name: str):
    promptvars = PromptVars(name, Missing)
    typer.echo(f"PromptVars {promptvars.name} created.")

@promptvars_app.command(short_help="(copy a promptvars from one name to another)")
def copyfrom(source_name: str, dest_name: str):
    """
    Example of this command line:
    >python cli.py promptvars copyfrom "source_name" "dest_name"
    """
    obj = dataclasses.replace(PromptVars(source_name, Missing), name=dest_name)
    typer.echo(f"Copied PromptVars {source_name} to {dest_name}:\n{obj}")

@promptvars_app.command(short_help="(update a single value)")
def update(name: str, val_name: str, val: str):
    """
    Example of this command line:
    >python cli.py promptvars update "name" "val_name" "val"
    """
    promptvars = PromptVars(name, Missing)
    d = promptvars.vars if promptvars.vars is not None else {}
    d[val_name] = val
    promptvars.vars = d
    typer.echo(f"PromptVars {name} updated:\n{promptvars}")  

# print out the promptvars
@promptvars_app.command(short_help="(print out the promptvars)")
def print(name: str):
    promptvars = PromptVars(name, Missing)
    typer.echo(f"PromptVars {name} is:\n{promptvars.vars}")

# do the same thing for params
@params_app.command()
def create(name: str):
    params = CompletionParams(name)
    typer.echo(f"Created CompletionParams {params.name}")

# print a params
@params_app.command()
def print(name: str):
    params = CompletionParams(name)
    # print all the attributes of params
    typer.echo(f"Printing CompletionParams {name}")
    typer.echo(params)

@params_app.command()
def copyfrom(source_name: str, dest_name: str):
    params = dataclasses.replace(CompletionParams(source_name), name=dest_name)
    typer.echo(f"Copied CompletionParams {source_name} to {dest_name}:\n{params}")

@params_app.command()
def change(name: str, value_name: str, value: str):
    params = CompletionParams(name)
    # convert value to the right type    
    
# do the same thing for petitions
@petitions_app.command()
def create(name: str, prompt_name: str, param_name: str, promptvars_name: Optional[str] = typer.Argument(None)):
    """
    Example of this command line:
    >python cli.py petitions create "name" "prompt_name" "param_name" ["promptvars_name"]
    """
    petition = Petition(name, prompt_name, param_name, promptvars_name)
    typer.echo(f"Created Petition {name}, {petition}")

@petitions_app.command()
def print(name: str):
    petition = Petition.get(name)
    petition.load_all()
    typer.echo(f"Printing Petition {name}, {petition}, {petition.prompt.text}")

@petitions_app.command()
def copyfrom(source_name: str, dest_name: str):
    petition = Petition.get(source_name)
    petition.load_all()
    pet2 = dataclasses.replace(petition, name=dest_name)
    typer.echo(f"Copied Petition {source_name} to {dest_name}:\n{pet2}")

# def interactive_main_menu():
#     # uses a questionary menu to get the user's input
#     # requires import questionary

# @interact_app.command()
# def start(name: str):
#     """
#     This command starts an interactive mode for a prompt with multiple completions.

#     Example of this command line:
#     >python cli.py interact start "name"
#     """
#     petition = Petition(name, Missing, Missing)
#     petition.load_all()
#     typer.echo(f"Starting Interaction {name}, {petition}")
#     # start the interactive mode
