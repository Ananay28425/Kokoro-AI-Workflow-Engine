# Starts the Jarvis command-line app by connecting this file to cli.commands.
from cli.commands import app


if __name__ == "__main__":
    # Hand control to the Typer app defined in cli.commands.
    app()
