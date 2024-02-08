# Geometry Dash Reinforcement Learning Agent

This project implements a Reinforcement Learning (RL) agent designed to play the game Geometry Dash. The agent uses an Actor-Critic Model (A2C) to determine the best actions to take within the game environment. Simple visual inputs are taken to determine whether the avatar is onscreen or offscreen. This is the basis for dead and alive, which drives the reward function.

## Project Structure

Here's a brief explanation of the key files and directories in this project:

- [`agent.py`](command:_github.copilot.openRelativePath?%5B%22agent.py%22%5D "agent.py"): This file contains the implementation of the RL agent.
- [`ai.py`](command:_github.copilot.openRelativePath?%5B%22ai.py%22%5D "ai.py"): This is the main script that runs the RL agent.
- [`assets/`](command:_github.copilot.openRelativePath?%5B%22assets%2F%22%5D "assets/"): This directory contains any necessary assets for the game or the RL agent.
- [`checkpoints/`](command:_github.copilot.openRelativePath?%5B%22checkpoints%2F%22%5D "checkpoints/"): This directory is where the trained models of the RL agent are saved. The current model is `geometry_dash_model.pt`.
- [`frames/`](command:_github.copilot.openRelativePath?%5B%22frames%2F%22%5D "frames/"): This directory is used to store frames from the game for debugging or visualization purposes.
- [`gdrlEnv/`](command:_github.copilot.openRelativePath?%5B%22gdrlEnv%2F%22%5D "gdrlEnv/"): This directory contains the implementation of the Geometry Dash environment that the RL agent interacts with.
- [`screen_capture.py`](command:_github.copilot.openRelativePath?%5B%22screen_capture.py%22%5D "screen_capture.py"): This script is used to capture screenshots from the game.
- [`requirements.txt`](command:_github.copilot.openRelativePath?%5B%22requirements.txt%22%5D "requirements.txt"): This file lists the Python dependencies required for this project.

## Setup

1. Clone this repository.
2. Install the necessary Python dependencies using pip:

```sh
pip install -r requirements.txt
```

## Running the RL Agent
to run the RL agent, simply execute the `ai.py` script:
```sh
python ai.py
```

## Contributing
Contributions to this project are welcome. Please feel free to open an issue. 
To contribute:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.