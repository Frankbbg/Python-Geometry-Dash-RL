# Geometry Dash Reinforcement Learning Agent

This project implements a Reinforcement Learning (RL) agent designed to play the game Geometry Dash. The agent uses a Convolutional Neural Network (CNN) to process visual inputs and determine the best actions to take within the game environment.

## Project Structure

- `bot.py`: The main script containing the RL loop where the agent learns to interact with the game.
- `image_processing.py`: Contains image preprocessing functionality and screenshot capturing from the game.
- `agent.py`: Defines the DQN agent and related functionality, such as the model architecture and the replay buffer.
- `requirements.txt`: Lists all Python libraries needed to run the agent.

## Setup

To get started, clone this repository and install the required Python dependencies:

```bash
git clone https://github.com/your-username/geometry-dash-rl-agent.git
cd geometry-dash-rl-agent
pip install -r requirements.txt
```

## Running the Agent

To start training the agent, simply run:

```bash
python bot.py
```

The script will automatically begin capturing frames from the game, preprocess them, and feed them to the agent to make decisions and learn through reinforcement learning.

## Image Processing

The `image_processing.py` script preprocesses the game frames for the agent to consume. It handles grayscale conversion, resizing to lower dimensions, and normalization of pixel values.

## DQN Agent

The `agent.py` script defines the DQN agent, including the neural network model, action selection, and learning mechanisms. It uses experience replay and a target network based on the DQN algorithm to stabilize training and improve learning efficiency.

## Contribution

Contributions to this project are welcome. To contribute:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name - [@your_twitter](https://twitter.com/your_twitter) - email@example.com

Project Link: [https://github.com/your-username/geometry-dash-rl-agent](https://github.com/your-username/geometry-dash-rl-agent)
