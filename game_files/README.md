# Board Game Project

This directory contains the files for our Board Game project, a web-based implementation of a strategic board game.

## Table of Contents

- [Files](#files)
- [Setup](#setup)
- [Running the Game](#running-the-game)
- [Running Tests](#running-tests)
- [Development](#development)
- [Future Improvements](#future-improvements)

## Files

- `gameBot.js`: Contains the main game logic for the Board Game.
- `gameBot.test.js`: Contains the test suite for the Board Game logic.
- `styles.css`: Contains the styles for the game's user interface.
- `index.html`: The main HTML file that renders the game in a web browser.
- `game_bot.py`: A Python version of the game bot (not currently in use).
- `script.js`: Contains the JavaScript code for handling user interactions and updating the UI.

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <project-directory>
   ```

2. Install dependencies:
   ```
   npm install
   ```

## Running the Game

To run the game, open the `index.html` file in a web browser. You can do this by double-clicking the file or using a local server.

If you have Python installed, you can start a simple HTTP server:

```
python -m http.server
```

Then open `http://localhost:8000` in your web browser.

## Running Tests

To run the tests, use the following command in the terminal:

```
npx jest
```

Make sure you have Jest installed as a dev dependency in your project. If not, you can install it with:

```
npm install --save-dev jest
```

## Development

When making changes to the game logic:

1. Update the tests in `gameBot.test.js` accordingly.
2. Run the tests before committing changes to ensure the game logic is working as expected.
3. Update this README if you add new features or change the project structure.

## Future Improvements

- Implement AI player using the `game_bot.py` file.
- Add more advanced game features and rules.
- Improve the user interface and styling.
- Add multiplayer support.
- Implement difficulty levels for AI.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
