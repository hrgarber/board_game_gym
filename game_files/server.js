const express = require('express');
const path = require('path');
const BoardGame = require('./gameBot');

const app = express();
const port = 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname)));

const games = {};

app.post('/game', (req, res) => {
  const { size } = req.body;
  const game = new BoardGame(size || 8);
  const gameId = `game-${Object.keys(games).length + 1}`;
  games[gameId] = game;
  res.status(201).json({ id: gameId, board: game.board });
});

app.post('/game/:id/move', (req, res) => {
  const { id } = req.params;
  const { row, col } = req.body;
  const game = games[id];
  if (!game) {
    return res.status(404).json({ error: 'Game not found' });
  }
  const moveResult = game.makeMove(row, col);
  if (moveResult) {
    res.json({ 
      board: game.board, 
      currentPlayer: game.currentPlayer, 
      gameOver: game.gameOver, 
      winner: game.winner 
    });
  } else {
    res.status(400).json({ error: 'Invalid move' });
  }
});

app.get('/game/:id', (req, res) => {
  const { id } = req.params;
  const game = games[id];
  if (!game) {
    return res.status(404).json({ error: 'Game not found' });
  }
  res.json({ 
    board: game.board, 
    currentPlayer: game.currentPlayer, 
    gameOver: game.gameOver, 
    winner: game.winner 
  });
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
