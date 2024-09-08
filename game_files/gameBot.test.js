const { BoardGame } = require('./gameBot');

describe('BoardGame', () => {
  let game;

  beforeEach(() => {
    game = new BoardGame(8);
  });

  test('initializes with correct board size', () => {
    expect(game.board.length).toBe(8);
    expect(game.board[0].length).toBe(8);
  });

  test('makes valid move', () => {
    expect(game.makeMove(0, 0)).toBe(true);
    expect(game.board[0][0]).toBe(1);
  });

  test('rejects invalid move', () => {
    game.makeMove(0, 0);
    expect(game.makeMove(0, 0)).toBe(false);
  });

  test('switches player after valid move', () => {
    game.makeMove(0, 0);
    expect(game.currentPlayer).toBe(-1);
  });

  test('detects horizontal win', () => {
    for (let i = 0; i < 5; i++) {
      game.makeMove(0, i);
      if (i < 4) game.makeMove(1, i);
    }
    expect(game.gameOver).toBe(true);
    expect(game.winner).toBe(1);
  });

  test('detects vertical win', () => {
    for (let i = 0; i < 5; i++) {
      game.makeMove(i, 0);
      if (i < 4) game.makeMove(i, 1);
    }
    expect(game.gameOver).toBe(true);
    expect(game.winner).toBe(1);
  });

  test('detects diagonal win', () => {
    for (let i = 0; i < 5; i++) {
      game.makeMove(i, i);
      if (i < 4) game.makeMove(i, i + 1);
    }
    expect(game.gameOver).toBe(true);
    expect(game.winner).toBe(1);
  });

  test('detects draw', () => {
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        game.makeMove(i, j);
      }
    }
    expect(game.gameOver).toBe(true);
    expect(game.winner).toBe(null);
  });

  test('prevents moves after game over', () => {
    for (let i = 0; i < 5; i++) {
      game.makeMove(0, i);
      if (i < 4) game.makeMove(1, i);
    }
    expect(game.makeMove(2, 0)).toBe(false);
  });

  test('resets game correctly', () => {
    game.makeMove(0, 0);
    game.resetGame();
    expect(game.board.every(row => row.every(cell => cell === 0))).toBe(true);
    expect(game.currentPlayer).toBe(1);
    expect(game.gameOver).toBe(false);
    expect(game.winner).toBe(null);
  });

  test('getGameStatus returns correct status', () => {
    expect(game.getGameStatus()).toBe('Current player: X');
    for (let i = 0; i < 5; i++) {
      game.makeMove(0, i);
      if (i < 4) game.makeMove(1, i);
    }
    expect(game.getGameStatus()).toBe('Player X wins!');
  });
});
