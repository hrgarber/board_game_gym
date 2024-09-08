const BoardGame = require('./gameBot');

describe('BoardGame', () => {
  let game;

  beforeEach(() => {
    game = new BoardGame();
  });

  test('should initialize the board correctly', () => {
    expect(game.board.length).toBe(8);
    expect(game.board[0].length).toBe(8);
    expect(game.board.every(row => row.every(cell => cell === 0))).toBe(true);
  });

  test('should make a valid move', () => {
    expect(game.makeMove(0, 0)).toBe(true);
    expect(game.board[0][0]).toBe(1);
  });

  test('should not allow a move on an occupied cell', () => {
    game.makeMove(0, 0);
    expect(game.makeMove(0, 0)).toBe(false);
  });

  test('should switch players after a move', () => {
    game.makeMove(0, 0);
    expect(game.currentPlayer).toBe(-1);
  });

  test('should detect a horizontal win', () => {
    for (let i = 0; i < 5; i++) {
      game.makeMove(0, i);
      if (i < 4) game.makeMove(1, i);
    }
    expect(game.gameOver).toBe(true);
    expect(game.winner).toBe(1);
  });

  test('should detect a vertical win', () => {
    for (let i = 0; i < 5; i++) {
      game.makeMove(i, 0);
      if (i < 4) game.makeMove(i, 1);
    }
    expect(game.gameOver).toBe(true);
    expect(game.winner).toBe(1);
  });

  test('should detect a diagonal win', () => {
    for (let i = 0; i < 5; i++) {
      game.makeMove(i, i);
      if (i < 4) game.makeMove(i, i + 1);
    }
    expect(game.gameOver).toBe(true);
    expect(game.winner).toBe(1);
  });

  test('should detect a draw', () => {
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        game.makeMove(i, j);
      }
    }
    expect(game.gameOver).toBe(true);
    expect(game.winner).toBe(null);
  });

  test('should reset the game', () => {
    game.makeMove(0, 0);
    game.resetGame();
    expect(game.board.every(row => row.every(cell => cell === 0))).toBe(true);
    expect(game.currentPlayer).toBe(1);
    expect(game.gameOver).toBe(false);
    expect(game.winner).toBe(null);
  });

  test('should return correct game status', () => {
    expect(game.getGameStatus()).toBe('Current player: X');
    game.makeMove(0, 0);
    expect(game.getGameStatus()).toBe('Current player: O');
    for (let i = 0; i < 5; i++) {
      game.makeMove(0, i);
      if (i < 4) game.makeMove(1, i);
    }
    expect(game.getGameStatus()).toBe('Player X wins!');
  });
});
