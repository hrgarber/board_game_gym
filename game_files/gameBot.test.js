const BoardGame = require('./gameBot');

describe('BoardGame', () => {
  let game;

  beforeEach(() => {
    game = new BoardGame();
  });

  test('should initialize the board correctly', () => {
    expect(game.board.length).toBe(12);
    expect(game.board[0].length).toBe(8);
    expect(game.board.every(row => row.every(cell => cell === 0))).toBe(true);
  });

  test('should start with white player', () => {
    expect(game.currentPlayer).toBe(1); // Assuming 1 represents white
  });

  test('should make a valid regular move', () => {
    expect(game.makeMove(0, 0, 'regular')).toBe(true);
    expect(game.board[0][0]).toBe(1);
  });

  test('should make a valid permanent move', () => {
    expect(game.makeMove(0, 0, 'permanent')).toBe(true);
    expect(game.board[0][0]).toBe(-1); // Assuming -1 represents black (opponent's color)
  });

  test('should not allow a move on an occupied cell', () => {
    game.makeMove(0, 0, 'regular');
    expect(game.makeMove(0, 0, 'regular')).toBe(false);
    expect(game.makeMove(0, 0, 'permanent')).toBe(false);
  });

  test('should switch players after a move', () => {
    game.makeMove(0, 0, 'regular');
    expect(game.currentPlayer).toBe(-1);
  });

  test('should flip adjacent regular pieces when placing a permanent piece', () => {
    game.makeMove(1, 1, 'regular');
    game.makeMove(0, 0, 'regular'); // Black's move
    game.makeMove(1, 0, 'permanent');
    expect(game.board[1][1]).toBe(-1); // Should be flipped to black
  });

  test('should not flip permanent pieces when placing a permanent piece', () => {
    game.makeMove(1, 1, 'permanent');
    game.makeMove(0, 0, 'regular'); // Black's move
    game.makeMove(1, 0, 'permanent');
    expect(game.board[1][1]).toBe(-1); // Should remain black (permanent)
  });

  test('should enforce edge rule for permanent pieces on sides A and B', () => {
    expect(game.makeMove(0, 0, 'permanent')).toBe(true);
    expect(game.board[0][0]).toBe(-1); // Should be black (opponent's color)
    
    game.makeMove(1, 0, 'regular'); // Black's move
    
    expect(game.makeMove(11, 0, 'permanent')).toBe(true);
    expect(game.board[11][0]).toBe(-1); // Should be black (opponent's color)
  });

  test('should detect a win for white connecting sides A and B', () => {
    // Create a path from side A to side B for white
    for (let i = 0; i < 12; i++) {
      game.makeMove(i, 0, 'regular');
      if (i < 11) game.makeMove(0, i, 'regular'); // Black's moves
    }
    expect(game.gameOver).toBe(true);
    expect(game.winner).toBe(1); // White wins
  });

  test('should detect a win for black connecting sides A and B', () => {
    game.makeMove(0, 1, 'regular'); // White's first move
    // Create a path from side A to side B for black
    for (let i = 0; i < 12; i++) {
      game.makeMove(i, 0, 'regular');
      if (i < 11) game.makeMove(0, i + 1, 'regular'); // White's moves
    }
    expect(game.gameOver).toBe(true);
    expect(game.winner).toBe(-1); // Black wins
  });

  test('should allow a diagonal path to win', () => {
    // Create a diagonal path for white
    for (let i = 0; i < 8; i++) {
      game.makeMove(i, i, 'regular');
      if (i < 7) game.makeMove(0, i + 1, 'regular'); // Black's moves
    }
    // Complete the path to side B
    game.makeMove(8, 7, 'regular');
    game.makeMove(1, 7, 'regular'); // Black's move
    game.makeMove(9, 7, 'regular');
    game.makeMove(2, 7, 'regular'); // Black's move
    game.makeMove(10, 7, 'regular');
    game.makeMove(3, 7, 'regular'); // Black's move
    game.makeMove(11, 7, 'regular');
    
    expect(game.gameOver).toBe(true);
    expect(game.winner).toBe(1); // White wins
  });

  test('should reset the game', () => {
    game.makeMove(0, 0, 'regular');
    game.resetGame();
    expect(game.board.every(row => row.every(cell => cell === 0))).toBe(true);
    expect(game.currentPlayer).toBe(1); // White starts again
    expect(game.gameOver).toBe(false);
    expect(game.winner).toBe(null);
  });

  test('should return correct game status', () => {
    expect(game.getGameStatus()).toBe('Current player: White');
    game.makeMove(0, 0, 'regular');
    expect(game.getGameStatus()).toBe('Current player: Black');
    // Create a winning path for white
    for (let i = 1; i < 12; i++) {
      game.makeMove(i, 0, 'regular');
      if (i < 11) game.makeMove(0, i, 'regular'); // Black's moves
    }
    expect(game.getGameStatus()).toBe('Player White wins!');
  });
});
