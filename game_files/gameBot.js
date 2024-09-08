// gameBot.js - Board Game Logic

console.log("Loading gameBot.js");

class BoardGame {
    constructor(boardSize = 8) {
        console.log(`Creating BoardGame with size ${boardSize}`);
        this.boardSize = boardSize;
        this.board = this.initializeBoard();
        this.currentPlayer = 1; // 1 for X, -1 for O
        this.gameOver = false;
        this.winner = null;
    }

    initializeBoard() {
        return Array(this.boardSize).fill().map(() => Array(this.boardSize).fill(0));
    }

    makeMove(row, col) {
        if (this.gameOver || this.board[row][col] !== 0) {
            return false;
        }
        this.board[row][col] = this.currentPlayer;
        if (this.checkWin(row, col)) {
            this.gameOver = true;
            this.winner = this.currentPlayer;
        } else if (this.isBoardFull() || !this.isWinPossible()) {
            this.gameOver = true;
            this.winner = null; // It's a draw
        } else {
            this.currentPlayer *= -1; // Switch player
        }
        return true;
    }

    checkWin(row, col) {
        const directions = [
            [0, 1], [1, 0], [1, 1], [1, -1] // horizontal, vertical, diagonal
        ];
        
        for (const [dx, dy] of directions) {
            let count = 1;
            for (const factor of [-1, 1]) {
                for (let i = 1; i < 5; i++) {
                    const newRow = row + factor * i * dx;
                    const newCol = col + factor * i * dy;
                    if (
                        newRow < 0 || newRow >= this.boardSize ||
                        newCol < 0 || newCol >= this.boardSize ||
                        this.board[newRow][newCol] !== this.board[row][col]
                    ) {
                        break;
                    }
                    count++;
                }
            }
            if (count >= 5) return true;
        }
        return false;
    }

    isBoardFull() {
        return this.board.every(row => row.every(cell => cell !== 0));
    }

    isWinPossible() {
        const directions = [
            [0, 1], [1, 0], [1, 1], [1, -1] // horizontal, vertical, diagonal
        ];
        
        for (let row = 0; row < this.boardSize; row++) {
            for (let col = 0; col < this.boardSize; col++) {
                if (this.board[row][col] !== 0) continue; // Skip occupied cells
                
                for (const [dx, dy] of directions) {
                    let xCount = 0, oCount = 0, emptyCount = 1; // Start with 1 empty (current cell)
                    
                    for (let i = -4; i <= 4; i++) {
                        const newRow = row + i * dx;
                        const newCol = col + i * dy;
                        
                        if (newRow < 0 || newRow >= this.boardSize || newCol < 0 || newCol >= this.boardSize) {
                            continue;
                        }
                        
                        if (this.board[newRow][newCol] === 1) xCount++;
                        else if (this.board[newRow][newCol] === -1) oCount++;
                        else emptyCount++;
                        
                        if (xCount + emptyCount >= 5 || oCount + emptyCount >= 5) {
                            return true; // A win is still possible
                        }
                    }
                }
            }
        }
        return false; // No win is possible
    }

    resetGame() {
        this.board = this.initializeBoard();
        this.currentPlayer = 1;
        this.gameOver = false;
        this.winner = null;
    }

    getGameStatus() {
        if (this.gameOver) {
            if (this.winner) {
                return `Player ${this.winner === 1 ? 'X' : 'O'} wins!`;
            } else {
                return "It's a draw!";
            }
        } else {
            return `Current player: ${this.currentPlayer === 1 ? 'X' : 'O'}`;
        }
    }
}

console.log("Exporting BoardGame");
module.exports = BoardGame;
