// gameBot.js - Board Game Logic

console.log("Loading gameBot.js");

class BoardGame {
    constructor() {
        console.log("Creating BoardGame");
        this.boardSize = { rows: 12, cols: 8 };
        this.board = this.initializeBoard();
        this.currentPlayer = 1; // 1 for White, -1 for Black
        this.gameOver = false;
        this.winner = null;
    }

    initializeBoard() {
        return Array(this.boardSize.rows).fill().map(() => Array(this.boardSize.cols).fill(0));
    }

    makeMove(row, col, pieceType) {
        if (this.gameOver || this.board[row][col] !== 0) {
            return false;
        }

        if (pieceType === 'permanent') {
            this.board[row][col] = -this.currentPlayer * 2; // Use 2 or -2 for permanent pieces
            this.flipAdjacentPieces(row, col);
        } else {
            this.board[row][col] = this.currentPlayer;
        }

        if (this.checkWin()) {
            this.gameOver = true;
            this.winner = this.currentPlayer;
        } else {
            this.currentPlayer *= -1; // Switch player
        }
        return true;
    }

    flipAdjacentPieces(row, col) {
        for (let i = -1; i <= 1; i++) {
            for (let j = -1; j <= 1; j++) {
                if (i === 0 && j === 0) continue;
                const newRow = row + i;
                const newCol = col + j;
                if (
                    newRow >= 0 && newRow < this.boardSize.rows &&
                    newCol >= 0 && newCol < this.boardSize.cols &&
                    Math.abs(this.board[newRow][newCol]) === 1 // Only flip regular pieces
                ) {
                    this.board[newRow][newCol] *= -1;
                }
            }
        }
    }

    checkWin() {
        // Check for a path from side A to side B
        const visited = Array(this.boardSize.rows).fill().map(() => Array(this.boardSize.cols).fill(false));
        for (let col = 0; col < this.boardSize.cols; col++) {
            if (Math.abs(this.board[0][col]) === Math.abs(this.currentPlayer)) {
                if (this.dfs(0, col, visited)) {
                    return true;
                }
            }
        }
        return false;
    }

    dfs(row, col, visited) {
        if (row === this.boardSize.rows - 1) return true; // Reached side B
        visited[row][col] = true;

        const directions = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
        for (const [dx, dy] of directions) {
            const newRow = row + dx;
            const newCol = col + dy;
            if (
                newRow >= 0 && newRow < this.boardSize.rows &&
                newCol >= 0 && newCol < this.boardSize.cols &&
                !visited[newRow][newCol] &&
                Math.abs(this.board[newRow][newCol]) === Math.abs(this.currentPlayer)
            ) {
                if (this.dfs(newRow, newCol, visited)) {
                    return true;
                }
            }
        }
        return false;
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
                return `Player ${this.winner === 1 ? 'White' : 'Black'} wins!`;
            } else {
                return "It's a draw!";
            }
        } else {
            return `Current player: ${this.currentPlayer === 1 ? 'White' : 'Black'}`;
        }
    }
}

console.log("Exporting BoardGame");
module.exports = BoardGame;
