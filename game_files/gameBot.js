// gameBot.js - Board Game AI Bot

class GameBot {
    constructor(boardSize = 8) {
        this.boardSize = boardSize;
        this.board = this.initializeBoard();
        this.currentPlayer = 1; // 1 for player, -1 for bot
    }

    initializeBoard() {
        // Initialize an empty board
        return Array(this.boardSize).fill().map(() => Array(this.boardSize).fill(0));
    }

    getMoves() {
        // Generate all possible moves
        const moves = [];
        for (let i = 0; i < this.boardSize; i++) {
            for (let j = 0; j < this.boardSize; j++) {
                if (this.board[i][j] === 0) {
                    moves.push({ row: i, col: j });
                }
            }
        }
        return moves;
    }

    makeMove(move) {
        if (this.board[move.row][move.col] === 0) {
            this.board[move.row][move.col] = this.currentPlayer;
            this.currentPlayer *= -1; // Switch player
            return true;
        }
        return false;
    }

    undoMove(move) {
        this.board[move.row][move.col] = 0;
        this.currentPlayer *= -1; // Switch player back
    }

    evaluate() {
        // Simple evaluation: difference in piece count
        let score = 0;
        for (let i = 0; i < this.boardSize; i++) {
            for (let j = 0; j < this.boardSize; j++) {
                score += this.board[i][j];
            }
        }
        return score;
    }

    minimax(depth, alpha, beta, maximizingPlayer) {
        if (depth === 0) {
            return this.evaluate();
        }

        const moves = this.getMoves();

        if (maximizingPlayer) {
            let maxEval = -Infinity;
            for (const move of moves) {
                this.makeMove(move);
                const eval = this.minimax(depth - 1, alpha, beta, false);
                this.undoMove(move);
                maxEval = Math.max(maxEval, eval);
                alpha = Math.max(alpha, eval);
                if (beta <= alpha) break;
            }
            return maxEval;
        } else {
            let minEval = Infinity;
            for (const move of moves) {
                this.makeMove(move);
                const eval = this.minimax(depth - 1, alpha, beta, true);
                this.undoMove(move);
                minEval = Math.min(minEval, eval);
                beta = Math.min(beta, eval);
                if (beta <= alpha) break;
            }
            return minEval;
        }
    }

    getBestMove(depth = 3) {
        let bestMove;
        let bestEval = -Infinity;
        const moves = this.getMoves();

        for (const move of moves) {
            this.makeMove(move);
            const eval = this.minimax(depth - 1, -Infinity, Infinity, false);
            this.undoMove(move);

            if (eval > bestEval) {
                bestEval = eval;
                bestMove = move;
            }
        }

        return bestMove;
    }

    // Training and evaluation methods
    train(numGames = 1000) {
        let wins = 0;
        for (let i = 0; i < numGames; i++) {
            this.board = this.initializeBoard();
            this.currentPlayer = 1;
            
            while (this.getMoves().length > 0) {
                const move = this.currentPlayer === 1 ? this.getBestMove() : this.getRandomMove();
                this.makeMove(move);
                
                if (this.checkWin()) {
                    if (this.currentPlayer === -1) wins++;
                    break;
                }
            }
        }
        return wins / numGames;
    }

    getRandomMove() {
        const moves = this.getMoves();
        return moves[Math.floor(Math.random() * moves.length)];
    }

    checkWin() {
        // Implement win condition check
        // This is a placeholder and should be implemented based on your game rules
        return false;
    }

    // Interface for interacting with the bot
    playGame() {
        while (this.getMoves().length > 0) {
            if (this.currentPlayer === 1) {
                console.log("Current board:");
                this.printBoard();
                const move = this.getBestMove();
                console.log(`Bot's move: Row ${move.row}, Col ${move.col}`);
                this.makeMove(move);
            } else {
                const move = this.getHumanMove();
                this.makeMove(move);
            }

            if (this.checkWin()) {
                console.log(`Player ${this.currentPlayer === 1 ? 2 : 1} wins!`);
                break;
            }
        }
        console.log("Game Over");
    }

    getHumanMove() {
        // In a real implementation, this would get input from the user
        // For now, we'll just return a random move
        return this.getRandomMove();
    }

    printBoard() {
        for (let i = 0; i < this.boardSize; i++) {
            console.log(this.board[i].map(cell => cell === 0 ? '.' : cell === 1 ? 'X' : 'O').join(' '));
        }
    }
}

// Example usage
const bot = new GameBot();
console.log("Training bot...");
const winRate = bot.train(1000);
console.log(`Bot win rate after training: ${winRate * 100}%`);
console.log("Starting game...");
bot.playGame();