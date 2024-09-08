// script.js
const boardElement = document.getElementById('board');
const resetBtn = document.getElementById('reset-btn');
const statusElement = document.getElementById('status');

let game = new BoardGame();

function updateBoard() {
    boardElement.innerHTML = '';
    for (let i = 0; i < game.boardSize; i++) {
        for (let j = 0; j < game.boardSize; j++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.row = i;
            cell.dataset.col = j;
            cell.textContent = game.board[i][j] === 1 ? 'X' : game.board[i][j] === -1 ? 'O' : '';
            cell.addEventListener('click', handleCellClick);
            boardElement.appendChild(cell);
        }
    }
}

function handleCellClick(event) {
    const row = parseInt(event.target.dataset.row);
    const col = parseInt(event.target.dataset.col);
    
    if (game.makeMove(row, col)) {
        updateBoard();
        if (game.checkWin(row, col)) {
            statusElement.textContent = `Player ${game.currentPlayer === 1 ? 'O' : 'X'} wins!`;
        } else if (game.isBoardFull()) {
            statusElement.textContent = "It's a draw!";
        } else {
            statusElement.textContent = `Current player: ${game.currentPlayer === 1 ? 'X' : 'O'}`;
        }
    }
}

resetBtn.addEventListener('click', () => {
    game.resetGame();
    updateBoard();
    statusElement.textContent = 'New game started. X plays first.';
});

// Initialize the game
updateBoard();
statusElement.textContent = 'Game started. X plays first.';
