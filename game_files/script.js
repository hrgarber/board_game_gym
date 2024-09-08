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
    statusElement.textContent = game.getGameStatus();
}

function handleCellClick(event) {
    const row = parseInt(event.target.dataset.row);
    const col = parseInt(event.target.dataset.col);
    
    if (game.makeMove(row, col)) {
        updateBoard();
    }
}

resetBtn.addEventListener('click', () => {
    game.resetGame();
    updateBoard();
});

// Initialize the game
updateBoard();
