// script.js
const boardElement = document.getElementById('board');
const resetBtn = document.getElementById('reset-btn');
const statusElement = document.getElementById('status');

let gameId;
let boardSize = 8;

async function createGame() {
    const response = await fetch('/game', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ size: boardSize }),
    });
    const data = await response.json();
    gameId = data.id;
    updateBoard(data.board);
}

async function makeMove(row, col) {
    const response = await fetch(`/game/${gameId}/move`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ row, col }),
    });
    const data = await response.json();
    updateBoard(data.board);
    updateStatus(data);
}

function updateBoard(board) {
    boardElement.innerHTML = '';
    for (let i = 0; i < boardSize; i++) {
        for (let j = 0; j < boardSize; j++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.row = i;
            cell.dataset.col = j;
            cell.textContent = board[i][j] === 1 ? 'X' : board[i][j] === -1 ? 'O' : '';
            cell.addEventListener('click', handleCellClick);
            boardElement.appendChild(cell);
        }
    }
}

function updateStatus(gameState) {
    if (gameState.gameOver) {
        if (gameState.winner) {
            statusElement.textContent = `Player ${gameState.winner === 1 ? 'X' : 'O'} wins!`;
        } else {
            statusElement.textContent = "It's a draw!";
        }
    } else {
        statusElement.textContent = `Current player: ${gameState.currentPlayer === 1 ? 'X' : 'O'}`;
    }
}

function handleCellClick(event) {
    const row = parseInt(event.target.dataset.row);
    const col = parseInt(event.target.dataset.col);
    makeMove(row, col);
}

resetBtn.addEventListener('click', createGame);

// Initialize the game
createGame();
