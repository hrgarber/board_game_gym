// script.js
const boardElement = document.getElementById('board')
const trainBtn = document.getElementById('train-btn')
const playBtn = document.getElementById('play-btn')
const resetBtn = document.getElementById('reset-btn')
const statusElement = document.getElementById('status')

let bot = new GameBot()
let isPlaying = false

function updateBoard () {
  boardElement.innerHTML = ''
  for (let i = 0; i < bot.boardSize; i++) {
    for (let j = 0; j < bot.boardSize; j++) {
      const cell = document.createElement('div')
      cell.className = 'cell'
      cell.dataset.row = i
      cell.dataset.col = j
      cell.textContent = bot.board[i][j] === 1 ? 'X' : bot.board[i][j] === -1 ? 'O' : ''
      cell.addEventListener('click', handleCellClick)
      boardElement.appendChild(cell)
    }
  }
}

function handleCellClick (event) {
  if (!isPlaying || bot.currentPlayer !== -1) return

  const row = parseInt(event.target.dataset.row)
  const col = parseInt(event.target.dataset.col)

  if (bot.makeMove({ row, col })) {
    updateBoard()
    checkGameState()
    if (isPlaying) {
      makeBotMove()
    }
  }
}

function makeBotMove () {
  if (!isPlaying || bot.currentPlayer !== 1) return

  const move = bot.getBestMove()
  bot.makeMove(move)
  updateBoard()
  checkGameState()
}

function checkGameState () {
  if (bot.checkWin()) {
    isPlaying = false
    statusElement.textContent = `Player ${bot.currentPlayer === 1 ? 'O' : 'X'} wins!`
  } else if (bot.getMoves().length === 0) {
    isPlaying = false
    statusElement.textContent = "It's a draw!"
  }
}

trainBtn.addEventListener('click', async () => {
  statusElement.textContent = 'Training...'
  trainBtn.disabled = true
  playBtn.disabled = true

  // Use setTimeout to allow the UI to update before starting the training
  setTimeout(async () => {
    const winRate = await bot.train(1000)
    statusElement.textContent = `Training complete. Bot win rate: ${(winRate * 100).toFixed(2)}%`
    trainBtn.disabled = false
    playBtn.disabled = false
  }, 100)
})

playBtn.addEventListener('click', () => {
  bot = new GameBot()
  isPlaying = true
  updateBoard()
  statusElement.textContent = 'Game started. Your turn (O)!'
})

resetBtn.addEventListener('click', () => {
  bot = new GameBot()
  isPlaying = false
  updateBoard()
  statusElement.textContent = ''
})

updateBoard()