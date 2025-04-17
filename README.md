# ♠️ Blackjack+

**Blackjack+** is a twist on classic blackjack. The goal is to get as close as possible to the **target score** without going over. Unlike traditional blackjack, the **target score changes each round**!

---
🎮 **Play the game here:** [https://black-jack4445.netlify.app/](https://black-jack4445.netlify.app/)  

---
## ⚙️ Technical Overview

### 🧱 Tech Stack
- **Frontend**: React + TypeScript + Material UI
- **Backend**: Django + Django Channels (for real-time WebSocket communication)
- **AI Model**: PyTorch model trained to simulate computer opponent logic

### 🤖 AI Logic
The computer dealer uses a **PyTorch-based model** trained on simulated Blackjack+ scenarios to make decisions during gameplay. To simulate human-like behavior, the AI plays faster when it’s more confident in its decision, and slower when it’s uncertain

---

## 🔢 How to Play

### 🎯 Starting the Round
- A new **target score** between 21 and 27 is randomly chosen at the beginning of each round and displayed on screen.
- Both you and the computer receive two cards.

### 🙋 Your Turn
You can choose to:
- **Hit**: Take another card.
- **Stand**: End your turn and keep your current total.

### 🧠 Computer's Turn
- The computer will choose to hit or stand.

### 🏆 Winning the Game
- If **you bust**, you lose the round.
- If **the computer busts** and you don't, you win!
- If **neither busts**, the player **closest to the target** without going over wins.
- If both are equally close, it's a **tie**.

---

## 🃏 Card Values
- **Number cards (2–10)**: Face value  
- **Face cards (J, Q, K)**: 10  
- **Aces**: 1 or 11 (whichever is more favorable to the player)

---


