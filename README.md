# 🌟 Terminal Typing Test: Twain’s AI Slop Typing Extravaganza! 🌟

Welcome to **Terminal-Typing-Test**, my **hello world** dive into the wild world of **generative AI**! 🚀 Ditch the browser—this terminal-based typing test challenges you to type quirky, AI-generated Mark Twain-inspired quotes as fast as your fingers can fly. Think typing practice meets Twain’s wit with a dash of *AI slop* for flavor. Oh, and there’s a **secret Easter egg** hidden in the codebase—can you find it? 👀

![Terminal Typing Test](https://img.shields.io/badge/Terminal-Typing%20Test-blueviolet?style=flat-square&logo=github)  
**Stars**: ⭐ {repo_stars} | **Forks**: 🍴 {repo_forks} | **Topics**: {repo_topics}

*Last updated: {last_commit_date}*  
*Latest commit: {last_commit_message}*

---

## 🎩 What’s This All About?

This project is my first adventure in generative AI, using a **character-level RNN** to create Mark Twain-esque sentences from a ~1M+ character dataset (think *Huck Finn* meets AI weirdness). The app serves up these sentences in a colorful **ncurses** terminal interface for a typing test that tracks your speed (WPM) and accuracy. It’s fun, nerdy, and browser-free—because terminals are where the cool kids code. 😎

> **Sample Quote**:  
> The river was mighty quiet that night, Huck said. "Reckon we oughta sneak past that old mill afore dawn breaks. Ain’t no tellin’ what trouble’s waitin’." Tom nodded, his eyes sharp. They paddled on, silent as ghosts.

---

## ✨ Features

- **AI-Generated Twain Quotes**: A `CharRNN` model (256 embedding size, 512 hidden units, 3 LSTM layers) generates sentences inspired by Mark Twain’s style. Expect witty dialogue, Southern charm, or gloriously weird “AI slop.”
- **Terminal Typing Test**: Type the generated text in a colorful ncurses interface. Correct characters glow **green**, mistakes burn **red**, and stats (WPM, accuracy) pop in **yellow**.
- **Text Wrapping**: Long sentences wrap across multiple lines to fit your terminal, keeping every Twain-inspired word intact.
- **Efficient Data Handling**: Preprocessed data (`chars`, `stoi`, `itos`, `data`, `vocab_size`) is saved to `mark_twain_preprocessed.pkl` for quick loading.
- **Dynamic GitHub Stats**: Live repo stats (stars, forks, topics) fetched via GitHub API, showing the project’s pulse.
- **Easter Egg Alert**: An unused file lurks in `TextGens/shakespeare_quotes.pt`. A hint of future bardic adventures? 🤔

---

## 📂 Project Structure

```
├── cleanup.py              # Preprocesses Mark Twain text
├── generate.py             # TextGenerator class for AI sentence generation
├── novels.txt             # Raw Mark Twain dataset (~1M+ characters)
├── README.md              # This fabulous file!
├── TextGens/
│   ├── mark_twain_preprocessed.pkl  # Preprocessed data
│   ├── mark_twain_story.pt         # Trained model checkpoint
│   ├── shakespeare_quotes.pt       # 🐣 Easter egg: Unused Shakespeare model
├── train.py               # Trains the CharRNN model
└── type.py                # Terminal typing test with ncurses
```

---

## 🚀 Get Started

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/{owner}/{repo}.git
   cd Terminal-Typing-Test
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch
   ```
   *(Note: `curses` is included in Python’s standard library on Unix-like systems. For Windows, install `windows-curses`.)*

3. **Preprocess Data**:
   ```bash
   python cleanup.py
   ```

4. **Train the Model** (if not using the provided checkpoint):
   ```bash
   python train.py
   ```

5. **Run the Typing Test**:
   ```bash
   python type.py
   ```

   - Type the AI-generated text.
   - Press **ESC** to quit, or **y** to try again.
   - Watch for **green** (correct) and **red** (incorrect) characters.

---

## 🛠️ Dynamic GitHub Stats

This README pulls live data from the GitHub API to keep things fresh:

- **Stars**: ⭐ {repo_stars} (e.g., 42)
- **Forks**: 🍴 {repo_forks} (e.g., 10)
- **Topics**: {repo_topics} (e.g., `generative-ai`, `typing-test`, `mark-twain`)
- **Latest Commit**: {last_commit_message} (e.g., "Add text wrapping to type.py") on {last_commit_date} (e.g., Aug 26, 2025)

**Manual Update**:
Fetch data using the GitHub API:
```bash
curl -