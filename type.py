import curses
import random
import time

# List of common words for the typing test
words = [
    "the", "of", "and", "a", "to", "in", "is", "you", "that", "it", "he", "was", "for", "on", "are", "as", "with", "his", "they", "I",
    "at", "be", "this", "have", "from", "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we", "when", "your", "can", "said",
    "there", "use", "an", "each", "which", "she", "do", "how", "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these", "so",
    "some", "her", "would", "make", "like", "him", "into", "time", "has", "look", "two", "more", "write", "go", "see", "number", "no", "way", "could", "people",
    "my", "than", "first", "water", "been", "call", "who", "oil", "its", "now", "find", "long", "down", "day", "did", "get", "come", "made", "may", "part"
]

def main(stdscr):
    # Initialize colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Correct
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)    # Incorrect
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Title
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Stats

    stdscr.nodelay(False)
    stdscr.keypad(True)

    while True:
        stdscr.clear()
        stdscr.attron(curses.color_pair(3) | curses.A_BOLD)
        stdscr.addstr(0, 0, "Terminal Typing Test")
        stdscr.attroff(curses.color_pair(3) | curses.A_BOLD)
        stdscr.addstr(1, 0, "Type the text below as fast and accurately as possible.")
        stdscr.addstr(2, 0, "Press ESC to quit at any time.")

        # Generate random sentence (20 words)
        sentence_list = random.choices(words, k=20)
        sentence = ' '.join(sentence_list)
        max_y, max_x = stdscr.getmaxyx()
        # Wrap sentence if too long, but for simplicity assume terminal is wide enough

        stdscr.addstr(4, 0, "Text to type:")
        stdscr.addstr(5, 0, sentence)
        stdscr.addstr(7, 0, "Start typing here (backspace enabled):")

        stdscr.refresh()

        typed = ""
        pos = 0
        start_time = None

        while len(typed) < len(sentence):
            ch = stdscr.getch()

            if ch == 27:  # ESC to quit
                return

            if start_time is None:
                start_time = time.time()

            if ch == curses.KEY_BACKSPACE or ch == 127:
                if pos > 0:
                    pos -= 1
                    typed = typed[:-1]
            elif 32 <= ch <= 126:  # Printable characters
                typed += chr(ch)
                pos += 1

            # Redraw the sentence with colors
            stdscr.move(5, 0)
            stdscr.clrtoeol()
            for i, char in enumerate(sentence):
                attr = 0
                if i < len(typed):
                    if typed[i] == char:
                        attr = curses.color_pair(1)  # Green for correct
                    else:
                        attr = curses.color_pair(2)  # Red for incorrect
                stdscr.addch(char, attr)

            # Show cursor position
            stdscr.move(5, min(pos, max_x - 1))
            stdscr.refresh()

        end_time = time.time()
        time_taken = end_time - start_time if start_time else 0

        # Calculate stats
        correct_chars = sum(1 for i in range(len(sentence)) if typed[i] == sentence[i])
        accuracy = (correct_chars / len(sentence)) * 100 if len(sentence) > 0 else 0
        wpm = (len(sentence) / 5) / (time_taken / 60) if time_taken > 0 else 0

        # Display stats
        stdscr.addstr(9, 0, "Test Complete!", curses.color_pair(4) | curses.A_BOLD)
        stdscr.addstr(10, 0, f"Time taken: {time_taken:.2f} seconds")
        stdscr.addstr(11, 0, f"Words per minute (WPM): {wpm:.2f}")
        stdscr.addstr(12, 0, f"Accuracy: {accuracy:.2f}%")
        stdscr.addstr(14, 0, "Press 'y' to try again, or any other key to quit.")

        stdscr.refresh()

        ch = stdscr.getch()
        if chr(ch).lower() != 'y':
            break

# Run the curses application
curses.wrapper(main)
