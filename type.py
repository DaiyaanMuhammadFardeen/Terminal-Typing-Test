import curses
from generate import TextGenerator
import time

def wrap_text(sentence, max_x):
    """Split sentence into lines of max_x - 1 characters."""
    lines = []
    for i in range(0, len(sentence), max_x - 1):
        lines.append(sentence[i:i + max_x - 1])
    return lines

def main(stdscr):
    # Initialize colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Correct
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)    # Incorrect
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Title
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Stats

    stdscr.nodelay(False)
    stdscr.keypad(True)

    # Initialize text generator
    generator = TextGenerator()

    while True:
        stdscr.clear()
        stdscr.attron(curses.color_pair(3) | curses.A_BOLD)
        stdscr.addstr(0, 0, "Terminal Typing Test")
        stdscr.attroff(curses.color_pair(3) | curses.A_BOLD)
        stdscr.addstr(1, 0, "Type the text below as fast and accurately as possible.")
        stdscr.addstr(2, 0, "Press ESC to quit at any time.")

        # Generate sentence using TextGenerator
        sentence = generator.generate(start="The", sentences=5, temp=1.0, max_length=1000)
        max_y, max_x = stdscr.getmaxyx()

        # Wrap sentence into lines
        lines = wrap_text(sentence, max_x)
        start_row = 5

        # Check if terminal has enough rows
        if start_row + len(lines) + 4 > max_y:
            stdscr.addstr(4, 0, "Error: Terminal too small to display full text.", curses.color_pair(2))
            stdscr.addstr(6, 0, "Press any key to quit.")
            stdscr.refresh()
            stdscr.getch()
            return

        # Display wrapped sentence
        stdscr.addstr(4, 0, "Text to type:")
        for i, line in enumerate(lines):
            stdscr.addstr(start_row + i, 0, line)

        stdscr.addstr(start_row + len(lines) + 1, 0, "Start typing here (backspace enabled):")
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

            # Determine cursor position (row, col)
            cursor_row = start_row + (pos // (max_x - 1))
            cursor_col = pos % (max_x - 1)

            # Redraw only the current line
            line_idx = pos // (max_x - 1)
            if line_idx < len(lines):  # Ensure line exists
                stdscr.move(start_row + line_idx, 0)
                stdscr.clrtoeol()
                line_start = line_idx * (max_x - 1)
                for i, char in enumerate(lines[line_idx]):
                    attr = 0
                    global_idx = line_start + i
                    if global_idx < len(typed):
                        if typed[global_idx] == char:
                            attr = curses.color_pair(1)  # Green for correct
                        else:
                            attr = curses.color_pair(2)  # Red for incorrect
                    stdscr.addch(char, attr)

            # Update cursor position
            stdscr.move(cursor_row, cursor_col)
            stdscr.refresh()

        end_time = time.time()
        time_taken = end_time - start_time if start_time else 0

        # Calculate stats
        correct_chars = sum(1 for i in range(len(sentence)) if typed[i] == sentence[i])
        accuracy = (correct_chars / len(sentence)) * 100 if len(sentence) > 0 else 0
        wpm = (len(sentence) / 5) / (time_taken / 60) if time_taken > 0 else 0

        # Display stats
        stats_row = start_row + len(lines) + 3
        stdscr.addstr(stats_row, 0, "Test Complete!", curses.color_pair(4) | curses.A_BOLD)
        stdscr.addstr(stats_row + 1, 0, f"Time taken: {time_taken:.2f} seconds")
        stdscr.addstr(stats_row + 2, 0, f"Words per minute (WPM): {wpm:.2f}")
        stdscr.addstr(stats_row + 3, 0, f"Accuracy: {accuracy:.2f}%")
        stdscr.addstr(stats_row + 5, 0, "Press 'y' to try again, or any other key to quit.")

        stdscr.refresh()

        ch = stdscr.getch()
        if chr(ch).lower() != 'y':
            break

# Run the curses application
curses.wrapper(main)
