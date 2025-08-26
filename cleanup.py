import re

def clean_text(input_file, output_file):
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Normalize newlines: Replace multiple newlines with a single newline
    text = re.sub(r'\n\s*\n+', '\n', text)

    # 2. Convert all-caps titles to title case
    # Assume titles are lines that are entirely uppercase (ignoring spaces/punctuation)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Check if line is all uppercase (excluding spaces/punctuation)
        if line.strip() and all(c.isupper() or not c.isalpha() for c in line.strip()):
            cleaned_line = line.title()  # Convert to title case
            cleaned_lines.append(cleaned_line)
        else:
            cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # 3. Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    # 4. (Optional) Convert to lowercase to reduce vocab_size
    # Uncomment if you want to use lowercase
    # text = text.lower()

    # 5. Normalize special characters (e.g., curly quotes, em dashes)
    text = text.replace('“', '"').replace('”', '"')  # Curly to straight quotes
    text = text.replace('—', '-')  # Em dash to hyphen
    text = text.replace('’', "'")  # Curly apostrophe to straight

    # 6. Trim leading/trailing whitespace from the entire text
    text = text.strip()

    # Write cleaned text to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

    return text

# Example usage
input_file = "novels.txt"
output_file = "mark_twain_cleaned.txt"
cleaned_text = clean_text(input_file, output_file)
print(f"Cleaned text saved to {output_file}")
print(f"First 100 characters of cleaned text: {cleaned_text[:100]}")
