"""
Convert the annotated corpus's <> and «» delimiters to the bracket
characters used elsewhere (see brackets.py):
  <...>  -> ⧔...⧕
  «...»  -> ⧨...⧩
"""
import os
import re

ROOT = os.path.dirname(__file__)
INPUT_PATH = os.path.join(ROOT, 'corpus_processed_annotated.txt')
OUTPUT_PATH = os.path.join(ROOT, 'corpus_processed_annotated_updated.txt')

REPLACEMENTS = [
    (re.compile('<'), '⧔'),
    (re.compile('>'), '⧕'),
    (re.compile('«'), '⧨'),
    (re.compile('»'), '⧩'),
    (re.compile('{'), '⦃'),
    (re.compile('}'), '⦄'),
]


def convert(text):
    for pattern, repl in REPLACEMENTS:
        text = pattern.sub(repl, text)
    return text


def main():
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    text = convert(text)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"wrote converted corpus to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
