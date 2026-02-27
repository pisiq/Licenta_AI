# -*- coding: utf-8 -*-
"""Strip all non-ASCII characters from Python source files."""
import sys

REPLACEMENTS = [
    # Emojis in train.py / trainer.py
    ('\U0001f4da', '[*]'),   # book
    ('\U0001f4ca', '[*]'),   # chart
    ('\U0001f389', '[BEST]'),
    ('\u26a0\ufe0f', '[WARN]'),
    ('\U0001f510', '[FROZEN]'),
    ('\U0001f513', '[UNFROZEN]'),
    ('\U0001f4a5', '[!]'),
    ('\u2713', '[OK]'),
    ('\u2714', '[OK]'),
    ('\u2718', '[X]'),
    ('\u2717', '[X]'),
    # Box-drawing / typography in data_preprocessing.py / trainer.py
    ('\u2500', '-'),    # box horizontal
    ('\u2501', '-'),
    ('\u2502', '|'),
    ('\u2014', '-'),    # em dash
    ('\u2013', '-'),    # en dash
    ('\u2190', '<-'),
    ('\u2192', '->'),
    ('\u2019', "'"),
    ('\u201c', '"'),
    ('\u201d', '"'),
    # Other common ones
    ('\u00b7', '*'),
    ('\u2022', '*'),
]

TARGET_FILES = ['train.py', 'trainer.py', 'data_preprocessing.py', 'model.py', 'metrics.py', 'config.py']

for fname in TARGET_FILES:
    try:
        with open(fname, encoding='utf-8') as f:
            content = f.read()
        original = content
        for old, new in REPLACEMENTS:
            content = content.replace(old, new)
        # Catch anything else > 127 that might be in print strings
        safe = []
        for ch in content:
            if ord(ch) > 127:
                # Replace with ASCII description
                safe.append('?')
            else:
                safe.append(ch)
        content = ''.join(safe)
        if content != original:
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(content)
            print('Fixed: ' + fname)
        else:
            print('Clean: ' + fname)
    except FileNotFoundError:
        print('Skip (not found): ' + fname)
    except Exception as e:
        print('ERROR ' + fname + ': ' + str(e))

print('Done.')

