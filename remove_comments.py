#!/usr/bin/env python3
import re
import sys

def remove_comments_and_docstrings(source_code):
    lines = source_code.split('\n')
    result = []
    in_docstring = False
    docstring_char = None
    skip_next_blank = False
    
    for line in lines:
        stripped = line.lstrip()
        
        if in_docstring:
            if docstring_char and docstring_char in line:
                in_docstring = False
                docstring_char = None
                skip_next_blank = True
            continue
        
        if stripped.startswith('"""') or stripped.startswith("'''"):
            docstring_char = '"""' if stripped.startswith('"""') else "'''"
            if stripped.count(docstring_char) >= 2:
                skip_next_blank = True
                continue
            else:
                in_docstring = True
                continue
        
        if '#' in line:
            code_part = line.split('#')[0].rstrip()
            if code_part:
                result.append(code_part)
            continue
        
        if skip_next_blank and not stripped:
            skip_next_blank = False
            continue
        
        if stripped or result:
            result.append(line)
    
    while result and not result[-1].strip():
        result.pop()
    
    return '\n'.join(result) + '\n'

if __name__ == '__main__':
    file_path = sys.argv[1]
    with open(file_path, 'r', encoding='utf-8') as f:
        original = f.read()
    
    cleaned = remove_comments_and_docstrings(original)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    
    print(f"Cleaned: {file_path}")
