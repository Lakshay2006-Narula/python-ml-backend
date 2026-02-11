import re

# Read the file
with open('src/pdf_generator.py', encoding='utf-8') as f:
    content = f.read()

# Replace all toc.addEntry calls with notify_toc
# Pattern: self.toc.addEntry(0, '1. Introduction', 1, key='sec1')
# Replace with: self.notify_toc(0, '1. Introduction', "sec1")
pattern = r"self\.toc\.addEntry\((\d+),\s*([^,]+),\s*1,\s*key='(\w+)'\)"
replacement = r'self.notify_toc(\1, \2, "\3")'

content = re.sub(pattern, replacement, content)

# Write back
with open('src/pdf_generator.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Replaced all addEntry calls with notify_toc")
