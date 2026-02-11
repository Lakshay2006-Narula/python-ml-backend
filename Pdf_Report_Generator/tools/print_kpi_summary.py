import json
import sys

try:
    data = json.load(open('data/processed/report_text.json', 'r', encoding='utf-8'))
except Exception as e:
    print(f'ERROR: cannot open report file: {e}', file=sys.stderr)
    sys.exit(2)

ks = data.get('KPI Summary')
if ks is None:
    print('KPI Summary not found', file=sys.stderr)
    sys.exit(1)

print(ks)
