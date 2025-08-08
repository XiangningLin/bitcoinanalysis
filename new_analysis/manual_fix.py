#!/usr/bin/env python3
import json
import re
from collections import Counter

# Read CETraS raw text
with open('outputs/cetras/cetras_role_predictions.part1.raw.txt', 'r') as f:
    raw = f.read()

# Extract JSON from markdown code block
match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
if match:
    json_str = match.group(1).strip()
    try:
        data = json.loads(json_str)
        print(f'✅ Successfully parsed {len(data)} CETraS roles')
        
        # Save fixed version
        with open('outputs/cetras/cetras_role_predictions.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Show distribution
        roles = [item.get('role', '') for item in data]
        print(f'📊 CETraS Role Distribution: {dict(Counter(roles))}')
        
        # Also check RWFB
        with open('outputs/rwfb/rwfb_role_predictions.json', 'r') as f:
            rwfb_data = json.load(f)
        rwfb_roles = [item.get('role', '') for item in rwfb_data]
        print(f'📊 RWFB Role Distribution: {dict(Counter(rwfb_roles))}')
        
    except Exception as e:
        print(f'❌ Parse error: {e}')
        print(f"Raw content preview: {raw[:200]}...")
else:
    print('❌ No JSON block found')
    print(f"Raw content preview: {raw[:200]}...")

