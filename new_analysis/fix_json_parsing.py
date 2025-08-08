#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix JSON parsing for role predictions
"""

import json
import re
import os


def extract_json_from_raw(raw_text: str):
    """Extract JSON from markdown code blocks or raw text"""
    # Try to find JSON in markdown code blocks
    json_match = re.search(r'```json\s*(.*?)\s*```', raw_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to find JSON array pattern
        json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return None
    
    try:
        return json.loads(json_str)
    except:
        return None


def fix_role_predictions(input_file: str, output_file: str):
    """Fix role predictions JSON file"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if len(data) == 1 and "_raw" in data[0]:
        # Extract JSON from raw text
        raw_text = data[0]["_raw"]
        parsed_json = extract_json_from_raw(raw_text)
        
        if parsed_json:
            print(f"✅ Successfully parsed {len(parsed_json)} role predictions")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, ensure_ascii=False, indent=2)
            return parsed_json
        else:
            print("❌ Failed to parse JSON from raw text")
            return None
    else:
        print("✅ JSON already properly formatted")
        return data


def main():
    # Fix RWFB roles
    rwfb_input = "outputs/rwfb/rwfb_role_predictions.json"
    rwfb_output = "outputs/rwfb/rwfb_role_predictions_fixed.json"
    
    print("🔧 Fixing RWFB role predictions...")
    rwfb_roles = fix_role_predictions(rwfb_input, rwfb_output)
    
    # Fix CETraS roles
    cetras_input = "outputs/cetras/cetras_role_predictions.json"
    cetras_output = "outputs/cetras/cetras_role_predictions_fixed.json"
    
    print("🔧 Fixing CETraS role predictions...")
    cetras_roles = fix_role_predictions(cetras_input, cetras_output)
    
    # Show role distribution
    if rwfb_roles:
        from collections import Counter
        rwfb_role_list = [item.get("role", "") for item in rwfb_roles if isinstance(item, dict)]
        rwfb_counter = Counter(rwfb_role_list)
        print(f"\n📊 RWFB Role Distribution: {dict(rwfb_counter)}")
    
    if cetras_roles:
        from collections import Counter
        cetras_role_list = [item.get("role", "") for item in cetras_roles if isinstance(item, dict)]
        cetras_counter = Counter(cetras_role_list)
        print(f"📊 CETraS Role Distribution: {dict(cetras_counter)}")
    
    # Replace original files with fixed versions
    if rwfb_roles:
        os.rename(rwfb_output, rwfb_input)
        print("✅ RWFB file updated")
    
    if cetras_roles:
        os.rename(cetras_output, cetras_input)
        print("✅ CETraS file updated")


if __name__ == "__main__":
    main()

