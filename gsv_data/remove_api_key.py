"""
Removes expired API key from all URL columns in gsv_data CSVs.
Will need to add &key=[key] back in to use the URLs
"""

import re
from pathlib import Path

data_dir = Path(__file__).parent
csv_files = list(data_dir.glob("*.csv"))

pattern = re.compile(r'&key=[^"]+')

for path in csv_files:
    text = path.read_text()
    cleaned, count = pattern.subn("", text)
    if count:
        path.write_text(cleaned)
        print(f"  cleaned ({count} replacements): {path.name}")
    else:
        print(f"  skipped (no key found): {path.name}")

print("Done.")
