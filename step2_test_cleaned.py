from pathlib import Path
import json

out_file = Path("./data/customer1/processed/clean.jsonl")

# 1️⃣ Check the file exists
if not out_file.exists():
    raise FileNotFoundError(f"❌ {out_file} not found!")

# 2️⃣ Check it’s not empty
if out_file.stat().st_size == 0:
    raise ValueError(f"❌ {out_file} is empty!")

# 3️⃣ Check a few sample lines
with out_file.open("r", encoding="utf-8") as f:
    first_line = f.readline()
    second_line = f.readline()

print("✅ File exists and is not empty.")
print("\nFirst two records:\n")
print(first_line.strip()[:400])
print(second_line.strip()[:400])

# 4️⃣ Optional: count total records
count = sum(1 for _ in open(out_file, "r", encoding="utf-8"))
print(f"\n📦 Total records: {count}")