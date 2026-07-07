"""
migrate_csv_v2.py
------------------------------------------------------------
ONE-TIME script, second migration. Your live_data.csv already
has 14 columns (from the first ML migration). This new version
of collector.py adds 3 more: flare_class, cme_alert, anomaly_flag.

Run this once, same way as migrate_csv.py:
    python3 migrate_csv_v2.py
    git add live_data.csv
    git commit -m "migrate: add flare/CME/anomaly columns"
    git push
------------------------------------------------------------
"""
import csv
import shutil
import os

IN_FILE = "live_data.csv"
BACKUP_FILE = "live_data.csv.backup2"
NEW_COLUMNS = ["flare_class", "cme_alert", "anomaly_flag"]


def main():
    if not os.path.exists(IN_FILE):
        print(f"{IN_FILE} not found in current directory. Run this from your repo root.")
        return

    shutil.copy(IN_FILE, BACKUP_FILE)
    print(f"Backed up original to {BACKUP_FILE}")

    with open(IN_FILE, "r", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        print("File is empty, nothing to migrate.")
        return

    header = rows[0]
    if all(col in header for col in NEW_COLUMNS):
        print("File already has the new columns. Nothing to do.")
        os.remove(BACKUP_FILE)
        return

    target_len = len(header) + len(NEW_COLUMNS)
    new_header = header + NEW_COLUMNS
    new_rows = [new_header]
    for row in rows[1:]:
        if len(row) < target_len:
            row = row + [""] * (target_len - len(row))
        new_rows.append(row)

    with open(IN_FILE, "w", newline="") as f:
        csv.writer(f).writerows(new_rows)

    print(f"Migrated {len(rows) - 1} existing rows.")
    print(f"New header ({len(new_header)} cols): {new_header}")
    print("Done. You can now deploy the updated collector.py + new modules safely.")


if __name__ == "__main__":
    main()
