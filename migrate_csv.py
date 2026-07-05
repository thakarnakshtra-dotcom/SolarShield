"""
migrate_csv.py
------------------------------------------------------------
ONE-TIME script. Your existing live_data.csv has 12 columns
(no ML fields yet). The updated collector.py writes 14 columns
(adds ml_storm_probability, ml_alert). Run this once to add
the new columns (blank for all your existing historical rows,
since they were collected before the ML model existed) so the
file format is consistent going forward.

Run this locally in your repo folder, then commit the result:
    python3 migrate_csv.py
    git add live_data.csv
    git commit -m "migrate: add ml_storm_probability, ml_alert columns"
    git push
------------------------------------------------------------
"""
import csv
import shutil
import os

IN_FILE = "live_data.csv"
BACKUP_FILE = "live_data.csv.backup"

NEW_COLUMNS = ["ml_storm_probability", "ml_alert"]


def main():
    if not os.path.exists(IN_FILE):
        print(f"{IN_FILE} not found in current directory. Run this from your repo root.")
        return

    shutil.copy(IN_FILE, BACKUP_FILE)
    print(f"Backed up original to {BACKUP_FILE} (just in case)")

    with open(IN_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("File is empty, nothing to migrate.")
        return

    header = rows[0]
    if all(col in header for col in NEW_COLUMNS):
        print("File already has the new columns. Nothing to do.")
        os.remove(BACKUP_FILE)
        return

    new_header = header + NEW_COLUMNS
    new_rows = [new_header]
    for row in rows[1:]:
        new_rows.append(row + [""] * len(NEW_COLUMNS))

    with open(IN_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    print(f"Migrated {len(rows) - 1} existing rows.")
    print(f"New header: {new_header}")
    print("Done. You can now deploy the updated collector.py safely.")


if __name__ == "__main__":
    main()
