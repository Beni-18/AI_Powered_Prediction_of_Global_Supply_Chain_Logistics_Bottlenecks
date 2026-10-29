"""Command line wrapper around common/contracts.py.

Usage:
    python scripts/validate_contract.py <schema_name> <csv_path>     check one file
    python scripts/validate_contract.py --self-test                  check every example file in contracts/fixtures/
    python scripts/validate_contract.py --list                       list schema names

Exit code 0 = file is valid, 1 = at least one rule is broken.
"""

import sys
from pathlib import Path

# Make "import common" work when this file is run as a plain script from any folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.contracts import validate_csv  # noqa: E402
from common.paths import FIXTURES_DIR, SCHEMAS_DIR  # noqa: E402


def _print_report(report, label: str) -> None:
    """Print a short human readable result for one file."""
    status = "PASS" if report.ok else "FAIL"
    print(f"{status}  {label}  ({report.n_rows} rows, schema '{report.schema}')")
    for violation in report.violations:
        print(f"      - {violation}")
    if report.extra_columns:
        print(f"      note: extra columns not in the contract: {report.extra_columns}")


def main(argv: list) -> int:
    if len(argv) == 2 and argv[1] == "--list":
        for p in sorted(SCHEMAS_DIR.glob("*.schema.json")):
            print(p.name.replace(".schema.json", ""))
        return 0

    if len(argv) == 2 and argv[1] == "--self-test":
        # Every fixture file is named after the schema it must satisfy.
        failed = 0
        for csv_path in sorted(FIXTURES_DIR.glob("*.csv")):
            report = validate_csv(csv_path.stem, csv_path)
            _print_report(report, csv_path.name)
            failed += 0 if report.ok else 1
        return 1 if failed else 0

    if len(argv) != 3:
        print(__doc__)
        return 2

    schema_name, csv_path = argv[1], argv[2]
    report = validate_csv(schema_name, csv_path)
    _print_report(report, csv_path)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
