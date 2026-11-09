import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
META_DIR = PROJECT_ROOT / "data" / "openml_meta"
OUTPUT = PROJECT_ROOT / "configs" / "openml_suite.json"
API = "https://www.openml.org/api/v1/json"
STUDIES = {"classification": 99, "regression": 353}
MAX_ROWS = 2000
MIN_ROWS = 150
MAX_FEATURES = 60
MAX_CLASSES = 10


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def qualities_for(data_ids: list[int]) -> list[dict]:
    rows = []
    for start in range(0, len(data_ids), 50):
        chunk = ",".join(str(i) for i in data_ids[start:start + 50])
        listing = fetch_json(f"{API}/data/list/data_id/{chunk}/limit/1000")["data"]["dataset"]
        for item in listing:
            quality = {q["name"]: q["value"] for q in item.get("quality", [])}
            rows.append({"data_id": int(item["did"]), "name": item["name"], "version": int(item["version"]), **quality})
    return rows


def as_int(value) -> int:
    return int(float(value)) if value not in (None, "") else -1


def task_details(task_ids: list[int], study_id: int) -> dict[int, dict]:
    cache = META_DIR / f"tasks_{study_id}.json"
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text(encoding="utf-8")).items()}
    details = {}
    for task_id in task_ids:
        task = fetch_json(f"{API}/task/{task_id}")["task"]
        inputs = {item["name"]: item for item in task["input"]}
        procedure = inputs["estimation_procedure"]["estimation_procedure"]
        parameters = {p["name"]: p.get("value") for p in procedure.get("parameter", [])}
        data_id = int(inputs["source_data"]["data_set"]["data_set_id"])
        details[data_id] = {
            "task_id": int(task_id),
            "target": inputs["source_data"]["data_set"]["target_feature"],
            "procedure": procedure["type"],
            "folds": int(parameters.get("number_folds") or 0),
            "repeats": int(parameters.get("number_repeats") or 0),
            "splits_url": procedure.get("data_splits_url"),
        }
    cache.write_text(json.dumps(details, indent=2), encoding="utf-8")
    return details


def select(task: str, study_id: int) -> tuple[list[dict], list[dict]]:
    study = json.loads((META_DIR / f"study_{study_id}.json").read_text(encoding="utf-8"))["study"]
    task_ids = study["tasks"]["task_id"]
    data_ids = study["data"]["data_id"]
    rows = qualities_for(data_ids)
    (META_DIR / f"qualities_{study_id}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    task_by_data = task_details(task_ids, study_id)
    selected, rejected = [], []
    for row in sorted(rows, key=lambda r: r["data_id"]):
        n_rows = as_int(row.get("NumberOfInstances"))
        n_features = as_int(row.get("NumberOfFeatures")) - 1
        n_symbolic = as_int(row.get("NumberOfSymbolicFeatures")) - (1 if task == "classification" else 0)
        n_missing = as_int(row.get("NumberOfMissingValues"))
        n_classes = as_int(row.get("NumberOfClasses"))
        reasons = []
        if not MIN_ROWS <= n_rows <= MAX_ROWS:
            reasons.append(f"rows={n_rows}")
        if n_features > MAX_FEATURES:
            reasons.append(f"features={n_features}")
        if n_symbolic > 0:
            reasons.append(f"categorical_features={n_symbolic}")
        if n_missing > 0:
            reasons.append(f"missing_values={n_missing}")
        if task == "classification" and n_classes > MAX_CLASSES:
            reasons.append(f"classes={n_classes}")
        record = {
            "data_id": row["data_id"],
            **task_by_data[row["data_id"]],
            "name": row["name"],
            "version": row["version"],
            "rows": n_rows,
            "features": n_features,
            "classes": n_classes if task == "classification" else None,
        }
        (rejected if reasons else selected).append({**record, "excluded_because": reasons} if reasons else record)
    return selected, rejected


def main() -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    suite = {
        "criteria": {
            "source": "OpenML-CC18 (study 99) and OpenML-CTR23 (study 353)",
            "rows": [MIN_ROWS, MAX_ROWS],
            "max_features": MAX_FEATURES,
            "categorical_features": 0,
            "missing_values": 0,
            "max_classes": MAX_CLASSES,
        },
    }
    for task, study_id in STUDIES.items():
        selected, rejected = select(task, study_id)
        suite[task] = selected
        suite[f"{task}_excluded"] = rejected
        print(f"{task}: {len(selected)} selected, {len(rejected)} excluded")
        for item in selected:
            print(f"  {item['data_id']:>6} {item['name']:<40} rows={item['rows']:<5} features={item['features']:<3} classes={item['classes']}")
    OUTPUT.write_text(json.dumps(suite, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
