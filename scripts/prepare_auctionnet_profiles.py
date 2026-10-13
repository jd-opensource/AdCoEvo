#!/usr/bin/env python3
"""Stream official General/Sparse period archives into calibration assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

REQUIRED = [
    "deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex", "budget",
    "CPAConstraint", "timeStepIndex", "remainingBudget", "pValue", "pValueSigma",
    "bid", "xi", "cost", "isExposed", "conversionAction", "leastWinningCost",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_archive(name: str, archive: Path, out: Path) -> dict:
    result = {"profile": name, "archive": str(archive), "archive_sha256": sha256(archive), "periods": {}}
    with zipfile.ZipFile(archive) as zf:
        members = {Path(item.filename).name: item for item in zf.infolist()
                   if item.filename.endswith(".csv") and not item.filename.startswith("__MACOSX")}
        for period in (7, 8):
            member = members.get(f"period-{period}.csv")
            if member is None:
                raise ValueError(f"{archive} is missing period-{period}.csv")
            stats = _profile_member(zf, member, period)
            output = out / f"{name}_period_{period}_agg.csv"
            _write_aggregate(output, stats.pop("aggregate"))
            stats["aggregate"] = str(output)
            stats["aggregate_sha256"] = sha256(output)
            stats["member"] = member.filename
            stats["member_crc32"] = f"{member.CRC:08x}"
            stats["member_uncompressed_bytes"] = member.file_size
            result["periods"][str(period)] = stats
    first, second = result["periods"]["7"], result["periods"]["8"]
    for key in ("budgets", "cpa_constraints", "categories"):
        if first[key] != second[key]:
            raise ValueError(f"{name} {key} changes between periods 7 and 8")
    result["budgets"] = first["budgets"]
    result["cpa_constraints"] = first["cpa_constraints"]
    result["categories"] = first["categories"]
    result["total_budget"] = sum(first["budgets"])
    return result


def _profile_member(zf: zipfile.ZipFile, member: zipfile.ZipInfo, expected_period: int) -> dict:
    agg = defaultdict(lambda: {"rows": 0, "cost_step": 0.0, "conv_step": 0,
                               "value_step": 0.0, "lwc_sum": 0.0, "pvalue_sum": 0.0,
                               "bid_sum": 0.0, "xi_sum": 0.0})
    rows = exposures = conversions = 0
    pvalue_sum = pvalue_sigma_sum = exposed_value = cost = lwc_sum = 0.0
    budgets, cpas, categories = {}, {}, {}
    with zf.open(member) as raw:
        reader = csv.reader(line.decode("utf-8") for line in raw)
        header = next(reader)
        missing = set(REQUIRED) - set(header)
        if missing:
            raise ValueError(f"{member.filename} is missing columns: {sorted(missing)}")
        ix = {column: header.index(column) for column in REQUIRED}
        for row in reader:
            rows += 1
            period = int(float(row[ix["deliveryPeriodIndex"]]))
            if period != expected_period:
                raise ValueError(f"unexpected period {period} in {member.filename}")
            advertiser = int(float(row[ix["advertiserNumber"]]))
            tick = int(float(row[ix["timeStepIndex"]]))
            category = int(float(row[ix["advertiserCategoryIndex"]]))
            budget = float(row[ix["budget"]]); cpa = float(row[ix["CPAConstraint"]])
            pvalue = float(row[ix["pValue"]]); sigma = float(row[ix["pValueSigma"]])
            exposed = int(float(row[ix["isExposed"]])); conversion = int(float(row[ix["conversionAction"]])) * exposed
            row_cost = float(row[ix["cost"]]) * exposed
            lwc = float(row[ix["leastWinningCost"]]); bid = float(row[ix["bid"]]); xi = float(row[ix["xi"]])
            budgets[advertiser] = budget; cpas[advertiser] = cpa; categories[advertiser] = category
            exposures += exposed; conversions += conversion; pvalue_sum += pvalue
            pvalue_sigma_sum += sigma; exposed_value += pvalue * exposed; cost += row_cost; lwc_sum += lwc
            item = agg[(advertiser, tick)]; item["rows"] += 1; item["cost_step"] += row_cost
            item["conv_step"] += conversion; item["value_step"] += pvalue * exposed
            item["lwc_sum"] += lwc; item["pvalue_sum"] += pvalue
            item["bid_sum"] += bid; item["xi_sum"] += xi
    if set(budgets) != set(range(48)) or len(agg) != 48 * 48:
        raise ValueError(f"{member.filename} does not contain 48 advertisers x 48 ticks")
    compact = []
    for (advertiser, tick), item in sorted(agg.items()):
        count = item["rows"]
        compact.append({"period": expected_period, "advertiserNumber": advertiser,
                        "timeStepIndex": tick, "category": categories[advertiser],
                        "budget": budgets[advertiser], "cpa_constraint": cpas[advertiser],
                        "cost_step": item["cost_step"], "conv_step": item["conv_step"],
                        "value_step": item["value_step"], "lwc_mean": item["lwc_sum"] / count,
                        "pvalue_mean": item["pvalue_sum"] / count,
                        "bid_mean": item["bid_sum"] / count, "win_rate": item["xi_sum"] / count,
                        "volume": count})
    return {"rows": rows, "exposures": exposures, "conversions": conversions,
            "cost": cost, "pvalue_mean": pvalue_sum / rows,
            "pvalue_sigma_mean": pvalue_sigma_sum / rows,
            "exposed_pvalue_sum": exposed_value, "least_winning_cost_mean": lwc_sum / rows,
            "budgets": [budgets[index] for index in range(48)],
            "cpa_constraints": [cpas[index] for index in range(48)],
            "categories": [categories[index] for index in range(48)], "aggregate": compact}


def _write_aggregate(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--general", type=Path, default=Path("autoBidding_aigb_track_data_period_7-8.zip"))
    parser.add_argument("--sparse", type=Path, default=Path("autoBidding_aigb_track_final_data_period_7-8.zip"))
    parser.add_argument("--out", type=Path, default=Path("data/auctionnet_profiles"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    profiles = {name: profile_archive(name, archive, args.out)
                for name, archive in (("general", args.general), ("sparse", args.sparse))}
    manifest = {"schema_version": "auctionnet-profile-calibration-v1", "profiles": profiles}
    path = args.out / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
