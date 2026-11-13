r"""
Preprocess the RCAEval TrainTicket RE3-TT dataset for use with UAM-AD.

Same pipeline, same file layout and same pkl format as preprocess_rcaeval_re3_ob.py
(this script reuses its RCAEvalRE3OBPreprocessor, including the Drain3 log templates);
only the service list and the fault types differ.

Dataset facts (verified on D:/RE3-TT/RE3-TT):
  - Scenarios: ts-auth-service_f1..f4 and ts-route-service_f1..f4 (code-level faults),
    plus ts-route-service_f3_1, which is skipped because its folder name does not end
    in a fault type
  - 3-4 runs per scenario (ts-auth-service has 4); the OB pipeline reads runs 1-3 only,
    so run 4 is not used
  - Each run: 1801 rows @ 1 s, fault injected at the middle (900 s before / 900 s after)
  - traces.csv holds 27 services (same list as RE2-TT)
  - logs.csv has raw messages (no template) -> Drain3, as in RE3-OB
  - traces.csv statusCode is empty everywhere, so trace feature 3 (error_rate) is always 0

Usage:
  python codes/common/preprocess_rcaeval_re3_tt.py \
      --data_root D:/RE3-TT/RE3-TT \
      --output_dir data/rcaeval_re3_tt
"""

import argparse

import preprocess_rcaeval_re3_ob as ob

TT_SERVICES = [
    "ts-admin-basic-info-service",
    "ts-admin-travel-service",
    "ts-assurance-service",
    "ts-auth-service",
    "ts-basic-service",
    "ts-config-service",
    "ts-consign-price-service",
    "ts-consign-service",
    "ts-contacts-service",
    "ts-food-map-service",
    "ts-food-service",
    "ts-inside-payment-service",
    "ts-order-other-service",
    "ts-order-service",
    "ts-payment-service",
    "ts-preserve-other-service",
    "ts-preserve-service",
    "ts-price-service",
    "ts-route-service",
    "ts-seat-service",
    "ts-security-service",
    "ts-station-service",
    "ts-ticketinfo-service",
    "ts-train-service",
    "ts-travel-service",
    "ts-travel2-service",
    "ts-user-service",
]

# The OB preprocessor reads these module-level names at call time.
ob.FAULT_TYPES = ["f1", "f2", "f3", "f4"]
ob.SERVICES = TT_SERVICES
ob.SERVICE2IDX = {s: i for i, s in enumerate(TT_SERVICES)}
ob.SERVICE_ALIASES = {}
ob.NUM_SERVICES = len(TT_SERVICES)


def main():
    p = argparse.ArgumentParser(
        description="Preprocess RCAEval TrainTicket RE3-TT -> UAM-AD pkl format"
    )
    p.add_argument("--data_root",     required=True,
                   help="Root dir of RE3-TT (contains ts-auth-service_f1/, ...)")
    p.add_argument("--output_dir",    default="../../data/rcaeval_re3_tt")
    p.add_argument("--anomaly_rate",  default=0.20, type=float)
    p.add_argument("--unlabel_ratio", default=0.80, type=float)
    p.add_argument("--val_ratio",     default=0.15, type=float)
    p.add_argument("--random_seed",   default=42, type=int)
    args = p.parse_args()

    ob.RCAEvalRE3OBPreprocessor(
        data_root     = args.data_root,
        output_dir    = args.output_dir,
        anomaly_rate  = args.anomaly_rate,
        unlabel_ratio = args.unlabel_ratio,
        val_ratio     = args.val_ratio,
        random_seed   = args.random_seed,
    ).run()


if __name__ == "__main__":
    main()
