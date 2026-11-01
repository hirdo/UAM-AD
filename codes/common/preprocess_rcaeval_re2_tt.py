r"""
Preprocess the RCAEval TrainTicket RE2-TT dataset for use with UAM-AD.

Same pipeline, same file layout and same pkl format as preprocess_rcaeval_re2_ob.py
(this script reuses its RCAEvalOBPreprocessor); only the service list differs.

Dataset facts (verified on D:/RE2-TT/RE2-TT):
  - 30 scenarios: 5 injected services (ts-auth/order/route/train/travel-service)
                  x 6 fault types (cpu, delay, disk, loss, mem, socket), 3 runs each
  - traces.csv holds 27 services in total (the union over all runs; one run has 20)
  - simple_metrics.csv has ~370 columns (cpu/mem/net/... per service and per mongo)
  - logs.csv has NO log_template column (only message, level, req_path, error), so the
    RE2-OB fallback applies: container_name is used as the log token
  - traces.csv statusCode is empty everywhere, so trace feature 3 (error_rate) is always 0

Usage:
  python codes/common/preprocess_rcaeval_re2_tt.py \
      --data_root D:/RE2-TT/RE2-TT \
      --output_dir data/rcaeval_re2_tt
"""

import argparse

import preprocess_rcaeval_re2_ob as ob

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
ob.SERVICES = TT_SERVICES
ob.SERVICE2IDX = {s: i for i, s in enumerate(TT_SERVICES)}
ob.SERVICE_ALIASES = {}
ob.NUM_SERVICES = len(TT_SERVICES)


def main():
    p = argparse.ArgumentParser(
        description="Preprocess RCAEval TrainTicket RE2-TT -> UAM-AD pkl format"
    )
    p.add_argument("--data_root",     required=True,
                   help="Root dir of RE2-TT (contains ts-auth-service_cpu/, ...)")
    p.add_argument("--output_dir",    default="../../data/rcaeval_re2_tt")
    p.add_argument("--anomaly_rate",  default=0.20, type=float)
    p.add_argument("--unlabel_ratio", default=0.80, type=float)
    p.add_argument("--val_ratio",     default=0.15, type=float)
    p.add_argument("--random_seed",   default=42, type=int)
    args = p.parse_args()

    ob.RCAEvalOBPreprocessor(
        data_root     = args.data_root,
        output_dir    = args.output_dir,
        anomaly_rate  = args.anomaly_rate,
        unlabel_ratio = args.unlabel_ratio,
        val_ratio     = args.val_ratio,
        random_seed   = args.random_seed,
    ).run()


if __name__ == "__main__":
    main()
