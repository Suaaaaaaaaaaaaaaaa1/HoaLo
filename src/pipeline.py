"""
Main pipeline runner — orchestrates: scrape → preprocess → sentiment → topics → visualize → strategy → email.
"""
import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STEPS = [
    ("scrape", "src/scraper.py", ["--output", "data/raw"]),
    ("preprocess", "src/preprocessing.py", ["--input", "data/raw", "--output", "data/processed"]),
    ("sentiment", "src/sentiment.py", ["--input", "data/processed", "--output", "data/results/sentiment"]),
    ("topics", "src/topic_modeling.py", ["--input", "data/processed", "--output", "data/results/topics"]),
    ("visualize", "src/visualize.py", ["--sentiment", "data/results/sentiment", "--topics", "data/results/topics", "--output", "reports/figures"]),
    ("strategy", "src/strategy.py", ["--raw", "data/raw", "--sentiment", "data/results/sentiment", "--topics", "data/results/topics", "--output", "reports/strategy_report.md"]),
    ("email", "src/email_sender.py", ["--report", "reports/strategy_report.md", "--figures", "reports/figures"]),
]


def run_step(name: str, script: str, args: list[str], config_path: str, dry_run: bool = False):
    cmd = [sys.executable, script, "--config", config_path] + args
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {name}")
    logger.info(f"CMD:  {' '.join(cmd)}")
    logger.info(f"{'='*60}")

    if dry_run:
        logger.info("[DRY RUN] Skipping execution")
        return True

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error(f"Step '{name}' failed with return code {result.returncode}")
        return False

    logger.info(f"Step '{name}' completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the full Hỏa Lò analysis pipeline")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--start-from", choices=[s[0] for s in STEPS], default=None)
    parser.add_argument("--stop-after", choices=[s[0] for s in STEPS], default=None)
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--only", choices=[s[0] for s in STEPS], default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    started = args.start_from is None
    for name, script, step_args in STEPS:
        if args.only and name != args.only:
            continue
        if not started:
            if name == args.start_from:
                started = True
            else:
                logger.info(f"Skipping step: {name}")
                continue
        if name in args.skip:
            logger.info(f"Skipping step: {name} (--skip)")
            continue

        success = run_step(name, script, step_args, args.config, args.dry_run)
        if not success:
            logger.error(f"Pipeline failed at step: {name}")
            sys.exit(1)

        if args.stop_after and name == args.stop_after:
            logger.info(f"Stopping after step: {name}")
            break

    logger.info("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
