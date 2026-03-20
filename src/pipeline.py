"""Pipeline runner: scrape → clean → nlp_analysis → email"""
import os
import sys
import logging
import argparse
import subprocess

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STEPS = [
    ("scrape", "src/scraper.py", ["--output", "data/raw"]),
    ("clean", "src/clean_data.py", ["--input", "data/raw", "--output", "data/cleaned"]),
    ("nlp", "src/nlp_analysis.py", ["--input", "data/cleaned", "--output", "data/results", "--figures", "reports/figures"]),
    ("insights", "src/generate_insights.py", ["--input", "data/results/nlp_analysis_report.json", "--output", "reports/strategy_report.md"]),
    ("email", "src/email_sender.py", ["--report", "reports/strategy_report.md", "--figures", "reports/figures"]),
]


def run_step(name, script, args, config, dry_run=False):
    cmd = [sys.executable, script, "--config", config] + args
    logger.info(f"\n{'=' * 60}\nSTEP: {name}\n{'=' * 60}")
    if dry_run:
        logger.info("[DRY RUN]")
        return True
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"Step '{name}' failed")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--start-from", choices=[s[0] for s in STEPS])
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--only", choices=[s[0] for s in STEPS])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started = args.start_from is None
    for name, script, step_args in STEPS:
        if args.only and name != args.only:
            continue
        if not started:
            if name == args.start_from:
                started = True
            else:
                continue
        if name in args.skip:
            continue
        if not run_step(name, script, step_args, args.config, args.dry_run):
            sys.exit(1)
    logger.info("\nPipeline complete!")


if __name__ == "__main__":
    main()
