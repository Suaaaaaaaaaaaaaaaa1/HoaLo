"""
Email sender for delivering analysis reports and visualizations.
"""
import os
import ssl
import logging
import argparse
import smtplib
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

import yaml
import markdown
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_email_body_html(report_path: Path) -> str:
    """Đọc file Markdown và chuyển đổi thành HTML với CSS cơ bản để hiển thị đẹp trên Gmail."""
    if report_path.exists():
        content = report_path.read_text(encoding="utf-8")
        # Chuyển đổi Markdown sang HTML
        html_content = markdown.markdown(content, extensions=['extra', 'tables'])
        
        # Template HTML bọc bên ngoài báo cáo
        html_template = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, Helvetica, sans-serif; line-height: 1.6; color: #333333; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                h1 {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; font-size: 24px; }}
                h2 {{ color: #2980b9; margin-top: 25px; }}
                ul, ol {{ margin-bottom: 20px; }}
                li {{ margin-bottom: 8px; }}
                strong {{ color: #d35400; }}
                .header-info {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin-bottom: 25px; }}
                .footer {{ margin-top: 40px; font-size: 0.85em; color: #7f8c8d; border-top: 1px solid #eeeeee; padding-top: 15px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header-info">
                <p style="margin: 0 0 10px 0;"><strong>Xin chào,</strong></p>
                <p style="margin: 0 0 5px 0;">Dưới đây là <b>Báo cáo Phân tích Chiến lược Facebook - Di tích Nhà tù Hỏa Lò</b> được tổng hợp tự động bởi hệ thống AI.</p>
                <p style="margin: 0; font-size: 0.9em; color: #555;"><i>Thời gian tạo báo cáo: {datetime.now().strftime('%d/%m/%Y %H:%M')}</i></p>
            </div>
            
            <div class="report-content">
                {html_content}
            </div>
            
            <div class="footer">
                <p>Đây là email tự động từ hệ thống Hoa Lo Facebook Analysis Pipeline.<br>Vui lòng xem các biểu đồ chi tiết trong file đính kèm.</p>
            </div>
        </body>
        </html>
        """
        return html_template
        
    return "<p>Báo cáo phân tích đính kèm. Vui lòng kiểm tra file attachment.</p>"


def attach_file(msg: MIMEMultipart, filepath: Path):
    if not filepath.exists():
        logger.warning(f"Attachment not found: {filepath}")
        return

    with open(filepath, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={filepath.name}")
    msg.attach(part)
    logger.info(f"Attached: {filepath.name}")


def send_report(config: dict, report_path: Path, figures_dir: Path):
    email_cfg = config.get("email", {})

    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    recipients_str = os.getenv("EMAIL_RECIPIENTS", "")

    if not sender or not password or not recipients_str:
        logger.error("Email credentials not set. Set EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENTS in .env")
        return False

    recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]

    msg = MIMEMultipart()
    prefix = email_cfg.get("subject_prefix", "[Hoa Lo Analysis]")
    msg["Subject"] = f"{prefix} Báo cáo Chiến lược AI - {datetime.now().strftime('%d/%m/%Y')}"
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    # Đính kèm nội dung HTML vào thân email
    html_body = build_email_body_html(report_path)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    # Vẫn giữ lại file .md đính kèm phòng trường hợp cần lưu trữ gốc
    if email_cfg.get("attach_report", True) and report_path.exists():
        attach_file(msg, report_path)

    # Đính kèm các biểu đồ hình ảnh
    if email_cfg.get("attach_figures", True) and figures_dir.exists():
        for fig_path in sorted(figures_dir.glob("*.png")):
            attach_file(msg, fig_path)

    smtp_host = email_cfg.get("smtp_host", "smtp.gmail.com")
    smtp_port = email_cfg.get("smtp_port", 587)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            if email_cfg.get("use_tls", True):
                server.starttls(context=context)
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        logger.info(f"Report sent to {', '.join(recipients)}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Send analysis report via email")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--report", default="reports/strategy_report.md")
    parser.add_argument("--figures", default="reports/figures")
    parser.add_argument("--dry-run", action="store_true", help="Print email content without sending")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    report_path = Path(args.report)
    figures_dir = Path(args.figures)

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        body = build_email_body_html(report_path)
        print(body)
        if figures_dir.exists():
            figs = list(figures_dir.glob("*.png"))
            logger.info(f"Would attach {len(figs)} figure(s)")
        return

    send_report(config, report_path, figures_dir)


if __name__ == "__main__":
    main()
