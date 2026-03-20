"""
Generate AI Insights using Gemini API based on NLP analysis results.
Input: data/results/nlp_analysis_report.json
Output: reports/strategy_report.md
"""
import os
import json
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

import google.generativeai as genai

# Cấu hình logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_data(json_path: Path) -> dict:
    """Đọc dữ liệu từ file JSON."""
    if not json_path.exists():
        logger.error(f"Không tìm thấy file {json_path}")
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_prompt(data: dict) -> str:
    """Tạo prompt chi tiết để nạp vào Gemini."""
    
    # Rút trích các số liệu quan trọng
    total_posts = data.get("dataset", {}).get("total_posts", 0)
    sentiment = data.get("sentiment", {})
    pos_pct = sentiment.get("positive", {}).get("pct", 0)
    neg_pct = sentiment.get("negative", {}).get("pct", 0)
    
    topics = data.get("topics", {}).get("labels", {})
    best_topic_id = str(data.get("topics", {}).get("best_topic", ""))
    best_topic_name = topics.get(best_topic_id, "Không rõ")
    
    top_words = data.get("keywords", {}).get("top_20_words", [])
    words_str = ", ".join([f"{w} ({c})" for w, c in top_words[:10]])

    prompt = f"""
    Bạn là một chuyên gia Chiến lược Mạng xã hội (Social Media Strategist) hàng đầu. 
    Dưới đây là dữ liệu phân tích Fanpage Facebook "Di tích Lịch sử Nhà tù Hỏa Lò" của 1 tuần vừa qua.
    
    TÓM TẮT DỮ LIỆU (JSON data):
    - Tổng số bài post đã phân tích: {total_posts}
    - Phân bổ cảm xúc (Sentiment): Tích cực {pos_pct}%, Tiêu cực {neg_pct}%.
    - Chủ đề (Topic) thu hút tương tác tốt nhất: "{best_topic_name}"
    - Các từ khóa xuất hiện nhiều nhất: {words_str}
    
    YÊU CẦU:
    Hãy viết một bản Báo cáo Phân tích và Đề xuất Chiến lược Nội dung thật chuyên sâu, trình bày dưới định dạng Markdown để tôi gửi cho Ban Giám đốc. 
    
    Báo cáo cần có CẤU TRÚC 4 PHẦN như sau:
    1. **Tổng quan Hiệu suất (Executive Summary):** Đánh giá ngắn gọn tình hình chung của Fanpage dựa trên dữ liệu tuần qua.
    2. **Phân tích Sâu sắc (Deep-dive Insights):** Phân tích tại sao chủ đề "{best_topic_name}" lại hiệu quả. Đánh giá về tỷ lệ cảm xúc của người dùng.
    3. **Phân tích Từ khóa (Keyword Analysis):** Những từ khóa này phản ánh điều gì về sở thích của tệp khán giả Hỏa Lò hiện tại?
    4. **Chiến lược Nội dung 7 ngày tới (Next 7-Day Action Plan):** - Đưa ra 3-4 hành động/tuyến bài CỤ THỂ cần triển khai ngay trong tuần này. 
       - Trọng tâm vào việc kế thừa chủ đề đang hot và khắc phục các điểm yếu.
       - Gợi ý 1 ý tưởng bài viết chi tiết (Tiêu đề, hướng nội dung) để team Content làm ngay.
    
    NGUYÊN TẮC TỐI THƯỢNG (CẦN TUÂN THỦ NGHIÊM NGẶT): 
    - Giữ giọng văn chuyên nghiệp, khách quan, sâu sắc nhưng mang tính thực chiến cao.
    - TUYỆT ĐỐI KHÔNG tự sáng tạo, bịa đặt ra các chương trình khuyến mãi, giảm giá vé, sự kiện ảo, hoặc các loại tour mới không có thật. Bạn chỉ được phép nhắc đến "Đêm thiêng liêng" nếu nó xuất hiện trong từ khóa.
    - Các gợi ý ở phần 4 CHỈ ĐƯỢC PHÉP tập trung vào KHÍA CẠNH NỘI DUNG (cách viết bài, cách kể chuyện, khai thác góc nhìn mới về lịch sử, thay đổi định dạng bài đăng), tuyệt đối không đề xuất thay đổi về mặt vận hành, chính sách hay kinh doanh của di tích.
    - Không bịa đặt thêm số liệu ngoài các thông tin đã cung cấp.
    """
    return prompt

def generate_report(prompt: str, api_key: str, output_path: Path):
    """Gọi Gemini API và lưu kết quả ra file Markdown."""
    genai.configure(api_key=api_key)
    
    # Sử dụng model Gemini 1.5 Flash cho tốc độ nhanh và khả năng tổng hợp text tốt
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    logger.info("Đang gửi yêu cầu phân tích tới Gemini API...")
    try:
        response = model.generate_content(prompt)
        report_content = response.text
        
        # Lưu file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info(f"Đã tạo báo cáo thành công tại: {output_path}")
        
    except Exception as e:
        logger.error(f"Lỗi khi gọi Gemini API: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/results/nlp_analysis_report.json")
    parser.add_argument("--output", default="reports/strategy_report.md")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logger.error("Không tìm thấy GEMINI_API_KEY trong biến môi trường (.env). Dừng thực thi.")
        return

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = load_data(inp)
    if not data:
        return

    prompt = generate_prompt(data)
    generate_report(prompt, api_key, out)

if __name__ == "__main__":
    main()
