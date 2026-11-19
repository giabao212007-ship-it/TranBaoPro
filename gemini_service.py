import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from agent.business_evidence_engine import BusinessEvidenceEngine

GEMINI_API_KEY = "AQ.Ab8RN6IFgMoBl29kDYt-te20zIL9gFs22U9iikQ6x0JeogO6uw"
MODELS = ["gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview"]

class GeminiRetailService:
    """
    Service integrating Google Gemini LLM for Retail Chain Analytics & Strategic Recommendations.
    Uses API Key: AQ.Ab8RN6IFgMoBl29kDYt-te20zIL9gFs22U9iikQ6x0JeogO6uw
    Enhanced with Business Evidence Engine inspired by DataLens Studio.
    """
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key
        self.evidence_engine = BusinessEvidenceEngine()

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None, timeout: int = 4) -> str:
        """Call Gemini API with automatic model fallback and timeout"""
        for model in MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            
            payload: Dict[str, Any] = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "topP": 0.8,
                    "maxOutputTokens": 2048
                }
            }
            if system_instruction:
                payload["systemInstruction"] = {
                    "parts": [{"text": system_instruction}]
                }

            try:
                data_bytes = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=data_bytes,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    candidates = resp_json.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            text = parts[0].get("text", "").strip()
                            if text:
                                return text
            except urllib.error.HTTPError as he:
                print(f"[GeminiService] Model {model} HTTP {he.code}: {he.reason}")
                continue
            except Exception as e:
                print(f"[GeminiService] Model {model} request failed: {e}")
                if "11001" in str(e) or "Name or service not known" in str(e):
                    break
                continue

        # Deterministic fallback if network is unreachable or quota exceeded
        return ""

    def generate_store_strategic_advice(self, store_data: Dict[str, Any], company_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses Gemini to generate executive analysis for a prioritized store,
        rigorously enforcing correlation vs causation rules.
        """
        system_prompt = (
            "Bạn là Chuyên gia Cố vấn Chiến lược Cấp cao (Enterprise AI Strategic Advisor) cho dự án Trần Bảo Project. "
            "Quy tắc tối thượng:\n"
            "1. Phân tích hoàn toàn dựa trên dữ liệu định lượng được cung cấp, không suy diễn nguyên nhân chắc chắn khi dữ liệu chỉ thể hiện mối liên hệ tương quan.\n"
            "2. Phân biệt rõ giữa Correlation (Tương quan) và Causation (Nhân quả).\n"
            "3. Đề xuất hành động phải mang tính khả thi cao, chia theo mốc 30 ngày cho quản lý chi nhánh và ban giám đốc.\n"
            "4. Giọng văn sắc bén, chuyên nghiệp, chuẩn phong cách báo cáo quản trị cấp C-level."
        )

        user_prompt = f"""
Hãy phân tích chuyên sâu cho Cửa hàng {store_data.get('store_id')} - {store_data.get('store_name')} ({store_data.get('city')}).
Quản lý cửa hàng: {store_data.get('manager')}

Số liệu hiệu quả kinh doanh tháng 10/2026:
- Doanh thu thuần: {store_data.get('net_revenue', 0):,.0f} đ / Chỉ tiêu: {store_data.get('target_revenue', 0):,.0f} đ (Đạt {store_data.get('rev_completion_pct', 0):.1f}%)
- Lợi nhuận gộp: {store_data.get('gross_profit', 0):,.0f} đ / Chỉ tiêu: {store_data.get('target_gp', 0):,.0f} đ (Đạt {store_data.get('gp_completion_pct', 0):.1f}%)
- Biên lợi nhuận gộp thực tế: {store_data.get('gross_margin_pct', 0):.1f}% (Chuẩn tối thiểu yêu cầu: 36.0%)
- Tỷ lệ hàng đổi trả: {store_data.get('return_rate_pct', 0):.2f}% (Ngưỡng kiểm soát tối đa: 3.0%)
- Tăng trưởng doanh thu MoM: {store_data.get('mom_growth_pct', 0):+.1f}% so với tháng trước
- Cơ cấu nhóm sản phẩm đóng góp: {store_data.get('category_mix', 'N/A')}
- Điểm khẩn cấp ưu tiên: {store_data.get('urgency_score', 0):.1f}

Bối cảnh toàn chuỗi 18 cửa hàng:
- Tổng doanh thu thuần chuỗi: {company_summary.get('total_net_revenue', 0):,.0f} đ
- Biên lợi nhuận gộp bình quân chuỗi: {company_summary.get('overall_gross_margin_pct', 0):.1f}%
- Tỷ lệ đổi trả bình quân chuỗi: {company_summary.get('overall_return_rate_pct', 0):.2f}%

Yêu cầu định dạng đầu ra (JSON hợp lệ):
{{
  "executive_summary": "1-2 câu tóm tắt bản chất vấn đề của cửa hàng này",
  "data_driven_findings": ["3 phát hiện cốt lõi từ số liệu"],
  "correlation_vs_causation_caveats": ["2 điểm cần kiểm chứng thực địa, không quy chụp nguyên nhân chủ quan"],
  "action_plan_30_days": ["3 hành động cụ thể để cải thiện tình hình"]
}}
Chỉ trả về JSON thuần túy, không có backticks hay markdown.
"""
        raw_res = self.generate_text(user_prompt, system_prompt)
        if raw_res:
            try:
                clean_json = raw_res.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json", 1)[1].split("```", 1)[0]
                elif "```" in clean_json:
                    clean_json = clean_json.split("```", 1)[1].split("```", 1)[0]
                
                start_b = clean_json.find("{")
                end_b = clean_json.rfind("}")
                if start_b != -1 and end_b != -1:
                    clean_json = clean_json[start_b:end_b+1]

                return json.loads(clean_json, strict=False)
            except Exception as ex:
                pass

        # Fallback structured data if offline
        return {
            "executive_summary": f"Cửa hàng {store_data.get('store_id')} gặp vấn đề nghiêm trọng về độ lệch mục tiêu hoặc tỷ lệ đổi trả, cần sự can thiệp từ ban lãnh đạo.",
            "data_driven_findings": store_data.get("data_backed_findings", []),
            "correlation_vs_causation_caveats": store_data.get("causation_caveats", []),
            "action_plan_30_days": store_data.get("action_recommendations", [])
        }

    def answer_copilot_question(self, question: str, context_data: Dict[str, Any]) -> str:
        """
        Interactive Q&A Copilot for the Sales Director.
        Uses Business Evidence Engine inspired by DataLens Studio with Gemini LLM reasoning.
        """
        # 1. Synthesize quantitative evidence dossier from context data
        dossier = self.evidence_engine.extract_evidence(question, context_data)
        findings_str = "\n".join([f"• {f}" for f in dossier.get("findings", [])])
        caveats_str = "\n".join([f"• {c}" for c in dossier.get("caveats", [])])
        recs_str = "\n".join([f"• {r.get('action')} ({r.get('priority')}): {r.get('reason')} [Đo lường: {r.get('measure')}]" for r in dossier.get("recommendations", [])])

        system_prompt = (
            "Bạn là Chuyên gia Cố vấn Chiến lược Cấp cao (Trần Bảo AI Copilot) tích hợp Data Intelligence của Trần Bảo Project. "
            "Bạn đang cố vấn trực tiếp cho Ban Giám đốc và các Quản lý doanh nghiệp. "
            "Quy tắc phản hồi chuẩn C-Level (Executive Briefing):\n"
            "1. Tuyệt đối bám sát BẰNG CHỨNG SỐ LIỆU ĐỊNH LƯỢNG được trích xuất từ dữ liệu hoạt động thực tế.\n"
            "2. Luôn phân định rõ ràng giữa Correlation (Tương quan thống kê) và Causation (Nhân quả) - không quy chụp nguyên nhân chủ quan khi chưa đối chứng thực tế.\n"
            "3. Bố cục câu trả lời sắc bén, chuyên nghiệp với 3 phần rõ ràng:\n"
            "   📌 1. Bằng Chứng & Số Liệu Thực Chứng (Empirical Evidence)\n"
            "   🔍 2. Phân Tích Nguyên Nhân & Giới Hạn Nhân Quả (Causal Caveats)\n"
            "   🎯 3. Kế Hoạch Hành Động Khuyến Nghị 30 Ngày (Prioritized Action Plan)\n"
            "4. Định dạng Markdown chỉn chu, gạch đầu dòng rõ ràng, số liệu in đậm, văn phong chuẩn phong thái báo cáo Giám đốc điều hành."
        )

        user_prompt = f"""
CÂU HỎI QUẢN TRỊ CỦA BAN GIÁM ĐỐC:
"{question}"

BẰNG CHỨNG SỐ LIỆU THỰC CHỨNG TỔNG HỢP TỪ DỮ LIỆU:
{findings_str}

GIỚI HẠN NHÂN QUẢ & CÁC YẾU TỐ CẦN ĐỐI CHỨNG:
{caveats_str}

KHUYẾN NGHỊ HÀNH ĐỘNG DỰ THẢO:
{recs_str}

SNAPSHOT CHỈ SỐ TOÀN CHUỖI:
- Doanh thu thuần: {dossier.get('metrics_snapshot', {}).get('total_net_revenue', 0):,.0f} đ
- Doanh thu gộp: {dossier.get('metrics_snapshot', {}).get('total_gross_sales', 0):,.0f} đ
- Lợi nhuận gộp: {dossier.get('metrics_snapshot', {}).get('total_gross_profit', 0):,.0f} đ
- Biên lợi nhuận gộp: {dossier.get('metrics_snapshot', {}).get('gross_margin_pct', 0):.1f}%
- Tỷ lệ đổi trả: {dossier.get('metrics_snapshot', {}).get('return_rate_pct', 0):.2f}%

HÃY CUNG CẤP BÁO CÁO CỐ VẤN CHIẾN LƯỢC TOÀN DIỆN, SẮC BÉN VÀ CÓ TÍNH HÀNH ĐỘNG CAO CHO CÂU HỎI TRÊN:
"""
        ans = self.generate_text(user_prompt, system_prompt, timeout=12)
        if ans and len(ans.strip()) > 50:
            return ans

        # Deterministic evidence engine fallback if API is unreachable or rate limited
        return self.evidence_engine.format_dossier_to_markdown(dossier)

    def revise_store_strategic_advice(self, feedback: str, top_3_stores: List[Dict[str, Any]], company_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Revises the strategic brief for top hotspot stores based on executive natural language feedback.
        Inspired by DataLens Studio's revision pipeline.
        """
        prompt = f"""
BAN GIÁM ĐỐC YÊU CẦU ĐIỀU CHỈNH CHIẾN LƯỢC:
"{feedback}"

THÔNG TIN TOP 3 ĐIỂM NÓNG HIỆN TẠI:
{json.dumps(top_3_stores, ensure_ascii=False, indent=2)}

BỐI CẢNH TOÀN CHUỖI:
- Doanh thu thuần: {company_summary.get('total_net_revenue', 0):,.0f} đ
- Biên lãi gộp: {company_summary.get('overall_gross_margin_pct', 0):.1f}%
- Tỷ lệ đổi trả: {company_summary.get('overall_return_rate_pct', 0):.2f}%

YÊU CẦU:
Hãy điều chỉnh và tinh chỉnh lại kế hoạch hành động 30 ngày cho từng cửa hàng trong top 3 theo sát chỉ đạo của Ban Giám đốc.
Trả về định dạng JSON mảng 3 phần tử tương ứng với 3 cửa hàng:
[
  {{
    "store_id": "Mã cửa hàng",
    "executive_summary": "Tóm tắt chiến lược đã điều chỉnh theo chỉ đạo của Sếp",
    "data_driven_findings": ["3 phát hiện số liệu"],
    "correlation_vs_causation_caveats": ["2 giới hạn nhân quả"],
    "action_plan_30_days": ["3 hành động cụ thể theo chỉ đạo"]
  }}
]
Chỉ trả về JSON thuần túy, không có giải thích thêm.
"""
        raw = self.generate_text(prompt, "Bạn là Cố vấn Chiến lược Cấp cao C-Level của Trần Bảo Project. Hãy cập nhật kế hoạch hành động theo đúng chỉ thị.")
        if raw:
            try:
                clean = raw.strip()
                if "```json" in clean:
                    clean = clean.split("```json", 1)[1].split("```", 1)[0]
                elif "```" in clean:
                    clean = clean.split("```", 1)[1].split("```", 1)[0]
                start_b = clean.find("[")
                end_b = clean.rfind("]")
                if start_b != -1 and end_b != -1:
                    clean = clean[start_b:end_b+1]
                revised_list = json.loads(clean, strict=False)
                if isinstance(revised_list, list) and len(revised_list) > 0:
                    for i, st in enumerate(top_3_stores):
                        if i < len(revised_list):
                            st["ai_enhanced_analysis"] = revised_list[i]
                            st["recommended_actions"] = revised_list[i].get("action_plan_30_days", st.get("recommended_actions", []))
                    return top_3_stores
            except Exception as e:
                print(f"[GeminiService] Revise parse error: {e}")
        
        # Local fallback if offline
        for st in top_3_stores:
            st["ai_enhanced_analysis"] = {
                "executive_summary": f"Chiến lược cho {st.get('store_id')} đã được cập nhật bám sát chỉ đạo: '{feedback}'.",
                "data_driven_findings": st.get("data_backed_findings", []),
                "correlation_vs_causation_caveats": st.get("causation_caveats", []),
                "action_plan_30_days": [f"Chỉ đạo điều hành: {feedback}"] + st.get("recommended_actions", [])[:2]
            }
        return top_3_stores

