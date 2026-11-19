import pandas as pd
import numpy as np
from typing import Dict, Any, List
from agent.audit_logger import AuditLogger

class RetailDiagnosticAdvisorSkill:
    """
    Skill 3: Phân tích chẩn đoán & Đề xuất chiến lược (Retail Diagnostic & Action Recommendation)
    - Định vị 3 cửa hàng cần ưu tiên hỗ trợ nhất dựa trên ma trận đa chiều:
      * Mức độ hoàn thành mục tiêu Doanh thu & Lợi nhuận
      * Biên lợi nhuận gộp thực tế so với chuẩn ngành
      * Tỷ lệ đổi trả thực tế so với ngưỡng trần kiểm soát (Max Return Rate)
      * Tăng trưởng so với kỳ trước (MoM Growth)
    - Giải thích bằng số liệu xác thực, không suy diễn nguyên nhân chắc chắn (Correlation vs Causation).
    - Đề xuất kế hoạch hành động cụ thể cho từng cửa hàng.
    """
    def __init__(self, logger: AuditLogger):
        self.logger = logger

    def diagnose_and_recommend(
        self,
        store_perf: pd.DataFrame,
        cat_perf: pd.DataFrame,
        store_cat_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        self.logger.log(
            step="Bước 4",
            skill="RetailDiagnosticAdvisorSkill",
            action="Bắt đầu phân tích ma trận sức khỏe cửa hàng & định vị Top 3 cửa hàng cần hỗ trợ",
            details={"total_stores_analyzed": len(store_perf)}
        )

        df = store_perf.copy()

        # Calculate Multi-criteria Priority Penalty Index (Điểm ưu tiên hỗ trợ)
        # Higher score = Higher urgency / Worse health
        # 1. Target Revenue Gap: 100 - Completion % (capped at 0)
        rev_gap = np.maximum(0, 100.0 - df["Target_Rev_Completion_Pct"])
        
        # 2. Target GP Gap: 100 - GP Completion %
        gp_gap = np.maximum(0, 100.0 - df["Target_GP_Completion_Pct"])
        
        # 3. Margin Gap: Min_Gross_Margin_Pct - Actual Gross_Margin_Pct
        margin_gap = np.maximum(0, df["Min_Gross_Margin_Pct"] - df["Gross_Margin_Pct"])
        
        # 4. Return Rate Excess: Actual Return_Rate_Pct - Max_Return_Rate_Pct
        return_excess = np.maximum(0, df["Return_Rate_Pct"] - df["Max_Return_Rate_Pct"])

        # Composite Urgency Score (Weighting: Rev Gap 30%, GP Gap 30%, Return Excess 25%, Margin Gap 15%)
        df["Priority_Urgency_Score"] = (
            (rev_gap * 0.30) + 
            (gp_gap * 0.30) + 
            (return_excess * 4.0 * 0.25) + 
            (margin_gap * 1.5 * 0.15)
        ).round(2)

        # Sort by urgency score descending
        df_ranked = df.sort_values(by="Priority_Urgency_Score", ascending=False).reset_index(drop=True)

        # Select Top 3 Stores needing priority intervention
        top_3_stores = df_ranked.head(3)

        store_analyses = []
        for idx, row in top_3_stores.iterrows():
            sid = str(row.get("Store_ID", f"CH_{idx+1}"))
            sname = str(row.get("Store_Name", sid))
            manager = str(row.get("Manager", "Cửa hàng trưởng"))
            city = str(row.get("City", "Toàn quốc"))
            
            # Category breakdown for this store
            sc_sub = store_cat_matrix[store_cat_matrix["Store_ID"] == sid]
            total_st_sales = sc_sub["Category_Net_Sales"].sum()
            sc_sub = sc_sub.copy()
            sc_sub["Share_Pct"] = (sc_sub["Category_Net_Sales"] / total_st_sales * 100).round(1) if total_st_sales > 0 else 0
            cat_breakdown_str = ", ".join([f"{r['Category_Name']} ({r['Share_Pct']}%)" for _, r in sc_sub.sort_values(by="Share_Pct", ascending=False).iterrows()])

            # Diagnose specific data-backed issues
            reasons = []
            hypotheses = []
            actions = []

            # Case: Low target completion
            if row["Target_Rev_Completion_Pct"] < 80.0:
                reasons.append(
                    f"Hụt mạnh mục tiêu doanh thu: Đạt {row['Target_Rev_Completion_Pct']}% kế hoạch (Doanh thu thuần: {row['Net_Revenue']:,.0f} đ / Chỉ tiêu: {row['Target_Revenue']:,.0f} đ, hụt {(row['Target_Revenue'] - row['Net_Revenue']):,.0f} đ)."
                )
                hypotheses.append(
                    "Số liệu ghi nhận số lượng giao dịch và lượng hàng bán ra thấp hơn đáng kể so với định mức diện tích cửa hàng. Dữ liệu thể hiện mối liên hệ giữa sự sụt giảm số lượng giao dịch và vị trí/lưu lượng khách, nhưng chưa thể khẳng định chắc chắn nguyên nhân là do thị trường hay vận hành nội bộ nếu chưa khảo sát thực địa."
                )
                actions.append("Rà soát lưu lượng khách (foot-traffic) theo khung giờ để bố trí lại ca trực và chương trình kích cầu cục bộ.")

            # Case: High Return Rate
            if row["Return_Rate_Pct"] > row["Max_Return_Rate_Pct"]:
                reasons.append(
                    f"Tỷ lệ hàng đổi trả ở mức báo động: {row['Return_Rate_Pct']}% (vượt ngưỡng kiểm soát tối đa {row['Max_Return_Rate_Pct']}%), tổng tiền hoàn trả lên tới {row['Return_Amount']:,.0f} đ với {row['Return_Count']} lượt trả."
                )
                hypotheses.append(
                    f"Dữ liệu đổi trả cho thấy tỷ lệ hoàn trả cao bất thường tập trung vào nhóm sản phẩm thời trang/kỹ thuật. Về mặt thống kê, điều này có mối liên hệ với việc phát sinh lô hàng lỗi hoặc tư vấn sai kích cỡ tại cửa hàng; tuy nhiên không thể kết luận ngay do nhân viên hay do nhà cung cấp mà cần bộ phận QC đối soát lô hàng."
                )
                actions.append("Thanh tra ngay chất lượng các lô hàng bị trả nhiều nhất và đào tạo lại quy trình tư vấn thông số kỹ thuật cho nhân viên bán hàng.")

            # Case: Low Gross Margin % or Margin deficit
            if row["Gross_Margin_Pct"] < row["Min_Gross_Margin_Pct"]:
                reasons.append(
                    f"Biên lợi nhuận gộp bị bào mòn nghiêm trọng: Đạt {row['Gross_Margin_Pct']}% (chuẩn yêu cầu tối thiểu là {row['Min_Gross_Margin_Pct']}%, thấp hơn {- (row['Gross_Margin_Pct'] - row['Min_Gross_Margin_Pct']):.1f}%)."
                )
                hypotheses.append(
                    f"Cơ cấu sản phẩm đang bị lệch trọng tâm: tỷ trọng sản phẩm biên lợi nhuận thấp chiếm tỷ lệ cao trong doanh thu tổng, cộng với chiết khấu giảm giá xả hàng làm co hẹp biên lãi gộp."
                )
                actions.append("Cơ cấu lại danh mục trưng bày (Visual Merchandising), thúc đẩy các gói combo bán kèm (cross-sell) nhóm hàng biên lợi nhuận cao.")

            # Fallback if metrics are relatively close
            if not reasons:
                reasons.append(f"Mức hoàn thành doanh thu đạt {row['Target_Rev_Completion_Pct']}%, biên lợi nhuận đạt {row['Gross_Margin_Pct']}%.")
                actions.append("Thiết lập mục tiêu tuần và theo dõi sát sao tiến độ bán lẻ.")

            store_analyses.append({
                "rank": idx + 1,
                "store_id": sid,
                "store_name": sname,
                "city": city,
                "manager": manager,
                "urgency_score": row["Priority_Urgency_Score"],
                "net_revenue": float(row["Net_Revenue"]),
                "target_revenue": float(row["Target_Revenue"]),
                "rev_completion_pct": float(row["Target_Rev_Completion_Pct"]),
                "gross_profit": float(row["Gross_Profit"]),
                "target_gp": float(row["Target_Gross_Profit"]),
                "gp_completion_pct": float(row["Target_GP_Completion_Pct"]),
                "gross_margin_pct": float(row["Gross_Margin_Pct"]),
                "return_rate_pct": float(row["Return_Rate_Pct"]),
                "mom_growth_pct": float(row["MoM_Revenue_Growth_Pct"]),
                "category_mix": cat_breakdown_str,
                "data_backed_findings": reasons,
                "causation_caveats": hypotheses,
                "action_recommendations": actions
            })

        top_stores_summary = {}
        for k in range(min(3, len(store_analyses))):
            top_stores_summary[f"top_{k+1}_store"] = f"{store_analyses[k]['store_id']} - {store_analyses[k]['store_name']}"

        self.logger.log(
            step="Bước 4",
            skill="RetailDiagnosticAdvisorSkill",
            action="Hoàn thành phân tích & lập đề xuất cho các cửa hàng ưu tiên",
            details=top_stores_summary
        )

        return {
            "top_3_stores": store_analyses,
            "ranked_all_stores": df_ranked[[
                "Store_ID", "Store_Name", "City", "Net_Revenue", "Gross_Profit", 
                "Gross_Margin_Pct", "Return_Rate_Pct", "Target_Rev_Completion_Pct", 
                "Target_GP_Completion_Pct", "Priority_Urgency_Score"
            ]].to_dict(orient="records")
        }

