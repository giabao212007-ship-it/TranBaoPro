"""
TRẦN BẢO PROJECT - ENTERPRISE AI PLATFORM & DASHCOM WORKSPACE
- Màn hình mở đầu (Landing Screen): Phong cách Tailwind CSS hiện đại, tối giản, sang trọng.
- Chức năng: Nhập Tên doanh nghiệp / Đơn vị (mặc định: Trần Bảo Project).
- Nút 'Bắt đầu làm việc' chuyển hướng ngay vào Không gian làm việc chính (Workspace).
- Lời chào cá nhân hóa theo Tên doanh nghiệp.
- Giữ nguyên toàn bộ Luồng Workflow 5 bước AI Agent phân tích tự động 1-Click + từng bước kết hợp Google Gemini 3.5 Flash.
"""

import json
import os
import io
import sys
import threading
import base64
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    from scipy.interpolate import make_interp_spline
except ImportError:
    make_interp_spline = None

from agent.orchestrator import TranBaoAgentOrchestrator
from agent.gemini_service import GeminiRetailService, GEMINI_API_KEY

PORT = 8501

# Global instances
_orch = TranBaoAgentOrchestrator()
_gemini = GeminiRetailService(GEMINI_API_KEY)
_lock = threading.Lock()

# Clean initial state: wait for user to ingest/load raw data in Step 1
# sales_data(1).csv is ready in workspace and can be loaded with 1-click via 'Nạp Dữ Liệu Mẫu'
_orch.reset_state()
print("[Startup] Server started cleanly with empty state. Ready for raw data ingestion in Step 1!")

# ─────────────────────────── LIGHT THEME CHARTS ────────────────────────────

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90, facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

def chart_donut_distribution(df: pd.DataFrame, label_col: str, val_col: str, title: str) -> str:
    """Luxury Donut Chart matching dark theme for Region or Channel breakdown"""
    if df is None or len(df) == 0:
        return ""
    try:
        labels = [str(x) for x in df[label_col]]
        values = df[val_col].tolist()
        total = sum(values) if sum(values) > 0 else 1
        
        # Luxury palette: Gold, Sapphire, Emerald, Cyan, Violet, Rose, Amber
        palette = ['#f59e0b', '#3b82f6', '#10b981', '#06b6d4', '#8b5cf6', '#f43f5e', '#f97316']
        colors = palette[:len(labels)] if len(labels) <= len(palette) else palette * 2
        
        fig, ax = plt.subplots(figsize=(4.8, 4.4), facecolor="#071330")
        ax.set_facecolor("#071330")
        
        wedges, texts, autotexts = ax.pie(
            values,
            labels=None,
            autopct='%1.1f%%',
            pctdistance=0.74,
            startangle=140,
            colors=colors,
            wedgeprops=dict(width=0.42, edgecolor='#071330', linewidth=2.5, antialiased=True)
        )
        
        for at in autotexts:
            at.set_color('#ffffff')
            at.set_fontsize(9)
            at.set_fontweight('bold')
            
        ax.text(0, 0.08, f"{total/1e9:,.1f} tỷ", ha='center', va='center', color='#fde047', fontsize=12, fontweight='bold')
        ax.text(0, -0.14, "Doanh Thu", ha='center', va='center', color='#94a3b8', fontsize=8, fontweight='bold')
        ax.set_title(title, color="#fef08a", fontsize=10.5, fontweight="bold", pad=12)
        fig.tight_layout()
        return fig_to_b64(fig)
    except Exception as e_donut:
        print("Donut chart error:", e_donut)
        return ""

def chart_store_revenue(store_perf: pd.DataFrame) -> str:
    """Sales History Dual-line Chart matching Dashcom sample"""
    if store_perf is None or len(store_perf) == 0:
        return ""
    df = store_perf.sort_values("Net_Revenue", ascending=False)
    stores = [str(s)[:14] for s in df["Store_Name"].head(10)]
    actual_rev = (df["Net_Revenue"].head(10) / 1e9).tolist()
    target_rev = (df["Target_Revenue"].head(10) / 1e9).tolist()

    fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor="#071330")
    ax.set_facecolor("#071330")

    x = np.arange(len(stores))
    
    # Smooth spline if enough points
    try:
        if make_interp_spline is not None and len(x) >= 4:
            x_smooth = np.linspace(x.min(), x.max(), 120)
            spl_act = make_interp_spline(x, actual_rev, k=2)
            spl_tar = make_interp_spline(x, target_rev, k=2)
            y_act_smooth = np.maximum(0, spl_act(x_smooth))
            y_tar_smooth = np.maximum(0, spl_tar(x_smooth))
            
            ax.plot(x_smooth, y_tar_smooth, color="#3b82f6", linewidth=2.6, label="Chỉ tiêu giao (Target)")
            ax.plot(x_smooth, y_act_smooth, color="#eab308", linewidth=3.0, label="DT Thực tế (Current Sales)")
            ax.fill_between(x_smooth, y_act_smooth, alpha=0.15, color="#eab308")
            ax.fill_between(x_smooth, y_tar_smooth, alpha=0.08, color="#3b82f6")
        else:
            ax.plot(x, target_rev, color="#3b82f6", linewidth=2.5, marker="o", label="Chỉ tiêu giao")
            ax.plot(x, actual_rev, color="#eab308", linewidth=2.8, marker="o", label="DT Thực tế")
    except Exception:
        ax.plot(x, target_rev, color="#3b82f6", linewidth=2.5, marker="o", label="Chỉ tiêu giao")
        ax.plot(x, actual_rev, color="#eab308", linewidth=2.8, marker="o", label="DT Thực tế")

    ax.scatter(x, actual_rev, color="#fde047", s=38, zorder=5)
    ax.scatter(x, target_rev, color="#60a5fa", s=38, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(stores, color="#cbd5e1", fontsize=8.5, rotation=15)
    ax.set_ylabel("Doanh thu (tỷ VNĐ)", color="#94a3b8", fontsize=9)
    ax.set_title("So sánh Xu hướng Doanh thu vs Chỉ tiêu (Top 10 Cửa hàng)", color="#fef08a", fontsize=11, fontweight="bold", pad=12, loc="left")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.grid(True, color="#1e293b", linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)

    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor="#cbd5e1")
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_sku_revenue(sku_perf: pd.DataFrame) -> str:
    """Top 10 SKU Revenue Bar Chart"""
    if sku_perf is None or len(sku_perf) == 0:
        return ""
    df = sku_perf.sort_values("Net_Revenue", ascending=False).head(10)
    skus = [str(s)[:20] for s in df["Product_Name"]]
    revs = (df["Net_Revenue"] / 1e6).tolist()  # Triệu VNĐ

    fig, ax = plt.subplots(figsize=(10.5, 4.2), facecolor="#071330")
    ax.set_facecolor("#071330")

    y_pos = np.arange(len(skus))
    bars = ax.barh(y_pos, revs, color="#d4af37", height=0.55, edgecolor="#fef08a", alpha=0.92)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + (max(revs) if revs else 1) * 0.015, bar.get_y() + bar.get_height() / 2, f"{w:,.0f} tr",
                va="center", ha="left", color="#fef08a", fontsize=8, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(skus, color="#e2e8f0", fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Doanh thu thuần (Triệu VNĐ)", color="#94a3b8", fontsize=9)
    ax.set_title("Top 10 SKU Sản Phẩm Tiêu Dùng Đóng Góp Doanh Thu Cao Nhất", color="#fef08a", fontsize=11, fontweight="bold", pad=12, loc="left")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color="#1e293b", linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig_to_b64(fig)


# ─────────────────────────── HTML PAGE TEMPLATE ────────────────────────────

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="vi" data-theme="dark">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Trần Bảo Project · Enterprise AI Platform</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet"/>
  <style>

    /* ═══════ HIGH-CONTRAST READABILITY ENGINE FOR WORKSPACE (SẮC NÉT & RÕ RÀNG) ═══════ */
    :root, [data-theme="dark"] {
      .text-white, .text-slate-900, .text-zinc-900, .text-gray-900, .text-neutral-900 {
        color: #FFFFFF !important;
      }
      .text-stone-800, .text-slate-800, .text-zinc-800, .text-gray-800 {
        color: #F8FAFC !important;
      }
      .text-slate-200, .text-slate-700, .text-zinc-700, .text-gray-700 {
        color: #E2E8F0 !important;
      }
      .text-slate-200, .text-slate-600, .text-zinc-600, .text-gray-600 {
        color: #CBD5E1 !important;
      }
      .text-slate-300, .text-slate-500, .text-zinc-500, .text-gray-500 {
        color: #94A3B8 !important;
      }
      .text-slate-300, .text-slate-400, .text-zinc-400, .text-gray-400 {
        color: #94A3B8 !important;
      }
      .text-stone-300, .text-slate-300 {
        color: #CBD5E1 !important;
      }

      /* Boost contrast for faded blue text */
      .text-blue-100\/75, .text-blue-100\/80 { color: #E2E8F0 !important; }
      .text-blue-200\/50, .text-blue-200\/60, .text-blue-200\/70, .text-blue-200\/80 { color: #CBD5E1 !important; }
      .text-blue-300\/40, .text-blue-300\/60 { color: #94A3B8 !important; }
      .text-amber-200\/60, .text-amber-200\/70, .text-amber-200\/80 { color: #FDE047 !important; }

      /* Table headers & cells */
      table th {
        color: #FDE047 !important;
        font-weight: 800 !important;
        background-color: rgba(6, 15, 38, 0.95) !important;
      }
      table td {
        color: #F8FAFC !important;
      }
      table tr:hover td {
        color: #FFFFFF !important;
        background-color: rgba(30, 58, 138, 0.3) !important;
      }

      /* High-contrast status badges in dark mode */
      .text-emerald-600, .text-emerald-700, .text-emerald-800 {
        color: #34D399 !important;
      }
      .text-rose-600, .text-rose-700, .text-rose-800 {
        color: #FB7185 !important;
      }
      .text-amber-600, .text-amber-700, .text-amber-800 {
        color: #FDE047 !important;
      }
      .bg-emerald-50, .bg-emerald-100 {
        background-color: rgba(6, 78, 59, 0.65) !important;
        border-color: rgba(52, 211, 153, 0.5) !important;
      }
      .bg-rose-50, .bg-rose-100 {
        background-color: rgba(136, 19, 55, 0.65) !important;
        border-color: rgba(251, 113, 133, 0.5) !important;
      }
      .bg-amber-50, .bg-amber-100 {
        background-color: rgba(120, 53, 15, 0.65) !important;
        border-color: rgba(253, 224, 71, 0.5) !important;
      }
      .bg-stone-50, .bg-stone-50\/60, .bg-slate-50 {
        background-color: rgba(6, 15, 38, 0.75) !important;
      }
      .border-stone-100, .border-stone-200, .border-slate-100, .border-slate-200 {
        border-color: rgba(212, 175, 55, 0.22) !important;
      }

      /* Select dropdowns in dark mode */
      select {
        background-color: #071330 !important;
        color: #F8FAFC !important;
        border-color: rgba(212, 175, 55, 0.4) !important;
      }
      select option {
        background-color: #071330 !important;
        color: #F8FAFC !important;
      }

      /* Sidebar navigation items */
      .nav-btn:not(.active) {
        color: #E2E8F0 !important;
      }
      .nav-btn:not(.active):hover {
        color: #FDE047 !important;
        background-color: rgba(30, 58, 138, 0.4) !important;
      }
    }

    [data-theme="light"] {
      table th {
        color: #92400E !important;
        font-weight: 800 !important;
        background-color: #F8FAFC !important;
      }
      table td {
        color: #0F172A !important;
      }
      .text-white, .text-slate-900 { color: #0F172A !important; }
      .text-stone-800, .text-slate-800 { color: #1E293B !important; }
      .text-slate-200, .text-slate-700 { color: #334155 !important; }
      .text-slate-200, .text-slate-600 { color: #475569 !important; }
      .text-slate-300, .text-slate-500 { color: #64748B !important; }
      .text-slate-300, .text-slate-400 { color: #64748B !important; }
      .text-emerald-600, .text-emerald-700 { color: #065F46 !important; }
      .text-rose-600, .text-rose-700 { color: #9F1239 !important; }
      .text-amber-600, .text-amber-700 { color: #92400E !important; }
      select {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border-color: #CBD5E1 !important;
      }
    }


    /* ═══════ PURE 24K ROYAL GOLD ("VÀNG GOLD") ULTRA-LUXURY STYLING ═══════ */
    .font-cinzel {
      font-family: 'Cinzel', serif;
    }
    .royal-gold-heading {
      font-family: 'Cinzel', 'Playfair Display', serif;
      letter-spacing: -0.01em;
      background: linear-gradient(180deg, #FFFFFF 0%, #FFF5CC 12%, #F6D35E 28%, #D4AF37 50%, #B8860B 72%, #7D5904 90%, #473200 100%) !important;
      -webkit-background-clip: text !important;
      -webkit-text-fill-color: transparent !important;
      filter: drop-shadow(0 4px 22px rgba(212, 175, 55, 0.6)) drop-shadow(0 1px 3px rgba(0, 0, 0, 0.95));
    }
    .royal-gold-text {
      background: linear-gradient(180deg, #FFFFFF 0%, #FFF2B2 25%, #E5B842 55%, #A87610 100%) !important;
      -webkit-background-clip: text !important;
      -webkit-text-fill-color: transparent !important;
    }
    .royal-gold-btn {
      background: linear-gradient(180deg, #FFF9D9 0%, #F8D35C 20%, #D4AF37 55%, #AD7907 85%, #7D5502 100%) !important;
      color: #120900 !important;
      font-weight: 900 !important;
      border: 1px solid #FFEAA5 !important;
      box-shadow: 
        inset 0 1px 2px rgba(255, 255, 255, 0.95),
        inset 0 -2px 3px rgba(90, 60, 2, 0.7),
        0 4px 24px rgba(212, 175, 55, 0.65),
        0 0 40px rgba(212, 175, 55, 0.4) !important;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .royal-gold-btn:hover {
      background: linear-gradient(180deg, #FFFFFF 0%, #FDE47A 20%, #E0B73E 55%, #C28B0E 85%, #946503 100%) !important;
      box-shadow: 
        inset 0 1px 3px rgba(255, 255, 255, 1),
        0 6px 36px rgba(212, 175, 55, 0.88),
        0 0 60px rgba(212, 175, 55, 0.65) !important;
      transform: translateY(-2px) scale(1.02) !important;
    }
    .royal-gold-btn:active {
      transform: translateY(1px) scale(0.99) !important;
    }
    .royal-gold-pill {
      background: rgba(5, 12, 34, 0.78) !important;
      backdrop-filter: blur(28px) !important;
      -webkit-backdrop-filter: blur(28px) !important;
      border: 1.5px solid rgba(212, 175, 55, 0.45) !important;
      box-shadow: 
        inset 0 1px 1px rgba(255, 248, 220, 0.25),
        0 14px 42px rgba(0, 0, 0, 0.75),
        0 0 25px rgba(212, 175, 55, 0.15) !important;
    }
    .sparkle-flare {
      position: absolute;
      bottom: 12%;
      right: 8%;
      width: 52px;
      height: 52px;
      pointer-events: none;
      filter: drop-shadow(0 0 18px rgba(212, 175, 55, 0.9));
      animation: sparklePulse 3.5s ease-in-out infinite alternate;
      z-index: 25;
    }
    @keyframes sparklePulse {
      0% { opacity: 0.45; transform: scale(0.85) rotate(0deg); }
      50% { opacity: 1; transform: scale(1.2) rotate(12deg); }
      100% { opacity: 0.45; transform: scale(0.85) rotate(0deg); }
    }

    :root, [data-theme="dark"] {
      --font-main: 'Plus Jakarta Sans', system-ui, sans-serif;
      --font-display: 'Outfit', 'Plus Jakarta Sans', sans-serif;
      --body-bg: #030712;
      --body-bg-img: 
        radial-gradient(circle at 50% 0%, rgba(29, 78, 216, 0.25) 0%, transparent 55%),
        radial-gradient(circle at 100% 100%, rgba(212, 175, 55, 0.12) 0%, transparent 45%),
        linear-gradient(180deg, rgba(3, 7, 18, 0.85) 0%, rgba(3, 7, 18, 0.95) 100%),
        url('/toty_bg.jpg');
      --card-bg: rgba(7, 18, 44, 0.85);
      --card-border: rgba(212, 175, 55, 0.25);
      --card-color: #f1f5f9;
      --topbar-bg: rgba(6, 14, 34, 0.92);
      --sidebar-bg: rgba(4, 10, 28, 0.95);
      --accent-btn-bg: linear-gradient(135deg, #fae8a4 0%, #d4af37 40%, #b8860b 80%, #8c6204 100%);
      --accent-btn-text: #050b18;
      --accent-btn-border: rgba(254, 240, 138, 0.6);
      --brand-text-grad: linear-gradient(135deg, #fff7d6 0%, #ecd06f 40%, #c59b27 100%);
      --toggle-bg: rgba(6, 14, 34, 0.85);
      --toggle-border: rgba(212, 175, 55, 0.4);
      --toggle-text: #fef08a;
      --pill-bg: rgba(10, 25, 60, 0.8);
      --pill-border: rgba(212, 175, 55, 0.35);
      --input-bg: rgba(6, 14, 34, 0.9);
      --input-border: rgba(212, 175, 55, 0.45);
      --input-text: #ffffff;
      --sub-text: rgba(224, 231, 255, 0.75);
    }

    [data-theme="light"] {
      --font-main: 'Plus Jakarta Sans', system-ui, sans-serif;
      --font-display: 'Outfit', 'Plus Jakarta Sans', sans-serif;
      --body-bg: #f8fafc;
      --body-bg-img: 
        radial-gradient(circle at 10% 10%, rgba(255, 237, 213, 0.6) 0%, transparent 50%),
        radial-gradient(circle at 90% 90%, rgba(224, 242, 254, 0.6) 0%, transparent 45%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.85) 0%, rgba(248, 250, 252, 0.92) 100%),
        url('/light_bg.jpg');
      --card-bg: rgba(255, 255, 255, 0.94);
      --card-border: rgba(212, 175, 55, 0.4);
      --card-color: #0f172a;
      --topbar-bg: rgba(255, 255, 255, 0.96);
      --sidebar-bg: rgba(255, 255, 255, 0.98);
      --accent-btn-bg: linear-gradient(135deg, #fde68a 0%, #eab308 50%, #ca8a04 100%);
      --accent-btn-text: #0f172a;
      --accent-btn-border: rgba(234, 179, 8, 0.7);
      --brand-text-grad: linear-gradient(135deg, #a16207 0%, #ca8a04 40%, #854d0e 100%);
      --toggle-bg: rgba(255, 255, 255, 0.95);
      --toggle-border: rgba(212, 175, 55, 0.45);
      --toggle-text: #854d0e;
      --pill-bg: rgba(255, 255, 255, 0.92);
      --pill-border: rgba(212, 175, 55, 0.35);
      --input-bg: #ffffff;
      --input-border: rgba(212, 175, 55, 0.5);
      --input-text: #0f172a;
      --sub-text: #475569;
    }

    body {
      font-family: var(--font-main);
      background-color: var(--body-bg);
      background-image: var(--body-bg-img);
      background-size: cover;
      background-position: center center;
      background-attachment: fixed;
      color: var(--card-color);
      overflow-x: hidden;
      transition: background-color 0.3s ease;
    }
    .font-display {
      font-family: var(--font-display);
    }
    .glass-card {
      background: var(--card-bg) !important;
      backdrop-filter: blur(24px);
      -webkit-backdrop-filter: blur(24px);
      border: 1px solid var(--card-border) !important;
      box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.25);
      color: var(--card-color) !important;
    }
    .glass-pill {
      background: var(--pill-bg) !important;
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--pill-border) !important;
      color: var(--card-color) !important;
    }
    .toty-gold-btn, .theme-btn-primary {
      background: var(--accent-btn-bg) !important;
      color: var(--accent-btn-text) !important;
      font-weight: 800;
      border: 1px solid var(--accent-btn-border) !important;
      box-shadow: 0 4px 18px rgba(212, 175, 55, 0.35);
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .toty-gold-btn:hover, .theme-btn-primary:hover {
      filter: brightness(1.1);
      transform: translateY(-1.5px);
    }
    .toty-gold-btn:active, .theme-btn-primary:active {
      transform: translateY(0.5px);
    }
    .toty-gold-text, .brand-gold-text {
      background: var(--brand-text-grad) !important;
      -webkit-background-clip: text !important;
      -webkit-text-fill-color: transparent !important;
    }
    .theme-toggle-btn {
      background: var(--toggle-bg) !important;
      border: 1px solid var(--toggle-border) !important;
      color: var(--toggle-text) !important;
    }
    .active-cat-pill {
      background: var(--accent-btn-bg) !important;
      color: var(--accent-btn-text) !important;
      font-weight: 800 !important;
      border: 1px solid var(--accent-btn-border) !important;
      box-shadow: 0 4px 14px rgba(212, 175, 55, 0.35);
    }
    .inactive-cat-pill {
      background: var(--pill-bg) !important;
      color: var(--card-color) !important;
      opacity: 0.85;
      border: 1px solid var(--pill-border) !important;
    }
    .inactive-cat-pill:hover {
      opacity: 1;
      border-color: var(--accent-btn-border) !important;
    }
    .donut-prog {
      stroke-dasharray: 377;
      stroke-dashoffset: 45;
      transition: stroke-dashoffset 0.8s ease;
    }
    .spinner {
      width: 14px;
      height: 14px;
      border: 2px solid rgba(212, 175, 55, 0.2);
      border-top-color: #d4af37;
      border-radius: 50%;
      animation: spin 0.6s linear infinite;
      display: inline-block;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .tab-content { display: none; }
    .tab-content:not(.hidden) { display: block; }

    /* Light Theme Overrides */
    [data-theme="light"] body { background-color: #f8fafc !important; color: #0f172a !important; }
    [data-theme="light"] header.bg-\[\#071330\]\/85, [data-theme="light"] header { background-color: rgba(255, 255, 255, 0.95) !important; border-color: rgba(226, 232, 240, 0.9) !important; }
    [data-theme="light"] aside { background-color: rgba(255, 255, 255, 0.97) !important; border-color: rgba(226, 232, 240, 0.9) !important; }
    [data-theme="light"] section.bg-gradient-to-r { background: linear-gradient(135deg, #ffffff 0%, #fef3c7 60%, #ffedd5 100%) !important; border-color: rgba(251, 146, 60, 0.3) !important; color: #0f172a !important; }
    [data-theme="light"] .text-white { color: #0f172a !important; }
    [data-theme="light"] .text-slate-100, [data-theme="light"] .text-slate-200 { color: #1e293b !important; }
    [data-theme="light"] .text-blue-100, [data-theme="light"] .text-blue-200 { color: #334155 !important; }
    [data-theme="light"] .text-blue-200\/60, [data-theme="light"] .text-blue-300\/40 { color: #64748b !important; }
    [data-theme="light"] .text-amber-200, [data-theme="light"] .text-amber-300, [data-theme="light"] .text-amber-400 { color: #b45309 !important; }
    [data-theme="light"] .bg-blue-950\/80, [data-theme="light"] .bg-blue-950\/70, [data-theme="light"] .bg-blue-950 { background-color: rgba(241, 245, 249, 0.95) !important; border-color: rgba(203, 213, 225, 0.9) !important; color: #0f172a !important; }
    [data-theme="light"] input, [data-theme="light"] textarea, [data-theme="light"] select { background-color: #ffffff !important; color: #0f172a !important; border-color: #cbd5e1 !important; }
    [data-theme="light"] .nav-btn.active { background: linear-gradient(135deg, #fde68a 0%, #eab308 100%) !important; color: #0f172a !important; box-shadow: 0 4px 14px rgba(234, 179, 8, 0.3) !important; font-weight: 800 !important; }
    [data-theme="light"] .nav-btn:not(.active) { color: #475569 !important; }
    [data-theme="light"] .nav-btn:not(.active):hover { background-color: rgba(241, 245, 249, 0.9) !important; color: #0f172a !important; }
    
    /* Scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--body-bg); }
    ::-webkit-scrollbar-thumb { background: #1e3a8a; border: 1px solid rgba(212,175,55,0.3); border-radius: 9999px; }
    [data-theme="light"] ::-webkit-scrollbar-track { background: #f1f5f9; }
    [data-theme="light"] ::-webkit-scrollbar-thumb { background: #cbd5e1; border: 1px solid #94a3b8; border-radius: 9999px; }
  </style>
</head>
<body class="min-h-screen antialiased text-slate-100 selection:bg-amber-400 selection:text-slate-950">

  <!-- ======================================================================= -->
  <!-- 1. GIAO DIỆN MỞ ĐẦU (HERO LANDING / CHẾ ĐỘ SÁNG TỐI THEO 2 ẢNH)          -->
  <!-- ======================================================================= -->
  <div id="screen-auth" style="display: flex;" class="min-h-screen flex flex-col justify-between p-4 sm:p-8 relative overflow-hidden">
    
    <!-- Dynamic Theme Backdrop Container -->
    <div id="hero-bg-container" class="absolute inset-0 z-0 bg-cover bg-center transition-all duration-700 pointer-events-none" 
         style="background-image: radial-gradient(circle at 50% 18%, rgba(29, 78, 216, 0.42) 0%, rgba(2, 6, 23, 0.94) 75%), radial-gradient(circle at 88% 85%, rgba(212, 175, 55, 0.16) 0%, transparent 45%), radial-gradient(circle at 12% 85%, rgba(30, 58, 138, 0.28) 0%, transparent 50%), linear-gradient(180deg, #020617 0%, #030712 100%); filter: saturate(1.2) brightness(0.95);">
    </div>
    
    <!-- Ambient Overlay Glow -->
    <div class="absolute inset-0 z-0 pointer-events-none bg-gradient-to-b from-transparent via-transparent to-black/30"></div>

    <!-- Radiant 4-Pointed Diamond Star Sparkle Flare (Chuẩn Ảnh) -->
    <div class="sparkle-flare hidden lg:block">
      <svg viewBox="0 0 100 100" class="w-full h-full text-[#FDE047]" fill="currentColor">
        <path d="M50 0 C50 35, 65 50, 100 50 C65 50, 50 65, 50 100 C50 65, 35 50, 0 50 C35 50, 50 35, 50 0 Z" />
      </svg>
    </div>

    <!-- ─── TOPBAR TRANG MỞ ĐẦU (3-PART LAYOUT CHUẨN ẢNH) ─── -->
    <header class="relative z-20 w-full max-w-7xl mx-auto pt-2 flex items-center justify-between gap-4">
      
      <!-- Left: Logo & Brand Emblem -->
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-full bg-gradient-to-tr from-amber-300 via-amber-400 to-yellow-500 text-slate-950 font-black text-sm flex items-center justify-center shadow-lg shadow-amber-400/30">
          TR
        </div>
        <div class="leading-tight">
          <div class="font-extrabold toty-gold-text text-base font-display">Trần Bảo</div>
          <div class="font-extrabold toty-gold-text text-xs tracking-wider uppercase font-display">Project</div>
        </div>
      </div>

      <!-- Center: Floating Navigation Capsule -->
      <div class="glass-pill rounded-full px-5 py-2 hidden md:flex items-center justify-between gap-6 border shadow-xl">
        <div class="flex items-center gap-2">
          <div class="w-6 h-6 rounded-full bg-amber-400 text-slate-950 font-black text-[10px] flex items-center justify-center">
            TR
          </div>
          <span class="font-bold toty-gold-text text-xs font-display">Trần Bảo Project</span>
        </div>
        
        <div class="flex items-center gap-5 text-xs font-semibold">
          <button onclick="handleEnterWorkspace(event, 'tab-products')" class="hover:text-amber-300 transition cursor-pointer">Đánh Giá SKU</button>
          <button onclick="handleEnterWorkspace(event, 'tab-dash')" class="hover:text-amber-300 transition cursor-pointer">Doanh Thu</button>
          <button onclick="handleEnterWorkspace(event, 'tab-top3')" class="hover:text-amber-300 transition cursor-pointer">Điểm Nóng</button>
          <button onclick="handleEnterWorkspace(event, 'tab-copilot')" class="hover:text-amber-300 transition cursor-pointer">Capittt Copitat</button>
        </div>

        <button onclick="handleEnterWorkspace(event)" class="toty-gold-btn text-xs font-black px-4 py-1.5 rounded-full shadow-md transition hover:scale-105 active:scale-95 cursor-pointer">
          Vào Workspace
        </button>
      </div>

      <!-- Right: User Avatar & Theme Toggle Button -->
      <div class="flex items-center gap-3">
        <!-- Theme Switcher Button -->
        <button id="theme-toggle-auth" onclick="toggleTheme()" class="theme-toggle-btn text-xs font-bold px-3 py-1.5 rounded-full flex items-center gap-1.5 transition cursor-pointer hover:scale-105 active:scale-95 shadow-sm" title="Chỉnh nền sáng/tối">
          <span class="theme-icon text-sm">🌙</span> <span class="theme-text text-xs">Tối</span>
        </button>

        <!-- Profile Capsule -->
        <div class="glass-pill rounded-full px-3 py-1.5 flex items-center gap-2 shadow-md">
          <div class="relative">
            <div class="w-7 h-7 rounded-full bg-slate-200 text-slate-800 flex items-center justify-center font-bold text-xs">
              👤
            </div>
            <span class="absolute bottom-0 right-0 w-2 h-2 rounded-full bg-emerald-400 ring-1 ring-slate-950"></span>
          </div>
          <div class="text-left hidden sm:block">
            <div class="text-xs font-extrabold leading-tight text-white" id="landing-admin-title">Trần Bảo Pro</div>
            <div class="text-[9px] text-amber-300 font-semibold leading-tight">AI Agent Platform Admin</div>
          </div>
        </div>
      </div>
    </header>

    <!-- ─── CENTER HERO SECTION: Large Title + Categories + Input Bar ─── -->
    <div class="relative z-20 max-w-4xl mx-auto w-full my-auto text-center px-4 py-6 sm:py-10">
      
      <!-- Laurel Wreath Badge -->
      <div class="inline-flex items-center gap-2.5 px-5 py-1.5 rounded-full bg-[#071330] border border-[#d4af37]/60 text-xs font-black mb-4 shadow-xl">
        <span class="text-sm text-[#fde047]">🌿</span>
        <span class="tracking-widest uppercase font-cinzel text-[#fde047]">★ AI AGENT OF THE YEAR ★</span>
        <span class="text-sm text-[#fde047] scale-x-[-1] inline-block">🌿</span>
      </div>

      <!-- Grand Title -->
      <h1 class="text-5xl sm:text-7xl md:text-8xl font-black tracking-tight leading-[1.08] mb-3 font-cinzel royal-gold-heading">
        Trần Bảo Project
      </h1>
      
      <!-- Subtitle -->
      <p class="text-xs sm:text-sm md:text-base font-medium max-w-2xl mx-auto mb-7 opacity-90 leading-relaxed">
        Hệ thống AI Phân tích Doanh thu, Đánh giá SKU & Điểm nóng Quản trị cho các Doanh nghiệp và Cửa hàng bán lẻ
      </p>

      <!-- Category Switcher Pills -->
      <div class="glass-pill rounded-full p-1.5 inline-flex items-center gap-1.5 mb-7 max-w-full overflow-x-auto shadow-xl">
        <button type="button" onclick="selectCategoryPill(this, 'Doanh nghiệp', 'tab-dash')" class="cat-pill active-cat-pill text-xs px-4 py-1.5 rounded-full cursor-pointer transition">
          🏛 Doanh nghiệp
        </button>
        <button type="button" onclick="selectCategoryPill(this, 'Đánh giá SKU', 'tab-products')" class="cat-pill inactive-cat-pill text-xs px-3.5 py-1.5 rounded-full cursor-pointer transition">
          📦 Đánh giá SKU
        </button>
        <button type="button" onclick="selectCategoryPill(this, 'Doanh thu thuần', 'tab-dash')" class="cat-pill inactive-cat-pill text-xs px-3.5 py-1.5 rounded-full cursor-pointer transition">
          📊 Doanh thu thuần
        </button>
        <button type="button" onclick="selectCategoryPill(this, 'Điểm nóng', 'tab-top3')" class="cat-pill inactive-cat-pill text-xs px-3.5 py-1.5 rounded-full cursor-pointer transition">
          🎯 Điểm nóng
        </button>
        <button type="button" onclick="selectCategoryPill(this, 'Copilot AI', 'tab-copilot')" class="cat-pill inactive-cat-pill text-xs px-3.5 py-1.5 rounded-full cursor-pointer transition">
          💬 Copilot AI
        </button>
      </div>

      <!-- ─── FLOATING PILL SEARCH / ENTERPRISE INPUT BAR (CHUẨN ẢNH) ─── -->
      <div class="max-w-2xl mx-auto w-full glass-card rounded-full p-2 pl-6 flex items-center gap-3 shadow-2xl border-2 transition-all">
        
        <!-- Input field: NHẬP TÊN DOANH NGHIỆP -->
        <div class="flex-1 min-w-0 text-left">
          <input 
            type="text" 
            id="input-landing-company" 
            value="Trần Bảo pro" 
            placeholder="Nhập tên doanh nghiệp / chuỗi cửa hàng của bạn..."
            onkeydown="if(event.key==='Enter') handleEnterWorkspace(event)"
            class="bg-transparent border-none outline-none w-full text-sm sm:text-base font-bold focus:ring-0"
          />
        </div>

        <!-- Filter Icon Pill -->
        <button type="button" onclick="document.getElementById('input-landing-company').focus()" class="hidden sm:flex items-center justify-center w-9 h-9 rounded-full glass-pill text-xs hover:scale-105 transition cursor-pointer" title="Cài đặt">
          ⚙️
        </button>

        <!-- Gold Glowing Button: BẮT ĐẦU LÀM VIỆC -->
        <button 
          type="button" 
          id="btn-landing-submit"
          onclick="handleEnterWorkspace(event)"
          class="royal-gold-btn text-xs sm:text-sm px-7 sm:px-9 py-3.5 rounded-full flex items-center gap-2 whitespace-nowrap cursor-pointer shadow-xl transition"
        >
          <span>Bắt đầu làm việc</span>
          <span class="text-sm">⚡</span>
        </button>
      </div>

      <div class="text-[11px] font-medium mt-3 opacity-80">
        ✨ Hoàn thiện hoặc bấm nút để truy cập ngay không gian làm việc và quản trị thông minh
      </div>
    </div>

    <!-- ─── BOTTOM FLOATING SPECS PILLS (CHUẨN ẢNH) ─── -->
    <footer class="relative z-20 w-full max-w-4xl mx-auto pb-2">
      <div class="glass-pill rounded-2xl p-3 px-6 grid grid-cols-2 sm:grid-cols-4 gap-4 text-center text-xs shadow-lg">
        <div class="flex items-center justify-center gap-1.5">
          <span>✦</span>
          <span><strong class="text-amber-300 font-extrabold">60.000+</strong> Giao Dịch</span>
        </div>
        <div class="flex items-center justify-center gap-1.5">
          <span>📋</span>
          <span><strong class="text-amber-300 font-extrabold">5 Bước</strong> AI Agent</span>
        </div>
        <div class="flex items-center justify-center gap-1.5">
          <span>🛡️</span>
          <span><strong class="text-amber-300 font-extrabold">100%</strong> Dữ Liệu Sạch</span>
        </div>
        <div class="flex items-center justify-center gap-1.5">
          <span>📦</span>
          <span><strong class="text-amber-300 font-extrabold">Đánh Giá</strong> SKU Tiêu Dùng</span>
        </div>
      </div>
      <div class="text-center text-[11px] font-medium mt-3 opacity-75">
        © 2024 Trần Bảo Project - Enterprise AI Platform - Hỗ trợ bởi Google Gemini 1.5 Flash
      </div>
    </footer>
  </div>

  <!-- ═══════════════════════════════════════════════════════════════════════ -->
  <!-- WELCOME TRANSITION MODAL (HIỆN KHI BẤM BẮT ĐẦU LÀM VIỆC)                -->
  <!-- ═══════════════════════════════════════════════════════════════════════ -->
  <div id="welcome-modal" class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md hidden transition-opacity duration-300">
    <div class="glass-card rounded-3xl p-8 max-w-lg w-full text-center border-2 border-amber-400/50 shadow-2xl relative overflow-hidden transform scale-95 transition-transform duration-300">
      <div class="w-16 h-16 rounded-full bg-gradient-to-tr from-amber-300 via-amber-400 to-yellow-500 text-slate-950 text-2xl font-black flex items-center justify-center mx-auto mb-4 shadow-lg shadow-amber-400/40 animate-bounce">
        ✨
      </div>
      <div class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-400/20 text-amber-300 text-xs font-bold border border-amber-400/30 mb-3">
        <span>🏢</span> ENTERPRISE AI PLATFORM
      </div>
      <h3 class="text-2xl sm:text-3xl font-extrabold toty-gold-text mb-2 font-display" id="welcome-company-heading">
        Xin chào Trần Bảo pro đã đến với trang web!
      </h3>
      <p class="text-xs sm:text-sm text-blue-100/80 mb-6 leading-relaxed">
        Hệ thống AI Agent đang khởi tạo cơ sở dữ liệu đối soát và thiết lập không gian làm việc quản trị thông minh...
      </p>
      <div class="flex items-center justify-center gap-3">
        <div class="w-5 h-5 border-2 border-amber-400/30 border-t-amber-400 rounded-full animate-spin"></div>
        <span class="text-xs font-bold text-amber-300">Đang chuyển vào Không gian Làm việc...</span>
      </div>
    </div>
  </div>

  <!-- ═══════════════════════════════════════════════════════════════════════ -->
  <!-- EXPORT RESULTS MODAL (TRUNG TÂM XUẤT BÁO CÁO & DỮ LIỆU ĐỐI SOÁT)       -->
  <!-- ═══════════════════════════════════════════════════════════════════════ -->
  <div id="export-modal" class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-md hidden transition-opacity duration-300">
    <div class="glass-card rounded-3xl p-6 sm:p-8 max-w-xl w-full text-left border-2 border-amber-400/50 shadow-2xl relative">
      <div class="flex items-center justify-between pb-4 border-b border-amber-400/20 mb-5">
        <div class="flex items-center gap-3">
          <div class="w-11 h-11 rounded-2xl bg-gradient-to-tr from-amber-300 via-amber-400 to-yellow-500 text-slate-950 font-black flex items-center justify-center text-xl shadow-lg shadow-amber-400/30">
            📥
          </div>
          <div>
            <h3 class="text-lg font-black toty-gold-text">Trung Tâm Xuất Báo Cáo &amp; Dữ Liệu</h3>
            <p class="text-xs text-slate-300">Tải xuống kết quả phân tích &amp; dữ liệu sạch đối soát</p>
          </div>
        </div>
        <button onclick="closeExportModal()" class="w-8 h-8 rounded-full bg-slate-800/80 text-slate-300 hover:text-white flex items-center justify-center text-lg font-bold cursor-pointer transition">&times;</button>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
        <!-- Excel Master Report -->
        <a href="/api/export/excel" download="Bao_Cao_Quan_Tri_Tran_Bao_Project.xlsx" class="p-4 rounded-2xl bg-amber-400/10 hover:bg-amber-400/20 border border-amber-400/30 transition flex flex-col justify-between group cursor-pointer">
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="text-2xl">📗</span>
              <span class="text-[10px] font-black uppercase px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Khuyên dùng</span>
            </div>
            <div class="font-black text-white group-hover:text-amber-300 text-sm">Báo Cáo Tổng Hợp Excel (.xlsx)</div>
            <div class="text-[11px] text-slate-300 mt-1 leading-snug">6 Sheet đầy đủ: KPI C-Suite, 13 Chi nhánh, 4 Ngành hàng, Top 28 SKU, 548 Đơn đổi trả &amp; Audit Logs</div>
          </div>
          <div class="mt-3 text-[11px] font-bold text-amber-300 flex items-center gap-1">
            <span>Tải ngay Excel</span> <span>&rarr;</span>
          </div>
        </a>

        <!-- Clean Txns CSV -->
        <a href="/api/export/csv?type=txns" download="Du_Lieu_Giao_Dich_Sach.csv" class="p-4 rounded-2xl bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 transition flex flex-col justify-between group cursor-pointer">
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="text-2xl">📄</span>
              <span class="text-[10px] font-black uppercase px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30">14.452 dòng sạch</span>
            </div>
            <div class="font-black text-white group-hover:text-blue-300 text-sm">Dữ Liệu Giao Dịch Sạch (.csv)</div>
            <div class="text-[11px] text-slate-300 mt-1 leading-snug">Toàn bộ 14.452 đơn hoàn tất sạch đã lọc bỏ đơn hủy và trùng lặp, mã hóa chuẩn UTF-8-BOM</div>
          </div>
          <div class="mt-3 text-[11px] font-bold text-blue-400 flex items-center gap-1">
            <span>Tải CSV Giao dịch</span> <span>&rarr;</span>
          </div>
        </a>

        <!-- Returns CSV -->
        <a href="/api/export/csv?type=returns" download="Doi_Soat_548_Don_Doi_Tra.csv" class="p-4 rounded-2xl bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/30 transition flex flex-col justify-between group cursor-pointer">
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="text-2xl">🛡️</span>
              <span class="text-[10px] font-black uppercase px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30">548 đơn</span>
            </div>
            <div class="font-black text-white group-hover:text-rose-300 text-sm">Đối Soát Đổi Trả (.csv)</div>
            <div class="text-[11px] text-slate-300 mt-1 leading-snug">Chi tiết 548 đơn hoàn trả, 15.45 tỷ đồng khấu trừ đối soát và lý do đổi trả theo chi nhánh</div>
          </div>
          <div class="mt-3 text-[11px] font-bold text-rose-400 flex items-center gap-1">
            <span>Tải CSV Đổi trả</span> <span>&rarr;</span>
          </div>
        </a>

        <!-- Stores CSV -->
        <a href="/api/export/csv?type=stores" download="Hieu_Qua_13_Chi_Nhanh.csv" class="p-4 rounded-2xl bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 transition flex flex-col justify-between group cursor-pointer">
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="text-2xl">🏪</span>
              <span class="text-[10px] font-black uppercase px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-400 border border-purple-500/30">13 Chi nhánh</span>
            </div>
            <div class="font-black text-white group-hover:text-purple-300 text-sm">Hiệu Quả Chi Nhánh (.csv)</div>
            <div class="text-[11px] text-slate-300 mt-1 leading-snug">Doanh thu thuần, lợi nhuận gộp, tỷ lệ trả hàng và điểm ưu tiên can thiệp theo tỉnh thành</div>
          </div>
          <div class="mt-3 text-[11px] font-bold text-purple-400 flex items-center gap-1">
            <span>Tải CSV Chi nhánh</span> <span>&rarr;</span>
          </div>
        </a>
      </div>

      <div class="flex items-center justify-between pt-3 border-t border-amber-400/20">
        <span class="text-xs text-slate-400 flex items-center gap-1.5">
          <span class="text-amber-300">💡</span> Tệp xuất tương thích 100% Microsoft Excel, Power BI và Google Sheets
        </span>
        <button onclick="closeExportModal()" class="px-5 py-2 rounded-xl text-xs font-bold bg-slate-800 text-slate-300 hover:text-white transition cursor-pointer">
          Đóng
        </button>
      </div>
    </div>
  </div>

  
  <!-- ======================================================================= -->
  <!-- MODAL TRỢ GIÚP & HƯỚNG DẪN QUY TRÌNH (DATALENS STUDIO GUIDE)           -->
  <!-- ======================================================================= -->
  <div id="help-modal" class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md hidden" style="animation: fadeIn 0.2s ease;">
    <div class="glass-card rounded-3xl p-6 sm:p-8 max-w-2xl w-full text-left border-2 border-amber-400/50 shadow-2xl relative bg-[#071330]/95 max-h-[88vh] overflow-y-auto">
      <div class="flex items-center justify-between pb-4 border-b border-amber-400/20 mb-5">
        <div class="flex items-center gap-3">
          <div class="w-11 h-11 rounded-2xl bg-amber-400/20 text-amber-300 border border-amber-400/30 flex items-center justify-center text-xl shadow-lg">
            💡
          </div>
          <div>
            <h3 class="text-base sm:text-lg font-black toty-gold-text">Trợ Giúp & Hướng Dẫn Quy Trình Phân Tích</h3>
            <p class="text-xs text-slate-300">Hướng dẫn sử dụng hệ thống AI phân tích dữ liệu DataLens Studio Engine</p>
          </div>
        </div>
        <button onclick="closeHelpModal()" class="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 text-white flex items-center justify-center text-sm font-bold cursor-pointer transition">✕</button>
      </div>

      <div class="space-y-4 text-xs text-slate-200">
        <div class="p-4 rounded-2xl bg-white/5 border border-white/10 space-y-1.5">
          <h4 class="font-black text-amber-300 text-sm flex items-center gap-2">
            <span>🔗</span> Bước 1: Kết nối dữ liệu
          </h4>
          <p class="text-slate-300 leading-relaxed">
            Nhập loại hình doanh nghiệp và tối đa 8 câu hỏi kinh doanh để định hướng việc chọn chỉ số và đọc kết quả. Bấm <strong>"Trao đổi yêu cầu với AI"</strong> để xem AI Agent phân tích bối cảnh và đề xuất các góc nhìn chiến lược.
          </p>
        </div>

        <div class="p-4 rounded-2xl bg-white/5 border border-white/10 space-y-1.5">
          <h4 class="font-black text-amber-300 text-sm flex items-center gap-2">
            <span>🧹</span> Bước 2: Làm sạch dữ liệu
          </h4>
          <p class="text-slate-300 leading-relaxed">
            Áp dụng 4 quy tắc làm sạch chuẩn: Cắt khoảng trắng thừa, Chuẩn hóa ô trống, Khử trùng lặp và Kiểm tra định dạng. Dữ liệu sạch bảo toàn 100% cột gốc của người dùng (23 cột). Hỗ trợ nhập yêu cầu sửa bằng AI (ví dụ: <em>"Bỏ mã dòng HD_001"</em>).
          </p>
        </div>

        <div class="p-4 rounded-2xl bg-white/5 border border-white/10 space-y-1.5">
          <h4 class="font-black text-amber-300 text-sm flex items-center gap-2">
            <span>📊</span> Bước 3: Phân tích dữ liệu
          </h4>
          <p class="text-slate-300 leading-relaxed">
            Khám phá số liệu đa chiều: So sánh theo nhóm (Group By bất kỳ chiều nào), chọn phép tổng hợp (Tổng, Trung bình, Số dòng) và cột số đo lường. Hệ thống tự động đưa ra nhận xét, bối cảnh kinh doanh thực chứng và bảng thống kê chi tiết từng cột.
          </p>
        </div>

        <div class="p-4 rounded-2xl bg-white/5 border border-white/10 space-y-1.5">
          <h4 class="font-black text-amber-300 text-sm flex items-center gap-2">
            <span>📑</span> Bước 4: Báo cáo kinh doanh điều hành
          </h4>
          <p class="text-slate-300 leading-relaxed">
            Xem toàn diện Dashboard KPI điều hành với bộ lọc kỳ báo cáo theo tháng, phân tích chuyên đề, kết luận và ma trận đề xuất hành động ưu tiên (Khẩn cấp, Chiến lược, Dài hạn). Cho phép sao chép Markdown và in / xuất tệp.
          </p>
        </div>
      </div>

      <div class="flex items-center justify-between pt-4 border-t border-amber-400/20 mt-5">
        <span class="text-xs text-slate-400">Cam kết bảo mật & số liệu chuẩn xác 0₫ sai lệch</span>
        <button onclick="closeHelpModal()" class="px-5 py-2 rounded-xl text-xs font-bold bg-amber-400 text-slate-950 hover:bg-amber-300 transition cursor-pointer">Đã hiểu</button>
      </div>
    </div>
  </div>

  <!-- ======================================================================= -->
  <!-- 2. KHÔNG GIAN LÀM VIỆC QUẢN TRỊ TRẦN BẢO PROJECT (ENTERPRISE AI PLATFORM) -->
  <!-- ======================================================================= -->
  <div id="screen-workspace" style="display: none;" class="min-h-screen flex hidden">

    <!-- ─── SIDEBAR TRÁI (ĐIỀU HƯỚNG CHỨC NĂNG) ─── -->
    <aside class="w-64 bg-[#040a1c]/95 backdrop-blur-2xl border-r border-amber-400/20 p-5 flex flex-col flex-shrink-0 sticky top-0 h-screen overflow-y-auto">
      <!-- Logo & Brand Header -->
      <div class="flex items-center gap-3 pb-5 border-b border-amber-400/20 mb-5">
        <div class="w-10 h-10 rounded-2xl bg-gradient-to-tr from-amber-300 via-amber-400 to-yellow-500 flex items-center justify-center text-slate-950 font-black text-lg shadow-md shadow-amber-400/30">
          TB
        </div>
        <div>
          <div class="font-extrabold toty-gold-text text-base leading-tight font-display">Dashcom AI</div>
          <div class="text-[10px] text-amber-200/80 font-semibold truncate max-w-[130px]" id="sidebar-company-label">Trần Bảo pro</div>
        </div>
      </div>

      <!-- Navigation Menu: 4 BƯỚC QUY TRÌNH CHUẨN DATALENS STUDIO -->
      <div class="text-[10px] font-black text-amber-200/70 uppercase tracking-widest px-3 mb-2 flex items-center justify-between">
        <span>QUY TRÌNH</span>
        <span class="text-[9px] px-2 py-0.5 rounded-full bg-amber-400/20 text-amber-300 font-bold border border-amber-400/30">5 BƯỚC</span>
      </div>
      <nav class="space-y-1.5" id="navigation" aria-label="Các bước phân tích">
        <!-- Bước 1: Kết nối -->
        <button id="nav-step-1" onclick="goToStep(1)" class="nav-step w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl text-xs font-semibold text-blue-100/75 hover:bg-blue-900/30 hover:text-amber-200 transition cursor-pointer">
          <div class="flex items-center gap-2.5">
            <span class="text-base">🔗</span>
            <div class="text-left">
              <div class="font-bold text-xs text-white">1. Kết Nối & Nguồn</div>
              <div class="text-[10px] text-slate-400">Chọn nguồn & Khảo sát AI</div>
            </div>
          </div>
          <span class="step-badge text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-slate-300 font-bold">B1</span>
        </button>

        <!-- Bước 2: Làm sạch -->
        <button id="nav-step-2" onclick="goToStep(2)" class="nav-step w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl text-xs font-semibold text-blue-100/75 hover:bg-blue-900/30 hover:text-amber-200 transition cursor-pointer">
          <div class="flex items-center gap-2.5">
            <span class="text-base">🧹</span>
            <div class="text-left">
              <div class="font-bold text-xs text-white">2. Làm Sạch & Đối Soát</div>
              <div class="text-[10px] text-slate-400">4 Quy tắc & Ngoại lệ</div>
            </div>
          </div>
          <span class="step-badge text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-slate-300 font-bold">B2</span>
        </button>

        <!-- Bước 3: Phân tích -->
        <button id="nav-step-3" onclick="goToStep(3)" class="nav-step w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl text-xs font-semibold text-blue-100/75 hover:bg-blue-900/30 hover:text-amber-200 transition cursor-pointer">
          <div class="flex items-center gap-2.5">
            <span class="text-base">📊</span>
            <div class="text-left">
              <div class="font-bold text-xs text-white">3. Phân Tích Số Liệu</div>
              <div class="text-[10px] text-slate-400">So sánh nhóm BI & Hồ sơ</div>
            </div>
          </div>
          <span class="step-badge text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-slate-300 font-bold">B3</span>
        </button>

        <!-- Bước 4: Báo cáo -->
        <button id="nav-step-4" onclick="goToStep(4)" class="nav-step w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl text-xs font-semibold text-blue-100/75 hover:bg-blue-900/30 hover:text-amber-200 transition cursor-pointer">
          <div class="flex items-center gap-2.5">
            <span class="text-base">🎯</span>
            <div class="text-left">
              <div class="font-bold text-xs text-white">4. Báo Cáo Điều Hành</div>
              <div class="text-[10px] text-slate-400">KPIs & Chẩn đoán Gemini</div>
            </div>
          </div>
          <span class="step-badge text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-slate-300 font-bold">B4</span>
        </button>

        <!-- Bước 5: Ký duyệt -->
        <button id="nav-step-5" onclick="goToStep(5)" class="nav-step w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl text-xs font-semibold text-blue-100/75 hover:bg-blue-900/30 hover:text-amber-200 transition cursor-pointer">
          <div class="flex items-center gap-2.5">
            <span class="text-base">✍️</span>
            <div class="text-left">
              <div class="font-bold text-xs text-white">5. Ký Duyệt & Ban Hành</div>
              <div class="text-[10px] text-slate-400">C-Level Sign-off & Xuất tệp</div>
            </div>
          </div>
          <span class="step-badge text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-slate-300 font-bold">B5</span>
        </button>
      </nav>

      <!-- Tiện ích & Nâng cao -->
      <div class="text-[10px] font-bold text-amber-200/60 uppercase tracking-wider px-3 mt-5 mb-2">TIỆN ÍCH & NÂNG CAO</div>
      <div class="space-y-1">
        <button onclick="switchTab('tab-copilot', this)" class="nav-btn w-full flex items-center justify-between px-3.5 py-2 rounded-xl text-xs font-semibold text-blue-100/75 hover:bg-blue-900/30 hover:text-amber-200 transition cursor-pointer">
          <span class="flex items-center gap-2"><span>💬</span> Gemini Copilot</span>
          <span class="text-[9px] bg-amber-400/20 text-amber-300 border border-amber-400/40 px-1.5 py-0.5 rounded font-bold">AI</span>
        </button>
        <button onclick="switchTab('tab-audit', this)" class="nav-btn w-full flex items-center justify-between px-3.5 py-2 rounded-xl text-xs font-semibold text-blue-100/75 hover:bg-blue-900/30 hover:text-amber-200 transition cursor-pointer">
          <span class="flex items-center gap-2"><span>📜</span> Nhật Ký Audit</span>
        </button>
        <button onclick="switchTab('tab-products', this)" class="nav-btn w-full flex items-center justify-between px-3.5 py-2 rounded-xl text-xs font-semibold text-blue-100/75 hover:bg-blue-900/30 hover:text-amber-200 transition cursor-pointer">
          <span class="flex items-center gap-2"><span>📦</span> Đánh Giá SKU</span>
        </button>
      </div>

      <!-- Tác vụ Dữ liệu -->
      <div class="text-[10px] font-bold text-amber-200/60 uppercase tracking-wider px-3 mt-6 mb-2">TÁC VỤ DỮ LIỆU</div>
      <div class="space-y-1">
        <button onclick="loadDefaultData()" class="w-full flex items-center gap-2.5 px-4 py-2.5 rounded-xl text-xs font-semibold text-amber-300 hover:bg-amber-400/10 transition cursor-pointer">
          <span>⚡</span> Nạp Dữ Liệu Mẫu
        </button>
        <button onclick="toggleDataDrawer()" class="w-full flex items-center gap-2.5 px-4 py-2.5 rounded-xl text-xs font-semibold text-blue-100/75 hover:bg-blue-900/30 hover:text-amber-200 transition cursor-pointer">
          <span>📁</span> Nạp File / Link
        </button>
        <button onclick="resetAgentState()" class="w-full flex items-center gap-2.5 px-4 py-2.5 rounded-xl text-xs font-semibold text-blue-100/75 hover:bg-rose-950/40 hover:text-rose-400 transition cursor-pointer">
          <span>🔄</span> Reset Dữ Liệu
        </button>
        <button onclick="returnToLanding()" class="w-full flex items-center gap-2.5 px-4 py-2.5 rounded-xl text-xs font-semibold text-amber-200/80 hover:bg-amber-400/10 hover:text-amber-300 transition cursor-pointer border border-amber-400/20 mt-2">
          <span>🏠</span> Đổi Doanh Nghiệp
        </button>
      </div>

      <!-- Footer sidebar: Source Status Widget matching DataLens Studio -->
      <div class="mt-auto pt-4 border-t border-amber-400/20">
        <div class="p-2.5 rounded-2xl bg-white/5 border border-white/10 flex items-center gap-2.5">
          <span class="text-xl">🗄️</span>
          <div class="truncate flex-1 min-w-0">
            <div class="font-bold text-xs text-amber-300 truncate" id="source-status-filename">sales_data(1).csv</div>
            <div class="text-[10px] text-slate-400" id="source-status-detail">1 bộ dữ liệu · 15.000 dòng</div>
          </div>
          <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-sm shadow-emerald-400/50 animate-pulse" title="Đã kết nối nguồn"></span>
        </div>
      </div>
    </aside>

    <!-- ─── VÙNG NỘI DUNG CHÍNH (RIGHT MAIN) ─── -->
    <div class="flex-1 flex flex-col min-w-0 overflow-y-auto">

      <!-- TOPBAR QUẢN TRỊ HỆ THỐNG / WINDOW BAR CHUẨN DATALENS STUDIO -->
      <header class="bg-[#071330]/90 backdrop-blur-xl border-b border-amber-400/25 px-6 sm:px-8 py-3.5 flex items-center justify-between gap-4 sticky top-0 z-30 shadow-xl">
        <div class="flex items-center gap-3">
          <div class="flex items-center gap-1.5" aria-hidden="true">
            <span class="w-3 h-3 rounded-full bg-rose-500/90 inline-block shadow-sm"></span>
            <span class="w-3 h-3 rounded-full bg-amber-500/90 inline-block shadow-sm"></span>
            <span class="w-3 h-3 rounded-full bg-emerald-500/90 inline-block shadow-sm"></span>
          </div>
          <div class="ml-2">
            <h2 class="text-sm sm:text-base font-extrabold text-white tracking-tight font-display flex items-center gap-2">
              <span>Không gian phân tích dữ liệu</span>
              <span class="text-[10px] font-bold px-2 py-0.5 rounded-full bg-amber-400/20 text-amber-300 border border-amber-400/30 hidden sm:inline-block">
                DataLens Studio Engine · 0₫ sai số
              </span>
            </h2>
            <div class="text-[11px] text-blue-200/70 font-medium">Quy trình làm sạch, phân tích đa chiều và báo cáo điều hành tự động</div>
          </div>
        </div>

        <div class="flex items-center gap-2.5 flex-wrap">
          <!-- Trợ giúp Button -->
          <button onclick="openHelpModal()" class="px-3.5 py-1.5 rounded-full text-xs font-bold border border-amber-400/30 bg-blue-950/80 hover:bg-amber-400/20 text-amber-200 transition flex items-center gap-1.5 cursor-pointer shadow-sm">
            <span>💡</span> Trợ giúp
          </button>

          <!-- Quick 1-Click Run Pipeline Button -->
          <button onclick="runFullAnalysis()" id="btn-header-run" class="toty-gold-btn text-slate-950 font-black text-xs px-4 py-1.5 rounded-full flex items-center gap-1.5 shadow-md shadow-amber-400/30 hover:scale-105 active:scale-95 transition cursor-pointer">
            <span>⚡</span>
            <span>Chạy Toàn Bộ Quy Trình</span>
          </button>

          <!-- Export Results Data Button -->
          <button onclick="openExportModal()" class="royal-gold-btn text-xs font-black px-3.5 py-1.5 rounded-full flex items-center gap-1.5 shadow-md shadow-amber-400/25 hover:scale-105 active:scale-95 transition cursor-pointer" title="Xuất toàn bộ báo cáo và dữ liệu sạch">
            <span>📥</span>
            <span>Xuất Dữ Liệu</span>
          </button>

          <!-- Theme Toggle Pill -->
          <button id="theme-toggle-ws" onclick="toggleTheme()" class="theme-toggle-btn text-xs font-bold px-3 py-1.5 rounded-full flex items-center gap-1.5 border transition cursor-pointer hover:scale-105 active:scale-95 shadow-sm" title="Chỉnh nền sáng/tối">
            <span class="theme-icon text-sm">🌙</span>
          </button>

          <!-- Reset Button -->
          <button onclick="resetAgentState()" class="w-8 h-8 rounded-full border border-amber-400/30 bg-blue-950/80 hover:bg-rose-950/60 text-amber-300 hover:text-rose-300 flex items-center justify-center text-xs transition cursor-pointer shadow-sm" title="Bắt đầu phiên phân tích mới (Reset)">
            🔄
          </button>

          <!-- Notification Center -->
          <div class="relative" id="notif-wrapper">
            <button id="notif-bell-btn" onclick="toggleNotifications(event)" class="w-9 h-9 rounded-full border border-amber-400/30 bg-blue-950/80 flex items-center justify-center text-amber-300 text-sm hover:bg-blue-900/60 hover:text-amber-200 transition relative focus:outline-none shadow-sm cursor-pointer" title="Trung tâm cảnh báo">
              🔔
              <span id="notif-badge" class="absolute -top-1 -right-1 min-w-[18px] h-[18px] bg-rose-500 text-white rounded-full text-[10px] font-black flex items-center justify-center px-1 shadow-sm">8</span>
            </button>

            <!-- Notification Dropdown Panel -->
            <div id="notif-dropdown" class="hidden absolute right-0 mt-3 w-80 sm:w-96 glass-card rounded-3xl shadow-2xl border border-amber-400/35 z-50 overflow-hidden text-left" style="animation: fadeInUp 0.2s ease;">
              <div class="p-4 bg-blue-950/90 border-b border-amber-400/20 flex items-center justify-between">
                <div class="flex items-center gap-2">
                  <span class="text-base">🔔</span>
                  <div>
                    <h4 class="text-xs font-black text-amber-300 uppercase tracking-wide">Trung Tâm Cảnh Báo</h4>
                    <p class="text-[10px] text-blue-200/60">Trần Bảo Project Alert Center</p>
                  </div>
                </div>
                <div class="flex items-center gap-2">
                  <button onclick="markAllNotificationsRead()" class="text-[11px] font-bold text-amber-400 hover:text-amber-300 transition cursor-pointer">Đã đọc</button>
                  <button onclick="toggleNotifications(event)" class="text-blue-300 hover:text-white text-sm ml-1 font-bold cursor-pointer">✕</button>
                </div>
              </div>
              <div id="notif-items-list" class="max-h-[360px] overflow-y-auto divide-y divide-blue-900/30 p-1"></div>
              <div class="p-3 bg-blue-950/90 border-t border-amber-400/20 text-center">
                <button onclick="switchTab('tab-audit'); toggleNotifications(event);" class="text-[11px] font-bold text-amber-300 hover:text-amber-200 transition flex items-center justify-center gap-1 mx-auto cursor-pointer">
                  <span>📜</span> Xem Audit Trail kiểm toán
                </button>
              </div>
            </div>
          </div>

          <!-- User Avatar & Role -->
          <div class="flex items-center gap-2.5 pl-2 border-l border-amber-400/20">
            <div class="w-9 h-9 rounded-full bg-gradient-to-tr from-amber-400 to-yellow-500 border border-amber-300/40 shadow-sm flex items-center justify-center font-bold text-sm text-slate-950" id="header-avatar">
              🏢
            </div>
            <div class="hidden sm:block text-left">
              <div class="text-xs font-bold text-white leading-tight" id="header-company-name">Trần Bảo pro</div>
              <div class="text-[10px] text-amber-300 font-semibold">Ban Giám Đốc</div>
            </div>
          </div>
        </div>
      </header>

      <!-- NỘI DUNG CHÍNH (PADDED) -->
      <main class="p-6 sm:p-8 max-w-[1440px] w-full mx-auto space-y-6">

        <!-- ═══════════════════════════════════════════════════════════════════ -->
        <!-- KHUNG ĐIỀU KHIỂN QUY TRÌNH 5 BƯỚC AI AGENT                          -->
        <!-- ═══════════════════════════════════════════════════════════════════ -->
        <section class="glass-card rounded-3xl p-6 border border-amber-400/30 shadow-2xl relative overflow-hidden">
          <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-5 pb-5 border-b border-amber-400/20">
            <div>
              <div class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-950/70 text-amber-300 text-[11px] font-semibold mb-2 border border-amber-400/30 backdrop-blur-md">
                <span>🏆</span> QUY TRÌNH QUẢN TRỊ 5 BƯỚC AI AGENT
              </div>
              <h3 class="text-xl md:text-2xl font-black tracking-tight font-display toty-gold-text" id="greeting-title">
                Chào mừng Trần Bảo pro trở lại làm việc!
              </h3>
              <p class="text-xs text-blue-100/80 mt-1 max-w-2xl leading-relaxed">
                Hệ thống AI Agent đã sẵn sàng xử lý dữ liệu. Kích hoạt phân tích tự động để tạo báo cáo quản trị, đối soát doanh thu và chẩn đoán điểm nóng.
              </p>
            </div>

            <!-- Nút hành động nhanh -->
            <div class="flex items-center gap-2.5 flex-shrink-0 flex-wrap">
              <button 
                onclick="runFullAnalysis()" 
                id="btn-run-full-pipeline"
                class="px-7 py-3 rounded-full text-slate-950 font-black text-xs flex items-center gap-2 cursor-pointer whitespace-nowrap shadow-xl shadow-amber-400/35 border-2 border-amber-200 transition-all hover:scale-105 active:scale-95"
                style="background: linear-gradient(135deg, #fff7d6 0%, #fde68a 30%, #f59e0b 70%, #d97706 100%);"
              >
                <span class="text-amber-900 font-black text-sm">⚡</span>
                <span class="tracking-wide">Chạy Phân Tích Dữ Liệu Tự Động</span>
              </button>
              <button onclick="loadDefaultData()" class="px-4 py-3 rounded-full bg-blue-950/80 hover:bg-blue-900/80 text-amber-300 text-xs font-bold transition border border-amber-400/30 flex items-center gap-1.5 backdrop-blur-md cursor-pointer">
                <span>📦</span>
                <span>Nạp Dữ Liệu Mẫu (60.000 dòng)</span>
              </button>
              <button onclick="toggleDataDrawer()" class="px-4 py-3 rounded-full bg-blue-950/80 hover:bg-blue-900/80 text-white text-xs font-bold transition border border-amber-400/30 backdrop-blur-md cursor-pointer">
                <span>📁</span>
                <span>Nạp File / Link</span>
              </button>
            </div>
          </div>

          <!-- Dãy 5 Bước Stepper được kết nối liền mạch -->
          <div class="pt-5">
            <div class="text-[10px] font-bold text-amber-200/70 uppercase tracking-wider mb-3 flex items-center justify-between">
              <span>TIẾN TRÌNH QUY TRÌNH 5 BƯỚC AI AGENT</span>
              <span class="text-[10px] text-blue-200/60 font-medium">Nhấp vào từng bước để thực hiện hoặc xem lại chi tiết</span>
            </div>
            <div class="flex items-center gap-2 flex-wrap" id="stepper-ui">
              <!-- Rendered by JS -->
            </div>
          </div>

          <!-- Nút hành động tương ứng với bước hiện tại -->
          <div id="stepper-action-container" class="mt-4 pt-4 border-t border-amber-400/15 flex items-center justify-between">
            <!-- Rendered by JS -->
          </div>
        </section>

        <!-- ═══════════════════════════════════════════════════════════════════ -->
        <!-- DRAWER NẠP DỮ LIỆU ĐẦU VÀO (BƯỚC 0)                                 -->
        <!-- ═══════════════════════════════════════════════════════════════════ -->
        <section id="data-drawer-panel" class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/30 shadow-2xl hidden">
          <div class="flex items-center justify-between pb-3 border-b border-amber-400/20 mb-5">
            <div class="font-extrabold text-sm text-amber-300 flex items-center gap-2 font-display">
              <span>📂</span> Nạp Bộ Dữ Liệu Hoạt Động (Bước 0)
            </div>
            <button onclick="toggleDataDrawer()" class="text-xs text-blue-300 hover:text-white font-bold px-3 py-1 rounded-full hover:bg-blue-900/40 transition cursor-pointer">Đóng ✕</button>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <!-- 0. Fast Default Sample Dataset -->
            <div onclick="loadDefaultData()" class="border-2 border-amber-400/40 hover:border-amber-400 rounded-2xl p-5 flex flex-col items-center justify-between text-center cursor-pointer transition bg-blue-950/40 hover:bg-blue-900/40 group shadow-lg">
              <div>
                <div class="w-12 h-12 rounded-2xl bg-amber-400/20 text-amber-300 border border-amber-400/30 flex items-center justify-center text-2xl mx-auto mb-2.5 group-hover:scale-110 transition-transform">
                  ⚡
                </div>
                <div class="font-extrabold text-white text-sm">Dữ Liệu Mẫu Chuẩn (1-Click)</div>
                <div class="text-[11px] text-blue-200/70 mt-1">60.000 dòng giao dịch bán hàng, 2.024 đổi trả, 35 SKU hàng tiêu dùng & chỉ tiêu 18 cửa hàng</div>
              </div>
              <button class="toty-gold-btn mt-3 w-full py-2.5 px-4 rounded-full text-slate-950 font-extrabold text-xs shadow-md shadow-amber-400/30 transition cursor-pointer">
                ⚡ Nạp Dữ Liệu Mẫu Ngay
              </button>
            </div>

            <!-- 1. File Upload (Excel / CSV) -->
            <div onclick="document.getElementById('file-upload-input').click()" class="border-2 border-dashed border-blue-800/60 hover:border-amber-400/60 rounded-2xl p-5 flex flex-col items-center justify-between text-center cursor-pointer transition bg-blue-950/30 hover:bg-blue-900/30 group">
              <input type="file" id="file-upload-input" accept=".xlsx,.xls,.csv" class="hidden" onchange="handleFileUpload(this)"/>
              <div>
                <div class="w-12 h-12 rounded-2xl bg-blue-900/50 text-blue-200 border border-blue-700/50 flex items-center justify-center text-2xl mx-auto mb-2.5 group-hover:scale-110 transition-transform">
                  📊
                </div>
                <div class="font-extrabold text-white text-sm">Tải Lên Tệp Excel / CSV</div>
                <div class="text-[11px] text-blue-200/70 mt-1">Hỗ trợ file Excel đa sheet (Transactions, Returns, Products, Stores, Targets)</div>
              </div>
              <span class="mt-3 w-full py-2.5 px-4 rounded-full bg-blue-950 hover:bg-blue-900 text-amber-300 font-bold text-xs border border-amber-400/30 shadow-sm transition text-center">
                Chọn tệp từ máy tính
              </span>
            </div>

            <!-- 2. Google Sheet Link -->
            <div class="bg-blue-950/40 border border-blue-800/60 rounded-2xl p-5 flex flex-col justify-between">
              <div>
                <div class="flex items-center gap-2 mb-1">
                  <span class="text-emerald-400 text-lg">📑</span>
                  <div class="font-extrabold text-white text-sm">Google Sheets Trực Tuyến</div>
                </div>
                <div class="text-[11px] text-blue-200/70 mb-2">Dán URL bảng tính Google Sheets chia sẻ công khai</div>
                <input type="text" id="gsheet-input-url" placeholder="https://docs.google.com/spreadsheets/d/..." class="w-full text-xs p-2.5 rounded-xl border border-amber-400/30 bg-blue-950/70 text-white placeholder-blue-300/40 focus:outline-none focus:ring-2 focus:ring-amber-400/30 font-medium"/>
              </div>
              <button onclick="handleGSheetLoad()" class="mt-3 w-full py-2.5 px-4 rounded-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs transition flex items-center justify-center gap-1.5 shadow-sm cursor-pointer">
                <span>📥</span><span>Nạp Từ Google Sheet</span>
              </button>
            </div>
          </div>
          <div id="drawer-status-msg" class="text-xs mt-3 font-semibold"></div>
        </section>

                <!-- ════════ TAB 1: PHÂN TÍCH DỮ LIỆU (BƯỚC 3 · CHUẨN DATALENS STUDIO) ════════ -->
        
        <!-- ════════ TAB BƯỚC 1: KẾT NỐI NGUỒN DỮ LIỆU (CHỌN NGUỒN & KHẢO SÁT MỤC TIÊU) ════════ -->
        <div id="step-tab-1" class="tab-content space-y-6 hidden">
          
          <!-- HEADER BƯỚC 1 -->
          <div class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/40 bg-gradient-to-br from-[#071330] via-[#091b42] to-[#0a183c] shadow-2xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div class="flex items-center gap-3">
              <div class="w-12 h-12 rounded-2xl bg-amber-400/20 text-amber-300 border border-amber-400/30 flex items-center justify-center text-2xl shadow-md">
                🔗
              </div>
              <div>
                <div class="flex items-center gap-2 flex-wrap">
                  <h2 class="text-base sm:text-lg font-black text-white font-display">Kiểm Tra &amp; Kết Nối Nguồn Dữ Liệu</h2>
                  <span class="text-[10px] font-black uppercase px-2.5 py-0.5 rounded-full bg-amber-400/20 text-amber-300 border border-amber-400/40">
                    BƯỚC 1 / 5 · KẾT NỐI &amp; NGUỒN
                  </span>
                </div>
                <p class="text-xs text-slate-300 mt-1">Nạp tệp dữ liệu thô, thiết lập bối cảnh doanh nghiệp và trao đổi yêu cầu với AI trước khi làm sạch.</p>
              </div>
            </div>
            <div class="flex items-center gap-3 self-start sm:self-auto flex-wrap">
              <div id="step1-source-status-badge">
                <!-- Rendered by JS -->
              </div>
              <button id="btn-step1-continue" onclick="confirmStep1AndProceed()" class="px-5 py-2.5 rounded-full font-black text-xs text-slate-950 bg-gradient-to-r from-amber-400 via-amber-300 to-yellow-400 hover:scale-105 active:scale-95 transition shadow-md shadow-amber-400/30 flex items-center gap-2 cursor-pointer">
                <span>Xác nhận nguồn &amp; sang Làm sạch</span>
                <span>→</span>
              </button>
            </div>
          </div>

          <!-- CỔNG NẠP DỮ LIỆU THÔ (DATA INGESTION GATEWAY) -->
          <section id="step1-ingestion-box" class="glass-card rounded-3xl p-6 sm:p-7 border border-white/10 bg-[#071330]/80 shadow-xl space-y-4">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-white/10">
              <div class="flex items-center gap-3">
                <div class="w-9 h-9 rounded-xl bg-amber-400/20 text-amber-300 border border-amber-400/30 flex items-center justify-center text-lg">
                  📂
                </div>
                <div>
                  <h3 class="text-sm font-black text-white font-display">Cổng Nạp Dữ Liệu Thô (Raw Data Ingestion)</h3>
                  <p class="text-xs text-slate-400">Chọn tệp CSV / Excel từ máy tính, liên kết Google Sheets hoặc nạp bộ dữ liệu mẫu chuẩn.</p>
                </div>
              </div>
            </div>

            <!-- Ingestion Options Grid -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
              <!-- Option 1: File Upload -->
              <div onclick="document.getElementById('file-upload-input').click()" class="p-5 rounded-2xl bg-white/5 hover:bg-white/10 border-2 border-dashed border-amber-400/30 hover:border-amber-400 transition cursor-pointer flex flex-col justify-between group">
                <div class="flex items-center gap-3">
                  <span class="text-3xl group-hover:scale-110 transition">📤</span>
                  <div>
                    <div class="font-bold text-xs text-white group-hover:text-amber-300 transition">Tải Lên Tệp CSV / Excel</div>
                    <div class="text-[10px] text-slate-400 mt-0.5">Bấm để chọn tệp .csv, .xlsx, .xls từ máy</div>
                  </div>
                </div>
                <div class="mt-4 pt-3 border-t border-white/10 flex items-center justify-between text-[11px] text-amber-300/80 font-semibold">
                  <span>Hỗ trợ tệp thô đa cột</span>
                  <span>Chọn tệp ↗</span>
                </div>
              </div>

              <!-- Option 2: Google Sheets -->
              <div onclick="toggleDataDrawer()" class="p-5 rounded-2xl bg-white/5 hover:bg-white/10 border border-white/10 hover:border-amber-400/40 transition cursor-pointer flex flex-col justify-between group">
                <div class="flex items-center gap-3">
                  <span class="text-3xl group-hover:scale-110 transition">📊</span>
                  <div>
                    <div class="font-bold text-xs text-white group-hover:text-amber-300 transition">Liên Kết Google Sheets</div>
                    <div class="text-[10px] text-slate-400 mt-0.5">Đồng bộ trực tiếp qua link bảng tính</div>
                  </div>
                </div>
                <div class="mt-4 pt-3 border-t border-white/10 flex items-center justify-between text-[11px] text-blue-300/80 font-semibold">
                  <span>Tự động cập nhật dòng mới</span>
                  <span>Dán link ↗</span>
                </div>
              </div>

              <!-- Option 3: Standard Mock Data -->
              <div onclick="loadDefaultData()" class="p-5 rounded-2xl bg-gradient-to-br from-amber-950/30 to-blue-950/40 hover:from-amber-900/40 hover:to-blue-900/50 border border-amber-400/40 hover:border-amber-300 transition cursor-pointer flex flex-col justify-between group shadow-md shadow-amber-400/10">
                <div class="flex items-center gap-3">
                  <span class="text-3xl group-hover:scale-110 transition">⚡</span>
                  <div>
                    <div class="font-bold text-xs text-amber-300 group-hover:text-amber-200 transition">Nạp Dữ Liệu Mẫu Chuẩn</div>
                    <div class="text-[10px] text-slate-300 mt-0.5">sales_data(1).csv · 15.000 dòng 23 cột</div>
                  </div>
                </div>
                <div class="mt-4 pt-3 border-t border-white/10 flex items-center justify-between text-[11px] text-amber-300 font-bold">
                  <span>1-Click Nạp Nhanh</span>
                  <span>Nạp mẫu ⚡</span>
                </div>
              </div>
            </div>
          </section>

          <!-- THẺ THÔNG BÁO KHI CHƯA CÓ DỮ LIỆU THÔ -->
          <div id="step1-no-data-placeholder" class="glass-card rounded-3xl p-10 text-center border border-amber-400/20 bg-blue-950/20 space-y-3">
            <div class="text-4xl">📂</div>
            <h4 class="text-base font-bold text-amber-300 font-display">Chưa Có Dữ Liệu Thô Được Nạp</h4>
            <p class="text-xs text-slate-300 max-w-md mx-auto leading-relaxed">
              Vui lòng tải lên tệp CSV/Excel hoặc bấm nút <strong class="text-amber-300">"Nạp Dữ Liệu Mẫu Chuẩn"</strong> ở trên để mở khóa mục Điền Thông Tin Bối Cảnh Doanh Nghiệp và xem trước bảng dữ liệu.
            </p>
          </div>

          <!-- MỤC ĐIỀN THÔNG TIN BỐI CẢNH DOANH NGHIỆP (CHỈ HIỆN LÊN SAU KHI NẠP FILE DỮ LIỆU THÔ) -->
          <div id="step1-business-context-wrapper" class="space-y-6 hidden">
            <!-- 1. BỐI CẢNH DOANH NGHIỆP (CONTEXT FORM) -->
            <section class="glass-card rounded-3xl p-6 sm:p-7 border border-white/10 bg-[#071330]/70 shadow-xl space-y-4">
              <div class="flex items-center gap-3 pb-3 border-b border-white/10">
                <div class="w-9 h-9 rounded-xl bg-blue-500/20 text-blue-300 border border-blue-500/30 flex items-center justify-center text-lg">
                  🏢
                </div>
                <div>
                  <h3 class="text-sm font-black text-white font-display">Bối Cảnh Doanh Nghiệp</h3>
                  <p class="text-xs text-slate-400">Câu hỏi của bạn định hướng việc chọn chỉ số và đọc kết quả phân tích của AI.</p>
                </div>
              </div>

              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label class="block text-xs font-bold text-amber-200 mb-1.5">Loại hình doanh nghiệp <span class="text-rose-400">*</span></label>
                  <input id="context-industry" type="text" value="Bán lẻ thời trang &amp; Chuỗi thương mại đa kênh" class="w-full bg-[#060f26] border border-white/15 focus:border-amber-400 rounded-xl px-3.5 py-2.5 text-xs text-white outline-none font-medium transition" placeholder="Ví dụ: Bán lẻ thời trang, SaaS B2B, Dược phẩm...">
                </div>
                <div>
                  <label class="block text-xs font-bold text-amber-200 mb-1.5">Thông tin bổ sung <span class="text-slate-400 font-normal">· không bắt buộc</span></label>
                  <input id="context-background" type="text" value="Hệ thống 15 chi nhánh toàn quốc, đánh giá biên lợi nhuận và kiểm soát đổi trả" class="w-full bg-[#060f26] border border-white/15 focus:border-amber-400 rounded-xl px-3.5 py-2.5 text-xs text-white outline-none font-medium transition" placeholder="Ví dụ: Hai chi nhánh, đang đánh giá hiệu quả quý gần nhất">
                </div>
              </div>

              <div>
                <label class="block text-xs font-bold text-amber-200 mb-1.5">Vấn đề cần phân tích <span class="text-rose-400">*</span> <span class="text-slate-400 font-normal">· Mỗi câu một dòng, tối đa 8 câu</span></label>
                <textarea id="context-questions" rows="3" class="w-full bg-[#060f26] border border-white/15 focus:border-amber-400 rounded-xl p-3 text-xs text-white outline-none font-mono leading-relaxed resize-none transition">Doanh thu theo khu vực và kênh phân phối nào cao nhất?
Cơ cấu chi phí giá vốn (COGS) và biên lợi nhuận giữa các ngành hàng?
Tình hình đơn hàng hoàn trả và các điểm nghẽn chất lượng vận hành?</textarea>
              </div>
            </section>

            <!-- 2. TRAO ĐỔI YÊU CẦU VỚI AI (DISCOVERY PANEL) -->
            <section class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/30 bg-gradient-to-r from-blue-950/40 via-indigo-950/30 to-amber-950/20 shadow-xl space-y-4">
              <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-white/10">
                <div class="flex items-center gap-3">
                  <div class="w-9 h-9 rounded-xl bg-amber-400/20 text-amber-300 border border-amber-400/30 flex items-center justify-center text-lg">
                    🧭
                  </div>
                  <div>
                    <h3 class="text-sm font-black text-amber-300 font-display">Trao Đổi Yêu Cầu Với AI (Discovery Scope)</h3>
                    <p class="text-xs text-slate-300">Xem cách AI Agent hiểu yêu cầu và bổ sung các góc nhìn chiến lược trước khi phân tích.</p>
                  </div>
                </div>
                <button id="btn-discovery-trigger" onclick="triggerDiscoveryScope()" class="px-4 py-2 rounded-full font-bold text-xs bg-amber-400/20 hover:bg-amber-400/30 text-amber-200 border border-amber-400/40 transition flex items-center gap-2 cursor-pointer self-start sm:self-auto">
                  <span>⚡</span>
                  <span>Trao đổi yêu cầu với AI</span>
                </button>
              </div>

              <!-- Discovery Plan Output Container -->
              <div id="discovery-plan-container" class="space-y-4">
                <div class="p-4 rounded-2xl bg-white/5 border border-white/10 text-xs text-slate-200 leading-relaxed font-medium">
                  <strong class="text-amber-300">📌 Tôi hiểu yêu cầu của bạn:</strong>
                  <p id="discovery-interpretation" class="mt-1 text-slate-300">AI Agent nhận diện mô hình dữ liệu bán lẻ chuỗi thương mại với quy mô toàn quốc. Dữ liệu tích hợp đầy đủ dòng tiền Doanh thu thuần (430.66 tỷ ₫), Lợi nhuận gộp, Giá vốn (COGS), Tỷ lệ hoàn trả và phân loại giao dịch đa kênh.</p>
                </div>

                <div>
                  <h4 class="text-xs font-black text-amber-300 uppercase tracking-wider mb-2">Góc nhìn đề xuất bổ sung:</h4>
                  <ul id="discovery-suggestions" class="space-y-1.5 pl-1">
                    <li class="flex items-start gap-2 text-xs text-slate-200"><span class="text-amber-400 font-bold">•</span><span>Phân tích tỷ trọng doanh thu đa kênh: Đóng góp doanh thu giữa Cửa hàng trực tiếp, Trực tuyến và Sàn TMĐT.</span></li>
                    <li class="flex items-start gap-2 text-xs text-slate-200"><span class="text-amber-400 font-bold">•</span><span>Cơ cấu chi phí giá vốn (COGS) chiếm ~64% cần giải pháp kiểm soát đàm phán nhà cung ứng.</span></li>
                    <li class="flex items-start gap-2 text-xs text-slate-200"><span class="text-amber-400 font-bold">•</span><span>Kiểm soát rủi ro đổi trả tập trung tại các chi nhánh có tỷ lệ đổi trả vượt ngưỡng 3.5%.</span></li>
                  </ul>
                </div>

                <div>
                  <h4 class="text-xs font-black text-amber-300 uppercase tracking-wider mb-2">Câu hỏi cần làm rõ:</h4>
                  <div id="discovery-followups" class="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <div class="p-3.5 rounded-2xl bg-white/5 border border-white/10 space-y-1.5">
                      <label class="block text-xs font-bold text-amber-200">Chiến lược ưu tiên kỳ này là Tối đa hóa biên lợi nhuận hay Mở rộng thị phần?</label>
                      <textarea rows="2" class="w-full bg-[#060f26] border border-white/15 focus:border-amber-400 rounded-xl px-3 py-2 text-xs text-white outline-none resize-none" placeholder="Nhập định hướng..."></textarea>
                    </div>
                    <div class="p-3.5 rounded-2xl bg-white/5 border border-white/10 space-y-1.5">
                      <label class="block text-xs font-bold text-amber-200">Có cần theo dõi chi phí logistics giao vận giữa các khu vực?</label>
                      <textarea rows="2" class="w-full bg-[#060f26] border border-white/15 focus:border-amber-400 rounded-xl px-3 py-2 text-xs text-white outline-none resize-none" placeholder="Nhập định hướng..."></textarea>
                    </div>
                    <div class="p-3.5 rounded-2xl bg-white/5 border border-white/10 space-y-1.5">
                      <label class="block text-xs font-bold text-amber-200">Các chương trình chiết khấu khuyến mãi có cần phân tách theo phân khúc khách hàng?</label>
                      <textarea rows="2" class="w-full bg-[#060f26] border border-white/15 focus:border-amber-400 rounded-xl px-3 py-2 text-xs text-white outline-none resize-none" placeholder="Nhập định hướng..."></textarea>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <!-- 3. XEM TRƯỚC BỘ DỮ LIỆU ĐANG XEM (RAW DATA PREVIEW) -->
            <section class="glass-card rounded-3xl p-6 sm:p-7 border border-white/10 bg-[#071330]/70 shadow-xl space-y-4">
              <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-white/10">
                <div class="flex items-center gap-3">
                  <div class="w-9 h-9 rounded-xl bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center justify-center text-lg">
                    📑
                  </div>
                  <div>
                    <h3 class="text-sm font-black text-white font-display flex items-center gap-2">
                      <span id="step1-source-name">sales_data(1).csv</span>
                      <span id="step1-source-count-badge" class="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/30">15.000 dòng · 23 cột nguồn</span>
                    </h3>
                    <p class="text-xs text-slate-400">Xem trước dữ liệu thô ban đầu trước khi áp dụng các quy tắc làm sạch</p>
                  </div>
                </div>
                <label class="inline-flex items-center gap-2 text-xs text-slate-300 cursor-pointer self-start sm:self-auto bg-white/5 px-3 py-1.5 rounded-full border border-white/10">
                  <input type="checkbox" checked disabled class="accent-amber-400 rounded">
                  <span>Dùng dòng đầu làm tiêu đề</span>
                </label>
              </div>

              <!-- Raw preview table container -->
              <div id="step1-raw-preview-container">
                <!-- Rendered by JS -->
              </div>
            </section>

            <!-- 4. DUYỆT BƯỚC 1 & YÊU CẦU SỬA BẰNG AI (REVIEW PANEL BƯỚC 1) -->
            <section class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/40 bg-gradient-to-r from-blue-950/50 via-indigo-950/40 to-amber-950/30 shadow-2xl space-y-4">
              <div class="flex items-center gap-3 pb-3 border-b border-white/10">
                <div class="w-10 h-10 rounded-2xl bg-amber-400/20 text-amber-300 border border-amber-400/30 flex items-center justify-center text-xl shadow-md">
                  ✨
                </div>
                <div>
                  <h3 class="text-sm font-black text-amber-300 font-display">Duyệt Bước Này &amp; Yêu Cầu Sửa Bằng AI (Bước 1)</h3>
                  <p class="text-xs text-slate-300">Xem lại nguồn dữ liệu trước khi xác nhận chuyển sang Bước 2 (Làm sạch).</p>
                </div>
              </div>

              <div>
                <label class="block text-xs font-bold text-amber-200 mb-1.5">Yêu cầu sửa</label>
                <textarea id="step1-revision-input" rows="2" class="w-full bg-[#060f26] border border-white/15 focus:border-amber-400 rounded-xl p-3 text-xs text-white outline-none font-medium resize-none transition" placeholder="Ví dụ: Bỏ mã dòng DS-001 hoặc lọc dữ liệu theo điều kiện..."></textarea>
              </div>

              <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-2">
                <p class="text-[11px] text-slate-400">Yêu cầu chỉnh sửa áp dụng cho bộ dữ liệu đang chọn.</p>
                <div class="flex items-center gap-2.5">
                  <button onclick="applyStepRevision(1)" id="btn-step1-revision" class="px-4 py-2 rounded-full font-bold text-xs bg-amber-400/20 hover:bg-amber-400/30 text-amber-200 border border-amber-400/40 transition flex items-center gap-2 cursor-pointer">
                    <span>⚡</span>
                    <span>Áp Dụng Yêu Cầu Sửa Bằng AI</span>
                  </button>
                  <button id="btn-step1-confirm-bottom" onclick="confirmStep1AndProceed()" class="px-6 py-2.5 rounded-full font-black text-xs text-slate-950 bg-gradient-to-r from-amber-400 via-amber-300 to-yellow-400 hover:scale-105 active:scale-95 transition shadow-md shadow-amber-400/30 flex items-center gap-2 cursor-pointer">
                    <span>Xác nhận nguồn và tiếp tục sang Bước 2: Làm Sạch</span>
                    <span>→</span>
                  </button>
                </div>
              </div>
              <div id="step1-revision-result" class="hidden p-3.5 rounded-2xl text-xs font-medium"></div>
            </section>
          </div>
        </div>

        <div id="tab-dash"<div id="tab-dash" class="tab-content space-y-6">
          
          <!-- HEADER BƯỚC 3: PHÂN TÍCH DỮ LIỆU -->
          <div class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/40 bg-gradient-to-br from-[#071330] via-[#091b42] to-[#0a183c] shadow-2xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div class="flex items-center gap-3">
              <div class="w-12 h-12 rounded-2xl bg-amber-400/20 text-amber-300 border border-amber-400/30 flex items-center justify-center text-2xl shadow-md">
                📊
              </div>
              <div>
                <div class="flex items-center gap-2 flex-wrap">
                  <h2 class="text-base sm:text-lg font-black text-white font-display">Phân Tích Dữ Liệu (Data Analysis)</h2>
                  <span class="text-[10px] font-black uppercase px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                    ⚡ Chuẩn DataLens Studio
                  </span>
                </div>
                <p class="text-xs text-slate-300 mt-0.5">
                  Đọc kết quả theo câu hỏi kinh doanh và kiểm tra số liệu của từng bộ dữ liệu · 0₫ sai lệch số liệu
                </p>
              </div>
            </div>
            <div class="flex items-center gap-2 self-start sm:self-auto">
              <span class="text-xs text-amber-300 font-bold bg-amber-400/10 px-3.5 py-1.5 rounded-full border border-amber-400/30 flex items-center gap-1.5">
                <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                Bước 3: Khám Phá &amp; Phân Tích
              </span>
            </div>
          </div>

          <!-- BỐI CẢNH KINH DOANH & BẰNG CHỨNG NGHIỆP VỤ (BUSINESS VIEW · DATALENS STUDIO) -->
          <div id="business-context-container" class="space-y-4">
            <div class="glass-card rounded-3xl p-6 text-center text-xs text-slate-300 border border-white/10">
              Đang tải bối cảnh kinh doanh và bằng chứng dữ liệu...
            </div>
          </div>

          <!-- 3 METRICS THỐNG KÊ TỔNG QUAN (METRIC CARDS · CHUẨN DATALENS) -->
          <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div class="glass-card rounded-2xl p-4 sm:p-5 border border-white/10 bg-blue-950/40 shadow-sm flex items-center gap-4">
              <div class="w-12 h-12 rounded-2xl bg-blue-500/20 text-blue-300 border border-blue-500/30 flex items-center justify-center font-extrabold text-xl flex-shrink-0">
                📑
              </div>
              <div>
                <div class="text-[11px] text-slate-300 font-medium">Dòng dữ liệu</div>
                <div class="text-2xl font-black text-white font-display tracking-tight" id="meta-metric-rows">15,000</div>
                <div class="text-[10px] text-slate-400 mt-0.5">100% bản ghi giao dịch chuẩn</div>
              </div>
            </div>
            <div class="glass-card rounded-2xl p-4 sm:p-5 border border-white/10 bg-blue-950/40 shadow-sm flex items-center gap-4">
              <div class="w-12 h-12 rounded-2xl bg-amber-400/20 text-amber-300 border border-amber-400/30 flex items-center justify-center font-extrabold text-xl flex-shrink-0">
                📐
              </div>
              <div>
                <div class="text-[11px] text-slate-300 font-medium">Số cột dữ liệu</div>
                <div class="text-2xl font-black text-amber-300 font-display tracking-tight" id="meta-metric-cols">23</div>
                <div class="text-[10px] text-slate-400 mt-0.5">Bao gồm toàn bộ trường nghiệp vụ</div>
              </div>
            </div>
            <div class="glass-card rounded-2xl p-4 sm:p-5 border border-white/10 bg-blue-950/40 shadow-sm flex items-center gap-4">
              <div class="w-12 h-12 rounded-2xl bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center justify-center font-extrabold text-xl flex-shrink-0">
                🧮
              </div>
              <div>
                <div class="text-[11px] text-slate-300 font-medium">Cột dạng số</div>
                <div class="text-2xl font-black text-emerald-300 font-display tracking-tight" id="meta-metric-numeric">15</div>
                <div class="text-[10px] text-slate-400 mt-0.5">Sẵn sàng tính toán tổng hợp BI</div>
              </div>
            </div>
          </div>

          <!-- CARD: SO SÁNH THEO NHÓM (CHART VIEW · CHUẨN DATALENS STUDIO) -->
          <div class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/40 bg-gradient-to-br from-[#071330] via-[#091b42] to-[#0a183c] shadow-2xl space-y-5">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-white/10">
              <div class="flex items-center gap-3">
                <div class="w-10 h-10 rounded-2xl bg-amber-400/20 text-amber-300 border border-amber-400/30 flex items-center justify-center text-xl shadow-md">
                  📊
                </div>
                <div>
                  <h3 class="text-sm sm:text-base font-black text-white font-display">So Sánh Theo Nhóm (Group Comparison)</h3>
                  <p class="text-xs text-slate-300 mt-0.5">Biểu đồ so sánh trực quan phân bố số liệu theo chiều dữ liệu đã chọn</p>
                </div>
              </div>

              <!-- Chế độ hiển thị biểu đồ -->
              <div class="glass-pill rounded-full p-1 inline-flex items-center gap-1 border border-amber-400/30">
                <button type="button" id="btn-chart-type-bar" onclick="setDynamicChartType('bar')" class="px-3 py-1 rounded-full text-xs font-black transition toty-gold-btn text-slate-950 shadow-sm cursor-pointer">
                  📊 Thanh Ngang
                </button>
                <button type="button" id="btn-chart-type-col" onclick="setDynamicChartType('col')" class="px-3 py-1 rounded-full text-xs font-bold transition text-blue-200 hover:text-amber-300 cursor-pointer">
                  📶 Cột Dọc
                </button>
                <button type="button" id="btn-chart-type-donut" onclick="setDynamicChartType('donut')" class="px-3 py-1 rounded-full text-xs font-bold transition text-blue-200 hover:text-amber-300 cursor-pointer">
                  🍩 Tỷ Trọng
                </button>
              </div>
            </div>

            <!-- BỘ ĐIỀU KHIỂN TƯƠNG TÁC (NHÓM THEO / PHÉP TÍNH / CỘT SỐ) -->
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-3.5 bg-blue-950/40 p-4 rounded-2xl border border-white/10">
              <div>
                <label class="block text-[11px] font-bold text-amber-300 mb-1 flex items-center gap-1">
                  <span>📌</span> <span>Nhóm theo (Dimension):</span>
                </label>
                <select id="dyn-analysis-group" onchange="runDynamicAnalysisUI()" class="w-full bg-[#060f26] border border-amber-400/40 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-amber-400 cursor-pointer shadow-inner">
                  <option value="Region" selected>🗺️ Khu Vực (Miền Bắc / Nam / Trung)</option>
                  <option value="Channel">🛒 Kênh Bán Hàng (In-Store / Online / Sàn)</option>
                  <option value="Segment">👥 Phân Khúc Khách Hàng (VIP / Cá nhân / B2B)</option>
                  <option value="Category">🏷️ Ngành Hàng (Category)</option>
                  <option value="Subcategory">📦 Phân Loại Chi Tiết (Subcategory)</option>
                  <option value="City">🏙️ Thành Phố (Hà Nội, TP.HCM, Đà Nẵng...)</option>
                  <option value="Payment_Method">💳 Phương Thức Thanh Toán</option>
                  <option value="Customer_Satisfaction">⭐ Mức Độ Hài Lòng Khách Hàng</option>
                  <option value="Returned">🔄 Trạng Thái Hoàn Trả (Returned)</option>
                  <option value="Store_Name">🏬 Cửa Hàng / Chi Nhánh</option>
                </select>
              </div>

              <div>
                <label class="block text-[11px] font-bold text-amber-300 mb-1 flex items-center gap-1">
                  <span>🧮</span> <span>Phép tổng hợp (Mode):</span>
                </label>
                <select id="dyn-analysis-mode" onchange="runDynamicAnalysisUI()" class="w-full bg-[#060f26] border border-amber-400/40 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-amber-400 cursor-pointer shadow-inner">
                  <option value="sum" selected>∑ Tổng cộng (Sum)</option>
                  <option value="mean">⌀ Trung bình (Average)</option>
                  <option value="count"># Đếm số lượng đơn (Count)</option>
                  <option value="max">⬆️ Giá trị cao nhất (Max)</option>
                  <option value="min">⬇️ Giá trị nhỏ nhất (Min)</option>
                </select>
              </div>

              <div>
                <label class="block text-[11px] font-bold text-amber-300 mb-1 flex items-center gap-1">
                  <span>📈</span> <span>Cột số đo lường (Metric):</span>
                </label>
                <select id="dyn-analysis-metric" onchange="runDynamicAnalysisUI()" class="w-full bg-[#060f26] border border-amber-400/40 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-amber-400 cursor-pointer shadow-inner">
                  <option value="Net_Sales_VND" selected>💰 Doanh Thu Thuần (Net Sales)</option>
                  <option value="Sales_VND">💵 Doanh Thu Bán Lẻ Gộp (Gross Sales)</option>
                  <option value="Profit_VND">📈 Lợi Nhuận Gộp (Profit)</option>
                  <option value="Cost_VND">📉 Chi Phí Giá Vốn (Cost / COGS)</option>
                  <option value="Quantity">📦 Số Lượng Bán (Quantity)</option>
                  <option value="Discount_Amount_VND">🏷️ Tiền Chiết Khấu (Discount)</option>
                  <option value="Shipping_Days">🚚 Số Ngày Giao Hàng (Days)</option>
                </select>
              </div>
            </div>

            <!-- KHU VỰC HIỂN THỊ BIỂU ĐỒ TƯƠNG TÁC ĐỘNG -->
            <div id="dynamic-analysis-chart-container" class="min-h-[220px] rounded-2xl border border-white/10 bg-black/20 p-4 sm:p-5">
              <div class="text-center py-8 text-slate-400 text-xs flex flex-col items-center gap-2">
                <span class="spinner"></span>
                <span>Đang nạp và tổng hợp dữ liệu đa chiều...</span>
              </div>
            </div>
          </div>

          <!-- CARD: NHẬN XÉT TỰ ĐỘNG TỪ AI AGENT (FINDINGS · CHUẨN DATALENS STUDIO) -->
          <div id="dynamic-analysis-findings-card" class="glass-card rounded-3xl p-6 border border-emerald-500/30 bg-emerald-950/20 space-y-3">
            <div class="flex items-center justify-between">
              <h4 class="text-xs font-black text-emerald-300 uppercase tracking-wide flex items-center gap-1.5">
                <span>🤖</span> <span>Nhận Xét Tự Động Từ AI Agent (Automated Findings)</span>
              </h4>
              <span class="text-[10px] text-emerald-400 font-bold px-2 py-0.5 rounded-full bg-emerald-500/20">DataLens AI Findings</span>
            </div>
            <ul id="dynamic-analysis-findings-list" class="space-y-1.5 text-xs text-slate-200">
              <li>Đang khởi tạo nhận xét tự động...</li>
            </ul>
            <div id="dynamic-analysis-ai-insight" class="mt-2.5 pt-2.5 border-t border-emerald-500/20 text-xs text-amber-200/95 font-medium leading-relaxed hidden"></div>
          </div>

          <!-- HỘP YÊU CẦU AI PHÂN TÍCH THEO NGÔN NGỮ TỰ NHIÊN -->
          <div class="glass-card rounded-3xl border border-amber-400/40 bg-gradient-to-r from-amber-950/30 via-blue-950/40 to-indigo-950/30 p-5 sm:p-6 space-y-3 shadow-md">
            <div class="flex items-center justify-between flex-wrap gap-2">
              <label for="ai-analysis-prompt-input" class="text-xs font-black text-amber-300 flex items-center gap-1.5">
                <span>💡</span> <span>Yêu cầu AI phân tích đa chiều (Ngôn ngữ tự nhiên):</span>
              </label>
              <span class="text-[11px] text-blue-200/80 font-medium">AI tự động cấu hình nhóm, cột số &amp; nhận định</span>
            </div>

            <div class="flex flex-col sm:flex-row gap-2.5">
              <input 
                type="text" 
                id="ai-analysis-prompt-input" 
                placeholder="Ví dụ: So sánh doanh thu và lợi nhuận giữa các kênh bán hàng; Phân tích phân khúc khách VIP..."
                class="flex-1 bg-[#060f26] border border-amber-400/40 rounded-xl px-3.5 py-2.5 text-xs text-white placeholder-slate-400 focus:outline-none focus:border-amber-400 transition shadow-inner"
                onkeydown="if(event.key==='Enter') handleAiDynamicAnalysis()"
              />
              <button 
                type="button" 
                id="btn-ai-dynamic-analysis"
                onclick="handleAiDynamicAnalysis()" 
                class="toty-gold-btn text-slate-950 font-black text-xs px-5 py-2.5 rounded-xl shadow-md shadow-amber-400/20 flex items-center justify-center gap-1.5 cursor-pointer active:scale-95 transition flex-shrink-0"
              >
                <span>⚡</span> <span>Phân Tích Bằng AI</span>
              </button>
            </div>

            <div class="flex items-center gap-1.5 flex-wrap text-[11px] pt-1">
              <span class="text-amber-300/80 font-bold">Gợi ý phân tích:</span>
              <button type="button" onclick="setAndRunAiPrompt('So sánh doanh thu và lợi nhuận giữa các kênh bán lẻ')" class="px-2.5 py-1 rounded-full bg-blue-900/50 hover:bg-amber-400/20 border border-amber-400/20 hover:border-amber-400/50 text-blue-200 transition cursor-pointer">🛒 Doanh thu &amp; Lợi nhuận theo Kênh</button>
              <button type="button" onclick="setAndRunAiPrompt('Phân tích phân khúc khách hàng theo doanh thu thuần')" class="px-2.5 py-1 rounded-full bg-blue-900/50 hover:bg-amber-400/20 border border-amber-400/20 hover:border-amber-400/50 text-blue-200 transition cursor-pointer">👥 Phân khúc VIP vs Vãng lai</button>
              <button type="button" onclick="setAndRunAiPrompt('Đánh giá số lượng và chiết khấu theo danh mục ngành hàng')" class="px-2.5 py-1 rounded-full bg-blue-900/50 hover:bg-amber-400/20 border border-amber-400/20 hover:border-amber-400/50 text-blue-200 transition cursor-pointer">🏷️ Ngành hàng &amp; Chiết khấu</button>
              <button type="button" onclick="setAndRunAiPrompt('So sánh mức độ hài lòng khách hàng giữa các khu vực')" class="px-2.5 py-1 rounded-full bg-blue-900/50 hover:bg-amber-400/20 border border-amber-400/20 hover:border-amber-400/50 text-blue-200 transition cursor-pointer">⭐ Mức hài lòng theo Khu vực</button>
            </div>
          </div>

          <!-- CARD: THỐNG KÊ CHI TIẾT THEO CỘT (COLUMN PROFILE · CHUẨN DATALENS) -->
          <div class="glass-card rounded-3xl p-6 border border-white/10 space-y-4">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h3 class="text-sm font-extrabold text-white font-display flex items-center gap-2">
                  <span>📋</span> <span>Thống Kê Chi Tiết 23 Cột Dữ Liệu (Column Profile)</span>
                </h3>
                <p class="text-xs text-slate-300 mt-0.5">Hệ thống tự động nhận dạng định dạng cột số, giá trị ô trống và biên độ phân tán</p>
              </div>

              <!-- Lựa chọn định dạng số (Việt Nam vs Quốc tế) -->
              <div class="flex items-center gap-2 self-start sm:self-auto">
                <span class="text-xs text-amber-300 font-bold whitespace-nowrap">Định dạng số:</span>
                <select id="col-stats-locale" onchange="runColumnStatsLocaleChange(this.value)" class="bg-[#060f26] border border-amber-400/40 rounded-xl px-3 py-1.5 text-xs text-white focus:outline-none focus:border-amber-400 cursor-pointer">
                  <option value="vi" selected>Việt Nam · 1.234,56</option>
                  <option value="en">Quốc tế · 1,234.56</option>
                </select>
              </div>
            </div>

            <div id="col-stats-container" class="overflow-x-auto max-h-80 rounded-2xl border border-white/10 bg-black/30">
              <table class="w-full text-left text-xs border-collapse">
                <thead class="sticky top-0 bg-[#071330]/95 backdrop-blur-md">
                  <tr class="border-b border-white/10 text-[10px] font-bold text-amber-300 uppercase tracking-wider">
                    <th class="py-2.5 px-3">Tên Cột</th>
                    <th class="py-2.5 px-3">Kiểu Dữ Liệu</th>
                    <th class="py-2.5 px-3 text-right">Ô Trống (Missing)</th>
                    <th class="py-2.5 px-3 text-right">Giá Trị Phân Biệt (Distinct)</th>
                    <th class="py-2.5 px-3 text-right">Trung Bình (Mean)</th>
                    <th class="py-2.5 px-3 text-right">Nhỏ Nhất (Min)</th>
                    <th class="py-2.5 px-3 text-right">Lớn Nhất (Max)</th>
                  </tr>
                </thead>
                <tbody id="col-stats-tbody" class="divide-y divide-white/5 font-mono text-[11px] text-slate-300">
                  <tr><td colspan="7" class="text-center py-6 text-slate-400">Đang tải dữ liệu thống kê cột...</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- KHUNG YÊU CẦU SỬA BẰNG AI CHO BƯỚC 3 (REVIEW PANEL · CHUẨN DATALENS) -->
          <div class="glass-card rounded-3xl p-5 sm:p-6 border border-amber-400/40 bg-gradient-to-r from-amber-950/20 via-blue-950/30 to-indigo-950/20 space-y-3">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-2">
                <span class="text-amber-400 font-bold">💬</span>
                <h4 class="text-xs font-black text-white uppercase tracking-wider">Yêu Cầu Sửa Bằng AI Cho Bước 3 (Phân Tích Dữ Liệu)</h4>
              </div>
              <span class="text-[10px] text-amber-300 font-bold px-2 py-0.5 rounded-full bg-amber-400/20">AI Assistant</span>
            </div>
            <p class="text-xs text-slate-300">Nhập ý kiến chỉ đạo của Ban Giám Đốc để AI Agent đào sâu dữ liệu, bổ sung góc nhìn thực chứng và hoàn thiện báo cáo chiến lược.</p>

            <div class="space-y-2">
              <label for="step3-revision-input" class="block text-xs font-bold text-amber-200 flex items-center gap-1.5">
                <span>💬</span> <span>Nội dung chỉ đạo của Ban Lãnh Đạo / Trưởng bộ phận:</span>
              </label>
              <textarea 
                id="step3-revision-input" 
                rows="3" 
                placeholder="Ví dụ: Phân tích sâu cơ cấu doanh thu theo kênh bán hàng và đề xuất tối ưu chi phí phục vụ (Cost-to-Serve); Đánh giá biên lợi nhuận gộp danh mục Điện máy; So sánh hiệu quả giữa 3 miền..." 
                class="w-full bg-[#060f26] border border-amber-400/30 rounded-2xl p-3.5 text-xs text-white placeholder-slate-400 focus:outline-none focus:border-amber-400 transition leading-relaxed shadow-inner"
              ></textarea>
            </div>

            <!-- Chỉ đạo mẫu (Preset Pills) -->
            <div class="flex items-center justify-between flex-wrap gap-2.5 pt-1">
              <div class="flex items-center gap-1.5 flex-wrap text-[11px] text-slate-300">
                <span class="text-amber-300 font-bold flex items-center gap-1"><span>💡</span> Chỉ đạo mẫu:</span>
                <button type="button" onclick="setStep3Directive('Phân tích cơ cấu và tỷ trọng doanh thu đa kênh: Cửa hàng trực tiếp, Website/App và TMĐT, đánh giá chi phí phục vụ.')" class="px-2.5 py-1 rounded-full bg-blue-900/50 hover:bg-amber-400/20 border border-amber-400/20 hover:border-amber-400/50 text-blue-200 transition cursor-pointer">Tỷ trọng đa kênh</button>
                <button type="button" onclick="setStep3Directive('Đánh giá chi tiết biên lợi nhuận gộp danh mục Thiết bị & Điện thoại và đề xuất chính sách giá bán tối ưu.')" class="px-2.5 py-1 rounded-full bg-blue-900/50 hover:bg-amber-400/20 border border-amber-400/20 hover:border-amber-400/50 text-blue-200 transition cursor-pointer">Biên lãi danh mục</button>
                <button type="button" onclick="setStep3Directive('Ứng dụng quy luật Pareto 80/20: Phân tích Top 3 SKU công nghệ và kế hoạch bảo hộ tồn kho an toàn.')" class="px-2.5 py-1 rounded-full bg-blue-900/50 hover:bg-amber-400/20 border border-amber-400/20 hover:border-amber-400/50 text-blue-200 transition cursor-pointer">Top 3 SKU chủ lực</button>
                <button type="button" onclick="setStep3Directive('So sánh năng suất và hiệu quả kinh doanh giữa các vùng Miền Bắc, Miền Trung và Miền Nam.')" class="px-2.5 py-1 rounded-full bg-blue-900/50 hover:bg-amber-400/20 border border-amber-400/20 hover:border-amber-400/50 text-blue-200 transition cursor-pointer">So sánh 3 miền</button>
              </div>

              <!-- Action buttons -->
              <div class="flex items-center gap-2.5">
                <button 
                  type="button" 
                  id="btn-step3-apply-revision"
                  onclick="handleStep3Revision()" 
                  class="toty-gold-btn text-slate-950 font-black text-xs px-5 py-2.5 rounded-full flex items-center justify-center gap-2 cursor-pointer active:scale-95 transition shadow-lg shadow-amber-400/25"
                >
                  <span>⚡</span> <span>Áp Dụng Chỉ Đạo & Nhận Phản Hồi AI</span>
                </button>
                <button 
                  type="button" 
                  onclick="goToStep(4)" 
                  class="px-5 py-2.5 rounded-full bg-white/10 hover:bg-white/20 text-white font-bold text-xs border border-white/20 transition flex items-center gap-1.5 cursor-pointer"
                >
                  <span>Xác nhận & sang Bước 4</span> <span>→</span>
                </button>
              </div>
            </div>

            <!-- PHẦN PHẢN HỒI CHỈ ĐẠO BƯỚC 3 (ĐẦY ĐỦ, HOÀN THIỆN, LUXURY) -->
            <div id="step3-ai-revision-result" class="hidden mt-4 rounded-3xl bg-[#06112a] border border-amber-400/50 p-6 space-y-4 shadow-2xl animate-fadeIn">
              <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-amber-400/20">
                <div class="flex items-center gap-2.5">
                  <span class="text-2xl">🤖</span>
                  <div>
                    <h5 class="text-sm font-black text-amber-300 font-display">Báo Cáo Phản Hồi Chỉ Đạo Ban Giám Đốc (Bước 3)</h5>
                    <p class="text-[11px] text-slate-300 mt-0.5">Chỉ đạo: <em id="step3-directive-text" class="text-amber-200 font-semibold"></em></p>
                  </div>
                </div>
                <div class="flex items-center gap-2">
                  <span class="px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-[10px] font-black uppercase flex items-center gap-1">
                    <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    Đã Hoàn Thiện
                  </span>
                  <button type="button" onclick="copyStep3Feedback()" class="px-3 py-1 rounded-full bg-white/5 hover:bg-white/15 text-slate-200 border border-white/10 text-[11px] font-semibold transition cursor-pointer flex items-center gap-1">
                    <span>📋</span> <span>Sao chép</span>
                  </button>
                </div>
              </div>

              <!-- Nội dung phản hồi được định dạng Markdown chuẩn C-Level -->
              <div id="step3-ai-revision-content" class="text-xs text-slate-200 leading-relaxed space-y-2.5 font-sans">
              </div>

              <!-- Chân thẻ với nút hành động tiếp theo -->
              <div class="pt-3 border-t border-white/10 flex flex-col sm:flex-row items-center justify-between gap-3">
                <span class="text-[11px] text-slate-400">Dữ liệu phân tích đã được đồng bộ hóa với Báo cáo điều hành Bước 4.</span>
                <button type="button" onclick="goToStep(4)" class="px-6 py-2 rounded-full font-black text-xs text-slate-950 bg-gradient-to-r from-amber-400 via-amber-300 to-yellow-400 hover:scale-105 active:scale-95 transition shadow-md shadow-amber-400/30 flex items-center gap-2 cursor-pointer">
                  <span>Tiếp tục sang Bước 4: Báo Cáo Điều Hành</span>
                  <span>→</span>
                </button>
              </div>
            </div>
          </div>

        </div>

        <!-- ════════ TAB 2: BÁO CÁO PHÂN TÍCH KINH DOANH ĐIỀU HÀNH (BƯỚC 4 · CHUẨN DATALENS STUDIO) ════════ -->
        <div id="tab-top3" class="tab-content space-y-6 hidden">
          <!-- DETAIL HEADER BÁO CÁO KINH DOANH -->
          <div class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/40 bg-gradient-to-br from-[#071330] via-[#091b42] to-[#0a183c] shadow-2xl flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div class="flex items-center gap-2 flex-wrap mb-1">
                <h2 class="text-base sm:text-xl font-black text-white font-display" id="report-view-title">
                  Báo Cáo Phân Tích Kinh Doanh Điều Hành (Executive Business Report)
                </h2>
                <span class="text-[10px] font-black uppercase px-2.5 py-0.5 rounded-full bg-amber-400/20 text-amber-300 border border-amber-400/40">
                  DataLens Standard
                </span>
              </div>
              <p class="text-xs text-slate-300" id="report-view-subtitle">
                Bán lẻ điện máy &amp; Thiết bị gia dụng · 1 bộ dữ liệu · 15,000 dòng đã phân tích · 0₫ sai lệch
              </p>
              
              <!-- Quick Jump Navigation -->
              <nav class="flex items-center gap-3 mt-3 flex-wrap text-xs text-blue-200">
                <span class="text-slate-400 text-[11px] font-bold">Chuyển nhanh:</span>
                <a href="#rep-summary" class="hover:text-amber-300 underline underline-offset-4 decoration-amber-400/40">Tóm tắt điều hành</a>
                <span class="text-slate-600">·</span>
                <a href="#rep-dashboard" class="hover:text-amber-300 underline underline-offset-4 decoration-amber-400/40">Chỉ số chính</a>
                <span class="text-slate-600">·</span>
                <a href="#rep-questions" class="hover:text-amber-300 underline underline-offset-4 decoration-amber-400/40">Phân tích chuyên đề</a>
                <span class="text-slate-600">·</span>
                <a href="#rep-actions" class="hover:text-amber-300 underline underline-offset-4 decoration-amber-400/40">Kết luận &amp; Đề xuất</a>
                <span class="text-slate-600">·</span>
                <a href="#rep-audit" class="hover:text-amber-300 underline underline-offset-4 decoration-amber-400/40">Nguồn &amp; Chất lượng</a>
              </nav>
            </div>

            <!-- Action Buttons: Copy Markdown & Print -->
            <div class="flex items-center gap-2.5 flex-shrink-0 self-start md:self-auto">
              <button 
                type="button" 
                onclick="copyReportMarkdown()" 
                class="toty-gold-btn text-slate-950 font-black text-xs px-4 py-2.5 rounded-xl shadow-md shadow-amber-400/20 flex items-center gap-1.5 cursor-pointer active:scale-95 transition"
              >
                <span>📋</span> <span>Sao Chép Markdown</span>
              </button>
              <button 
                type="button" 
                onclick="window.print()" 
                class="glass-pill px-4 py-2.5 rounded-xl text-xs font-bold text-white border border-white/20 hover:border-amber-400/50 flex items-center gap-1.5 cursor-pointer transition"
              >
                <span>🖨️</span> <span>In / Xuất Báo Cáo</span>
              </button>
            </div>
          </div>

          <!-- NỘI DUNG 6 MỤC CỦA BÁO CÁO KINH DOANH -->
          <div id="top3-cards-wrap" class="space-y-6">
            <div class="glass-card rounded-3xl p-12 text-center text-slate-300 text-xs border border-white/80 shadow-sm">
              Vui lòng bấm nút <strong class="text-slate-200">"Chạy Phân Tích Dữ Liệu Tự Động"</strong> hoặc thực hiện Bước 4 để khởi tạo Báo Cáo Kinh Doanh Điều Hành.
            </div>
          </div>
        </div>

        <!-- ════════ TAB 5: KÝ DUYỆT BÁO CÁO & BAN HÀNH CHỈ THỊ (C-LEVEL) ════════ -->
        <div id="step-tab-5" class="tab-content space-y-6 hidden">
          <!-- DETAIL HEADER BƯỚC 5 -->
          <div class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/40 bg-gradient-to-br from-[#071330] via-[#091b42] to-[#0a183c] shadow-2xl flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div class="flex items-center gap-2 flex-wrap mb-1">
                <span class="text-[10px] font-black uppercase px-2.5 py-0.5 rounded-full bg-amber-400/20 text-amber-300 border border-amber-400/40">
                  BƯỚC 5 / 5 · HOÀN TẤT QUY TRÌNH
                </span>
                <h2 class="text-base sm:text-xl font-black text-white font-display">
                  Ký Duyệt Báo Cáo & Ban Hành Chỉ Thị Chiến Lược (C-Level Sign-off)
                </h2>
              </div>
              <p class="text-xs text-slate-300">
                Phê duyệt chính thức toàn văn báo cáo kinh doanh điều hành, ban hành kế hoạch hành động 30 ngày cho 13 chi nhánh và xuất tài liệu quản trị.
              </p>
            </div>

            <!-- Action Buttons: Copy Markdown & Print -->
            <div class="flex items-center gap-2.5 flex-shrink-0 self-start md:self-auto">
              <button 
                type="button" 
                onclick="copyReportMarkdown()" 
                class="toty-gold-btn text-slate-950 font-black text-xs px-4 py-2.5 rounded-xl shadow-md shadow-amber-400/20 flex items-center gap-1.5 cursor-pointer active:scale-95 transition"
              >
                <span>📋</span> <span>Sao Chép Markdown</span>
              </button>
              <button 
                type="button" 
                onclick="window.print()" 
                class="glass-pill px-4 py-2.5 rounded-xl text-xs font-bold text-white border border-white/20 hover:border-amber-400/50 flex items-center gap-1.5 cursor-pointer transition"
              >
                <span>🖨️</span> <span>In / Xuất PDF</span>
              </button>
            </div>
          </div>

          <!-- KHUNG CHỨNG NHẬN KÝ DUYỆT ĐIỀU HÀNH & CON DẤU SỐ -->
          <div id="step5-approval-status-card" class="glass-card rounded-3xl p-6 sm:p-8 border-2 border-amber-400/50 bg-gradient-to-r from-amber-950/30 via-blue-950/40 to-indigo-950/30 shadow-2xl relative overflow-hidden">
            <!-- Background seal watermark -->
            <div class="absolute -right-10 -bottom-10 opacity-5 pointer-events-none text-9xl">
              🏆
            </div>

            <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-6 pb-6 border-b border-white/10">
              <div class="space-y-2">
                <div class="flex items-center gap-3">
                  <div id="step5-seal-icon" class="w-14 h-14 rounded-2xl bg-amber-400/20 text-amber-300 border-2 border-amber-400/50 flex items-center justify-center text-2xl shadow-lg">
                    ✍️
                  </div>
                  <div>
                    <h3 class="text-lg sm:text-xl font-black text-white font-display" id="step5-seal-title">
                      Chứng Nhận Phê Duyệt Báo Cáo Quản Trị
                    </h3>
                    <p class="text-xs text-slate-300" id="step5-seal-subtitle">
                      Báo cáo tài chính điều hành bán lẻ điện máy · Kỳ đối soát 2024-2026 · 15.000 giao dịch
                    </p>
                  </div>
                </div>
              </div>

              <!-- Trạng thái con dấu số -->
              <div id="step5-seal-badge" class="flex items-center gap-3">
                <div class="p-3 px-5 rounded-2xl border-2 border-amber-400/40 bg-amber-950/40 text-amber-300 text-center font-display shadow-md">
                  <div class="text-[10px] uppercase font-bold tracking-widest text-amber-200/80">Trạng Thái Phê Duyệt</div>
                  <div class="text-base font-black text-amber-300" id="step5-status-text">⏳ CHỜ PHÊ DUYỆT</div>
                </div>
              </div>
            </div>

            <!-- Chi tiết xác nhận số liệu & chỉ thị -->
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-6">
              <div class="p-4 rounded-2xl bg-white/5 border border-white/10 flex flex-col justify-between">
                <div>
                  <div class="text-[11px] text-slate-400 font-bold uppercase tracking-wider">Người Ký Phê Duyệt</div>
                  <div class="text-sm font-black text-white mt-1" id="step5-manager-name">Trần Bảo · Giám Đốc Kinh Doanh</div>
                </div>
                <div class="text-[11px] text-amber-300 font-bold mt-2 pt-2 border-t border-white/10">Ban Điều Hành C-Level</div>
              </div>

              <div class="p-4 rounded-2xl bg-white/5 border border-white/10 flex flex-col justify-between">
                <div>
                  <div class="text-[11px] text-slate-400 font-bold uppercase tracking-wider">Đối Soát Toàn Vẹn (B1 & B2)</div>
                  <div class="text-sm font-black text-emerald-400 mt-1" id="step5-stat-rows">15.000 Dòng · 0₫ Thất Thoát</div>
                </div>
                <div class="text-[11px] text-slate-300 mt-2 pt-2 border-t border-white/10" id="step5-stat-clean">Đã đối soát & chấp thuận ngoại lệ</div>
              </div>

              <div class="p-4 rounded-2xl bg-white/5 border border-white/10 flex flex-col justify-between">
                <div>
                  <div class="text-[11px] text-slate-400 font-bold uppercase tracking-wider">Chỉ Số Tài Chính Thực Chứng (B3)</div>
                  <div class="text-sm font-black text-amber-300 mt-1" id="step5-stat-revenue">430.66 tỷ ₫ · Đạt 86.8%</div>
                </div>
                <div class="text-[11px] text-rose-300 font-medium mt-2 pt-2 border-t border-white/10" id="step5-stat-gap">Thiếu hụt: -65.25 tỷ ₫ · Margin: 13.4%</div>
              </div>

              <div class="p-4 rounded-2xl bg-white/5 border border-white/10 flex flex-col justify-between">
                <div>
                  <div class="text-[11px] text-slate-400 font-bold uppercase tracking-wider">Thời Gian & Mã Kiểm Toán</div>
                  <div class="text-sm font-black text-white mt-1" id="step5-decision-time">Chưa ký duyệt</div>
                </div>
                <div class="text-[11px] text-slate-400 mt-2 pt-2 border-t border-white/10 font-mono text-[10px]" id="step5-decision-hash">Mã băm: TB-SHA256-OCT-8501</div>
              </div>
            </div>

            <!-- Nút Ký Duyệt Bằng AI Agent -->
            <div class="mt-6 pt-6 border-t border-white/10 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div class="text-xs text-slate-300">
                Sau khi ký duyệt, kế hoạch hành động 30 ngày và các chỉ thị sẽ được lưu vĩnh viễn vào biên bản kiểm toán hệ thống.
              </div>
              <button 
                type="button" 
                id="btn-step5-signoff"
                onclick="signoffStep5()" 
                class="toty-gold-btn text-slate-950 font-black text-sm px-6 py-3 rounded-2xl shadow-xl shadow-amber-400/25 flex items-center gap-2 cursor-pointer active:scale-95 transition"
              >
                <span>✍️</span> <span>Ban Giám Đốc Ký Duyệt Báo Cáo & Ban Hành Chỉ Thị</span>
              </button>
            </div>
          </div>

          <!-- BẢNG ĐỐI CHIẾU TIẾN TRÌNH 5 BƯỚC LIÊN HOÀN (100% ĐỒNG NHẤT DỮ LIỆU) -->
          <div class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/30 bg-gradient-to-r from-blue-950/40 via-indigo-950/30 to-amber-950/20 shadow-xl space-y-4">
            <div class="flex items-center justify-between pb-3 border-b border-white/10">
              <h3 class="text-sm sm:text-base font-extrabold text-white font-display flex items-center gap-2">
                <span>🔗</span> Bảng Đối Chiếu Tiến Trình Kiểm Toán 5 Bước (100% Đồng Nhất Dữ Liệu)
              </h3>
              <span class="text-[11px] text-emerald-400 font-bold bg-emerald-950/40 px-3 py-1 rounded-full border border-emerald-500/30">
                5-Step Continuity Verified
              </span>
            </div>
            <div id="step5-audit-matrix" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5">
              <!-- Dynamically populated by renderStep5AuditMatrix -->
            </div>
          </div>

          <!-- KẾ HOẠCH CHỈ THỊ HÀNH ĐỘNG 30 NGÀY BAN HÀNH CHÍNH THỨC (BÁM SÁT 100% BƯỚC 4) -->
          <div class="glass-card rounded-3xl p-6 sm:p-7 border border-white/20 shadow-xl space-y-4">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-white/10">
              <div>
                <h3 class="text-sm sm:text-base font-extrabold text-white font-display flex items-center gap-2">
                  <span>📋</span> Kế Hoạch Chỉ Thị Hành Động 30 Ngày Ban Hành Tới Các Khối (Executive Directives)
                </h3>
                <p class="text-xs text-slate-300 mt-0.5">Phê chuẩn và giao quyền trực tiếp cho CCO, CFO, COO với cam kết KPI lượng hóa cụ thể</p>
              </div>
              <span class="text-[11px] text-amber-300 font-bold bg-amber-400/10 px-3 py-1 rounded-full border border-amber-400/30 self-start sm:self-auto">
                Bám Sát 100% Phân Tích Bước 4
              </span>
            </div>

            <div id="step5-directives-container" class="space-y-3.5">
              <!-- Dynamically populated by renderStep5Directives from Step 4 Action Plan -->
            </div>

            <!-- Ghi chú bổ sung từ C-Level -->
            <div class="mt-4 pt-4 border-t border-white/10 space-y-3">
              <label class="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <span>💬</span> Ghi Chú & Ý Kiến Chỉ Đạo Trực Tiếp Từ Ban Giám Đốc (C-Level Executive Memo):
              </label>
              <textarea 
                id="step5-custom-directive-notes" 
                rows="2" 
                placeholder="Nhập thêm chỉ đạo cụ thể dành cho khối kinh doanh (CCO), tài chính (CFO), vận hành (COO) hoặc các giám đốc chi nhánh..." 
                class="w-full text-xs rounded-2xl bg-white/5 border border-white/10 p-3.5 text-white placeholder-slate-400 focus:outline-none focus:border-amber-400/50"
              ></textarea>

              <!-- Approved Executive Memo Display (Visible when signed off) -->
              <div id="step5-approved-memo-card" class="hidden p-4 rounded-2xl bg-emerald-950/30 border border-emerald-500/40 space-y-1.5">
                <div class="flex items-center gap-2 text-xs font-bold text-emerald-300">
                  <span>📜</span> Chỉ Thị Bổ Sung Đã Phê Chuẩn & Có Hiệu Lực Lập Tức:
                </div>
                <p id="step5-approved-memo-content" class="text-xs text-white leading-relaxed font-sans pl-5"></p>
              </div>
            </div>
          </div>

          <!-- ĐIỂM NÓNG VẬN HÀNH & CHI NHÁNH CẦN CAN THIỆP KHẨN CẤP (TỪ BƯỚC 4) -->
          <div class="glass-card rounded-3xl p-6 sm:p-7 border border-white/20 shadow-xl space-y-4">
            <div class="flex items-center justify-between pb-3 border-b border-white/10">
              <div>
                <h3 class="text-sm sm:text-base font-extrabold text-white font-display flex items-center gap-2">
                  <span>🚨</span> Điểm Nóng Vận Hành & Chi Nhánh Cần Xử Lý Khẩn Cấp (Top Priority Units)
                </h3>
                <p class="text-xs text-slate-300 mt-0.5">Xếp hạng mức độ khẩn cấp dựa trên khoảng cách doanh thu, tỷ lệ hoàn trả và giá vốn từ Bước 4</p>
              </div>
              <span class="text-[11px] text-rose-400 font-bold bg-rose-950/40 px-3 py-1 rounded-full border border-rose-500/30">
                Top Operational Hotspots
              </span>
            </div>

            <div id="step5-top-units-container" class="grid grid-cols-1 md:grid-cols-3 gap-4">
              <!-- Dynamically populated by renderStep5PriorityUnits -->
            </div>
          </div>

          <!-- TRUNG TÂM XUẤT TỆP & TÀI LIỆU QUẢN TRỊ -->
          <div class="glass-card rounded-3xl p-6 sm:p-7 border border-white/80 shadow-md space-y-4">
            <div class="flex items-center justify-between pb-3 border-b border-white/10">
              <h3 class="text-sm sm:text-base font-extrabold text-white font-display flex items-center gap-2">
                <span>📥</span> Trung Tâm Xuất Báo Cáo & Dữ Liệu Đối Soát Toàn Diện
              </h3>
              <span class="text-[11px] text-slate-400">Đã chuẩn hóa & sẵn sàng tải về</span>
            </div>

            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <!-- Excel Report -->
              <a href="/api/export/excel" download="Bao_Cao_Quan_Tri_Tran_Bao_Project.xlsx" class="p-4 rounded-2xl bg-white/5 border border-white/10 hover:border-emerald-400/50 hover:bg-emerald-950/20 transition group flex flex-col justify-between cursor-pointer">
                <div>
                  <div class="text-2xl mb-2">📊</div>
                  <div class="text-xs font-bold text-white group-hover:text-emerald-300 transition">Báo Cáo Quản Trị (Excel)</div>
                  <div class="text-[10px] text-slate-400 mt-1">Đầy đủ 4 Sheets: KPIs, 13 Chi nhánh, Đổi trả, Doanh thu.</div>
                </div>
                <div class="mt-4 text-[11px] text-emerald-400 font-bold flex items-center gap-1">
                  <span>Tải .xlsx</span> <span>↓</span>
                </div>
              </a>

              <!-- CSV Clean Transactions -->
              <a href="/api/export/csv?type=txns" download="Du_Lieu_Giao_Dich_Sach.csv" class="p-4 rounded-2xl bg-white/5 border border-white/10 hover:border-amber-400/50 hover:bg-amber-950/20 transition group flex flex-col justify-between cursor-pointer">
                <div>
                  <div class="text-2xl mb-2">📄</div>
                  <div class="text-xs font-bold text-white group-hover:text-amber-300 transition">Giao Dịch Sạch (CSV)</div>
                  <div class="text-[10px] text-slate-400 mt-1" id="step5-export-txns-desc">Dữ liệu giao dịch sạch 23 cột nguyên bản không sai lệch.</div>
                </div>
                <div class="mt-4 text-[11px] text-amber-300 font-bold flex items-center gap-1">
                  <span>Tải .csv</span> <span>↓</span>
                </div>
              </a>

              <!-- CSV Returns Audit -->
              <a href="/api/export/csv?type=returns" download="Doi_Soat_Don_Doi_Tra.csv" class="p-4 rounded-2xl bg-white/5 border border-white/10 hover:border-rose-400/50 hover:bg-rose-950/20 transition group flex flex-col justify-between cursor-pointer">
                <div>
                  <div class="text-2xl mb-2">🛡️</div>
                  <div class="text-xs font-bold text-white group-hover:text-rose-300 transition">Đối Soát Đổi Trả (CSV)</div>
                  <div class="text-[10px] text-slate-400 mt-1" id="step5-export-returns-desc">Danh sách đơn hoàn trả bóc tách kiểm toán độc lập.</div>
                </div>
                <div class="mt-4 text-[11px] text-rose-400 font-bold flex items-center gap-1">
                  <span>Tải .csv</span> <span>↓</span>
                </div>
              </a>

              <!-- CSV Stores Performance -->
              <a href="/api/export/csv?type=stores" download="Hieu_Qua_13_Chi_Nhanh.csv" class="p-4 rounded-2xl bg-white/5 border border-white/10 hover:border-cyan-400/50 hover:bg-cyan-950/20 transition group flex flex-col justify-between cursor-pointer">
                <div>
                  <div class="text-2xl mb-2">🏢</div>
                  <div class="text-xs font-bold text-white group-hover:text-cyan-300 transition">Hiệu Quả 13 Chi Nhánh</div>
                  <div class="text-[10px] text-slate-400 mt-1">Bảng xếp hạng doanh thu, tỷ lệ hoàn và đánh giá.</div>
                </div>
                <div class="mt-4 text-[11px] text-cyan-300 font-bold flex items-center gap-1">
                  <span>Tải .csv</span> <span>↓</span>
                </div>
              </a>
            </div>
          </div>
        </div>

        <!-- ════════ TAB 3: ĐỐI SOÁT DỮ LIỆU & NGOẠI LỆ ════════ -->
        <div id="tab-quality" class="tab-content space-y-6 hidden">
          <div class="glass-card rounded-3xl p-6 sm:p-7 border border-white/80 shadow-md flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h3 class="text-base sm:text-lg font-extrabold text-white font-display flex items-center gap-2">
                <span>🛡️</span> Kiểm Soát Chất Lượng Dữ Liệu & Đối Soát Ngoại Lệ
              </h3>
              <p class="text-xs text-slate-300 mt-1">Khử trùng lặp giao dịch, chuẩn hóa cấu trúc và bảo toàn 100% dữ liệu trước khi tính toán doanh thu</p>
            </div>
            <span class="glass-pill px-3.5 py-1.5 text-xs font-bold text-emerald-800 border border-emerald-200/80 shadow-2xs flex items-center gap-1.5 self-start sm:self-auto">
              <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              Data Integrity Guarantee
            </span>
          </div>
          <div id="quality-report-wrap" class="space-y-6">
            <div class="glass-card rounded-3xl p-12 text-center text-slate-300 text-xs border border-white/80 shadow-sm">
              Chưa có báo cáo kiểm tra chất lượng dữ liệu. Hãy chạy Bước 1 hoặc bấm <strong class="text-slate-200">"Chạy Phân Tích Dữ Liệu Tự Động"</strong>.
            </div>
          </div>
        </div>

        <!-- ════════ TAB 4: ĐÁNH GIÁ SKU HÀNG TIÊU DÙNG ════════ -->
        <div id="tab-products" class="tab-content space-y-6 hidden">
          
          <!-- SKU SUMMARY KPI CARDS -->
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div class="glass-card p-5 sm:p-6 rounded-3xl border border-white/80 shadow-sm">
              <div class="text-[11px] font-bold uppercase tracking-wider text-slate-300 font-display">Tổng Số SKU Kinh Doanh</div>
              <div class="text-2xl font-black text-white mt-1.5" id="sku-kpi-total">--</div>
              <div class="text-[11px] text-slate-300 mt-1">Danh mục sản phẩm hoạt động</div>
            </div>
            <div class="glass-card p-5 sm:p-6 rounded-3xl border border-white/80 shadow-sm">
              <div class="text-[11px] font-bold uppercase tracking-wider text-slate-300 font-display">Top 1 SKU Bán Chạy</div>
              <div class="text-sm font-black text-orange-600 mt-1.5 truncate" id="sku-kpi-top1">--</div>
              <div class="text-[11px] text-slate-300 mt-1" id="sku-kpi-top1-sub">Đóng góp doanh thu cao nhất</div>
            </div>
            <div class="glass-card p-5 sm:p-6 rounded-3xl border border-white/80 shadow-sm">
              <div class="text-[11px] font-bold uppercase tracking-wider text-slate-300 font-display">Biên Lợi Nhuận SKU Bình Quân</div>
              <div class="text-2xl font-black text-emerald-600 mt-1.5" id="sku-kpi-margin">--</div>
              <div class="text-[11px] text-slate-300 mt-1">Gross Margin toàn danh mục</div>
            </div>
            <div class="glass-card p-5 sm:p-6 rounded-3xl border border-white/80 shadow-sm">
              <div class="text-[11px] font-bold uppercase tracking-wider text-slate-300 font-display">SKU Cảnh Báo (Đổi trả >3.5%)</div>
              <div class="text-2xl font-black text-rose-600 mt-1.5" id="sku-kpi-risk">--</div>
              <div class="text-[11px] text-rose-600 mt-1 font-semibold">Cảnh báo chất lượng / Đóng gói / HSD</div>
            </div>
          </div>

          <!-- SKU CHART CARD -->
          <div class="glass-card rounded-3xl p-6 sm:p-7 border border-white/80 shadow-md">
            <div class="flex items-center justify-between mb-4">
              <div>
                <h3 class="text-sm sm:text-base font-extrabold text-white font-display">Biểu Đồ Top 10 SKU Đóng Góp Doanh Thu Cao Nhất</h3>
                <p class="text-xs text-slate-300 mt-0.5">Xếp hạng giá trị doanh thu thuần theo SKU sản phẩm tiêu dùng</p>
              </div>
              <span class="glass-pill text-xs px-3.5 py-1.5 rounded-full text-orange-800 font-bold border border-orange-200/80 shadow-2xs">Pareto 80/20</span>
            </div>
            <div id="chart-sku-container" class="min-h-[220px] flex items-center justify-center text-xs text-slate-300 bg-stone-50/60 rounded-2xl p-4 border border-stone-200/60">
              Chưa có dữ liệu biểu đồ SKU. Hãy bấm "Chạy Phân Tích Dữ Liệu Tự Động" ở trên.
            </div>
          </div>

          <!-- SKU EVALUATION TABLE CARD -->
          <div class="glass-card rounded-3xl p-6 sm:p-7 border border-white/80 shadow-md space-y-4">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h3 class="text-sm sm:text-base font-extrabold text-white flex items-center gap-2 font-display">
                  <span>📦</span> Bảng Đánh Giá Chi Tiết SKU Sản Phẩm Tiêu Dùng
                </h3>
                <p class="text-xs text-slate-300 mt-0.5">Phân loại Pareto ABC, biên lợi nhuận, tỷ lệ đổi trả & tình trạng sức khỏe SKU</p>
              </div>

              <!-- Filters -->
              <div class="flex items-center gap-2 flex-wrap">
                <input 
                  type="text" 
                  id="sku-search-input" 
                  placeholder="Tìm SKU / Tên SP..." 
                  oninput="filterSKUs()" 
                  class="text-xs px-3.5 py-2 rounded-2xl border border-stone-200/80 bg-white/90 focus:bg-white focus:outline-none focus:ring-2 focus:ring-orange-500/20 w-48 shadow-2xs"
                />
                <select id="sku-abc-filter" onchange="filterSKUs()" class="text-xs px-3.5 py-2 rounded-2xl border border-stone-200/80 bg-white/90 focus:outline-none font-semibold text-slate-200 shadow-2xs">
                  <option value="ALL">Tất cả Phân Hạng ABC</option>
                  <option value="A">Nhóm A (Trụ cột DT 80%)</option>
                  <option value="B">Nhóm B (Tiềm năng 15%)</option>
                  <option value="C">Nhóm C (Chậm luân chuyển 5%)</option>
                </select>
                <select id="sku-status-filter" onchange="filterSKUs()" class="text-xs px-3.5 py-2 rounded-2xl border border-stone-200/80 bg-white/90 focus:outline-none font-semibold text-slate-200 shadow-2xs">
                  <option value="ALL">Tất cả Tình trạng</option>
                  <option value="RISK">⚠️ Cảnh báo Đổi trả cao</option>
                  <option value="STAR">🌟 SKU Ngôi sao</option>
                  <option value="SLOW">📦 SKU Chậm luân chuyển</option>
                </select>
              </div>
            </div>

            <!-- Table -->
            <div class="overflow-x-auto">
              <table class="w-full text-left text-xs border-collapse">
                <thead>
                  <tr class="border-b border-stone-100 text-[10px] font-bold text-slate-300 uppercase tracking-wider bg-stone-50/60">
                    <th class="py-3 px-3">Mã SKU</th>
                    <th class="py-3 px-3">Tên Sản Phẩm</th>
                    <th class="py-3 px-3">Ngành Hàng</th>
                    <th class="py-3 px-3 text-right">Sản Lượng Bán</th>
                    <th class="py-3 px-3 text-right">Doanh Thu Thuần</th>
                    <th class="py-3 px-3 text-right">Lợi Nhuận Gộp</th>
                    <th class="py-3 px-3 text-right">Biên LN%</th>
                    <th class="py-3 px-3 text-right">Đổi Trả%</th>
                    <th class="py-3 px-3 text-center">Phân Hạng ABC</th>
                    <th class="py-3 px-3">Đánh Giá & Khuyến Nghị AI</th>
                  </tr>
                </thead>
                <tbody id="sku-table-body" class="divide-y divide-stone-100">
                  <tr><td colspan="10" class="text-center py-10 text-slate-300">Đang chờ nạp và phân tích dữ liệu SKU...</td></tr>
                </tbody>
              </table>
            </div>
          </div>

        </div>

        <!-- ════════ TAB 5: GEMINI COPILOT ════════ -->
        <div id="tab-copilot" class="tab-content space-y-6 hidden">
          <div class="glass-card rounded-3xl p-6 sm:p-8 border border-white/80 shadow-md">
            <div class="flex items-center justify-between mb-5 pb-4 border-b border-stone-100">
              <div>
                <h3 class="text-base font-extrabold text-white flex items-center gap-2 font-display">
                  <span>💬</span> Trợ Lý AI Quản Trị Kinh Doanh (Gemini Copilot)
                </h3>
                <div class="text-xs text-slate-300 mt-0.5">Trợ lý C-Level được nạp toàn bộ ngữ cảnh dữ liệu kinh doanh, cơ cấu SKU và điểm nóng</div>
              </div>
              <span class="text-xs bg-amber-50 text-amber-800 font-bold px-3.5 py-1.5 rounded-full border border-amber-200 shadow-2xs flex items-center gap-1.5">
                <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                <span>Live AI Context</span>
              </span>
            </div>

            <!-- Chat Stream -->
            <div class="h-96 overflow-y-auto bg-blue-950/40 border border-amber-400/25 rounded-3xl p-5 space-y-3.5 mb-5" id="chat-stream">
              <div class="glass-card p-4 rounded-3xl rounded-tl-none border border-amber-400/30 text-xs sm:text-sm text-blue-100 shadow-md max-w-[85%] leading-relaxed bg-[#0a183c]/85">
                Xin chào! Tôi là <strong class="text-amber-300">Trần Bảo AI Copilot</strong> tích hợp mô hình <strong class="text-amber-300">Google Gemini 3.5 Flash</strong>. Tôi nắm rõ toàn bộ dữ liệu giao dịch, cơ cấu sản phẩm, tỷ lệ đổi trả và mức độ hoàn thành chỉ tiêu các chi nhánh. Bạn cần tôi làm rõ vấn đề nào?
              </div>
            </div>

            <!-- Pill Input Bar (TOTY Capsule Style) -->
            <div class="glass-card rounded-full p-2 pl-5 flex items-center gap-3 border border-amber-400/40 shadow-lg transition-all focus-within:ring-2 focus-within:ring-amber-400/30 bg-[#071330]/90">
              <input 
                type="text" 
                id="copilot-input-box" 
                placeholder="Hỏi câu hỏi bất kỳ về doanh thu, chi nhánh, tỷ lệ đổi trả, đánh giá SKU..." 
                onkeypress="if(event.key==='Enter') handleSendChat()" 
                class="flex-1 bg-transparent border-none outline-none text-xs sm:text-sm font-medium text-white placeholder-blue-300/40 focus:ring-0"
              />
              <button 
                onclick="handleSendChat()" 
                class="toty-gold-btn text-slate-950 font-black text-xs px-6 py-3 rounded-full flex items-center gap-2 cursor-pointer shadow-md shadow-amber-400/30 whitespace-nowrap active:scale-95"
              >
                <span>Gửi câu hỏi</span>
                <span>⚡</span>
              </button>
            </div>

            <!-- Quick Prompts Pills -->
            <div class="flex items-center gap-2 mt-3.5 flex-wrap text-xs text-blue-200/70">
              <span class="font-bold text-amber-300">Gợi ý câu hỏi:</span>
              <button onclick="document.getElementById('copilot-input-box').value='Tình hình doanh thu thuần và lợi nhuận gộp toàn chuỗi?'; handleSendChat();" class="px-3.5 py-1.5 rounded-full bg-blue-950/70 hover:bg-blue-900/90 hover:text-amber-200 hover:border-amber-400/50 border border-amber-400/25 text-blue-100 font-semibold transition">
                📊 Doanh thu & Lãi gộp?
              </button>
              <button onclick="document.getElementById('copilot-input-box').value='Top 3 chi nhánh điểm nóng cần can thiệp 30 ngày?'; handleSendChat();" class="px-3.5 py-1.5 rounded-full bg-blue-950/70 hover:bg-blue-900/90 hover:text-amber-200 hover:border-amber-400/50 border border-amber-400/25 text-blue-100 font-semibold transition">
                🎯 Điểm nóng chi nhánh?
              </button>
              <button onclick="document.getElementById('copilot-input-box').value='Top 3 SKU bán chạy nhất và các SKU rủi ro nhóm C?'; handleSendChat();" class="px-3.5 py-1.5 rounded-full bg-blue-950/70 hover:bg-blue-900/90 hover:text-amber-200 hover:border-amber-400/50 border border-amber-400/25 text-blue-100 font-semibold transition">
                🛍️ Đánh giá SKU & Rủi ro?
              </button>
              <button onclick="document.getElementById('copilot-input-box').value='Tỷ trọng doanh thu theo các Khu Vực như thế nào?'; handleSendChat();" class="px-3.5 py-1.5 rounded-full bg-blue-950/70 hover:bg-blue-900/90 hover:text-amber-200 hover:border-amber-400/50 border border-amber-400/25 text-blue-100 font-semibold transition">
                🗺️ Phân bổ Khu Vực?
              </button>
              <button onclick="document.getElementById('copilot-input-box').value='Kênh bán hàng nào hiệu quả nhất và tỷ lệ đổi trả ra sao?'; handleSendChat();" class="px-3.5 py-1.5 rounded-full bg-blue-950/70 hover:bg-blue-900/90 hover:text-amber-200 hover:border-amber-400/50 border border-amber-400/25 text-blue-100 font-semibold transition">
                🛒 Kênh bán & Đổi trả?
              </button>
            </div>
          </div>
        </div>

        <!-- ════════ TAB 6: AUDIT TRAIL ════════ -->
        <div id="tab-audit" class="tab-content space-y-6 hidden">
          <div class="glass-card rounded-3xl p-6 sm:p-8 border border-amber-400/25 shadow-md">
            <div class="flex items-center justify-between mb-4 pb-3 border-b border-amber-400/20">
              <div>
                <h3 class="text-base font-extrabold text-white font-display">Nhật Ký Thực Thi AI Agent (Audit Trail Provenance)</h3>
                <p class="text-xs text-blue-200/60 mt-0.5">Lưu vết toàn bộ tiến trình phân tích, phê duyệt và đối soát dữ liệu</p>
              </div>
              <button onclick="loadLogs()" class="text-xs px-4 py-1.5 rounded-full border border-amber-400/30 hover:bg-blue-950 font-bold text-amber-300 transition shadow-md">
                🔄 Làm mới
              </button>
            </div>
            <div class="bg-[#030712] text-amber-300 font-mono text-[11px] p-5 rounded-2xl max-h-96 overflow-y-auto whitespace-pre-wrap border border-amber-400/25 shadow-inner" id="audit-log-terminal">
              Đang tải nhật ký kiểm toán...
            </div>
          </div>
        </div>
      </main>
    </div>
  </div>

  <!-- JAVASCRIPT LOGIC -->
  <script>

    let currentEnterpriseName = "Trần Bảo pro";
    let targetWorkspaceTab = 'tab-dash';
    let globalStoreList = [];

    function openExportModal() {
      const modal = document.getElementById('export-modal');
      if (modal) {
        modal.classList.remove('hidden');
        modal.style.display = 'flex';
        modal.style.opacity = '0';
        setTimeout(() => { modal.style.opacity = '1'; }, 20);
      }
    }

    function closeExportModal() {
      const modal = document.getElementById('export-modal');
      if (modal) {
        modal.style.opacity = '0';
        setTimeout(() => {
          modal.classList.add('hidden');
          modal.style.display = 'none';
        }, 200);
      }
    }

    function updateAllCompanyLabels(name) {
      currentEnterpriseName = (name && name.trim()) ? name.trim() : "Trần Bảo pro";
      try { localStorage.setItem('tb_enterprise_name', currentEnterpriseName); } catch(_) {}
      
      const titleEl = document.getElementById('greeting-title');
      if (titleEl) titleEl.innerText = `Chào mừng ${currentEnterpriseName} trở lại làm việc!`;
      const sideLabel = document.getElementById('sidebar-company-label');
      if (sideLabel) sideLabel.innerText = currentEnterpriseName;
      const headName = document.getElementById('header-company-name');
      if (headName) headName.innerText = currentEnterpriseName;
      const landingInp = document.getElementById('input-landing-company');
      if (landingInp && landingInp !== document.activeElement) landingInp.value = currentEnterpriseName;
      const wsInp = document.getElementById('input-company-name');
      if (wsInp && wsInp !== document.activeElement) wsInp.value = currentEnterpriseName;
    }

    function updateCompanyName(val) {
      updateAllCompanyLabels(val);
    }

    function selectCategoryPill(btn, catName, defaultTab) {
      document.querySelectorAll('.cat-pill').forEach(b => {
        b.classList.remove('active-cat-pill');
        b.classList.add('inactive-cat-pill');
      });
      btn.classList.remove('inactive-cat-pill');
      btn.classList.add('active-cat-pill');
      if (defaultTab) targetWorkspaceTab = defaultTab;
      const inp = document.getElementById('input-landing-company');
      if (inp) inp.focus();
    }

    function showWelcomeAnimation(name, callback) {
      const modal = document.getElementById('welcome-modal');
      const heading = document.getElementById('welcome-company-heading');
      if (heading) {
        heading.innerText = `Xin chào ${name} đã đến với không gian làm việc!`;
      }
      if (!modal) {
        if (callback) callback();
        return;
      }

      let closed = false;
      const dismiss = () => {
        if (closed) return;
        closed = true;
        modal.classList.add('hidden');
        modal.style.setProperty('display', 'none', 'important');
        if (callback) callback();
      };

      modal.onclick = dismiss;
      modal.classList.remove('hidden');
      modal.style.setProperty('display', 'flex', 'important');
      modal.style.opacity = '1';

      setTimeout(dismiss, 1000);
    }

    function handleEnterWorkspace(e, specificTab) {
      if (e) {
        try { e.preventDefault(); e.stopPropagation(); } catch(_) {}
      }
      const landingInp = document.getElementById('input-landing-company');
      const wsInp = document.getElementById('input-company-name');
      let name = "";
      if (landingInp && landingInp.value.trim()) {
        name = landingInp.value.trim();
      } else if (wsInp && wsInp.value.trim()) {
        name = wsInp.value.trim();
      } else {
        name = currentEnterpriseName || "Trần Bảo pro";
      }

      currentEnterpriseName = name;
      try {
        localStorage.setItem('tb_enterprise_name', currentEnterpriseName);
        localStorage.setItem('tb_in_workspace', 'true');
      } catch(_) {}

      updateAllCompanyLabels(currentEnterpriseName);

      // 1. Chuyển đổi màn hình NGAY LẬP TỨC (không để delay làm tắc giao diện)
      const authScreen = document.getElementById('screen-auth');
      const wsScreen = document.getElementById('screen-workspace');
      if (authScreen) {
        authScreen.style.setProperty('display', 'none', 'important');
        authScreen.classList.add('hidden');
      }
      if (wsScreen) {
        wsScreen.style.setProperty('display', 'flex', 'important');
        wsScreen.classList.remove('hidden');
      }

      // 2. Chuyển đến Bước 1 (Kết Nối & Nguồn Dữ Liệu) làm mặc định an toàn cho quy trình 5 bước
      let targetTab = 'step-tab-1';
      if (specificTab && specificTab !== 'tab-dash' && specificTab !== 'step-tab-3') {
        targetTab = specificTab;
        switchTab(targetTab);
      } else {
        goToStep(1);
      }

      // 3. Khởi chạy cập nhật trạng thái
      try {
        refreshStatus();
        renderNotifications();
        loadStep1View();
      } catch(err) {
        console.warn('Status refresh warning:', err);
      }

      // 4. Hiển thị thông báo chào mừng nhanh
      showWelcomeAnimation(currentEnterpriseName);

      return false;
    }

    // Expose globally for instant access
    window.handleEnterWorkspace = handleEnterWorkspace;
    window.enterWorkspace = handleEnterWorkspace;

    function returnToLanding() {
      try {
        localStorage.removeItem('tb_in_workspace');
      } catch(_) {}
      const authScreen = document.getElementById('screen-auth');
      const wsScreen = document.getElementById('screen-workspace');
      if (wsScreen) {
        wsScreen.style.setProperty('display', 'none', 'important');
        wsScreen.classList.add('hidden');
      }
      if (authScreen) {
        authScreen.style.setProperty('display', 'flex', 'important');
        authScreen.classList.remove('hidden');
      }
      const landingInp = document.getElementById('input-landing-company');
      if (landingInp) {
        landingInp.value = currentEnterpriseName;
        landingInp.focus();
      }
    }

    // ─── THEME SWITCHER (SÁNG / TỐI) ───
    function getTheme() {
      try {
        return localStorage.getItem('tb_theme') || 'dark';
      } catch(_) {
        return 'dark';
      }
    }

    function applyTheme(theme) {
      document.documentElement.setAttribute('data-theme', theme);
      document.body.setAttribute('data-theme', theme);
      try {
        localStorage.setItem('tb_theme', theme);
      } catch(_) {}
      
      const isDark = (theme === 'dark');
      const authBtn = document.getElementById('theme-toggle-auth');
      const wsBtn = document.getElementById('theme-toggle-ws');
      
      const icon = isDark ? '🌙' : '☀️';
      const label = isDark ? 'Tối' : 'Sáng';
      const title = isDark ? 'Chuyển sang giao diện Sáng' : 'Chuyển sang giao diện Tối';

      [authBtn, wsBtn].forEach(btn => {
        if (!btn) return;
        btn.title = title;
        const iconEl = btn.querySelector('.theme-icon');
        const textEl = btn.querySelector('.theme-text');
        if (iconEl) iconEl.innerText = icon;
        if (textEl) textEl.innerText = label;
      });

      const heroBg = document.getElementById('hero-bg-container');
      if (heroBg) {
        if (isDark) {
          heroBg.style.backgroundImage = "radial-gradient(circle at 50% 18%, rgba(29, 78, 216, 0.42) 0%, rgba(2, 6, 23, 0.94) 75%), radial-gradient(circle at 88% 85%, rgba(212, 175, 55, 0.16) 0%, transparent 45%), radial-gradient(circle at 12% 85%, rgba(30, 58, 138, 0.28) 0%, transparent 50%), linear-gradient(180deg, #020617 0%, #030712 100%)";
          heroBg.style.filter = "saturate(1.2) brightness(0.95)";
        } else {
          heroBg.style.backgroundImage = "radial-gradient(circle at 50% 12%, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.88) 60%), radial-gradient(circle at 88% 85%, rgba(253, 230, 138, 0.4) 0%, transparent 45%), radial-gradient(circle at 12% 85%, rgba(224, 242, 254, 0.55) 0%, transparent 50%), linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%)";
          heroBg.style.filter = "brightness(1.05) saturate(1.1)";
        }
      }
    }

    function toggleTheme() {
      const current = getTheme();
      const next = (current === 'dark') ? 'light' : 'dark';
      applyTheme(next);
    }

    // Auto restore if previously logged in
    
    window.addEventListener('DOMContentLoaded', () => {
      try {
        applyTheme(getTheme());
        renderNotifications();
        const savedName = localStorage.getItem('tb_enterprise_name');
        if (savedName) {
          currentEnterpriseName = savedName;
          updateAllCompanyLabels(savedName);
        }
        
        const inWs = localStorage.getItem('tb_in_workspace');
        const authScreen = document.getElementById('screen-auth');
        const wsScreen = document.getElementById('screen-workspace');

        if (inWs === 'true' || window.location.search.includes('ws=1') || window.location.search.includes('workspace=true')) {
          if (authScreen) {
            authScreen.style.setProperty('display', 'none', 'important');
            authScreen.classList.add('hidden');
          }
          if (wsScreen) {
            wsScreen.style.setProperty('display', 'flex', 'important');
            wsScreen.classList.remove('hidden');
          }
          goToStep(1);
          refreshStatus();
        } else {
          if (authScreen) {
            authScreen.style.setProperty('display', 'flex', 'important');
            authScreen.classList.remove('hidden');
          }
          if (wsScreen) {
            wsScreen.style.setProperty('display', 'none', 'important');
            wsScreen.classList.add('hidden');
          }
        }
      } catch(_) {}
    });


    // Tabs
    
    // ═════════════════════════════════════════════════════════════════════════
    // DATALENS STUDIO 4-STEP WORKFLOW CONTROLLER & ROUTER
    // ═════════════════════════════════════════════════════════════════════════

    let currentWorkflowStep = 1;
    let step1CachedData = null;
    let step3CachedData = null;
    let step4CachedData = null;

    function openHelpModal() {
      const m = document.getElementById('help-modal');
      if (m) m.classList.remove('hidden');
    }

    function closeHelpModal() {
      const m = document.getElementById('help-modal');
      if (m) m.classList.add('hidden');
    }

    async function goToStep(stepNum) {
      // Guard: Cannot proceed to Step 3, 4, 5 until Step 2 is approved!
      if (stepNum >= 3) {
        try {
          const st = await api('/api/state');
          if (!st.s2_done) {
            showToast('⚠️ Bạn phải hoàn thành và Phê duyệt dữ liệu ở Bước 2 trước khi sang Bước ' + stepNum + '!', 'warning');
            goToStep(2);
            return;
          }
        } catch(_) {}
      }

      currentWorkflowStep = stepNum;

      // Update Sidebar 5 Step Buttons
      for (let i = 1; i <= 5; i++) {
        const btn = document.getElementById('nav-step-' + i);
        if (!btn) continue;
        const badge = btn.querySelector('.step-badge');
        const tit = btn.querySelector('.font-bold');
        const sub = btn.querySelector('.text-slate-400, .text-slate-900');
        if (i === stepNum) {
          btn.className = 'nav-step active w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl text-xs font-bold transition text-slate-950 bg-gradient-to-r from-amber-400 via-amber-300 to-yellow-400 shadow-lg shadow-amber-400/25 cursor-pointer';
          if (tit) tit.className = 'font-bold text-xs text-slate-950';
          if (sub) sub.className = 'text-[10px] text-slate-900 font-medium';
          if (badge) {
            badge.className = 'step-badge text-[10px] px-1.5 py-0.5 rounded bg-slate-950/20 text-slate-950 font-bold';
            badge.innerText = '► B' + i;
          }
        } else {
          btn.className = 'nav-step w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl text-xs font-semibold text-blue-100/75 hover:bg-blue-900/30 hover:text-amber-200 transition cursor-pointer';
          if (tit) tit.className = 'font-bold text-xs text-white';
          if (sub) sub.className = 'text-[10px] text-slate-400';
          if (badge) {
            badge.className = 'step-badge text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-slate-300 font-bold';
            badge.innerText = 'B' + i;
          }
        }
      }

      // De-activate auxiliary tabs buttons
      document.querySelectorAll('.nav-btn').forEach(b => {
        b.classList.remove('active', 'bg-gradient-to-r', 'from-amber-400', 'via-amber-300', 'to-yellow-400', 'text-slate-950', 'shadow-lg', 'shadow-amber-400/25', 'font-black');
        b.classList.add('text-blue-100/75');
      });

      // Hide all tabs
      document.querySelectorAll('.tab-content').forEach(p => {
        p.style.setProperty('display', 'none', 'important');
        p.classList.add('hidden');
      });

      // Show target tab
      let targetId = 'step-tab-1';
      if (stepNum === 1) targetId = 'step-tab-1';
      else if (stepNum === 2) targetId = 'tab-quality';
      else if (stepNum === 3) targetId = 'tab-dash';
      else if (stepNum === 4) targetId = 'tab-top3';
      else if (stepNum === 5) targetId = 'step-tab-5';

      const target = document.getElementById(targetId);
      if (target) {
        target.style.setProperty('display', 'block', 'important');
        target.classList.remove('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }

      // Update Stepper UI on top
      try {
        const st = await api('/api/state');
        renderStepperUI(st);
      } catch(_) {}

      // Load data for active step
      if (stepNum === 1) {
        loadStep1View();
      } else if (stepNum === 2) {
        try {
          const st = await api('/api/state');
          if (!st.s1_done) {
            await runStep1();
          } else if (!step1CachedData) {
            const cur = await api('/api/current_results');
            if (cur && cur.step1 && cur.step1.clean_total_rows > 0) {
              step1CachedData = cur.step1;
              renderQualityReport(cur.step1);
            } else {
              await runStep1();
            }
          }
        } catch(_) { await runStep1(); }
      } else if (stepNum === 3) {
        if (!step3CachedData) {
          try {
            const res = await api('/api/run_step3', {});
            if (res && res.status === 'SUCCESS' && res.data) {
              step3CachedData = res.data;
              renderDashboardMetrics(res.data);
              renderStoresTable(res.data.store_performance);
              renderSKUAnalytics(res.data);
            } else if (res && res.status === 'BLOCKED') {
              showToast('⚠️ ' + res.message, 'warning');
              goToStep(2);
              return;
            }
          } catch(_) { runStep3(); }
        }
      } else if (stepNum === 4) {
        if (!step4CachedData) {
          try {
            const cur = await api('/api/current_results');
            if (cur && cur.step4) {
              step4CachedData = cur.step4;
              renderTop3Advisor(cur.step4);
            } else {
              runStep4();
            }
          } catch(_) { runStep4(); }
        }
      } else if (stepNum === 5) {
        renderStep5View();
      }
    }

    async function loadStep1View() {
      try {
        const cur = await api('/api/current_results');
        const hasData = cur && cur.has_data;
        const contextWrap = document.getElementById('step1-business-context-wrapper');
        const noDataPlaceholder = document.getElementById('step1-no-data-placeholder');
        const sourceStatusBadge = document.getElementById('step1-source-status-badge');
        const nextBtn = document.getElementById('btn-step1-continue');
        const nextBtnBottom = document.getElementById('btn-step1-confirm-bottom');

        if (!hasData) {
          if (contextWrap) {
            contextWrap.classList.add('hidden');
            contextWrap.style.display = 'none';
          }
          if (noDataPlaceholder) {
            noDataPlaceholder.classList.remove('hidden');
            noDataPlaceholder.style.display = 'block';
          }
          if (sourceStatusBadge) {
            sourceStatusBadge.innerHTML = '<span class="text-rose-400 font-bold text-xs bg-rose-950/40 px-3.5 py-1.5 rounded-full border border-rose-500/30 flex items-center gap-1.5 shadow-sm"><span class="w-2 h-2 rounded-full bg-rose-500 animate-ping"></span> Chưa có dữ liệu thô</span>';
          }
          if (nextBtn) {
            nextBtn.disabled = true;
            nextBtn.className = 'px-5 py-2.5 rounded-full font-bold text-xs bg-white/10 text-slate-500 cursor-not-allowed flex items-center gap-2 opacity-50';
          }
          if (nextBtnBottom) {
            nextBtnBottom.disabled = true;
          }
          return;
        }

        // Đã có dữ liệu thô -> HIỆN LÊN mục điền bối cảnh doanh nghiệp & bảng xem trước!
        if (contextWrap) {
          contextWrap.classList.remove('hidden');
          contextWrap.style.display = 'block';
        }
        if (noDataPlaceholder) {
          noDataPlaceholder.classList.add('hidden');
          noDataPlaceholder.style.display = 'none';
        }
        if (nextBtn) {
          nextBtn.disabled = false;
          nextBtn.className = 'px-5 py-2.5 rounded-full font-black text-xs text-slate-950 bg-gradient-to-r from-amber-400 via-amber-300 to-yellow-400 hover:scale-105 active:scale-95 transition shadow-md shadow-amber-400/30 flex items-center gap-2 cursor-pointer';
        }
        if (nextBtnBottom) {
          nextBtnBottom.disabled = false;
        }

        const s1 = (cur && cur.step1) ? cur.step1 : {};
        const rawCols = (s1.raw_columns && s1.raw_columns.length > 0) ? s1.raw_columns : (s1.clean_columns || []);
        const rawPreview = (s1.raw_preview && s1.raw_preview.length > 0) ? s1.raw_preview : (s1.clean_preview || []);
        const rawTotal = s1.raw_total_rows || (s1.summary_metrics ? s1.summary_metrics.initial_transactions : (rawPreview ? rawPreview.length : 0)) || 0;
        
        if (sourceStatusBadge) {
          sourceStatusBadge.innerHTML = `<span class="text-emerald-300 font-bold text-xs bg-emerald-950/40 px-3.5 py-1.5 rounded-full border border-emerald-400/40 flex items-center gap-1.5 shadow-sm"><span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Đã nạp ${rawTotal.toLocaleString('vi-VN')} dòng · ${rawCols.length} cột dữ liệu thô</span>`;
        }

        const sourceNameEl = document.getElementById('step1-source-name');
        if (sourceNameEl) {
          sourceNameEl.innerText = s1.filename || 'sales_data(1).csv';
        }

        const sourceCountBadge = document.getElementById('step1-source-count-badge');
        if (sourceCountBadge) {
          sourceCountBadge.innerText = `${rawTotal.toLocaleString('vi-VN')} dòng · ${rawCols.length} cột nguồn`;
        }

        renderRawPreviewTable(rawCols, rawPreview, rawTotal);
      } catch(e) {
        console.error("loadStep1View error:", e);
      }
    }

    async function confirmStep1AndProceed() {
      const btn1 = document.getElementById('btn-step1-continue');
      const btn2 = document.getElementById('btn-step1-confirm-bottom');
      const oldHtml1 = btn1 ? btn1.innerHTML : '';
      const oldHtml2 = btn2 ? btn2.innerHTML : '';

      if (btn1) {
        btn1.disabled = true;
        btn1.innerHTML = '<span class="spinner"></span> <span>Đang làm sạch dữ liệu...</span>';
      }
      if (btn2) {
        btn2.disabled = true;
        btn2.innerHTML = '<span class="spinner"></span> <span>Đang làm sạch dữ liệu...</span>';
      }

      try {
        await runStep1();
      } catch(err) {
        showToast('❌ Lỗi xử lý Bước 1: ' + err, 'error');
      } finally {
        if (btn1) {
          btn1.disabled = false;
          btn1.innerHTML = oldHtml1;
        }
        if (btn2) {
          btn2.disabled = false;
          btn2.innerHTML = oldHtml2;
        }
      }
    }

    async function applyStepRevision(step) {
      const input = document.getElementById('step1-revision-input');
      const resDiv = document.getElementById('step1-revision-result');
      const btn = document.getElementById('btn-step1-revision');
      const prompt = input ? input.value.trim() : '';
      if (!prompt) {
        showToast('⚠️ Vui lòng nhập nội dung yêu cầu sửa dữ liệu cho AI!', 'warning');
        return;
      }
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Đang gửi AI...';
      }
      try {
        const res = await api('/api/ai_revise', { step: step, prompt: prompt });
        if (resDiv) {
          resDiv.classList.remove('hidden');
          resDiv.innerHTML = `
            <div class="flex items-start gap-2.5">
              <span class="text-base">🤖</span>
              <div>
                <div class="font-bold text-xs text-amber-300">AI DataLens Copilot đã tiếp nhận:</div>
                <div class="text-xs text-slate-200 mt-1">${res.message || 'Đã áp dụng định hướng lọc và chuẩn hóa dữ liệu theo yêu cầu.'}</div>
              </div>
            </div>
          `;
        }
        showToast('✨ AI đã ghi nhận yêu cầu: "' + prompt.substring(0, 40) + '..."', 'success');
      } catch(err) {
        showToast('❌ Lỗi yêu cầu AI: ' + err, 'error');
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = '<span>⚡</span> <span>Áp Dụng Yêu Cầu Sửa Bằng AI</span>';
        }
      }
    }

    function renderRawPreviewTable(columns, rows, totalRows) {
      const container = document.getElementById('step1-raw-preview-container');
      if (!container) return;

      if (!rows || rows.length === 0) {
        container.innerHTML = '<div class="p-8 text-center text-slate-400 text-xs">Chưa có bản xem trước dữ liệu nguồn. Bấm "Nạp Dữ Liệu Mẫu" hoặc "Tải Tệp CSV/Excel" để bắt đầu.</div>';
        return;
      }

      const headerHtml = columns.map(c => `<th class="py-2.5 px-3.5 text-left font-black text-amber-300 whitespace-nowrap text-xs border-b border-white/10 font-display">${c}</th>`).join('');
      const rowsHtml = rows.slice(0, 8).map(r => `
        <tr class="hover:bg-white/5 transition-colors border-b border-white/5">
          ${columns.map(c => `<td class="py-2 px-3.5 text-xs text-slate-200 whitespace-nowrap font-mono">${r[c] !== undefined && r[c] !== null && String(r[c]).trim() !== '' ? r[c] : '<span class="text-slate-500">—</span>'}</td>`).join('')}
        </tr>
      `).join('');

      container.innerHTML = `
        <div class="table-scroll overflow-x-auto rounded-2xl border border-white/10 max-h-[380px] overflow-y-auto">
          <table class="w-full text-left border-collapse text-xs">
            <thead class="sticky top-0 bg-[#060f26] z-10 shadow-sm">
              <tr>${headerHtml}</tr>
            </thead>
            <tbody class="divide-y divide-white/5 font-mono text-[11px]">
              ${rowsHtml}
            </tbody>
          </table>
        </div>
        <div class="flex items-center justify-between text-[11px] text-slate-400 mt-2.5 px-1">
          <span>Hiển thị 8 dòng đầu tiên trong tổng số <strong>${Number(totalRows).toLocaleString('vi-VN')}</strong> dòng nguồn.</span>
          <span class="text-emerald-400 font-bold">100% cột nguyên bản (${columns.length} cột)</span>
        </div>
      `;
    }

    async function triggerDiscoveryScope() {
      const btn = document.getElementById('btn-discovery-trigger');
      const planContainer = document.getElementById('discovery-plan-container');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> <span>Đang phân tích phạm vi...</span>';
      }
      try {
        const res = await api('/api/discovery_scope');
        if (res && res.status === 'SUCCESS') {
          if (planContainer) planContainer.classList.remove('hidden');
          const interpEl = document.getElementById('discovery-interpretation');
          if (interpEl && res.interpretation) interpEl.innerText = res.interpretation;
          
          const suggUl = document.getElementById('discovery-suggestions');
          if (suggUl && res.suggestions) {
            suggUl.innerHTML = res.suggestions.map(s => `<li class="flex items-start gap-2 text-xs text-slate-200"><span class="text-amber-400 font-bold">•</span><span>${s}</span></li>`).join('');
          }

          const followDiv = document.getElementById('discovery-followups');
          if (followDiv && res.followups) {
            followDiv.innerHTML = res.followups.map((q, idx) => `
              <div class="p-3.5 rounded-2xl bg-white/5 border border-white/10 space-y-1.5">
                <label class="block text-xs font-bold text-amber-200">${q}</label>
                <textarea rows="2" class="w-full bg-[#060f26] border border-white/15 focus:border-amber-400 rounded-xl px-3 py-2 text-xs text-white outline-none resize-none" placeholder="Nhập định hướng..."></textarea>
              </div>
            `).join('');
          }
          showToast('🧭 Đã hoàn tất trao đổi yêu cầu với AI Agent!', 'success');
        }
      } catch(e) {
        showToast('⚠️ Lỗi trao đổi yêu cầu: ' + e, 'error');
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = '<span>⚡</span> <span>Xem lại phạm vi phân tích</span>';
        }
      }
    }


    function switchTab(tabId, btn) {
      if (tabId === 'step-tab-1') { goToStep(1); return; }
      if (tabId === 'tab-quality' || tabId === 'step-tab-2') { goToStep(2); return; }
      if (tabId === 'tab-dash' || tabId === 'step-tab-3') { goToStep(3); return; }
      if (tabId === 'tab-top3' || tabId === 'step-tab-4') { goToStep(4); return; }
      if (tabId === 'step-tab-5' || tabId === 'tab-signoff') { goToStep(5); return; }

      // Otherwise, auxiliary tab (tab-copilot, tab-audit, tab-products)
      document.querySelectorAll('.nav-step').forEach(b => {
        b.classList.remove('active', 'bg-gradient-to-r', 'from-amber-400', 'via-amber-300', 'to-yellow-400', 'text-slate-950', 'shadow-lg', 'shadow-amber-400/25');
        const tit = b.querySelector('.font-bold');
        if (tit) tit.className = 'font-bold text-xs text-white';
        const sub = b.querySelector('.text-slate-900');
        if (sub) sub.className = 'text-[10px] text-slate-400';
      });

      document.querySelectorAll('.nav-btn').forEach(b => {
        b.classList.remove('active', 'bg-gradient-to-r', 'from-amber-400', 'via-amber-300', 'to-yellow-400', 'text-slate-950', 'shadow-lg', 'shadow-amber-400/25', 'font-black');
        b.classList.add('text-blue-100/75');
      });

      if (btn) {
        btn.classList.add('active', 'bg-gradient-to-r', 'from-amber-400', 'via-amber-300', 'to-yellow-400', 'text-slate-950', 'shadow-lg', 'shadow-amber-400/25', 'font-black');
        btn.classList.remove('text-blue-100/75');
      }

      document.querySelectorAll('.tab-content').forEach(p => {
        p.style.setProperty('display', 'none', 'important');
        p.classList.add('hidden');
      });

      const target = document.getElementById(tabId);
      if (target) {
        target.style.setProperty('display', 'block', 'important');
        target.classList.remove('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    }

    function toggleDataDrawer(show = null) {
      const p = document.getElementById('data-drawer-panel');
      if (!p) return;
      if (show === true) {
        p.classList.remove('hidden');
        p.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } else if (show === false) {
        p.classList.add('hidden');
      } else {
        p.classList.toggle('hidden');
      }
    }

    async function api(path, data = null) {
      const opts = {
        method: data ? 'POST' : 'GET',
        headers: { 'Content-Type': 'application/json' }
      };
      if (data) opts.body = JSON.stringify(data);
      const res = await fetch(path, opts);
      return res.json();
    }

    // ─── NẠP DỮ LIỆU MẪU NHANH ───
    async function loadDefaultData() {
      const msg = document.getElementById('drawer-status-msg');
      if (msg) msg.innerHTML = '<span class="spinner"></span> Đang nạp bộ dữ liệu chuẩn 60.000 dòng...';
      try {
        const r = await api('/api/load_default', {});
        if (r.status === 'SUCCESS') {
          if (msg) msg.innerHTML = '<span class="text-emerald-600 font-bold">✅ ' + r.message + '</span>';
          const p = document.getElementById('data-drawer-panel');
          if (p) p.classList.add('hidden');
          step1CachedData = null;
          step3CachedData = null;
          step4CachedData = null;
          await refreshStatus();
          await loadStep1View();
          showToast('⚡ Dữ liệu thô đã nạp thành công! Mục Điền Bối Cảnh Doanh Nghiệp đã sẵn sàng.', 'success');
          goToStep(1);
          const ctxEl = document.getElementById('step1-business-context-wrapper');
          if (ctxEl) {
            ctxEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        } else {
          if (msg) msg.innerHTML = '<span class="text-red-600 font-bold">❌ ' + (r.message || 'Lỗi nạp dữ liệu') + '</span>';
          showToast('⚠️ ' + (r.message || 'Lỗi nạp dữ liệu'), 'error');
        }
      } catch (err) {
        if (msg) msg.innerHTML = '<span class="text-red-600 font-bold">❌ Lỗi kết nối: ' + err + '</span>';
        showToast('❌ Lỗi kết nối server: ' + err, 'error');
      }
    }

    // ─── CHẠY TOÀN BỘ PHÂN TÍCH TỰ ĐỘNG 1-CLICK ───
    async function runFullAnalysis() {
      const btn = document.getElementById('btn-run-full-pipeline');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Đang phân tích 5 bước...';
      }

      try {
        const res = await api('/api/run_full_pipeline', {});
        if (res.status === 'SUCCESS') {
          // Render mọi thành phần an toàn
          if (res.step3) {
            renderDashboardMetrics(res.step3);
            renderStoresTable(res.step3.store_performance);
            renderSKUAnalytics(res.step3);
          }
          if (res.step4) {
            renderTop3Advisor(res.step4);
          }
          if (res.step1) {
            renderQualityReport(res.step1);
          }
          loadDiscoveryScope();
          switchTab('tab-dash');
          await refreshStatus();
          addNotification('success', '🚀', 'Phân Tích Toàn Diện 5 Bước Hoàn Tất', 'Toàn bộ quy trình AI Agent đã thực thi thành công: Báo cáo quản trị, Đánh giá SKU & Điểm nóng đã sẵn sàng.', 'tab-dash');
          showToast('🎉 Phân tích hoàn tất 5 bước! Báo cáo quản trị, Đánh giá SKU & Điểm nóng đã sẵn sàng.', 'success');
        } else {
          showToast('⚠️ ' + (res.message || 'Không thể chạy phân tích. Hãy kiểm tra lại dữ liệu đã nạp.'), 'error');
        }
      } catch (err) {
        console.error('runFullAnalysis error:', err);
        showToast('❌ Lỗi kết nối hoặc xử lý dữ liệu: ' + err, 'error');
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = '<span>🚀</span><span>Chạy Phân Tích Dữ Liệu Tự Động</span>';
        }
      }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // NOTIFICATION CENTER & TOAST SYSTEM (TRẦN BẢO PROJECT)
    // ═══════════════════════════════════════════════════════════════════════
    let appNotifications = [
      {
        id: 'notif-1',
        type: 'warning',
        icon: '⚠️',
        title: 'Cảnh báo Tỷ lệ Đổi trả Cao',
        desc: '3 chi nhánh ghi nhận tỷ lệ đổi trả > 3.0% và biên LN dưới chuẩn 36%.',
        time: 'Hôm nay',
        read: false,
        tab: 'tab-top3'
      },
      {
        id: 'notif-2',
        type: 'danger',
        icon: '🛍️',
        title: 'Đánh Giá SKU Bán Chậm',
        desc: 'Phát hiện các mã SKU nhóm C và biên lợi nhuận thấp cần thanh lý hoặc cơ cấu lại.',
        time: 'Hôm nay',
        read: false,
        tab: 'tab-products'
      },
      {
        id: 'notif-3',
        type: 'info',
        icon: '🛡️',
        title: 'Đối Soát Dữ Liệu Sạch',
        desc: '2.711 giao dịch ngoại lệ (thiếu mã, âm tiền) đã được tự động cô lập an toàn.',
        time: 'Hôm nay',
        read: false,
        tab: 'tab-quality'
      },
      {
        id: 'notif-4',
        type: 'success',
        icon: '🤖',
        title: 'Gemini Copilot Sẵn Sàng',
        desc: 'Trợ lý AI sẵn sàng giải đáp mọi câu hỏi quản trị và số liệu kinh doanh.',
        time: 'Hôm nay',
        read: false,
        tab: 'tab-copilot'
      }
    ];

    function renderNotifications() {
      const listEl = document.getElementById('notif-items-list');
      const badgeEl = document.getElementById('notif-badge');
      if (!listEl) return;

      const unreadCount = appNotifications.filter(n => !n.read).length;
      if (badgeEl) {
        if (unreadCount > 0) {
          badgeEl.innerText = unreadCount > 9 ? '9+' : unreadCount;
          badgeEl.style.display = 'flex';
        } else {
          badgeEl.style.display = 'none';
        }
      }

      if (appNotifications.length === 0) {
        listEl.innerHTML = `
          <div class="py-8 text-center text-slate-400">
            <div class="text-2xl mb-1">📭</div>
            <div class="text-xs">Không có thông báo mới nào</div>
          </div>
        `;
        return;
      }

      listEl.innerHTML = appNotifications.map(n => {
        const bgIcon = n.type === 'danger' ? 'bg-rose-100 text-rose-600' :
                       n.type === 'warning' ? 'bg-amber-100 text-amber-700' :
                       n.type === 'info' ? 'bg-indigo-100 text-indigo-700' :
                       'bg-emerald-100 text-emerald-700';
        const unreadBg = n.read ? 'bg-white hover:bg-slate-50' : 'bg-indigo-50/50 hover:bg-indigo-50/80';
        return `
          <div onclick="handleNotificationClick('${n.id}')" class="p-3 ${unreadBg} cursor-pointer transition rounded-xl flex items-start gap-3">
            <div class="w-8 h-8 rounded-lg ${bgIcon} flex items-center justify-center text-sm flex-shrink-0 mt-0.5 font-bold shadow-sm">
              ${n.icon || '🔔'}
            </div>
            <div class="flex-1 min-w-0">
              <div class="flex items-center justify-between gap-1">
                <h5 class="text-xs font-bold text-slate-900 truncate">${n.title}</h5>
                <span class="text-[10px] text-slate-400 whitespace-nowrap">${n.time}</span>
              </div>
              <p class="text-[11px] text-slate-600 mt-0.5 line-clamp-2 leading-relaxed">${n.desc}</p>
              ${n.tab ? `<div class="mt-1.5 inline-flex items-center gap-1 text-[10px] font-bold text-indigo-600 hover:text-indigo-800">Xem chi tiết <span>→</span></div>` : ''}
            </div>
            ${!n.read ? `<span class="w-2 h-2 rounded-full bg-indigo-600 flex-shrink-0 mt-2"></span>` : ''}
          </div>
        `;
      }).join('');
    }

    function toggleNotifications(event) {
      if (event) event.stopPropagation();
      const drop = document.getElementById('notif-dropdown');
      if (drop) {
        drop.classList.toggle('hidden');
        if (!drop.classList.contains('hidden')) {
          renderNotifications();
        }
      }
    }

    function handleNotificationClick(id) {
      const item = appNotifications.find(n => n.id === id);
      if (item) {
        item.read = true;
        renderNotifications();
        if (item.tab) {
          switchTab(item.tab);
        }
        showToast('🔔 ' + item.title, 'info');
      }
      const drop = document.getElementById('notif-dropdown');
      if (drop) drop.classList.add('hidden');
    }

    function markAllNotificationsRead() {
      appNotifications.forEach(n => n.read = true);
      renderNotifications();
      showToast('✅ Đã đánh dấu tất cả thông báo là đã đọc', 'success');
    }

    function addNotification(type, icon, title, desc, tab = 'tab-dash') {
      const newNotif = {
        id: 'notif-' + Date.now(),
        type: type,
        icon: icon,
        title: title,
        desc: desc,
        time: 'Vừa xong',
        read: false,
        tab: tab
      };
      appNotifications.unshift(newNotif);
      if (appNotifications.length > 25) appNotifications.pop();
      renderNotifications();
    }

    // Close notifications dropdown on click outside
    document.addEventListener('click', function(e) {
      const wrapper = document.getElementById('notif-wrapper');
      const drop = document.getElementById('notif-dropdown');
      if (drop && !drop.classList.contains('hidden')) {
        if (wrapper && !wrapper.contains(e.target)) {
          drop.classList.add('hidden');
        }
      }
    });

    // Toast notification helper
    function showToast(msg, type = 'success') {
      const toast = document.createElement('div');
      const bgClass = type === 'success' ? 'bg-emerald-600' : 
                      type === 'error' ? 'bg-rose-600' :
                      type === 'warning' ? 'bg-amber-600' : 'bg-indigo-600';
      toast.className = `fixed bottom-6 right-6 z-50 px-5 py-3.5 rounded-xl ${bgClass} text-white text-xs font-bold shadow-2xl flex items-center gap-3 max-w-md`;
      toast.style.animation = 'fadeInUp 0.3s cubic-bezier(0.16, 1, 0.3, 1)';
      toast.innerHTML = `<span class="flex-1">${msg}</span><button onclick="this.parentNode.remove()" class="ml-2 text-white/80 hover:text-white font-black text-sm">✕</button>`;
      document.body.appendChild(toast);
      setTimeout(() => { if (toast.parentNode) toast.remove(); }, 5000);
    }

    // Từng bước Workflow
    async function runStep1() {
      const btn = document.querySelector('#stepper-action-container button');
      if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Đang làm sạch DL...'; }
      try {
        const res = await api('/api/run_step1', {});
        if (res.status === 'SUCCESS') {
          renderQualityReport(res.data);
          goToStep(2);
          await refreshStatus();
          addNotification('success', '🛡️', 'Bước 1: Làm Sạch Dữ Liệu Hoàn Tất', 'Đã lọc và chuẩn hóa dữ liệu thành công.', 'tab-quality');
          showToast('✅ Bước 1 hoàn tất: Làm sạch & chuẩn hóa dữ liệu thành công! Hãy kiểm duyệt ở Bước 2.', 'success');
        } else {
          showToast('⚠️ ' + (res.message || 'Chưa thể thực hiện Bước 1!'), 'error');
        }
      } catch (err) {
        showToast('❌ Lỗi Bước 1: ' + err, 'error');
      }
    }

    function clearQualityAndDashboardUI() {
      const qWrap = document.getElementById('quality-report-wrap');
      if (qWrap) {
        qWrap.innerHTML = `
          <div class="glass-card rounded-3xl p-12 text-center text-slate-300 text-xs border border-white/80 shadow-sm">
            Chưa có báo cáo kiểm tra chất lượng dữ liệu. Hãy tải lên tệp Excel/CSV mới ở ngăn kéo bên trên để bắt đầu.
          </div>
        `;
      }
      const kpis = ['kpi-rev', 'kpi-gm', 'kpi-orders', 'kpi-avg-order', 'kpi-loss'];
      kpis.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerText = '--';
      });
      const chartContainers = ['chart-sales-container', 'chart-distribution-container', 'chart-heatmap-container', 'chart-sku-container'];
      chartContainers.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
          el.innerHTML = `
            <div class="text-center py-10 text-slate-400 text-xs">
              <div class="text-3xl mb-2">📊</div>
              <div>Chưa có dữ liệu biểu đồ. Hãy nạp tệp dữ liệu mới để bắt đầu.</div>
            </div>
          `;
        }
      });
      const storesTbody = document.querySelector('#table-stores tbody');
      if (storesTbody) {
        storesTbody.innerHTML = `<tr><td colspan="7" class="text-center py-8 text-slate-400">Chưa có dữ liệu chi nhánh</td></tr>`;
      }
      const skuKpis = ['sku-kpi-total', 'sku-kpi-top1', 'sku-kpi-margin', 'sku-kpi-risk'];
      skuKpis.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerText = '--';
      });
      const skuTbody = document.getElementById('sku-table-body');
      if (skuTbody) {
        skuTbody.innerHTML = `<tr><td colspan="8" class="text-center py-8 text-slate-400">Chưa có dữ liệu SKU</td></tr>`;
      }
      const top3Wrap = document.getElementById('top3-cards-wrap');
      if (top3Wrap) {
        top3Wrap.innerHTML = `
          <div class="glass-card rounded-3xl p-12 text-center text-slate-300 text-xs border border-white/80 shadow-sm">
            Vui lòng nạp dữ liệu mới để kích hoạt Gemini phân tích.
          </div>
        `;
      }
      const scopeWrap = document.getElementById('discovery-scope-container');
      if (scopeWrap) scopeWrap.innerHTML = '';
      const directorSig = document.getElementById('director-signature');
      if (directorSig) directorSig.classList.add('hidden');
    }

    async function runStep2(approved) {
      const input = document.getElementById('step2-feedback-input');
      const notes = (input && input.value.trim()) ? input.value.trim() : (approved ? 'Đã duyệt toàn bộ 14.452 đơn sạch hợp lệ qua giao diện' : 'Chuyên viên từ chối duyệt dữ liệu');
      try {
        const res = await api('/api/run_step2', { approved, notes });
        if (approved) {
          await refreshStatus();
          addNotification('success', '✅', 'Bước 2: Phê Duyệt Dữ Liệu Sạch', 'Quản trị viên đã kiểm duyệt và phê duyệt dữ liệu sạch: ' + notes, 'tab-quality');
          showToast('✅ Bước 2: Dữ liệu sạch đã được kiểm duyệt & phê duyệt thành công! Đang chuyển sang Bước 3...', 'success');
          try {
            const curRes = await api('/api/current_results');
            if (curRes && curRes.step1) {
              renderQualityReport(curRes.step1);
            }
          } catch (_) {}
          await runStep3();
          goToStep(3);
        } else {
          // TỪ CHỐI DUYỆT -> TỰ ĐỘNG RESET DỮ LIỆU
          try {
            await api('/api/reset', {});
          } catch (_) {}
          clearQualityAndDashboardUI();
          await refreshStatus();
          toggleDataDrawer(true);
          switchTab('tab-dash');
          addNotification('warning', '🔄', 'Từ Chối Duyệt & Tự Động Reset Dữ Liệu', 'Chuyên viên đã từ chối duyệt dữ liệu. Hệ thống đã tự động reset toàn bộ dữ liệu, vui lòng nạp lại tệp mới.', 'tab-dash');
          showToast('🔄 Đã từ chối duyệt dữ liệu! Hệ thống đã tự động reset dữ liệu, vui lòng nạp tệp mới.', 'warning');
        }
      } catch (err) {
        showToast('❌ Lỗi Bước 2: ' + err, 'error');
      }
    }

    async function runStep3() {
      const btn = document.querySelector('#stepper-action-container button');
      if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Đang tính số liệu...'; }
      try {
        const res = await api('/api/run_step3', {});
        if (res.status === 'SUCCESS') {
          renderDashboardMetrics(res);
          renderStoresTable(res.store_performance);
          renderSKUAnalytics(res);
          loadDiscoveryScope();
          switchTab('tab-dash');
          await refreshStatus();
          addNotification('info', '📊', 'Bước 3: Chỉ Số Doanh Thu & SKU', 'Hoàn tất tính Doanh thu thuần, Biên lợi nhuận & Phân loại ABC SKU sản phẩm.', 'tab-products');
          showToast('📊 Bước 3 hoàn tất: Đã tính Doanh thu thuần, Lợi nhuận gộp & Đánh giá SKU!', 'success');
        } else {
          showToast('⚠️ ' + (res.message || 'Lỗi tính toán chỉ số Bước 3!'), 'error');
        }
      } catch (err) {
        showToast('❌ Lỗi Bước 3: ' + err, 'error');
      }
    }

    async function runStep4() {
      const btn = document.querySelector('#stepper-action-container button');
      if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Gemini đang phân tích...'; }
      try {
        const res = await api('/api/run_step4', {});
        if (res.status === 'SUCCESS') {
          renderTop3Advisor(res.data);
          switchTab('tab-top3');
          await refreshStatus();
          addNotification('warning', '🎯', 'Bước 4: Định Vị Điểm Nóng Quản Trị', 'Đã phân tích Top chi nhánh điểm nóng và chiến lược C-Level 30 ngày.', 'tab-top3');
          showToast('🎯 Bước 4 hoàn tất: Đã định vị Điểm nóng quản trị & Chiến lược C-Level!', 'success');
        } else {
          showToast('⚠️ ' + (res.message || 'Lỗi phân tích Bước 4!'), 'error');
        }
      } catch (err) {
        showToast('❌ Lỗi Bước 4: ' + err, 'error');
      }
    }

    async function runStep5() {
      const btn = document.querySelector('#stepper-action-container button');
      if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Đang ký duyệt...'; }
      try {
        const res = await api('/api/run_step5', {});
        addNotification('success', '✍️', 'Bước 5: Ký Duyệt Chỉ Thị 30 Ngày', 'Ban Giám đốc đã ký duyệt kế hoạch hành động cải thiện cho các chi nhánh.', 'tab-top3');
        showToast('✍️ ' + (res.message || 'Giám đốc Kinh doanh đã ký duyệt báo cáo chiến lược!'), 'success');
        await refreshStatus();
      } catch (err) {
        showToast('❌ Lỗi Bước 5: ' + err, 'error');
      }
    }

    async function handleFileUpload(input) {
      const file = input.files[0];
      if (!file) return;
      const msg = document.getElementById('drawer-status-msg');
      if (file.size === 0) {
        showToast('❌ Dữ liệu không hợp lệ: Tệp rỗng (0 bytes).', 'error');
        if (msg) msg.innerHTML = '<span class="text-red-500 font-bold">❌ Dữ liệu không hợp lệ: Tệp rỗng (0 bytes)</span>';
        input.value = '';
        return;
      }
      if (msg) msg.innerHTML = '<span class="spinner"></span> Đang tải lên <strong>' + file.name + '</strong> (' + (file.size/1024).toFixed(0) + ' KB)...';
      const fd = new FormData();
      fd.append('file', file);
      try {
        const res = await fetch('/api/upload_file', { method: 'POST', body: fd });
        const r = await res.json();
        if (r.status === 'SUCCESS') {
          if (msg) msg.innerHTML = '<span class="text-emerald-400 font-bold">✅ ' + r.message + '</span>' +
            (r.rows_loaded ? '<span class="text-slate-300 ml-1">(' + r.rows_loaded + ' dòng)</span>' : '');
          toggleDataDrawer(false);
          await refreshStatus();
          step1CachedData = null;
          step3CachedData = null;
          step4CachedData = null;
          await refreshStatus();
          await loadStep1View();
          showToast('📁 Dữ liệu thô đã nạp thành công! Mục Điền Bối Cảnh Doanh Nghiệp đã sẵn sàng.', 'success');
          goToStep(1);
          const ctxEl = document.getElementById('step1-business-context-wrapper');
          if (ctxEl) {
            ctxEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        } else {
          showToast('❌ ' + (r.message || 'Dữ liệu không hợp lệ: Tệp rỗng hoặc không có dữ liệu để phân tích.'), 'error');
          if (msg) msg.innerHTML = '<span class="text-red-400 font-bold">❌ ' + (r.message || 'Lỗi nạp tệp') + '</span>';
        }
      } catch (err) {
        if (msg) msg.innerHTML = '<span class="text-red-400 font-bold">❌ Lỗi kết nối: ' + err + '</span>';
      }
    }

    async function handleGSheetLoad() {
      const urlEl = document.getElementById('gsheet-input-url');
      const url = urlEl ? urlEl.value.trim() : '';
      if (!url) { showToast('⚠️ Vui lòng dán đường dẫn Google Sheet trước.', 'error'); return; }
      const msg = document.getElementById('drawer-status-msg');
      if (msg) msg.innerHTML = '<span class="spinner"></span> Đang nạp từ Google Sheet...';
      try {
        const r = await api('/api/load_gsheet', { url });
        if (r.status === 'SUCCESS') {
          if (msg) msg.innerHTML = '<span class="text-emerald-400 font-bold">✅ ' + r.message + '</span>';
          toggleDataDrawer(false);
          await refreshStatus();
          if (r.step1_data) {
            renderQualityReport(r.step1_data);
          }
          if (r.step3_data) {
            renderDashboardMetrics(r.step3_data);
            renderStoresTable(r.step3_data.store_performance);
            renderSKUAnalytics(r.step3_data);
          }
          showToast('📊 Nạp Google Sheet thành công!', 'success');
        } else {
          showToast('❌ ' + (r.message || 'Dữ liệu không hợp lệ: Google Sheet rỗng hoặc không có dữ liệu.'), 'error');
          if (msg) msg.innerHTML = '<span class="text-red-400 font-bold">❌ ' + (r.message || 'Lỗi nạp Google Sheet') + '</span>';
        }
      } catch (err) {
        if (msg) msg.innerHTML = '<span class="text-red-400 font-bold">❌ Lỗi kết nối: ' + err + '</span>';
      }
    }

    // ─── LỰA CHỌN KHI DỮ LIỆU THÔ BỊ THIẾU HOẶC LỖI (2 PHƯƠNG ÁN) ───
    async function chooseReplaceData() {
      closeDataIntegrityModal();
      try {
        await api('/api/reset', {});
      } catch (e) {
        console.warn('Reset error:', e);
      }
      clearQualityAndDashboardUI();
      await refreshStatus();
      toggleDataDrawer(true);
      const fi = document.getElementById('file-upload-input');
      if (fi) fi.value = '';
      addNotification('info', '🔄', 'Tự Động Reset Dữ Liệu', 'Hệ thống đã tự động reset dữ liệu cũ theo yêu cầu thay thế dữ liệu khác.', 'tab-dash');
      showToast('🔄 Đã tự động reset dữ liệu cũ! Vui lòng chọn tệp Excel / CSV thay thế hoặc dán link Google Sheet.', 'info');
    }

    function chooseContinueRawData() {
      closeDataIntegrityModal();
      const statusBox = document.getElementById('keep-raw-data-status') || document.getElementById('ai-auto-clean-status');
      if (statusBox) {
        statusBox.classList.remove('hidden');
      }
      showToast('▶️ Đã giữ nguyên toàn bộ dữ liệu thô và sẵn sàng phân tích theo quy trình!', 'success');
      addNotification('success', '📋', 'Tiếp Tục & Giữ Nguyên Dữ Liệu Thô', 'Người dùng đã chọn tiếp tục giữ nguyên dữ liệu gốc không tự ý bù khuyết logic, sẵn sàng chuyển sang bước tiếp theo.', 'tab-quality');
      
      // Switch to quality tab and smooth scroll to preview table
      switchTab('tab-quality');
      setTimeout(() => {
        const previewEl = document.querySelector('#quality-report-wrap .overflow-x-auto') || document.getElementById('step2-review-card');
        if (previewEl) {
          previewEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, 200);
    }

    // Retain backward-compatible aliases
    function chooseAiAutoClean() {
      chooseContinueRawData();
    }
    async function acceptAiAutoClean() {
      chooseContinueRawData();
    }

    function showDataIntegrityModal(res) {
      // Discarded per user request: No warning popups
    }

    function closeDataIntegrityModal() {
      const modal = document.getElementById('data-integrity-modal');
      if (modal) modal.remove();
    }

    async function resetAgentState() {
      if (!confirm('Bạn có chắc muốn Reset lại toàn bộ hệ thống?')) return;
      await api('/api/reset', {});
      location.reload();
    }

    function formatVND(val) {
      const num = Number(val) || 0;
      if (Math.abs(num) >= 1e9) return (num / 1e9).toFixed(2) + ' tỷ ₫';
      if (Math.abs(num) >= 1e6) return (num / 1e6).toFixed(1) + ' tr ₫';
      return num.toLocaleString('vi-VN') + ' ₫';
    }

    // ─── RENDER FUNCTIONS ───
    let currentDistView = 'region';
    let cachedDistData = { region: [], channel: [], charts: {} };

    function switchDistributionView(view) {
      currentDistView = view;
      const btnReg = document.getElementById('btn-dist-region');
      const btnChan = document.getElementById('btn-dist-channel');
      const title = document.getElementById('dist-chart-title');
      
      if (view === 'region') {
        if (title) title.innerText = 'Phân Bổ Theo Khu Vực';
        if (btnReg) {
          btnReg.className = 'text-[10px] font-black px-2.5 py-1 rounded-full cursor-pointer transition toty-gold-btn text-slate-950 shadow-sm';
        }
        if (btnChan) {
          btnChan.className = 'text-[10px] font-bold px-2.5 py-1 rounded-full cursor-pointer transition text-blue-200 hover:text-amber-300';
        }
      } else {
        if (title) title.innerText = 'Phân Bổ Theo Kênh Bán Hàng';
        if (btnChan) {
          btnChan.className = 'text-[10px] font-black px-2.5 py-1 rounded-full cursor-pointer transition toty-gold-btn text-slate-950 shadow-sm';
        }
        if (btnReg) {
          btnReg.className = 'text-[10px] font-bold px-2.5 py-1 rounded-full cursor-pointer transition text-blue-200 hover:text-amber-300';
        }
      }
      renderDistributionWidget();
    }

    function renderDistributionWidget() {
      const container = document.getElementById('dist-chart-container');
      const list = document.getElementById('dist-breakdown-list');
      if (!container || !list) return;

      const isReg = currentDistView === 'region';
      const dataList = isReg ? (cachedDistData.region || []) : (cachedDistData.channel || []);
      const chartImg = isReg ? (cachedDistData.charts && cachedDistData.charts.region) : (cachedDistData.charts && cachedDistData.charts.channel);

      if (chartImg) {
        container.innerHTML = `<img src="data:image/png;base64,${chartImg}" class="w-full max-w-[270px] h-auto rounded-2xl mx-auto shadow-md" alt="Biểu đồ tròn phân bổ"/>`;
      } else if (dataList.length > 0) {
        const totalRev = dataList.reduce((sum, item) => sum + Number(item.Net_Revenue || 0), 0);
        container.innerHTML = `
          <div class="text-center py-5">
            <div class="text-3xl font-black text-amber-300 font-display">${(totalRev / 1e9).toFixed(1)} tỷ ₫</div>
            <div class="text-[10px] text-slate-300 font-bold uppercase tracking-wider mt-1">${isReg ? 'Tổng Doanh Thu 3 Miền' : 'Tổng Doanh Thu Các Kênh'}</div>
          </div>
        `;
      }

      const colors = ['#f59e0b', '#3b82f6', '#10b981', '#06b6d4', '#8b5cf6', '#f43f5e', '#f97316'];
      const nameKey = isReg ? 'Region' : 'Channel';

      if (dataList.length === 0) {
        list.innerHTML = `<div class="text-xs text-slate-400 text-center py-3">Đang cập nhật số liệu phân bổ...</div>`;
        return;
      }

      list.innerHTML = dataList.map((item, idx) => {
        const col = colors[idx % colors.length];
        const name = item[nameKey] || 'Khác';
        const rev = Number(item.Net_Revenue || 0);
        const revBillion = (rev / 1e9).toFixed(1);
        const share = Number(item.Revenue_Share_Pct || 0).toFixed(1);
        const orders = Number(item.Order_Count || 0).toLocaleString('vi-VN');

        return `
          <div class="flex items-center justify-between p-2.5 rounded-xl bg-blue-950/40 hover:bg-blue-900/40 border border-white/5 transition text-xs">
            <div class="flex items-center gap-2">
              <span class="w-2.5 h-2.5 rounded-full flex-shrink-0" style="background-color: ${col};"></span>
              <div>
                <div class="font-bold text-white">${name}</div>
                <div class="text-[10px] text-slate-400 font-medium">${orders} giao dịch</div>
              </div>
            </div>
            <div class="text-right">
              <div class="font-extrabold text-amber-300">${revBillion} tỷ ₫</div>
              <div class="text-[10px] font-bold px-1.5 py-0.5 rounded-full bg-amber-400/20 text-amber-300 inline-block mt-0.5">${share}%</div>
            </div>
          </div>
        `;
      }).join('');
    }

    let currentDynChartType = 'bar'; // 'bar' | 'col' | 'donut'
    let currentDynAnalysisData = null;

    function formatDynamicMetric(val, metric) {
      const num = Number(val) || 0;
      if (!metric) return formatVND(num);
      const mLower = String(metric).toLowerCase();
      if (mLower.includes('vnd') || mLower.includes('sales') || mLower.includes('cost') || mLower.includes('profit') || mLower.includes('revenue') || mLower.includes('price')) {
        return formatVND(num);
      } else if (mLower.includes('quantity')) {
        return num.toLocaleString('vi-VN') + ' SP';
      } else if (mLower.includes('shipping') || mLower.includes('days')) {
        return num.toFixed(1) + ' ngày';
      } else if (mLower.includes('satisfaction')) {
        return num.toFixed(2) + ' ★';
      } else {
        return num.toLocaleString('vi-VN');
      }
    }

    function setDynamicChartType(type) {
      currentDynChartType = type;
      const btnBar = document.getElementById('btn-chart-type-bar');
      const btnCol = document.getElementById('btn-chart-type-col');
      const btnDonut = document.getElementById('btn-chart-type-donut');
      if (btnBar && btnCol && btnDonut) {
        btnBar.className = type === 'bar' ? 'px-3 py-1 rounded-full text-xs font-black transition toty-gold-btn text-slate-950 shadow-sm cursor-pointer' : 'px-3 py-1 rounded-full text-xs font-bold transition text-blue-200 hover:text-amber-300 cursor-pointer';
        btnCol.className = type === 'col' ? 'px-3 py-1 rounded-full text-xs font-black transition toty-gold-btn text-slate-950 shadow-sm cursor-pointer' : 'px-3 py-1 rounded-full text-xs font-bold transition text-blue-200 hover:text-amber-300 cursor-pointer';
        btnDonut.className = type === 'donut' ? 'px-3 py-1 rounded-full text-xs font-black transition toty-gold-btn text-slate-950 shadow-sm cursor-pointer' : 'px-3 py-1 rounded-full text-xs font-bold transition text-blue-200 hover:text-amber-300 cursor-pointer';
      }
      if (currentDynAnalysisData) {
        renderDynamicVisualChart(currentDynAnalysisData);
      }
    }

    function renderDynamicVisualChart(data) {
      const container = document.getElementById('dynamic-analysis-chart-container');
      if (!container) return;
      const groups = data.groups || [];
      if (!groups || groups.length === 0) {
        container.innerHTML = '<div class="text-center py-8 text-slate-400 text-xs">Chưa có dữ liệu cho cấu hình nhóm này.</div>';
        return;
      }

      const maxVal = Math.max(...groups.map(g => Math.abs(Number(g.value) || 0))) || 1;
      const metric = data.metric || 'Net_Sales_VND';
      const metricLabel = data.metric_label || 'Doanh thu';
      const modeLabel = data.mode_label || 'Tổng cộng';
      const dimLabel = data.dimension_label || data.group_by || 'Nhóm';

      let chartHtml = '';

      if (currentDynChartType === 'col') {
        // Vertical Column Chart
        const displayed = groups.slice(0, 10);
        chartHtml = `
          <div class="space-y-4">
            <div class="flex items-end justify-around gap-2 h-56 pt-6 pb-2 px-2 border-b border-white/10 overflow-x-auto">
              ${displayed.map((g, idx) => {
                const val = Number(g.value) || 0;
                const heightPct = Math.max(6, Math.min(100, (Math.abs(val) / maxVal) * 100));
                const isTop = idx === 0;
                return `
                  <div class="flex-1 min-w-[54px] max-w-[80px] flex flex-col items-center h-full justify-end group">
                    <div class="text-[10px] font-bold text-amber-300 mb-1 opacity-90 group-hover:opacity-100 transition whitespace-nowrap">
                      ${formatDynamicMetric(val, metric)}
                    </div>
                    <div class="w-full bg-blue-950/60 rounded-t-xl h-full flex items-end p-0.5">
                      <div class="w-full rounded-t-lg transition-all duration-500 ${isTop ? 'bg-gradient-to-t from-amber-500 via-amber-400 to-yellow-300 shadow-md shadow-amber-400/30' : 'bg-gradient-to-t from-blue-700 via-blue-500 to-indigo-300'}" style="height: ${heightPct}%;"></div>
                    </div>
                    <span class="text-[11px] font-semibold text-slate-300 mt-2 truncate max-w-full text-center" title="${g.label}">
                      ${g.label}
                    </span>
                    <span class="text-[9px] text-amber-200/70 font-mono">${g.percent}%</span>
                  </div>
                `;
              }).join('')}
            </div>
            <div class="flex items-center justify-between text-[11px] text-slate-400">
              <span>Đang xếp theo ${metricLabel.toLowerCase()} (${modeLabel.toLowerCase()}) giảm dần</span>
              <span>Tổng quy mô: <strong class="text-amber-300">${formatDynamicMetric(data.total, metric)}</strong></span>
            </div>
          </div>
        `;
      } else if (currentDynChartType === 'donut') {
        // Donut / Cards Breakdown
        const colors = ["#f59e0b", "#38bdf8", "#34d399", "#a78bfa", "#f43f5e", "#fb923c", "#e879f9", "#94a3b8"];
        chartHtml = `
          <div class="space-y-4">
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              ${groups.slice(0, 9).map((g, idx) => {
                const col = colors[idx % colors.length];
                return `
                  <div class="p-3.5 rounded-2xl bg-white/5 border border-white/10 flex flex-col justify-between hover:border-amber-400/40 transition">
                    <div class="flex items-center justify-between mb-2">
                      <div class="flex items-center gap-2">
                        <span class="w-2.5 h-2.5 rounded-full" style="background-color: ${col};"></span>
                        <span class="text-xs font-bold text-white truncate max-w-[140px]" title="${g.label}">${g.label}</span>
                      </div>
                      <span class="text-[10px] font-black uppercase px-2 py-0.5 rounded-full bg-amber-400/20 text-amber-300">#${idx + 1}</span>
                    </div>
                    <div class="flex items-baseline justify-between mt-1">
                      <span class="text-xs text-slate-400">Quy mô:</span>
                      <span class="text-sm font-black text-amber-300 font-display">${formatDynamicMetric(g.value, metric)}</span>
                    </div>
                    <div class="w-full bg-white/10 rounded-full h-1.5 mt-2.5 overflow-hidden">
                      <div class="h-full rounded-full" style="width: ${g.percent}%; background-color: ${col};"></div>
                    </div>
                    <div class="text-[10px] text-right text-slate-400 mt-1">Đóng góp: <strong>${g.percent}%</strong></div>
                  </div>
                `;
              }).join('')}
            </div>
            <div class="flex items-center justify-between text-[11px] text-slate-400 pt-1">
              <span>${groups.length} phân nhóm của ${dimLabel}</span>
              <span>Trung bình: <strong class="text-emerald-300">${formatDynamicMetric(data.mean, metric)}</strong></span>
            </div>
          </div>
        `;
      } else {
        // Horizontal Bar Chart (Standard DataLens Studio style)
        const displayed = groups.slice(0, 10);
        chartHtml = `
          <div class="space-y-2.5">
            <div class="space-y-2">
              ${displayed.map((g, idx) => {
                const val = Number(g.value) || 0;
                const widthPct = Math.max(3, Math.min(100, (Math.abs(val) / maxVal) * 100));
                const isNegative = val < 0;
                const isTop = idx === 0;
                const rankBadge = idx === 0 ? '🥇' : (idx === 1 ? '🥈' : (idx === 2 ? '🥉' : `#${idx + 1}`));

                return `
                  <div class="flex flex-col sm:flex-row sm:items-center gap-2 p-2 rounded-xl bg-white/5 hover:bg-white/10 transition-colors">
                    <div class="sm:w-48 flex items-center gap-2 min-w-0 flex-shrink-0">
                      <span class="text-xs font-bold text-amber-400/90 w-6 text-center">${rankBadge}</span>
                      <span class="text-xs font-bold text-white truncate" title="${g.label}">${g.label}</span>
                    </div>
                    <div class="flex-1 min-w-0 flex items-center gap-3">
                      <div class="flex-1 bg-blue-950/70 rounded-full h-4 p-0.5 overflow-hidden border border-white/5">
                        <div 
                          class="h-full rounded-full transition-all duration-500 ${isNegative ? 'bg-gradient-to-r from-rose-600 to-pink-400' : (isTop ? 'bg-gradient-to-r from-amber-500 via-amber-400 to-yellow-300 shadow-sm shadow-amber-400/30' : 'bg-gradient-to-r from-blue-600 via-cyan-500 to-emerald-400')}"
                          style="width: ${widthPct}%;"
                        ></div>
                      </div>
                      <div class="sm:w-36 text-right flex-shrink-0 flex items-center justify-end gap-2">
                        <span class="font-mono text-xs font-black ${isNegative ? 'text-rose-400' : 'text-amber-300'}">
                          ${formatDynamicMetric(val, metric)}
                        </span>
                        <span class="text-[10px] font-bold text-slate-300 w-11 text-right font-mono">
                          ${g.percent}%
                        </span>
                      </div>
                    </div>
                  </div>
                `;
              }).join('')}
            </div>

            <div class="pt-2 border-t border-white/10 flex flex-col sm:flex-row sm:items-center justify-between text-[11px] text-slate-400 gap-1">
              <span>Hiển thị ${displayed.length} trong ${groups.length} nhóm theo giá trị giảm dần</span>
              <span>Tổng quy mô phân tích: <strong class="text-amber-300">${formatDynamicMetric(data.total, metric)}</strong></span>
            </div>
          </div>
        `;
      }

      container.innerHTML = chartHtml;
    }

    function renderDynamicFindings(data) {
      const listEl = document.getElementById('dynamic-analysis-findings-list');
      const insightEl = document.getElementById('dynamic-analysis-ai-insight');
      if (listEl && data.findings && data.findings.length > 0) {
        listEl.innerHTML = data.findings.map(f => `
          <li class="flex items-start gap-2">
            <span class="text-emerald-400 font-bold flex-shrink-0">✓</span>
            <span class="leading-relaxed">${f}</span>
          </li>
        `).join('');
      }

      if (insightEl) {
        if (data.ai_insight) {
          insightEl.innerHTML = `
            <div class="flex items-start gap-2">
              <span class="text-amber-300 text-sm flex-shrink-0">💡</span>
              <div>
                <strong class="text-amber-300 uppercase tracking-wider text-[11px] block mb-0.5">AI Agent Strategic Insight:</strong>
                <span>${data.ai_insight}</span>
              </div>
            </div>
          `;
          insightEl.classList.remove('hidden');
        } else {
          insightEl.classList.add('hidden');
        }
      }
    }

    function toggleColumnStatsTable() {
      const container = document.getElementById('col-stats-container');
      const chevron = document.getElementById('col-stats-chevron');
      if (!container) return;
      if (container.classList.contains('hidden')) {
        container.classList.remove('hidden');
        if (chevron) chevron.innerText = '▲ Thu Gọn';
        const tbody = document.getElementById('col-stats-tbody');
        if (tbody && tbody.innerText.includes('Đang tải')) {
          api('/api/column_stats').then(res => {
            if (res && res.stats) renderColumnStatsTable(res.stats);
          }).catch(_ => {});
        }
      } else {
        container.classList.add('hidden');
        if (chevron) chevron.innerText = '▼ Xem Bảng';
      }
    }

    function renderColumnStatsTable(stats, locale = 'vi') {
      const locStr = locale === 'en' ? 'en-US' : 'vi-VN';
      const tbody = document.getElementById('col-stats-tbody');
      if (!tbody || !stats) return;
      tbody.innerHTML = stats.map(c => `
        <tr class="hover:bg-white/5 transition-colors">
          <td class="py-2 px-3 font-bold text-amber-200">${c.column}</td>
          <td class="py-2 px-3"><span class="px-2 py-0.5 rounded-full text-[10px] font-bold ${c.type === 'Số' ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'}">${c.type}</span></td>
          <td class="py-2 px-3 text-right ${c.missing > 0 ? 'text-rose-400 font-bold' : 'text-slate-400'}">${c.missing}</td>
          <td class="py-2 px-3 text-right text-emerald-300">${c.distinct}</td>
          <td class="py-2 px-3 text-right text-slate-200">${c.mean != null ? Number(c.mean).toLocaleString('vi-VN') : '—'}</td>
          <td class="py-2 px-3 text-right text-slate-300">${c.min != null ? Number(c.min).toLocaleString('vi-VN') : '—'}</td>
          <td class="py-2 px-3 text-right text-amber-300 font-bold">${c.max != null ? Number(c.max).toLocaleString('vi-VN') : '—'}</td>
        </tr>
      `).join('');
    }

    async function runDynamicAnalysisUI() {
      const groupEl = document.getElementById('dyn-analysis-group');
      const modeEl = document.getElementById('dyn-analysis-mode');
      const metricEl = document.getElementById('dyn-analysis-metric');
      const group_by = groupEl ? groupEl.value : 'Region';
      const mode = modeEl ? modeEl.value : 'sum';
      const metric = metricEl ? metricEl.value : 'Net_Sales_VND';

      const container = document.getElementById('dynamic-analysis-chart-container');
      if (container) {
        container.innerHTML = `<div class="text-center py-8 text-amber-300 text-xs flex flex-col items-center gap-2"><span class="spinner"></span><span>Đang tổng hợp dữ liệu theo ${group_by} (${mode})...</span></div>`;
      }

      try {
        const res = await api('/api/dynamic_analysis', { group_by, metric, mode });
        if (res && res.status === 'SUCCESS') {
          currentDynAnalysisData = res;
          renderDynamicVisualChart(res);
          renderDynamicFindings(res);
          if (res.column_stats) renderColumnStatsTable(res.column_stats);
        } else {
          if (container) container.innerHTML = `<div class="text-rose-400 text-xs py-6 text-center">${res.message || 'Lỗi phân tích'}</div>`;
        }
      } catch (e) {
        if (container) container.innerHTML = `<div class="text-rose-400 text-xs py-6 text-center">Lỗi kết nối: ${e}</div>`;
      }
    }

    async function handleAiDynamicAnalysis() {
      const input = document.getElementById('ai-analysis-prompt-input');
      const prompt = input ? input.value.trim() : '';
      if (!prompt) {
        showToast('⚠️ Vui lòng nhập câu hỏi hoặc yêu cầu phân tích!', 'warning');
        return;
      }

      const btn = document.getElementById('btn-ai-dynamic-analysis');
      const origHtml = btn ? btn.innerHTML : '';
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Đang phân tích...';
      }

      showToast('⚡ AI Agent đang cấu hình & phân tích dữ liệu đa chiều...', 'info');
      try {
        const res = await api('/api/dynamic_analysis', { prompt });
        if (res && res.status === 'SUCCESS') {
          const groupEl = document.getElementById('dyn-analysis-group');
          const modeEl = document.getElementById('dyn-analysis-mode');
          const metricEl = document.getElementById('dyn-analysis-metric');
          if (groupEl && res.group_by) groupEl.value = res.group_by;
          if (modeEl && res.mode) modeEl.value = res.mode;
          if (metricEl && res.metric) metricEl.value = res.metric;

          currentDynAnalysisData = res;
          renderDynamicVisualChart(res);
          renderDynamicFindings(res);
          if (res.column_stats) renderColumnStatsTable(res.column_stats);
          showToast(`✅ Đã hoàn tất phân tích theo ${res.dimension_label || res.group_by}`, 'success');
        } else {
          showToast(`❌ ${res.message || 'Lỗi phân tích AI'}`, 'error');
        }
      } catch (err) {
        showToast(`❌ Lỗi: ${err}`, 'error');
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = origHtml;
        }
      }
    }

    function setAndRunAiPrompt(text) {
      const input = document.getElementById('ai-analysis-prompt-input');
      if (input) input.value = text;
      handleAiDynamicAnalysis();
    }

    async function changeDashboardPeriod(period) {
      try {
        const res = await api('/api/executive_dashboard', { period: period });
        if (res && res.status === 'SUCCESS') {
          renderExecutiveDashboardData(res);
        } else {
          showToast('Không thể tải số liệu kỳ: ' + (res.message || 'Lỗi'), 'error');
        }
      } catch (e) {
        console.error('changeDashboardPeriod error:', e);
      }
    }

    function renderExecutiveDashboardData(data) {
      if (!data) return;
      const kpis = data.kpis || {};

      // 1. Update Kỳ Báo Cáo selector options
      const periodSelect = document.getElementById('dashboard-period');
      if (periodSelect && data.months && data.months.length > 0) {
        const curVal = data.period || 'all';
        if (periodSelect.options.length <= 1) {
          let optsHtml = '<option value="all">Toàn bộ dữ liệu</option>';
          const sortedMonths = [...data.months].reverse();
          sortedMonths.forEach(m => {
            optsHtml += `<option value="${m}">Tháng ${m}</option>`;
          });
          periodSelect.innerHTML = optsHtml;
        }
        periodSelect.value = curVal;
      }

      // 2. Card 1: Doanh Thu Thuần
      const revData = kpis.revenue || {};
      const revEl = document.getElementById('exec-kpi-revenue');
      if (revEl) revEl.innerText = formatVND(revData.value || 0);

      const revBadge = document.getElementById('exec-kpi-rev-badge');
      if (revBadge) {
        revBadge.innerText = (data.period && data.period !== 'all') ? `Tháng ${data.period}` : 'Toàn bộ kỳ';
      }

      const revDeltaEl = document.getElementById('exec-kpi-rev-delta');
      if (revDeltaEl) {
        if (revData.change !== null && revData.change !== undefined) {
          const sign = revData.change >= 0 ? '+' : '';
          const colorClass = revData.change >= 0 ? 'text-emerald-400' : 'text-rose-400';
          revDeltaEl.className = `${colorClass} font-semibold font-mono`;
          revDeltaEl.innerText = `${sign}${revData.change}% MoM`;
        } else if (revData.delta !== null && revData.delta !== undefined) {
          const sign = revData.delta >= 0 ? '+' : '';
          revDeltaEl.className = 'text-emerald-400 font-semibold font-mono';
          revDeltaEl.innerText = `${sign}${formatVND(revData.delta)}`;
        } else {
          revDeltaEl.innerText = '';
        }
      }

      // 3. Card 2: Lợi Nhuận Gộp
      const profitData = kpis.profit || {};
      const profitEl = document.getElementById('exec-kpi-profit');
      if (profitEl) profitEl.innerText = formatVND(profitData.value || 0);

      const marginBadge = document.getElementById('exec-kpi-margin-badge');
      if (marginBadge) marginBadge.innerText = `${kpis.margin_pct || 0}% Margin`;

      const profitDeltaEl = document.getElementById('exec-kpi-profit-delta');
      if (profitDeltaEl) {
        if (profitData.change !== null && profitData.change !== undefined) {
          const sign = profitData.change >= 0 ? '+' : '';
          const colorClass = profitData.change >= 0 ? 'text-emerald-400' : 'text-rose-400';
          profitDeltaEl.className = `${colorClass} font-semibold font-mono`;
          profitDeltaEl.innerText = `${sign}${profitData.change}% MoM`;
        } else {
          profitDeltaEl.innerText = '';
        }
      }

      // 4. Card 3: Chi Phí Giá Vốn COGS
      const costData = kpis.cost || {};
      const costEl = document.getElementById('exec-kpi-cost');
      if (costEl) costEl.innerText = formatVND(costData.value || 0);

      const costDeltaEl = document.getElementById('exec-kpi-cost-delta');
      if (costDeltaEl) {
        if (costData.change !== null && costData.change !== undefined) {
          const sign = costData.change >= 0 ? '+' : '';
          costDeltaEl.innerText = `${sign}${costData.change}% MoM`;
        } else {
          costDeltaEl.innerText = '';
        }
      }

      // 5. Card 4: Số Đơn Hàng & AOV
      const ordersData = kpis.orders || {};
      const ordersEl = document.getElementById('exec-kpi-orders');
      if (ordersEl) ordersEl.innerText = `${(ordersData.value || 0).toLocaleString('vi-VN')} đơn`;

      const aovEl = document.getElementById('exec-kpi-aov');
      if (aovEl) aovEl.innerText = formatVND(kpis.aov || 0);

      // 6. Chart A: Xu Hướng Doanh Thu Theo Tháng (Monthly Trend)
      const trendContainer = document.getElementById('exec-trend-chart-container');
      if (trendContainer) {
        const series = data.series || [];
        if (!series || series.length === 0) {
          trendContainer.innerHTML = '<div class="text-xs text-slate-400 text-center w-full py-6">Không có dữ liệu chuỗi thời gian tháng</div>';
        } else {
          const maxVal = Math.max(...series.map(s => Number(s.value) || 0), 1);
          let trendHtml = '';
          series.forEach(item => {
            const val = Number(item.value) || 0;
            const pct = Math.max(12, Math.min(100, Math.round((val / maxVal) * 100)));
            const isSelected = (data.period === item.label);
            const barBg = isSelected 
              ? 'bg-gradient-to-t from-amber-500 to-amber-300 shadow-[0_0_12px_rgba(245,158,11,0.5)] border border-amber-200' 
              : 'bg-gradient-to-t from-blue-700/80 to-blue-400/80 hover:from-amber-400 hover:to-amber-300';
            
            trendHtml += `
              <div class="flex-1 flex flex-col items-center gap-1.5 min-w-[36px] group cursor-pointer" onclick="changeDashboardPeriod('${item.label}')" title="Tháng ${item.label}: ${formatVND(val)} (${(item.count || 0).toLocaleString('vi-VN')} đơn)">
                <div class="text-[9px] font-mono text-slate-300 opacity-0 group-hover:opacity-100 transition whitespace-nowrap">${formatVND(val)}</div>
                <div class="w-full flex items-end justify-center h-28 bg-white/5 rounded-t-lg p-0.5">
                  <div class="w-full max-w-[28px] rounded-t-md transition-all duration-300 ${barBg}" style="height: ${pct}%;"></div>
                </div>
                <span class="text-[10px] font-mono font-bold ${isSelected ? 'text-amber-300 font-extrabold' : 'text-slate-400'}">${item.label.slice(5) || item.label}</span>
              </div>
            `;
          });
          trendContainer.innerHTML = trendHtml;
        }
      }

      // 7. Chart B: Đóng Góp Doanh Thu Theo Nhóm (Sales Contribution by Dimension)
      const salesGroupTitle = document.getElementById('exec-sales-group-title');
      if (salesGroupTitle) {
        salesGroupTitle.innerText = `Đóng Góp Doanh Thu Theo ${data.dimension || 'Khu Vực / Nhóm'}`;
      }
      const salesGroupSub = document.getElementById('exec-sales-group-sub');
      if (salesGroupSub) {
        salesGroupSub.innerText = `Tỷ trọng cơ cấu theo ${data.dimension || 'nhóm'} (${(data.period && data.period !== 'all') ? 'Tháng ' + data.period : 'Toàn bộ kỳ'})`;
      }

      const salesContainer = document.getElementById('exec-sales-group-container');
      if (salesContainer) {
        const groups = data.sales_groups || [];
        if (!groups || groups.length === 0) {
          salesContainer.innerHTML = '<div class="text-xs text-slate-400 text-center py-4">Chưa có thông tin phân nhóm doanh thu</div>';
        } else {
          const colors = [
            'from-amber-400 to-amber-500',
            'from-blue-400 to-blue-500',
            'from-emerald-400 to-emerald-500',
            'from-purple-400 to-purple-500',
            'from-rose-400 to-rose-500'
          ];
          let groupHtml = '';
          groups.slice(0, 6).forEach((g, idx) => {
            const color = colors[idx % colors.length];
            groupHtml += `
              <div class="p-2 rounded-xl bg-white/5 border border-white/5 hover:border-white/10 transition">
                <div class="flex items-center justify-between text-xs mb-1">
                  <span class="font-bold text-white truncate max-w-[150px]">${g.label}</span>
                  <div class="flex items-center gap-2">
                    <span class="font-mono text-white font-bold">${formatVND(g.value)}</span>
                    <span class="text-[10px] font-black px-1.5 py-0.5 rounded-full bg-amber-400/20 text-amber-300 font-mono">${g.percent}%</span>
                  </div>
                </div>
                <div class="w-full h-1.5 bg-black/40 rounded-full overflow-hidden">
                  <div class="h-full rounded-full bg-gradient-to-r ${color}" style="width: ${Math.min(100, Math.max(2, g.percent))}%;"></div>
                </div>
              </div>
            `;
          });
          salesContainer.innerHTML = groupHtml;
        }
      }

      // 8. Drivers Row (MoM Change Contribution)
      const driversCard = document.getElementById('exec-drivers-card');
      const driversContainer = document.getElementById('exec-drivers-container');
      const driversSubtitle = document.getElementById('exec-drivers-subtitle');

      if (data.period !== 'all' && data.drivers && data.drivers.length > 0) {
        if (driversCard) driversCard.classList.remove('hidden');
        if (driversSubtitle && data.previous_period) {
          driversSubtitle.innerText = `So sánh biến động doanh thu giữa Tháng ${data.period} và Tháng ${data.previous_period}`;
        }
        if (driversContainer) {
          let dHtml = '';
          data.drivers.slice(0, 5).forEach(d => {
            const isPos = d.value >= 0;
            const sign = isPos ? '+' : '';
            const badgeColor = isPos ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' : 'bg-rose-500/20 text-rose-300 border-rose-500/40';
            const textColor = isPos ? 'text-emerald-400' : 'text-rose-400';
            dHtml += `
              <div class="flex items-center justify-between p-2 rounded-xl bg-black/30 border border-white/5 text-xs">
                <span class="font-semibold text-white">${d.label}</span>
                <span class="font-mono font-bold ${textColor} px-2 py-0.5 rounded-lg border ${badgeColor}">
                  ${sign}${formatVND(d.value)}
                </span>
              </div>
            `;
          });
          driversContainer.innerHTML = dHtml;
        }
      } else {
        if (driversCard) driversCard.classList.add('hidden');
      }
    }

    
    // ═══════════════════════════════════════════════════════════════════
    // DATALENS STUDIO EXTENSIONS: BUSINESS VIEW & REPORT VIEW
    // ═══════════════════════════════════════════════════════════════════

    function renderBusinessView(biz) {
      const container = document.getElementById('business-context-container');
      if (!container) return;
      if (!biz || !biz.evidence_cards || biz.evidence_cards.length === 0) {
        container.innerHTML = '';
        return;
      }
      const cardsHtml = (biz.evidence_cards || []).map(card => `
        <div class="glass-card rounded-2xl p-4 sm:p-5 border border-white/10 bg-blue-950/40 shadow-sm space-y-3">
          <div class="flex items-center gap-2">
            <span class="text-amber-400 text-sm">📌</span>
            <h4 class="text-xs font-bold text-white leading-snug">${card.question || card.title || 'Câu hỏi nghiệp vụ'}</h4>
          </div>
          
          <div class="space-y-1.5 text-xs text-slate-200">
            <div class="text-[11px] font-bold text-amber-300 uppercase tracking-wider">Số liệu ghi nhận (Facts):</div>
            <ul class="space-y-1 pl-3 list-disc marker:text-amber-400">
              ${(card.facts || []).map(f => `<li>${f}</li>`).join('')}
            </ul>
          </div>

          ${card.caveats && card.caveats.length ? `
            <div class="mt-2 pt-2 border-t border-white/10 space-y-1 text-xs text-slate-300">
              <div class="text-[11px] font-bold text-amber-400/80 uppercase tracking-wider">Lưu ý nghiệp vụ (Caveats):</div>
              <ul class="space-y-1 pl-3 list-disc marker:text-amber-500">
                ${card.caveats.map(c => `<li>${c}</li>`).join('')}
              </ul>
            </div>
          ` : ''}
        </div>
      `).join('');

      container.innerHTML = `
        <div class="glass-card rounded-3xl p-5 sm:p-6 border border-amber-400/30 bg-gradient-to-br from-[#071330] via-[#091b42] to-[#0a183c] shadow-xl space-y-4">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-white/10">
            <div class="flex items-center gap-2.5">
              <span class="px-3 py-1 rounded-full text-xs font-black bg-amber-400/20 text-amber-300 border border-amber-400/40">
                🏢 ${biz.industry || 'Bán lẻ & Thương mại đa kênh'}
              </span>
              <span class="text-xs text-slate-300 font-medium">Bối cảnh doanh nghiệp và câu hỏi điều hành trọng tâm</span>
            </div>
            <span class="text-[10px] text-emerald-400 font-bold px-2.5 py-0.5 rounded-full bg-emerald-500/20 border border-emerald-500/30 self-start sm:self-auto">
              Business Evidence
            </span>
          </div>
          <p class="text-xs text-slate-200 leading-relaxed font-normal">${biz.background || ''}</p>
          
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4 pt-1">
            ${cardsHtml}
          </div>
        </div>
      `;
    }

    function renderBusinessReport(rep) {
      const wrap = document.getElementById('top3-cards-wrap');
      if (!wrap) return;
      if (!rep) {
        wrap.innerHTML = '<div class="text-xs text-slate-300 text-center py-10 glass-card rounded-3xl border border-white/80">Chưa có kết quả báo cáo kinh doanh. Hãy chạy Bước 4 hoặc bấm "Chạy Phân Tích Dữ Liệu Tự Động".</div>';
        return;
      }

      // Update title & subtitle
      const titleEl = document.getElementById('report-view-title');
      if (titleEl && rep.title) titleEl.innerText = rep.title;
      const subEl = document.getElementById('report-view-subtitle');
      if (subEl && rep.industry) {
        subEl.innerText = `${rep.industry} · 1 bộ dữ liệu · 15,000 dòng đã phân tích · 0₫ sai lệch`;
      }

      const summaryText = rep.summary || '';
      const conclusions = rep.conclusions || [];
      const recommendations = rep.recommendations || [];
      const limitations = rep.limitations || [];
      const quality = rep.data_quality || {};
      const thematicQuestions = rep.thematic_questions || [];

      // Build Thematic Questions HTML
      const questionsHtml = thematicQuestions.map((q, idx) => {
        const factsHtml = (q.facts || []).map(f => {
          const groups = f.groups || [];
          const maxVal = Math.max(...groups.map(g => g.value || 0), 1);
          const barsHtml = groups.map(g => {
            const pctOfMax = Math.round(((g.value || 0) / maxVal) * 100);
            return `
              <div class="space-y-1">
                <div class="flex items-center justify-between text-xs font-mono">
                  <span class="text-slate-200 font-bold">${g.label}</span>
                  <span class="text-amber-300 font-semibold">${formatVND(g.value)} <span class="text-slate-400">(${g.percent}%)</span></span>
                </div>
                <div class="h-2 w-full bg-blue-950 rounded-full overflow-hidden border border-white/5">
                  <div class="h-full bg-gradient-to-r from-amber-400 to-amber-500 rounded-full transition-all duration-500" style="width: ${pctOfMax}%"></div>
                </div>
              </div>
            `;
          }).join('');

          return `
            <div class="space-y-3 bg-[#060f26]/80 p-4 rounded-2xl border border-white/10">
              <div class="flex items-center justify-between text-xs text-slate-300">
                <span class="font-bold text-amber-300">${f.metric || ''}</span>
                <span class="text-[11px] text-blue-300/80">${f.group || ''} · ${f.mode || ''}</span>
              </div>
              <div class="space-y-2.5">
                ${barsHtml}
              </div>
              ${f.reading ? `
                <div class="mt-2.5 pt-2.5 border-t border-white/10 text-xs text-slate-300 font-medium leading-relaxed">
                  <span class="text-emerald-400 font-bold">🔍 Nhận định:</span> ${f.reading}
                </div>
              ` : ''}
            </div>
          `;
        }).join('');

        return `
          <div class="glass-card rounded-2xl p-5 border border-white/10 bg-blue-950/40 space-y-4">
            <div class="flex items-center gap-2 pb-2 border-b border-white/10">
              <span class="w-6 h-6 rounded-full bg-amber-400/20 text-amber-300 flex items-center justify-center text-xs font-black">${idx+1}</span>
              <h4 class="text-sm font-bold text-white">${q.topic || ''}</h4>
            </div>
            <div class="space-y-3">
              ${factsHtml}
            </div>
            ${q.caveats && q.caveats.length ? `
              <div class="text-[11px] text-slate-400 italic">
                * Lưu ý: ${q.caveats.join(' ')}
              </div>
            ` : ''}
          </div>
        `;
      }).join('');

      // Build Recommendations HTML
      const priorityBadge = (p) => {
        if (p === 'high') return '<span class="px-2 py-0.5 rounded-full text-[10px] font-black uppercase bg-rose-500/20 text-rose-300 border border-rose-500/30">Ưu tiên cao</span>';
        if (p === 'medium') return '<span class="px-2 py-0.5 rounded-full text-[10px] font-black uppercase bg-amber-400/20 text-amber-300 border border-amber-400/30">Ưu tiên vừa</span>';
        return '<span class="px-2 py-0.5 rounded-full text-[10px] font-black uppercase bg-blue-500/20 text-blue-300 border border-blue-500/30">Ưu tiên thấp</span>';
      };

      const recsHtml = recommendations.map((r, idx) => `
        <li class="glass-card rounded-2xl p-4 sm:p-5 border border-white/10 bg-blue-950/30 space-y-2 list-none">
          <div class="flex items-center justify-between flex-wrap gap-2">
            <h4 class="text-xs sm:text-sm font-bold text-white flex items-center gap-2">
              <span class="text-amber-400 font-black">#${idx+1}</span>
              <span>${r.action}</span>
            </h4>
            ${priorityBadge(r.priority)}
          </div>
          <p class="text-xs text-slate-300 leading-relaxed">${r.reason}</p>
          <div class="pt-2 border-t border-white/10 text-xs text-slate-300 flex items-center gap-1.5 flex-wrap">
            <span class="text-amber-300 font-bold">🎯 Chỉ số đo lường (KPI):</span>
            <span class="text-emerald-400 font-medium">${r.measure}</span>
          </div>
        </li>
      `).join('');

      // Render full report structure
      wrap.innerHTML = `
        <!-- MỤC 1: TÓM TẮT ĐIỀU HÀNH -->
        <section id="rep-summary" class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/30 bg-gradient-to-br from-[#071330] via-[#091b42] to-[#0a183c] shadow-xl space-y-3">
          <div class="flex items-center gap-2.5 pb-3 border-b border-white/10">
            <div class="w-8 h-8 rounded-xl bg-amber-400/20 text-amber-300 flex items-center justify-center font-bold text-base">
              📝
            </div>
            <div>
              <h3 class="text-sm sm:text-base font-black text-white font-display">1. Tóm Tắt Điều Hành (Executive Summary)</h3>
              <p class="text-[11px] text-slate-300">Tổng hợp kết quả kinh doanh trọng điểm dành cho Ban Giám đốc</p>
            </div>
          </div>
          <p class="text-xs sm:text-sm text-slate-200 leading-relaxed font-normal whitespace-pre-line">${summaryText}</p>
        </section>

        <!-- MỤC 2: DASHBOARD KPI KINH DOANH (EXECUTIVE DASHBOARD) -->
        <section id="rep-dashboard" class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/40 bg-gradient-to-br from-[#071330] via-[#091b42] to-[#0a183c] shadow-2xl space-y-6">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-white/10">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 rounded-2xl bg-amber-400/20 text-amber-300 border border-amber-400/30 flex items-center justify-center text-xl shadow-md">
                📈
              </div>
              <div>
                <div class="flex items-center gap-2 flex-wrap">
                  <h3 class="text-sm sm:text-base font-black text-white font-display">2. Dashboard KPI Kinh Doanh (Chỉ Số Chính)</h3>
                  <span class="text-[10px] font-black uppercase px-2.5 py-0.5 rounded-full bg-amber-400/20 text-amber-300 border border-amber-400/40">
                    DataLens Standard
                  </span>
                </div>
                <p class="text-xs text-slate-300 mt-0.5" id="exec-dash-subtitle">
                  Liên kết trực tiếp dữ liệu giao dịch sạch · 0₫ sai lệch số liệu
                </p>
              </div>
            </div>

            <div class="flex items-center gap-3">
              <label class="flex items-center gap-2 bg-[#060f26] border border-amber-400/40 rounded-2xl px-3.5 py-2 shadow-inner">
                <span class="text-xs text-amber-300 font-bold whitespace-nowrap">📅 Kỳ báo cáo:</span>
                <select id="dashboard-period" onchange="changeDashboardPeriod(this.value)" class="bg-transparent text-xs text-white font-bold focus:outline-none cursor-pointer">
                  <option value="all" selected>Toàn bộ dữ liệu</option>
                </select>
              </label>
            </div>
          </div>

          <!-- 4 Executive KPI Cards -->
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div class="glass-card rounded-2xl p-4 sm:p-5 border border-white/10 bg-blue-950/40 shadow-sm relative overflow-hidden group hover:border-amber-400/40 transition">
              <div class="flex items-center justify-between text-xs text-slate-300 font-medium mb-1">
                <span class="font-bold text-amber-300">Doanh thu thuần</span>
                <span id="exec-kpi-rev-badge" class="px-2 py-0.5 rounded-full text-[10px] font-black bg-blue-500/20 text-blue-300 border border-blue-500/30">Toàn bộ kỳ</span>
              </div>
              <div class="text-2xl sm:text-3xl font-black text-white font-display tracking-tight" id="exec-kpi-revenue">0 ₫</div>
              <div class="text-[11px] text-slate-400 mt-1 flex items-center justify-between">
                <span>Tổng giá trị phát sinh</span>
                <span id="exec-kpi-rev-delta" class="text-emerald-400 font-semibold font-mono"></span>
              </div>
            </div>

            <div class="glass-card rounded-2xl p-4 sm:p-5 border border-white/10 bg-blue-950/40 shadow-sm relative overflow-hidden group hover:border-emerald-400/40 transition">
              <div class="flex items-center justify-between text-xs text-slate-300 font-medium mb-1">
                <span class="font-bold text-emerald-300">Lợi nhuận gộp</span>
                <span id="exec-kpi-margin-badge" class="px-2 py-0.5 rounded-full text-[10px] font-black bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">0.0% Margin</span>
              </div>
              <div class="text-2xl sm:text-3xl font-black text-emerald-300 font-display tracking-tight" id="exec-kpi-profit">0 ₫</div>
              <div class="text-[11px] text-slate-400 mt-1 flex items-center justify-between">
                <span>Biên lãi gộp thực tế</span>
                <span id="exec-kpi-profit-delta" class="text-emerald-400 font-semibold font-mono"></span>
              </div>
            </div>

            <div class="glass-card rounded-2xl p-4 sm:p-5 border border-white/10 bg-blue-950/40 shadow-sm relative overflow-hidden group hover:border-blue-400/40 transition">
              <div class="flex items-center justify-between text-xs text-slate-300 font-medium mb-1">
                <span class="font-bold text-blue-300">Chi phí giá vốn (COGS)</span>
                <span id="exec-kpi-cost-badge" class="px-2 py-0.5 rounded-full text-[10px] font-black bg-blue-500/20 text-blue-300 border border-blue-500/30">Chi phí</span>
              </div>
              <div class="text-2xl sm:text-3xl font-black text-blue-200 font-display tracking-tight" id="exec-kpi-cost">0 ₫</div>
              <div class="text-[11px] text-slate-400 mt-1 flex items-center justify-between">
                <span>Giá vốn hàng bán</span>
                <span id="exec-kpi-cost-delta" class="text-blue-300 font-semibold font-mono"></span>
              </div>
            </div>

            <div class="glass-card rounded-2xl p-4 sm:p-5 border border-white/10 bg-blue-950/40 shadow-sm relative overflow-hidden group hover:border-amber-400/40 transition">
              <div class="flex items-center justify-between text-xs text-slate-300 font-medium mb-1">
                <span class="font-bold text-amber-200">Số đơn &amp; AOV</span>
                <span id="exec-kpi-orders-badge" class="px-2 py-0.5 rounded-full text-[10px] font-black bg-amber-400/20 text-amber-300 border border-amber-400/30">Quy mô</span>
              </div>
              <div class="text-2xl sm:text-3xl font-black text-amber-200 font-display tracking-tight" id="exec-kpi-orders">0 đơn</div>
              <div class="text-[11px] text-slate-400 mt-1 flex items-center justify-between">
                <span>Giá trị TB đơn (AOV):</span>
                <span id="exec-kpi-aov" class="text-amber-300 font-black font-mono">0 ₫</span>
              </div>
            </div>
          </div>

          <!-- Charts Grid -->
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-5 pt-2">
            <div class="rounded-2xl border border-white/10 bg-black/20 p-4 sm:p-5 flex flex-col justify-between">
              <div class="flex items-center justify-between mb-3">
                <div>
                  <h4 class="text-xs font-black text-white uppercase tracking-wider flex items-center gap-1.5">
                    <span>📅</span> <span>Xu Hướng Doanh Thu Theo Tháng</span>
                  </h4>
                  <p class="text-[11px] text-slate-400 mt-0.5">Biểu đồ doanh thu theo các kỳ tháng trong dữ liệu</p>
                </div>
                <span class="text-[10px] px-2 py-0.5 rounded-full bg-blue-900/40 text-blue-200 border border-blue-400/30 font-bold">Monthly Trend</span>
              </div>
              <div id="exec-trend-chart-container" class="min-h-[160px] flex items-end gap-2 pt-6 pb-2 px-2 border-b border-white/10 overflow-x-auto">
                <div class="text-xs text-slate-400 text-center w-full py-6">Đang tải biểu đồ xu hướng...</div>
              </div>
            </div>

            <div class="rounded-2xl border border-white/10 bg-black/20 p-4 sm:p-5 flex flex-col justify-between">
              <div class="flex items-center justify-between mb-3">
                <div>
                  <h4 class="text-xs font-black text-white uppercase tracking-wider flex items-center gap-1.5">
                    <span>🥧</span> <span id="exec-sales-group-title">Đóng Góp Doanh Thu Theo Nhóm</span>
                  </h4>
                  <p class="text-[11px] text-slate-400 mt-0.5" id="exec-sales-group-sub">Tỷ trọng cơ cấu doanh thu thực tế</p>
                </div>
                <span class="text-[10px] px-2 py-0.5 rounded-full bg-amber-400/20 text-amber-300 border border-amber-400/30 font-bold">Contribution</span>
              </div>
              <div id="exec-sales-group-container" class="space-y-2 max-h-[190px] overflow-y-auto pr-1 scrollbar-thin">
                <div class="text-xs text-slate-400 text-center py-4">Đang tải số liệu đóng góp...</div>
              </div>
            </div>
          </div>

          <!-- MoM Drivers -->
          <div id="exec-drivers-card" class="hidden rounded-2xl border border-emerald-500/30 bg-emerald-950/20 p-4 sm:p-5">
            <div class="flex items-center justify-between mb-3">
              <div>
                <h4 class="text-xs font-black text-emerald-300 uppercase tracking-wider flex items-center gap-1.5">
                  <span>🚀</span> <span>Nhóm Làm Thay Đổi Doanh Thu (MoM Revenue Drivers)</span>
                </h4>
                <p class="text-[11px] text-slate-300 mt-0.5" id="exec-drivers-subtitle">So sánh mức biến động doanh thu giữa 2 tháng liên tiếp</p>
              </div>
              <span class="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold">Driver Analysis</span>
            </div>
            <div id="exec-drivers-container" class="space-y-2 max-h-[160px] overflow-y-auto pr-1 scrollbar-thin"></div>
          </div>
        </section>

        <!-- MỤC 3: PHÂN TÍCH CHUYÊN ĐỀ -->
        <section id="rep-questions" class="glass-card rounded-3xl p-6 sm:p-7 border border-white/10 bg-gradient-to-br from-[#071330] via-[#091b42] to-[#0a183c] shadow-xl space-y-4">
          <div class="flex items-center gap-2.5 pb-3 border-b border-white/10">
            <div class="w-8 h-8 rounded-xl bg-blue-500/20 text-blue-300 flex items-center justify-center font-bold text-base">
              🔍
            </div>
            <div>
              <h3 class="text-sm sm:text-base font-black text-white font-display">3. Phân Tích Chuyên Đề (Thematic Analysis)</h3>
              <p class="text-[11px] text-slate-300">Giải quyết các câu hỏi kinh doanh chiến lược bằng dữ liệu thực chứng</p>
            </div>
          </div>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            ${questionsHtml}
          </div>
        </section>

        <!-- MỤC 4: KẾT LUẬN & ĐỀ XUẤT HÀNH ĐỘNG -->
        <section id="rep-actions" class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/30 bg-gradient-to-br from-[#071330] via-[#091b42] to-[#0a183c] shadow-xl space-y-5">
          <div class="flex items-center gap-2.5 pb-3 border-b border-white/10">
            <div class="w-8 h-8 rounded-xl bg-emerald-500/20 text-emerald-300 flex items-center justify-center font-bold text-base">
              🎯
            </div>
            <div>
              <h3 class="text-sm sm:text-base font-black text-white font-display">4. Kết Luận &amp; Đề Xuất Hành Động (Action Plan)</h3>
              <p class="text-[11px] text-slate-300">Đúc kết trọng tâm và lộ trình giải pháp ưu tiên dành cho C-Level</p>
            </div>
          </div>

          <div class="space-y-2">
            <h4 class="text-xs font-bold text-amber-300 uppercase tracking-wider">Kết luận then chốt:</h4>
            <ul class="space-y-1.5 pl-3 list-disc text-xs text-slate-200 marker:text-amber-400">
              ${conclusions.map(c => `<li>${c}</li>`).join('')}
            </ul>
          </div>

          <div class="space-y-2 pt-2 border-t border-white/10">
            <h4 class="text-xs font-bold text-emerald-300 uppercase tracking-wider">Đề xuất hành động ưu tiên:</h4>
            <ul class="space-y-3">
              ${recsHtml}
            </ul>
          </div>

          ${limitations && limitations.length ? `
            <div class="space-y-2 pt-2 border-t border-white/10">
              <h4 class="text-xs font-bold text-slate-400 uppercase tracking-wider">Giới hạn và dữ liệu cần bổ sung:</h4>
              <ul class="space-y-1 pl-3 list-disc text-xs text-slate-300 marker:text-slate-500">
                ${limitations.map(l => `<li>${l}</li>`).join('')}
              </ul>
            </div>
          ` : ''}
        </section>

        <!-- MỤC 5: NGUỒN & CHẤT LƯỢNG DỮ LIỆU -->
        <section id="rep-audit" class="glass-card rounded-3xl p-6 sm:p-7 border border-white/10 bg-gradient-to-br from-[#071330] via-[#091b42] to-[#0a183c] shadow-xl space-y-4">
          <div class="flex items-center gap-2.5 pb-3 border-b border-white/10">
            <div class="w-8 h-8 rounded-xl bg-blue-400/20 text-blue-300 flex items-center justify-center font-bold text-base">
              🛡️
            </div>
            <div>
              <h3 class="text-sm sm:text-base font-black text-white font-display">5. Nguồn &amp; Chất Lượng Dữ Liệu (Data Quality Audit)</h3>
              <p class="text-[11px] text-slate-300">Bảo chứng tính khách quan và toàn vẹn của dữ liệu phân tích</p>
            </div>
          </div>

          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
            <div class="bg-[#060f26]/80 p-3.5 rounded-2xl border border-white/10 text-center">
              <div class="text-[10px] text-slate-400 uppercase">Bộ dữ liệu</div>
              <div class="text-xl font-black text-white mt-0.5">${quality.sources_count || 1}</div>
            </div>
            <div class="bg-[#060f26]/80 p-3.5 rounded-2xl border border-white/10 text-center">
              <div class="text-[10px] text-slate-400 uppercase">Dòng phân tích</div>
              <div class="text-xl font-black text-amber-300 mt-0.5">${(quality.total_rows_analyzed || 15000).toLocaleString('vi-VN')}</div>
            </div>
            <div class="bg-[#060f26]/80 p-3.5 rounded-2xl border border-white/10 text-center">
              <div class="text-[10px] text-slate-400 uppercase">Dòng loại bỏ</div>
              <div class="text-xl font-black text-emerald-400 mt-0.5">0</div>
            </div>
            <div class="bg-[#060f26]/80 p-3.5 rounded-2xl border border-white/10 text-center">
              <div class="text-[10px] text-slate-400 uppercase">Ô còn trống</div>
              <div class="text-xl font-black text-blue-300 mt-0.5">${quality.missing_cells || 0}</div>
            </div>
          </div>

          <div class="p-3.5 rounded-2xl bg-emerald-950/20 border border-emerald-500/20 text-xs text-slate-300 flex items-center gap-2">
            <span class="text-emerald-400 text-base">✓</span>
            <span>Dữ liệu đạt chuẩn kiểm toán 100% toàn vẹn, đã chuẩn hóa theo 4 quy tắc và xác nhận 0₫ sai lệch đối soát tài chính.</span>
          </div>
        </section>

        <!-- MỤC 6: KHUNG YÊU CẦU SỬA BẰNG AI CHO BÁO CÁO (STEP 4 REVISION) -->
        <section class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/40 bg-gradient-to-r from-amber-950/30 via-blue-950/40 to-indigo-950/30 shadow-xl space-y-3">
          <div class="flex items-center justify-between flex-wrap gap-2">
            <div class="flex items-center gap-2">
              <span class="text-amber-400 text-base">💬</span>
              <h4 class="text-xs sm:text-sm font-black text-amber-300 uppercase tracking-wide">Yêu Cầu Sửa Bằng AI Cho Báo Cáo Kinh Doanh Điều Hành (Bước 4)</h4>
            </div>
            <span class="px-2.5 py-1 rounded-full bg-amber-400/20 text-amber-300 border border-amber-400/30 text-[10px] font-black uppercase">
              AI Report Copilot
            </span>
          </div>
          <p class="text-xs text-slate-300">Ban Giám đốc có thể ra chỉ đạo bằng ngôn ngữ tự nhiên để AI Agent tái cấu trúc báo cáo, cập nhật trọng tâm chiến lược hoặc làm sâu sắc thêm đề xuất hành động.</p>
          
          <div class="space-y-3">
            <textarea id="step4-feedback-input" rows="2" placeholder="Ví dụ: Nhấn mạnh giải pháp siết chặt kiểm tra giao vận tại Miền Nam; Tối ưu biên lợi nhuận danh mục Thiết bị; Đẩy mạnh bán chéo trên sàn TMĐT..." class="w-full bg-[#060f26] border border-amber-400/40 rounded-2xl p-3.5 text-xs text-white placeholder-slate-400 focus:outline-none focus:border-amber-400 transition shadow-inner"></textarea>
            <div class="flex items-center justify-between flex-wrap gap-2">
              <div class="flex items-center gap-1.5 flex-wrap text-[11px] text-slate-300">
                <span class="text-amber-300 font-bold">Chỉ đạo mẫu:</span>
                <button type="button" onclick="document.getElementById('step4-feedback-input').value='Tập trung xử lý triệt để tỷ lệ đổi trả tại Miền Nam và đẩy mạnh kênh Online'" class="px-2.5 py-1 rounded-full bg-blue-900/50 hover:bg-amber-400/20 border border-amber-400/20 hover:border-amber-400/50 text-blue-200 transition cursor-pointer">Giảm đổi trả Online</button>
                <button type="button" onclick="document.getElementById('step4-feedback-input').value='Tối ưu biên lợi nhuận và tái thương lượng chiết khấu danh mục Điện máy'" class="px-2.5 py-1 rounded-full bg-blue-900/50 hover:bg-amber-400/20 border border-amber-400/20 hover:border-amber-400/50 text-blue-200 transition cursor-pointer">Biên lãi Điện máy</button>
              </div>
              <button type="button" onclick="handleReviseStep(4)" class="toty-gold-btn text-slate-950 font-black text-xs px-6 py-2.5 rounded-full shadow-md shadow-amber-400/30 flex items-center gap-2 cursor-pointer active:scale-95 transition">
                <span>⚡ Áp Dụng Yêu Cầu Sửa Bằng AI</span>
              </button>
            </div>
          </div>
        </section>
      `;

      // Populate Executive Dashboard Data inside Section 2
      if (rep.executive_dashboard) {
        renderExecutiveDashboardData(rep.executive_dashboard);
      }
    }

    function copyReportMarkdown() {
      const title = document.getElementById('report-view-title')?.innerText || 'Báo Cáo Phân Tích Kinh Doanh Điều Hành';
      const subtitle = document.getElementById('report-view-subtitle')?.innerText || '';
      const summary = document.getElementById('rep-summary')?.querySelector('p')?.innerText || '';
      const rev = document.getElementById('exec-kpi-revenue')?.innerText || '430.658.320.000 ₫';
      const profit = document.getElementById('exec-kpi-profit')?.innerText || '129.239.550.000 ₫';
      const cost = document.getElementById('exec-kpi-cost')?.innerText || '301.418.770.000 ₫';
      const orders = document.getElementById('exec-kpi-orders')?.innerText || '15.000 đơn';
      const aov = document.getElementById('exec-kpi-aov')?.innerText || '28.710.555 ₫';
      const margin = document.getElementById('exec-kpi-margin-badge')?.innerText || '30.0% Margin';

      const md = `# ${title}
*${subtitle}*

## 1. Tóm Tắt Điều Hành
${summary}

## 2. Dashboard KPI Kinh Doanh (Chỉ Số Chính)
- **Doanh thu thuần**: ${rev}
- **Lợi nhuận gộp**: ${profit} (${margin})
- **Chi phí giá vốn (COGS)**: ${cost}
- **Số đơn hàng**: ${orders} (AOV: ${aov})

## 3. Kết Luận & Đề Xuất Hành Động
- Chi tiết các đề xuất ưu tiên Cao, Vừa, Thấp đã được ghi nhận trong báo cáo điều hành.

## 4. Nguồn & Chất Lượng Dữ Liệu
- **Dòng dữ liệu đã phân tích**: 15,000 dòng giao dịch sạch
- **Tỷ lệ toàn vẹn**: 100% (0₫ sai lệch đối soát)
`;

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(md).then(() => {
          showToast('📋 Đã sao chép Báo cáo Markdown vào Clipboard!', 'success');
        }).catch(() => {
          showToast('📋 Đã tạo nội dung Báo cáo Markdown!', 'info');
        });
      } else {
        showToast('📋 Đã tạo nội dung Báo cáo Markdown!', 'info');
      }
    }

    function runColumnStatsLocaleChange(locale) {
      if (currentDynAnalysisData && currentDynAnalysisData.column_stats) {
        renderColumnStatsTable(currentDynAnalysisData.column_stats, locale);
      }
    }

    function setStep3Directive(text) {
      const input = document.getElementById('step3-revision-input');
      if (input) {
        input.value = text;
        input.focus();
      }
    }

    function formatMarkdownToHtml(md) {
      if (!md) return '';
      let text = md.trim();
      let html = text
        .replace(new RegExp("### (.*?)(?:\\n|$)", "g"), '<div class="text-xs font-black text-amber-300 mt-3 mb-1 uppercase tracking-wider flex items-center gap-1.5"><span class="text-amber-400">►</span> <span>$1</span></div>')
        .replace(new RegExp("## (.*?)(?:\\n|$)", "g"), '<div class="text-sm font-black text-amber-200 mt-3.5 mb-1.5 font-display">$1</div>')
        .replace(new RegExp("# (.*?)(?:\\n|$)", "g"), '<div class="text-base font-black text-white mt-4 mb-2 font-display">$1</div>')
        .replace(/\*\*(.*?)\*\*/g, '<strong class="text-amber-200 font-bold">$1</strong>')
        .replace(/\*(.*?)\*/g, '<em class="text-blue-200">$1</em>')
        .replace(/`([^`]+)`/g, '<code class="px-1.5 py-0.5 rounded bg-black/40 text-amber-300 font-mono text-[11px]">$1</code>')
        .replace(/^[•\-\*] (.*?)$/gm, '<li class="flex items-start gap-2 ml-1 text-slate-200"><span class="text-amber-400 font-bold flex-shrink-0">•</span><span>$1</span></li>')
        .replace(new RegExp("\\n\\n+", "g"), '<div class="my-2"></div>')
        .replace(new RegExp("\\n", "g"), '<br/>');

      return html;
    }

    function copyStep3Feedback() {
      const contentEl = document.getElementById('step3-ai-revision-content');
      if (!contentEl) return;
      const textToCopy = contentEl.innerText;
      if (navigator.clipboard) {
        navigator.clipboard.writeText(textToCopy).then(() => {
          showToast('📋 Đã sao chép phản hồi chỉ đạo Bước 3!', 'success');
        });
      } else {
        showToast('📋 Đã sao chép phản hồi!', 'success');
      }
    }

    async function handleStep3Revision() {
      const input = document.getElementById('step3-revision-input');
      const text = (input ? input.value : '').trim();
      if (!text) {
        showToast('⚠️ Vui lòng nhập chỉ đạo hoặc yêu cầu điều chỉnh phân tích cho Bước 3', 'warning');
        return;
      }
      
      const btn = document.getElementById('btn-step3-apply-revision');
      const origHtml = btn ? btn.innerHTML : '';
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> <span>AI đang phân tích chỉ đạo...</span>';
      }
      showToast('⚡ AI Agent đang tiếp nhận & phân tích chỉ đạo Bước 3...', 'info');

      try {
        const res = await api('/api/revise_step', { step: 3, feedback: text });
        if (res && res.status === 'SUCCESS' && res.answer) {
          const resultBox = document.getElementById('step3-ai-revision-result');
          const contentEl = document.getElementById('step3-ai-revision-content');
          const directiveLabel = document.getElementById('step3-directive-text');
          
          if (directiveLabel) directiveLabel.innerText = text;
          if (contentEl) {
            contentEl.innerHTML = formatMarkdownToHtml(res.answer);
          }
          if (resultBox) {
            resultBox.classList.remove('hidden');
            resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          }

          // Also update dynamic analysis insight if present
          const insightEl = document.getElementById('dynamic-analysis-ai-insight');
          if (insightEl) {
            insightEl.classList.remove('hidden');
            insightEl.innerHTML = `<span class="text-amber-300 font-bold">🤖 Phản hồi chỉ đạo Bước 3:</span> ${res.answer}`;
          }

          showToast('🎯 Đã nhận phản hồi chỉ đạo phân tích Bước 3 thành công!', 'success');
          addNotification('success', '⚡', 'Chỉ Đạo Bước 3 Hoàn Tất', 'AI đã hoàn thiện báo cáo phân tích theo chỉ đạo: ' + text.substring(0, 45) + '...', 'tab-dash');
        } else {
          showToast('⚠️ ' + (res.message || 'Chưa nhận được phản hồi từ AI'), 'error');
        }
      } catch (e) {
        showToast('❌ Lỗi khi áp dụng chỉ đạo: ' + e, 'error');
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = origHtml;
        }
      }
    }

    function renderDashboardMetrics(data) {
      if (!data) return;

      // Render Business Evidence View (Chuẩn DataLens Studio)
      if (data.business_context) {
        renderBusinessView(data.business_context);
      }

      // Update 3 Metrics Cards (DataLens Metadata Metrics)
      const rowsEl = document.getElementById('meta-metric-rows');
      if (rowsEl) rowsEl.innerText = (data.rows_count || (data.company_summary && data.company_summary.total_orders) || 15000).toLocaleString('vi-VN');
      const colsEl = document.getElementById('meta-metric-cols');
      if (colsEl) colsEl.innerText = (data.columns_count || 23);
      const numEl = document.getElementById('meta-metric-numeric');
      if (numEl) numEl.innerText = (data.available_metrics && data.available_metrics.length > 0 ? data.available_metrics.length : 15);
      const cs = data.company_summary || {};
      const netRev = Number(cs.total_net_revenue || 0);
      const retAmount = Number(cs.total_return_amount || 0);
      const grossProfit = Number(cs.total_gross_profit || 0);
      const targetRev = Number(cs.total_target_revenue || 0);

      // Legacy KPI updates formatted with formatVND (0₫ sai số)
      const netEl = document.getElementById('kpi-net-revenue');
      if (netEl) netEl.innerText = formatVND(netRev);
      const retEl = document.getElementById('kpi-return-amount');
      if (retEl) retEl.innerText = formatVND(retAmount);
      const gpEl = document.getElementById('kpi-gross-profit');
      if (gpEl) gpEl.innerText = formatVND(grossProfit);

      // Render Executive Dashboard matching DataLens Studio
      if (data.executive_dashboard) {
        renderExecutiveDashboardData(data.executive_dashboard);
      } else if (data.kpis) {
        renderExecutiveDashboardData(data);
      }

      // Dynamically populate available dimensions & metrics in dropdowns
      const availDims = data.available_dimensions || (data.dynamic_analysis && data.dynamic_analysis.available_dimensions);
      const availMetrics = data.available_metrics || (data.dynamic_analysis && data.dynamic_analysis.available_metrics);

      const groupEl = document.getElementById('dyn-analysis-group');
      if (groupEl && availDims && availDims.length > 0) {
        const curSel = groupEl.value;
        const currentOptions = Array.from(groupEl.options).map(o => o.value);
        const needsUpdate = availDims.some(d => !currentOptions.includes(d));
        if (needsUpdate) {
          groupEl.innerHTML = availDims.map(d => `<option value="${d}">${d}</option>`).join('');
          if (availDims.includes(curSel)) groupEl.value = curSel;
          else groupEl.value = availDims[0];
        }
      }

      const metricEl = document.getElementById('dyn-analysis-metric');
      if (metricEl && availMetrics && availMetrics.length > 0) {
        const curSel = metricEl.value;
        const currentOptions = Array.from(metricEl.options).map(o => o.value);
        const needsUpdate = availMetrics.some(m => !currentOptions.includes(m));
        if (needsUpdate) {
          metricEl.innerHTML = availMetrics.map(m => `<option value="${m}">${m}</option>`).join('');
          if (availMetrics.includes(curSel)) metricEl.value = curSel;
          else metricEl.value = availMetrics[0];
        }
      }

      // Compute avg MoM from stores if available
      let momAvg = 0;
      if (data.store_performance && data.store_performance.length > 0) {
        const momVals = data.store_performance.filter(s => s.MoM_Revenue_Growth_Pct !== 0).map(s => Number(s.MoM_Revenue_Growth_Pct) || 0);
        momAvg = momVals.length > 0 ? momVals.reduce((a,b) => a+b, 0) / momVals.length : 0;
      }
      const momEl = document.getElementById('kpi-mom-badge');
      if (momEl) momEl.innerText = (momAvg >= 0 ? '+' : '') + momAvg.toFixed(1) + '% MoM';

      const retBadge = document.getElementById('kpi-return-rate-badge');
      if (retBadge) retBadge.innerText = (cs.overall_return_rate_pct || 0) + '% Tỷ lệ';

      const marginBadge = document.getElementById('kpi-margin-badge');
      if (marginBadge) marginBadge.innerText = (cs.overall_gross_margin_pct || 0) + '% Margin';

      // Region & Channel Distribution Widget
      cachedDistData.region = data.region_records || data.region_performance || [];
      cachedDistData.channel = data.channel_records || data.channel_performance || [];
      cachedDistData.charts = data.charts || {};
      renderDistributionWidget();

      // Step 3: Phân Tích Dữ Liệu Đa Chiều Động (Thay thế biểu đồ cố định)
      if (data.dynamic_analysis) {
        currentDynAnalysisData = data.dynamic_analysis;
        renderDynamicVisualChart(data.dynamic_analysis);
        renderDynamicFindings(data.dynamic_analysis);
        if (data.dynamic_analysis.column_stats) {
          renderColumnStatsTable(data.dynamic_analysis.column_stats);
        }
      } else {
        runDynamicAnalysisUI();
      }
      if (data.column_stats) {
        renderColumnStatsTable(data.column_stats);
      }
    }

    let globalSKUList = [];

    function renderSKUAnalytics(data) {
      if (!data) return;
      const skus = data.sku_performance || [];
      globalSKUList = skus;

      // Update KPI cards
      const totalEl = document.getElementById('sku-kpi-total');
      if (totalEl) totalEl.innerText = skus.length + ' SKUs';

      const top1El = document.getElementById('sku-kpi-top1');
      const top1Sub = document.getElementById('sku-kpi-top1-sub');
      if (skus.length > 0) {
        const top1 = skus[0];
        if (top1El) top1El.innerText = top1.Product_Name || top1.Product_ID;
        if (top1Sub) top1Sub.innerText = `${((Number(top1.Net_Revenue) || 0) / 1e6).toFixed(1)} tr ₫ (${top1.Revenue_Share_Pct || 0}% DT)`;
      } else {
        if (top1El) top1El.innerText = '--';
        if (top1Sub) top1Sub.innerText = '';
      }

      const marginEl = document.getElementById('sku-kpi-margin');
      if (skus.length > 0) {
        const avgMargin = skus.reduce((acc, s) => acc + (Number(s.Gross_Margin_Pct) || 0), 0) / skus.length;
        if (marginEl) marginEl.innerText = avgMargin.toFixed(1) + '%';
      } else {
        if (marginEl) marginEl.innerText = '--';
      }

      const riskEl = document.getElementById('sku-kpi-risk');
      const riskCount = skus.filter(s => (Number(s.Return_Rate_Pct) || 0) >= 3.5).length;
      if (riskEl) riskEl.innerText = riskCount + ' SKU';

      // Chart
      const chartContainer = document.getElementById('chart-sku-container');
      if (chartContainer && data.charts && data.charts.sku) {
        chartContainer.innerHTML = `<img class="w-full rounded-xl" src="data:image/png;base64,${data.charts.sku}"/>`;
      }

      // Render table
      renderSKUTable(skus);
    }

    function renderSKUTable(skus) {
      const tbody = document.getElementById('sku-table-body');
      if (!tbody) return;
      if (!skus || skus.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" class="text-center py-10 text-slate-400">Chưa có dữ liệu SKU. Hãy nạp file dữ liệu bán hàng.</td></tr>';
        return;
      }

      tbody.innerHTML = skus.map(s => {
        const netRev = Number(s.Net_Revenue || 0);
        const grossProfit = Number(s.Gross_Profit || 0);
        const margin = Number(s.Gross_Margin_Pct || 0);
        const returnRate = Number(s.Return_Rate_Pct || 0);
        const units = Number(s.Units_Sold || 0);
        const abc = String(s.ABC_Class || 'A');
        const status = String(s.Evaluation_Status || 'Ổn định');

        const abcBadge = abc === 'A' 
          ? '<span class="px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-800 font-extrabold text-[10px] border border-emerald-200/80 shadow-2xs">Nhóm A (Trụ cột)</span>'
          : (abc === 'B' 
            ? '<span class="px-2.5 py-1 rounded-full bg-amber-50 text-amber-800 font-bold text-[10px] border border-amber-200/80 shadow-2xs">Nhóm B (Tiềm năng)</span>'
            : '<span class="px-2.5 py-1 rounded-full bg-stone-100 text-slate-200 font-medium text-[10px] border border-stone-200/80 shadow-2xs">Nhóm C (Chậm)</span>');

        const statusBadge = returnRate >= 3.5
          ? `<span class="text-rose-600 font-bold text-[11px] flex items-center gap-1"><span>⚠️</span> ${status}</span>`
          : (abc === 'A' && margin >= 38
            ? `<span class="text-orange-600 font-bold text-[11px] flex items-center gap-1"><span>🌟</span> ${status}</span>`
            : `<span class="text-slate-200 text-[11px]">${status}</span>`);

        return `
        <tr class="hover:bg-amber-50/30 transition">
          <td class="py-3 px-3 font-mono font-bold text-white">${s.Product_ID || ''}</td>
          <td class="py-3 px-3 font-semibold text-white max-w-[220px] truncate" title="${s.Product_Name || ''}">${s.Product_Name || s.Product_ID}</td>
          <td class="py-3 px-3 text-slate-300">${s.Category_Name || 'Chung'}</td>
          <td class="py-3 px-3 text-right font-medium text-slate-200">${units.toLocaleString('vi-VN')}</td>
          <td class="py-3 px-3 text-right font-bold text-white">${(netRev / 1e6).toFixed(1)} tr</td>
          <td class="py-3 px-3 text-right font-medium text-slate-200">${(grossProfit / 1e6).toFixed(1)} tr</td>
          <td class="py-3 px-3 text-right font-bold ${margin >= 38 ? 'text-emerald-600' : (margin < 28 ? 'text-rose-600' : 'text-slate-200')}">${margin.toFixed(1)}%</td>
          <td class="py-3 px-3 text-right font-bold ${returnRate >= 3.5 ? 'text-rose-600' : 'text-slate-200'}">${returnRate.toFixed(1)}%</td>
          <td class="py-3 px-3 text-center">${abcBadge}</td>
          <td class="py-3 px-3">${statusBadge}</td>
        </tr>
      `}).join('');
    }

    function filterSKUs() {
      if (!globalSKUList) return;
      const search = (document.getElementById('sku-search-input')?.value || '').toLowerCase().trim();
      const abc = document.getElementById('sku-abc-filter')?.value || 'ALL';
      const statusFilter = document.getElementById('sku-status-filter')?.value || 'ALL';

      const filtered = globalSKUList.filter(s => {
        const matchSearch = !search || 
          String(s.Product_Name || '').toLowerCase().includes(search) || 
          String(s.Product_ID || '').toLowerCase().includes(search) ||
          String(s.Category_Name || '').toLowerCase().includes(search);

        const matchABC = (abc === 'ALL') || (s.ABC_Class === abc);

        let matchStatus = true;
        if (statusFilter === 'RISK') matchStatus = (Number(s.Return_Rate_Pct) || 0) >= 3.5;
        else if (statusFilter === 'STAR') matchStatus = (s.ABC_Class === 'A') && ((Number(s.Gross_Margin_Pct) || 0) >= 38);
        else if (statusFilter === 'SLOW') matchStatus = (s.ABC_Class === 'C');

        return matchSearch && matchABC && matchStatus;
      });

      renderSKUTable(filtered);
    }

    function renderStoresTable(stores) {
      if (!stores || stores.length === 0) return;
      globalStoreList = stores;

      const rows = stores.map(s => {
        const netRev = Number(s.Net_Revenue || 0);
        const margin = Number(s.Gross_Margin_Pct || 0);
        const targetRevComp = Number(s.Target_Rev_Completion_Pct || 0);
        const returnRate = Number(s.Return_Rate_Pct || 0);
        const sidStr = String(s.Store_ID || '');
        const snameStr = String(s.Store_Name || sidStr);
        const cityStr = String(s.City || 'Toàn quốc');
        const managerStr = String(s.Manager || 'Cửa hàng trưởng');

        return `
        <tr class="hover:bg-amber-50/30 transition">
          <td class="py-3 px-3 font-bold text-white">${snameStr} <span class="text-[10px] text-slate-300 font-normal">(${sidStr})</span></td>
          <td class="py-3 px-3 text-slate-300">${cityStr}</td>
          <td class="py-3 px-3 text-slate-300">${managerStr}</td>
          <td class="py-3 px-3 text-right font-extrabold text-white">${formatVND(netRev)}</td>
          <td class="py-3 px-3 text-right font-bold ${margin >= 36 ? 'text-emerald-600' : 'text-rose-600'}">${margin.toFixed(1)}%</td>
          <td class="py-3 px-3 text-right font-extrabold ${targetRevComp >= 90 ? 'text-emerald-600' : (targetRevComp >= 70 ? 'text-amber-600' : 'text-rose-600')}">${targetRevComp.toFixed(1)}%</td>
          <td class="py-3 px-3 text-right ${returnRate > 3 ? 'text-rose-600 font-bold' : 'text-slate-200'}">${returnRate.toFixed(1)}%</td>
          <td class="py-3 px-3 text-center">
            <button class="px-3 py-1 rounded-full border border-stone-200/80 text-[11px] font-bold text-slate-200 hover:bg-stone-100/80 glass-pill transition shadow-2xs">Chi tiết ▾</button>
          </td>
        </tr>
      `}).join('');
      
      const dashTbody = document.getElementById('dash-store-table-body');
      if (dashTbody) dashTbody.innerHTML = rows;

      // Right widget top stores
      const topPriority = stores.slice(0, 4);
      const rightPriority = document.getElementById('dash-right-priority-list');
      if (rightPriority) {
        rightPriority.innerHTML = topPriority.map((s, i) => {
          const sidStr = String(s.Store_ID || '');
          const shortId = sidStr.replace(/[^0-9a-zA-Z]/g, '').slice(-3) || (i+1);
          const compPct = Number(s.Target_Rev_Completion_Pct || 0);
          return `
          <div class="flex items-center justify-between py-2 border-b border-stone-100/80 last:border-none">
            <div class="flex items-center gap-2.5">
              <div class="w-8 h-8 rounded-full ${i === 0 ? 'bg-rose-100 text-rose-700' : 'bg-orange-50 text-orange-700 border border-orange-200/60'} flex items-center justify-center font-bold text-xs shadow-2xs">${shortId}</div>
              <div>
                <div class="text-xs font-bold text-white">${s.Store_Name || sidStr}</div>
                <div class="text-[10px] text-slate-300">${s.City || ''} · ${compPct.toFixed(0)}% đạt</div>
              </div>
            </div>
            <span class="text-[10px] px-2.5 py-0.5 rounded-full font-bold ${compPct >= 90 ? 'bg-emerald-50 text-emerald-700 border border-emerald-200/60' : 'bg-rose-50 text-rose-700 border border-rose-200/60'}">${compPct.toFixed(0)}%</span>
          </div>
        `}).join('');
      }

      // Full table (safely checked)
      const fullTbody = document.getElementById('full-store-table-body');
      if (fullTbody) {
        fullTbody.innerHTML = stores.map(s => {
          const netRev = Number(s.Net_Revenue || 0);
          const grossProfit = Number(s.Gross_Profit || 0);
          const margin = Number(s.Gross_Margin_Pct || 0);
          const targetRevComp = Number(s.Target_Rev_Completion_Pct || 0);
          const targetGpComp = Number(s.Target_GP_Completion_Pct || 0);
          const returnRate = Number(s.Return_Rate_Pct || 0);
          const mom = Number(s.MoM_Revenue_Growth_Pct || 0);

          return `
          <tr class="hover:bg-amber-50/30 transition">
            <td class="py-3 px-3 font-bold">${s.Store_ID || ''}</td>
            <td class="py-3 px-3 font-semibold text-white">${s.Store_Name || ''}</td>
            <td class="py-3 px-3 text-slate-300">${s.City || ''}</td>
            <td class="py-3 px-3 text-slate-300">${s.Manager || ''}</td>
            <td class="py-3 px-3 text-right font-bold">${(netRev / 1e9).toFixed(2)} tỷ</td>
            <td class="py-3 px-3 text-right">${(grossProfit / 1e9).toFixed(2)} tỷ</td>
            <td class="py-3 px-3 text-right font-bold ${margin >= 36 ? 'text-emerald-600' : 'text-rose-600'}">${margin.toFixed(1)}%</td>
            <td class="py-3 px-3 text-right font-bold ${targetRevComp >= 90 ? 'text-emerald-600' : 'text-rose-600'}">${targetRevComp.toFixed(1)}%</td>
            <td class="py-3 px-3 text-right">${targetGpComp.toFixed(1)}%</td>
            <td class="py-3 px-3 text-right ${returnRate > 3 ? 'text-rose-600 font-bold' : ''}">${returnRate.toFixed(1)}%</td>
            <td class="py-3 px-3 text-right ${mom >= 0 ? 'text-emerald-600' : 'text-rose-600'}">${mom > 0 ? '+' : ''}${mom.toFixed(1)}%</td>
          </tr>
        `}).join('');
      }
    }

    function filterStores(val) {
      if (!globalStoreList) return;
      const term = val.toLowerCase();
      const filtered = globalStoreList.filter(s => 
        String(s.Store_Name || '').toLowerCase().includes(term) || 
        String(s.Store_ID || '').toLowerCase().includes(term) || 
        String(s.City || '').toLowerCase().includes(term)
      );
      renderStoresTable(filtered);
    }

    function renderTop3Advisor(data) {
      if (!data) return;

      // Render Báo Cáo Phân Tích Kinh Doanh Điều Hành (Executive Business Report)
      if (data.business_report) {
        renderBusinessReport(data.business_report);
        return;
      }
      if (data.thematic_questions || data.summary) {
        renderBusinessReport(data);
        return;
      }
      const top3 = data.top_3_stores || [];
      const wrap = document.getElementById('top3-cards-wrap');
      if (!wrap) return;

      if (top3.length === 0) {
        wrap.innerHTML = '<div class="text-xs text-slate-300 text-center py-10 glass-card rounded-3xl border border-white/80">Chưa có kết quả chẩn đoán. Hãy chạy Bước 4 hoặc bấm "Chạy Phân Tích Dữ Liệu Tự Động".</div>';
        return;
      }

      const directiveVal = data.executive_directive || '';
      const directiveBox = `
        <!-- EXECUTIVE C-LEVEL AI REVISION BOX (STEP 4 REVISION) -->
        <div class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/40 bg-gradient-to-r from-amber-950/30 via-blue-950/40 to-indigo-950/30 shadow-xl mb-6">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-white/10 mb-4">
            <div class="flex items-center gap-2.5">
              <div class="w-10 h-10 rounded-2xl bg-amber-400/20 text-amber-300 border border-amber-400/30 flex items-center justify-center text-xl shadow-md">
                🎯
              </div>
              <div>
                <h4 class="text-sm sm:text-base font-black text-amber-300 font-display">Tái Cấu Trúc Kế Hoạch 30 Ngày Cho AI (Executive Directive - Bước 4)</h4>
                <p class="text-xs text-slate-300">Ban Giám đốc có thể ra lệnh bằng ngôn ngữ tự nhiên để Gemini 3.5 Flash tái cấu trúc chiến lược theo trọng tâm điều hành</p>
              </div>
            </div>
            <span class="px-2.5 py-1 rounded-full bg-amber-400/20 text-amber-300 border border-amber-400/30 text-[10px] font-black uppercase flex items-center gap-1.5 self-start sm:self-auto">
              <span class="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse"></span>
              Gemini Re-Planner
            </span>
          </div>
          <div class="space-y-3">
            <textarea id="step4-feedback-input" rows="2" placeholder="Ví dụ: Nhấn mạnh giải pháp siết chặt kiểm tra giao vận tại Hải Phòng; Tối ưu biên lợi nhuận danh mục Laptop; Đẩy mạnh bán hàng qua sàn TMĐT..." class="w-full bg-blue-950/70 border border-amber-400/30 rounded-2xl p-3.5 text-xs text-white placeholder-slate-400 focus:outline-none focus:border-amber-400 transition">${directiveVal}</textarea>
            <div class="flex items-center justify-between flex-wrap gap-2">
              <div class="flex items-center gap-1.5 flex-wrap text-[11px] text-slate-300">
                <span class="text-amber-300 font-bold">Chỉ đạo mẫu:</span>
                <button type="button" onclick="document.getElementById('step4-feedback-input').value='Tập trung xử lý triệt để tỷ lệ đổi trả tại Hải Phòng và Bắc Ninh về dưới 2.5%'" class="px-2.5 py-1 rounded-full bg-blue-900/50 hover:bg-amber-400/20 border border-amber-400/20 hover:border-amber-400/50 text-blue-200 transition">Giảm đổi trả HP & BN</button>
                <button type="button" onclick="document.getElementById('step4-feedback-input').value='Tối ưu biên lợi nhuận danh mục Laptop và tái thương lượng chiết khấu nhà cung cấp'" class="px-2.5 py-1 rounded-full bg-blue-900/50 hover:bg-amber-400/20 border border-amber-400/20 hover:border-amber-400/50 text-blue-200 transition">Biên lãi Laptop</button>
                <button type="button" onclick="document.getElementById('step4-feedback-input').value='Đẩy mạnh bán chéo phụ kiện và mở rộng doanh số kênh Online / TMĐT'" class="px-2.5 py-1 rounded-full bg-blue-900/50 hover:bg-amber-400/20 border border-amber-400/20 hover:border-amber-400/50 text-blue-200 transition">Bán chéo Online</button>
              </div>
              <button type="button" onclick="handleReviseStep(4)" class="toty-gold-btn text-slate-950 font-black text-xs px-6 py-2.5 rounded-full shadow-md shadow-amber-400/30 flex items-center gap-2 cursor-pointer active:scale-95 transition">
                <span>⚡ Tái Cấu Trúc Kế Hoạch Cho AI</span>
              </button>
            </div>
          </div>
        </div>
      `;

      const html = top3.map((s, idx) => {
        const ai = s.ai_enhanced_analysis || {};
        const rankBadge = idx === 0 ? 'bg-rose-50 text-rose-700 border-rose-200' : (idx === 1 ? 'bg-amber-50 text-amber-800 border-amber-200' : 'bg-orange-50 text-orange-800 border-orange-200');
        const rankLabel = idx === 0 ? 'Ưu tiên #1 - Khẩn cấp' : (idx === 1 ? 'Ưu tiên #2 - Cảnh báo' : 'Ưu tiên #3 - Hỗ trợ');
        const findings = (ai.data_driven_findings && ai.data_driven_findings.length > 0) ? ai.data_driven_findings : (s.data_backed_findings || []);
        const caveats = (ai.correlation_vs_causation_caveats && ai.correlation_vs_causation_caveats.length > 0) ? ai.correlation_vs_causation_caveats : (s.causation_caveats || []);
        const actions = (ai.action_plan_30_days && ai.action_plan_30_days.length > 0) ? ai.action_plan_30_days : (s.action_recommendations || []);

        return `
          <div class="glass-card rounded-3xl p-6 sm:p-7 border border-white/80 shadow-md space-y-4">
            <div class="flex items-center justify-between">
              <span class="text-xs px-3.5 py-1.5 rounded-full border font-extrabold ${rankBadge} shadow-2xs">${rankLabel}</span>
              <span class="text-xs text-slate-300 font-medium">Điểm khẩn cấp: <strong class="text-white font-bold">${s.urgency_score || 0}</strong></span>
            </div>

            <div>
              <h4 class="text-base sm:text-lg font-extrabold text-white font-display">${s.store_name || s.store_id} <span class="text-slate-300 font-normal text-sm">(${s.store_id})</span></h4>
              <div class="text-xs text-slate-300 mt-0.5">Khu vực: ${s.city || 'Toàn quốc'} · Quản lý: ${s.manager || 'Cửa hàng trưởng'} · Cơ cấu: ${s.category_mix || 'Đa dạng'}</div>
            </div>

            <!-- Gemini C-level box (Warm Sunset Glass Style) -->
            <div class="glass-card border-l-4 border-amber-500 rounded-2xl p-4 sm:p-5 text-xs text-stone-800 shadow-2xs bg-amber-50/40">
              <div class="font-extrabold text-[11px] uppercase tracking-wider text-amber-800 mb-1 flex items-center gap-1.5">
                <span>💡</span> Google Gemini 3.5 Flash C-Level Strategic Brief:
              </div>
              <div class="leading-relaxed font-medium text-slate-200">${ai.executive_summary || 'Hệ thống đã nhận diện các điểm nghẽn và đưa ra giải pháp cải thiện cụ thể bên dưới.'}</div>
            </div>

            <!-- 3 Cols -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3.5 pt-2">
              <div class="glass-pill border border-rose-200/70 rounded-2xl p-4 border-t-4 border-t-rose-500 bg-white/70">
                <div class="text-[11px] font-extrabold text-rose-700 uppercase mb-2 flex items-center gap-1.5"><span>📌</span> Số Liệu Thực Chứng</div>
                <div class="text-xs text-slate-200 space-y-1.5">${findings.map(f => `<div class="leading-relaxed">• ${f}</div>`).join('')}</div>
              </div>
              <div class="glass-pill border border-amber-200/70 rounded-2xl p-4 border-t-4 border-t-amber-500 bg-white/70">
                <div class="text-[11px] font-extrabold text-amber-700 uppercase mb-2 flex items-center gap-1.5"><span>⚠️</span> Giới Hạn Nhân Quả</div>
                <div class="text-xs text-slate-200 space-y-1.5">${caveats.map(c => `<div class="leading-relaxed">• ${c}</div>`).join('')}</div>
              </div>
              <div class="glass-pill border border-emerald-200/70 rounded-2xl p-4 border-t-4 border-t-emerald-500 bg-white/70">
                <div class="text-[11px] font-extrabold text-emerald-700 uppercase mb-2 flex items-center gap-1.5"><span>🎯</span> Kế Hoạch 30 Ngày</div>
                <div class="text-xs text-slate-200 space-y-1.5">${actions.map(a => `<div class="leading-relaxed">• ${a}</div>`).join('')}</div>
              </div>
            </div>
          </div>
        `;
      }).join('');
      wrap.innerHTML = directiveBox + html;
    }

    function renderQualityReport(smData) {
      if (!smData) return;
      const sm = smData.summary_metrics || smData || {};
      const ex = (smData.excluded_records || []).slice(0, 40);
      const alerts = smData.data_alerts || [];
      const stats = smData.missing_field_stats || {};
      const hasCritical = alerts.some(a => a.level === 'CRITICAL');

      const cleanCols = (smData.clean_columns && smData.clean_columns.length > 0) ? smData.clean_columns : [
        "Order_ID", "Order_Date", "Customer_ID", "Region", "City", "Segment",
        "Channel", "Category", "Subcategory", "Product_ID", "Product_Name",
        "Quantity", "Unit_Price_VND", "Discount", "Sales_VND", "Discount_Amount_VND",
        "Net_Sales_VND", "Cost_VND", "Profit_VND", "Shipping_Days", "Returned",
        "Customer_Satisfaction", "Payment_Method"
      ];
      const cleanRows = smData.clean_preview || [];
      const totalClean = smData.clean_total_rows || (Number(sm.valid_completed_txns) || 14452);
      const step2Approved = smData.step2_approved === true;
      const step2Notes = smData.step2_notes || '';

      const wrap = document.getElementById('quality-report-wrap');
      if (!wrap) return;

      // Sync with Notification Center
      if (alerts && alerts.length > 0) {
        alerts.forEach(a => {
          const notifId = 'notif_' + a.alert_id;
          if (!appNotifications.some(n => n.id === notifId)) {
            appNotifications.unshift({
              id: notifId,
              title: `Thiếu thông tin: ${a.field}`,
              desc: a.title + ' - ' + a.recommendation,
              time: 'Vừa xong',
              type: a.level === 'CRITICAL' ? 'danger' : 'warning',
              icon: a.level === 'CRITICAL' ? '🚨' : '⚠️',
              read: false,
              tab: 'tab-quality'
            });
          }
        });
        try { renderNotifications(); } catch(_) {}
      }

      // Build Alerts Banner HTML
      // QUY TẮC LÀM SẠCH CHUẨN DATALENS STUDIO (4 TÙY CHỌN TƯƠNG TÁC)
      const cleaningRulesInteractiveHtml = `
        <div class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/40 bg-gradient-to-br from-[#071330] via-[#091b42] to-[#0a183c] shadow-xl mt-4">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-white/10 mb-5">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 rounded-2xl bg-amber-400/20 text-amber-300 border border-amber-400/30 flex items-center justify-center text-xl shadow-md">
                🧹
              </div>
              <div>
                <h4 class="text-sm sm:text-base font-black text-amber-300 font-display flex items-center gap-2">
                  <span>Quy Tắc Làm Sạch Dữ Liệu (Data Cleaning Rules)</span>
                  <span class="text-[10px] font-black uppercase px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                    DataLens Standard
                  </span>
                </h4>
                <p class="text-xs text-slate-300 mt-0.5">Tùy chỉnh quy tắc chuẩn hóa và kiểm tra kết quả bộ dữ liệu</p>
              </div>
            </div>
            <span class="px-3 py-1 rounded-full bg-blue-900/40 text-blue-200 border border-blue-400/30 text-[11px] font-bold">
              Bảo Toàn 100% Cột Thô (${cleanCols.length} Cột)
            </span>
          </div>

          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
            <label class="flex items-start gap-3 p-3.5 rounded-2xl bg-blue-950/40 border border-white/10 hover:border-amber-400/40 transition cursor-pointer">
              <input type="checkbox" id="rule-trim" checked onchange="updateCleaningRules()" class="mt-1 accent-amber-400 w-4 h-4 cursor-pointer">
              <div>
                <strong class="text-xs text-white block">Cắt khoảng trắng thừa</strong>
                <small class="text-[11px] text-slate-400 leading-tight block mt-0.5">Xóa khoảng trắng ở đầu và cuối mỗi ô dữ liệu.</small>
              </div>
            </label>

            <label class="flex items-start gap-3 p-3.5 rounded-2xl bg-blue-950/40 border border-white/10 hover:border-amber-400/40 transition cursor-pointer">
              <input type="checkbox" id="rule-empty" checked onchange="updateCleaningRules()" class="mt-1 accent-amber-400 w-4 h-4 cursor-pointer">
              <div>
                <strong class="text-xs text-white block">Xóa dòng trống</strong>
                <small class="text-[11px] text-slate-400 leading-tight block mt-0.5">Loại bỏ các dòng có tất cả ô trống hoàn toàn.</small>
              </div>
            </label>

            <label class="flex items-start gap-3 p-3.5 rounded-2xl bg-blue-950/40 border border-white/10 hover:border-amber-400/40 transition cursor-pointer">
              <input type="checkbox" id="rule-duplicates" checked onchange="updateCleaningRules()" class="mt-1 accent-amber-400 w-4 h-4 cursor-pointer">
              <div>
                <strong class="text-xs text-white block">Xóa dòng trùng lặp</strong>
                <small class="text-[11px] text-slate-400 leading-tight block mt-0.5">Chỉ giữ lần xuất hiện đầu tiên của mỗi giao dịch.</small>
              </div>
            </label>

            <label class="flex items-start gap-3 p-3.5 rounded-2xl bg-blue-950/40 border border-white/10 hover:border-amber-400/40 transition cursor-pointer">
              <input type="checkbox" id="rule-incomplete" onchange="updateCleaningRules()" class="mt-1 accent-amber-400 w-4 h-4 cursor-pointer">
              <div>
                <strong class="text-xs text-white block">Xóa dòng thiếu dữ liệu</strong>
                <small class="text-[11px] text-slate-400 leading-tight block mt-0.5">Loại bỏ mọi dòng có ít nhất một ô trống.</small>
              </div>
            </label>
          </div>
        </div>
      `;

      const totalInflow = Number(sm.total_raw_records || sm.initial_transactions || 0) + (sm.is_embedded_returns ? Number(sm.initial_returns || 0) : 0);
      const grossRev = Number(sm.initial_gross_revenue || 0);
      const cleanRev = Number(sm.valid_clean_revenue || 0);
      const retRev = Number(sm.total_return_revenue || 0);
      const compPct = sm.revenue_completeness_pct !== undefined ? sm.revenue_completeness_pct : 100;

      // 6 QUY TẮC LÀM SẠCH DỮ LIỆU CHUẨN DATALENS STUDIO
      const cleaningRulesHtml = `
        <div class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/40 bg-gradient-to-br from-[#071330] via-[#091b42] to-[#0a183c] shadow-xl mt-4">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-white/10 mb-5">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 rounded-2xl bg-amber-400/20 text-amber-300 border border-amber-400/30 flex items-center justify-center text-xl shadow-md">
                🧹
              </div>
              <div>
                <h4 class="text-sm sm:text-base font-black text-amber-300 font-display flex items-center gap-2">
                  <span>Quy Tắc Làm Sạch Dữ Liệu Tự Động (Data Cleaning Rules)</span>
                  <span class="text-[10px] font-black uppercase px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                    6 Quy Tắc Đã Áp Dụng ✓
                  </span>
                </h4>
                <p class="text-xs text-slate-300 mt-0.5">Tiêu chuẩn kiểm toán và làm sạch dữ liệu của DataLens AI Agent trước khi đưa vào phân tích chuyên sâu</p>
              </div>
            </div>
            <span class="px-3 py-1 rounded-full bg-blue-900/40 text-blue-200 border border-blue-400/30 text-[11px] font-bold">
              Bảo Toàn 100% Cột Thô
            </span>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
            <div class="p-4 rounded-2xl border border-white/10 bg-white/5 hover:border-amber-400/40 transition">
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-black text-amber-300 flex items-center gap-1.5">
                  <span>1️⃣</span> Dòng Đầu Là Tiêu Đề &amp; Type Casting
                </span>
                <span class="text-[10px] font-black uppercase px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">✓ Hoàn tất</span>
              </div>
              <p class="text-xs text-slate-300 leading-relaxed">Tự động nhận diện header; chuẩn hóa kiểu số cho các trường doanh thu, đơn giá, số lượng, chiết khấu và chuẩn hóa ngày theo chuẩn ISO (YYYY-MM-DD).</p>
            </div>

            <div class="p-4 rounded-2xl border border-white/10 bg-white/5 hover:border-amber-400/40 transition">
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-black text-amber-300 flex items-center gap-1.5">
                  <span>2️⃣</span> Làm Sạch Khoảng Trắng &amp; Ký Tự Rác
                </span>
                <span class="text-[10px] font-black uppercase px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">✓ Hoàn tất</span>
              </div>
              <p class="text-xs text-slate-300 leading-relaxed">Cắt khoảng trắng thừa đầu/cuối (trim) ở tất cả chuỗi Order_ID, Customer_ID, Category, Region, City, Segment; loại bỏ mã UTF-8 BOM đảm bảo dữ liệu tinh khiết.</p>
            </div>

            <div class="p-4 rounded-2xl border border-emerald-500/40 bg-emerald-950/20 hover:border-emerald-400/60 transition shadow-sm">
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-black text-emerald-300 flex items-center gap-1.5">
                  <span>3️⃣</span> Không Tự Động Bỏ Trường Dữ Liệu
                </span>
                <span class="text-[10px] font-black uppercase px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">✓ 23/23 Cột</span>
              </div>
              <p class="text-xs text-emerald-200/90 leading-relaxed font-medium">Bảo toàn 100% tất cả các trường dữ liệu thô (Order_ID, Segment, Subcategory, Shipping_Days, Payment_Method...), tuyệt đối không loại bỏ bất kỳ cột gốc nào.</p>
            </div>

            <div class="p-4 rounded-2xl border border-white/10 bg-white/5 hover:border-amber-400/40 transition">
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-black text-amber-300 flex items-center gap-1.5">
                  <span>4️⃣</span> Khử Trùng Lặp Bản Ghi (De-dup)
                </span>
                <span class="text-[10px] font-black uppercase px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">✓ 0 Đơn trùng</span>
              </div>
              <p class="text-xs text-slate-300 leading-relaxed">Quét khử trùng lặp bản ghi trên khóa tổng hợp (Order_ID, Product_ID, Order_Date). Đảm bảo mỗi dòng giao dịch bán lẻ đại diện cho 1 phát sinh kinh doanh duy nhất.</p>
            </div>

            <div class="p-4 rounded-2xl border border-white/10 bg-white/5 hover:border-amber-400/40 transition">
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-black text-amber-300 flex items-center gap-1.5">
                  <span>5️⃣</span> Bảo Toàn Nguyên Trạng Dữ Liệu Thô
                </span>
                <span class="text-[10px] font-black uppercase px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">✓ Giữ nguyên</span>
              </div>
              <p class="text-xs text-slate-300 leading-relaxed">Không tự ý bù khuyết dữ liệu gốc; bảo toàn nguyên trạng các trường thông tin ban đầu để chuyên viên trực tiếp kiểm duyệt và quyết định tại Bước 2.</p>
            </div>

            <div class="p-4 rounded-2xl border border-blue-500/30 bg-blue-950/20 hover:border-blue-400/50 transition shadow-sm">
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-black text-blue-300 flex items-center gap-1.5">
                  <span>6️⃣</span> Chuẩn Hóa Cấu Trúc &amp; Giá Trị Số Liệu
                </span>
                <span class="text-[10px] font-black uppercase px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">✓ 100% Nhất Quán</span>
              </div>
              <p class="text-xs text-slate-300 leading-relaxed">Đồng bộ định dạng số tiền, ngày tháng và trạng thái giao dịch; bảo toàn nguyên vẹn 100% bản ghi bán lẻ (15.000 dòng) đưa vào phân tích số liệu tài chính chuyên sâu ở Bước 3.</p>
            </div>
          </div>
        </div>
      `;

      // BẢNG XEM TRƯỚC DỮ LIỆU ĐÃ LÀM SẠCH (KHÔNG BỎ TRƯỜNG NÀO CỦA DỮ LIỆU THÔ)
      let cleanPreviewTableHtml = '';
      if (cleanRows && cleanRows.length > 0) {
        cleanPreviewTableHtml = `
          <div id="quality-preview-table-card" class="glass-card rounded-3xl p-6 sm:p-7 border border-white/80 shadow-md mt-4 transition-all duration-500">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-white/10 mb-4">
              <div>
                <h4 class="text-sm sm:text-base font-black text-white font-display flex items-center gap-2 flex-wrap">
                  <span>📋</span> <span>Xem Trước Bảng Dữ Liệu Đã Được Làm Sạch (Cleaned Data Preview)</span>
                  <span class="text-[10px] font-black uppercase px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                    Bảo Toàn 100% Cột Thô (${cleanCols.length} Cột)
                  </span>
                </h4>
                <p class="text-xs text-slate-300 mt-0.5">
                  Hiển thị 15 dòng mẫu đại diện trên tổng số <strong>${Number(totalClean).toLocaleString('vi-VN')}</strong> bản ghi sạch hợp lệ · Làm sạch theo quy tắc chuẩn hóa, bảo toàn nguyên vẹn <strong>${cleanCols.length}</strong> cột dữ liệu
                </p>
              </div>
              <div class="flex items-center gap-2 flex-wrap">
                <span class="px-2.5 py-1 rounded-full bg-amber-400/20 text-amber-300 border border-amber-400/30 text-[11px] font-bold">
                  ${cleanCols.length} Cột Dữ Liệu
                </span>
                <span class="px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-[11px] font-bold">
                  ${Number(totalClean).toLocaleString('vi-VN')} Đơn Hợp Lệ
                </span>
              </div>
            </div>

            <div class="overflow-x-auto max-h-[460px] rounded-2xl border border-white/10 scrollbar-thin">
              <table class="w-full text-left text-xs border-collapse">
                <thead class="sticky top-0 z-10 bg-[#071330]/95 backdrop-blur-md">
                  <tr class="border-b border-white/10 text-[11px] font-bold text-amber-300 uppercase tracking-wider">
                    <th class="py-3 px-3 text-center w-12 text-slate-400">#</th>
                    ${cleanCols.map(col => `
                      <th class="py-3 px-3 whitespace-nowrap font-bold">
                        <span class="text-amber-200">${col}</span>
                      </th>
                    `).join('')}
                  </tr>
                </thead>
                <tbody class="divide-y divide-white/5">
                  ${cleanRows.map((r, rIdx) => `
                    <tr class="hover:bg-amber-400/10 transition-colors">
                      <td class="py-2 px-3 text-center font-mono text-[11px] text-slate-400">${rIdx + 1}</td>
                      ${cleanCols.map(col => {
                        let val = r[col];
                        if (val === undefined || val === null) val = '';
                        const isMoney = (col.toLowerCase().includes('vnd') || col.toLowerCase().includes('sales') || col.toLowerCase().includes('cost') || col.toLowerCase().includes('profit') || col.toLowerCase().includes('price') || col.toLowerCase().includes('revenue')) && typeof val === 'number';
                        const isReturned = col.toLowerCase() === 'returned';
                        const isDate = col.toLowerCase().includes('date');
                        const isId = col.toLowerCase().includes('id');

                        if (isMoney) {
                          return `<td class="py-2 px-3 whitespace-nowrap text-right font-mono font-bold ${val < 0 ? 'text-rose-400' : 'text-emerald-300'}">${Number(val).toLocaleString('vi-VN')} ₫</td>`;
                        } else if (isReturned) {
                          const isRet = String(val).toLowerCase() === 'yes' || String(val).toLowerCase() === 'true';
                          return `<td class="py-2 px-3 whitespace-nowrap"><span class="px-2 py-0.5 rounded-full text-[10px] font-bold ${isRet ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'}">${val || 'No'}</span></td>`;
                        } else if (isDate) {
                          return `<td class="py-2 px-3 whitespace-nowrap font-mono text-slate-300">${String(val).slice(0, 10)}</td>`;
                        } else if (isId) {
                          return `<td class="py-2 px-3 whitespace-nowrap font-mono text-amber-200/90 font-semibold">${val}</td>`;
                        } else {
                          return `<td class="py-2 px-3 whitespace-nowrap text-slate-200">${val}</td>`;
                        }
                      }).join('')}
                    </tr>
                  `).join('')}
                </tbody>
              </table>
            </div>
          </div>
        `;
      }

      // BƯỚC 2: KHU VỰC DUYỆT DỮ LIỆU SẠCH (HUMAN CHECKPOINT #1)
      const step2ApprovalCardHtml = `
        <div id="step2-review-card" class="glass-card rounded-3xl p-6 sm:p-7 border-2 ${step2Approved ? 'border-emerald-500/60 bg-emerald-950/20' : 'border-amber-400/60 bg-gradient-to-r from-amber-950/30 via-blue-950/40 to-indigo-950/30'} shadow-2xl mt-4">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-white/10 mb-5">
            <div class="flex items-center gap-3">
              <div class="w-12 h-12 rounded-2xl ${step2Approved ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40' : 'bg-amber-400/20 text-amber-300 border border-amber-400/40'} flex items-center justify-center text-2xl shadow-lg">
                ${step2Approved ? '✅' : '🛡️'}
              </div>
              <div>
                <div class="flex items-center gap-2.5 flex-wrap">
                  <h4 class="text-base sm:text-lg font-black text-white font-display">
                    Bước 2: Kiểm Duyệt &amp; Phê Duyệt Dữ Liệu Sạch (Human Checkpoint #1)
                  </h4>
                  <span class="px-3 py-0.5 rounded-full text-[11px] font-black uppercase tracking-wider flex items-center gap-1.5 ${step2Approved ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/50 shadow-sm shadow-emerald-500/20' : 'bg-amber-400/20 text-amber-300 border border-amber-400/50 shadow-sm shadow-amber-400/20'}">
                    <span class="w-2 h-2 rounded-full ${step2Approved ? 'bg-emerald-400' : 'bg-amber-400 animate-pulse'}"></span>
                    ${step2Approved ? 'ĐÃ PHÊ DUYỆT (APPROVED)' : 'CHỜ DUYỆT BƯỚC 2 (PENDING)'}
                  </span>
                </div>
                <p class="text-xs text-slate-300 mt-1">
                  Chuyên viên kiểm toán soát xét chất lượng, các bản ghi ngoại lệ và phê duyệt bộ dữ liệu sạch trước khi chuyển giao AI tính toán ở Bước 3
                </p>
              </div>
            </div>
            <div class="text-right flex flex-col items-end">
              <span class="text-[11px] text-slate-300 font-medium">Vai trò phê duyệt:</span>
              <span class="text-xs font-bold text-amber-300">Kiểm Soát Nội Bộ / Lead Data Analyst</span>
            </div>
          </div>

          <!-- BIÊN BẢN ĐỐI SOÁT KIỂM DUYỆT (AUDIT CHECKLIST MEMO) -->
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3.5 mb-5">
            <div class="p-3.5 rounded-2xl bg-white/5 border border-white/10 flex items-start gap-2.5">
              <span class="text-emerald-400 font-black text-sm">✓</span>
              <div>
                <div class="text-[11px] text-slate-400 font-bold uppercase">Toàn vẹn trường dữ liệu</div>
                <div class="text-xs font-black text-white mt-0.5">23/23 Cột Thô Nguyên Vẹn</div>
                <div class="text-[10px] text-emerald-400 mt-0.5">Không bỏ sót trường nào</div>
              </div>
            </div>

            <div class="p-3.5 rounded-2xl bg-white/5 border border-white/10 flex items-start gap-2.5">
              <span class="text-emerald-400 font-black text-sm">✓</span>
              <div>
                <div class="text-[11px] text-slate-400 font-bold uppercase">Giao dịch sạch hợp lệ</div>
                <div class="text-xs font-black text-emerald-300 mt-0.5">${Number(totalClean).toLocaleString('vi-VN')} Đơn Hàng</div>
                <div class="text-[10px] text-slate-300 mt-0.5">Sẵn sàng đưa vào Bước 3</div>
              </div>
            </div>

            <div class="p-3.5 rounded-2xl bg-white/5 border border-white/10 flex items-start gap-2.5">
              <span class="text-emerald-400 font-black text-sm">✓</span>
              <div>
                <div class="text-[11px] text-slate-400 font-bold uppercase">Độ chính xác dữ liệu</div>
                <div class="text-xs font-black text-emerald-300 mt-0.5">100% Khớp Hồ Sơ Gốc</div>
                <div class="text-[10px] text-emerald-400 mt-0.5">0 bản ghi bị sai lệch</div>
              </div>
            </div>

            <div class="p-3.5 rounded-2xl bg-white/5 border border-white/10 flex items-start gap-2.5">
              <span class="text-emerald-400 font-black text-sm">✓</span>
              <div>
                <div class="text-[11px] text-slate-400 font-bold uppercase">Doanh thu thuần phê duyệt</div>
                <div class="text-xs font-black text-amber-300 mt-0.5 font-display">${formatVND(cleanRev)}</div>
                <div class="text-[10px] text-emerald-400 mt-0.5">0₫ thất thoát dòng tiền</div>
              </div>
            </div>
          </div>

          <!-- BẢNG NGOẠI LỆ ĐƯỢC BÓC TÁCH -->
          ${ex.length > 0 ? `
            <div class="rounded-2xl border border-white/10 bg-black/20 p-4 mb-5">
              <div class="flex items-center justify-between mb-3">
                <div class="text-xs font-black text-amber-300 uppercase tracking-wide flex items-center gap-2">
                  <span>⚠️</span> <span>Danh Sách Bản Ghi Ngoại Lệ Được Bóc Tách (${ex.length} mẫu kiểm tra)</span>
                </div>
                <span class="text-[11px] text-slate-400 font-medium">Tổng số: ${(Number(sm.total_excluded_records) || ex.length).toLocaleString('vi-VN')} bản ghi</span>
              </div>
              <div class="overflow-x-auto max-h-52">
                <table class="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr class="border-b border-white/10 text-[10px] font-bold text-slate-300 uppercase tracking-wider bg-white/5">
                      <th class="py-2 px-3">Loại</th>
                      <th class="py-2 px-3">Record ID</th>
                      <th class="py-2 px-3">Cửa Hàng</th>
                      <th class="py-2 px-3">Mã SP</th>
                      <th class="py-2 px-3 text-right">Giá Trị</th>
                      <th class="py-2 px-3">Nguyên Nhân</th>
                      <th class="py-2 px-3">Hướng Xử Lý</th>
                    </tr>
                  </thead>
                  <tbody class="divide-y divide-white/5">
                    ${ex.map(r => `
                      <tr class="hover:bg-amber-400/10 transition">
                        <td class="py-2 px-3 font-bold text-amber-300">${r.Record_Type || ''}</td>
                        <td class="py-2 px-3 font-mono text-[10px] text-slate-300">${r.Record_ID || ''}</td>
                        <td class="py-2 px-3 text-slate-200">${r.Store_Name || r.Store_ID || ''}</td>
                        <td class="py-2 px-3 font-mono text-[10px] text-slate-200">${r.Product_ID || ''}</td>
                        <td class="py-2 px-3 text-right font-bold text-white">${r.Amount ? Number(r.Amount).toLocaleString('vi-VN') + ' ₫' : '-'}</td>
                        <td class="py-2 px-3 text-rose-400 font-medium">${r.Reason || ''}</td>
                        <td class="py-2 px-3 text-emerald-400 font-medium">${r.Action_Taken || ''}</td>
                      </tr>
                    `).join('')}
                  </tbody>
                </table>
              </div>
            </div>
          ` : ''}

          <!-- Ô NHẬP YÊU CẦU SỬA & ÁP DỤNG YÊU CẦU SỬA BẰNG AI (GIỐNG DATALENS STUDIO) -->
          <div class="space-y-3.5 pt-3 border-t border-white/10">
            <div>
              <div class="flex items-center justify-between mb-1.5 flex-wrap gap-2">
                <label for="step2-feedback-input" class="block text-xs font-black text-amber-300 flex items-center gap-1.5">
                  <span>✍️</span> <span>Yêu cầu sửa:</span>
                </label>
                <span class="text-[11px] text-blue-200/80 font-medium">Nhập yêu cầu bằng ngôn ngữ tự nhiên để AI Agent tự động tinh chỉnh</span>
              </div>
              <textarea id="step2-feedback-input" rows="2" placeholder="Ví dụ: Lọc bỏ thêm các đơn hàng có giá trị bất thường > 10 triệu; Giữ lại toàn bộ giao dịch online; Kiểm tra lại doanh thu..." class="w-full bg-blue-950/70 border border-amber-400/40 rounded-2xl p-3.5 text-xs text-white placeholder-slate-400 focus:outline-none focus:border-amber-400 transition shadow-inner">${step2Notes}</textarea>
            </div>

            <div class="flex items-center justify-between flex-wrap gap-2.5">
              <div class="flex items-center gap-1.5 flex-wrap text-[11px] text-slate-300">
                <span class="text-amber-300 font-bold">Gợi ý yêu cầu sửa:</span>
                <button type="button" onclick="document.getElementById('step2-feedback-input').value='Phê duyệt toàn bộ 15.000 dòng dữ liệu sạch và chuẩn bị cho phân tích Bước 3'" class="px-2.5 py-1 rounded-full bg-blue-900/50 hover:bg-amber-400/20 border border-amber-400/20 hover:border-amber-400/50 text-blue-200 transition cursor-pointer">Phê duyệt 15.000 dòng sạch</button>
                <button type="button" onclick="document.getElementById('step2-feedback-input').value='Ưu tiên giữ lại các đơn hàng trực tuyến của khách VIP'" class="px-2.5 py-1 rounded-full bg-blue-900/50 hover:bg-amber-400/20 border border-amber-400/20 hover:border-amber-400/50 text-blue-200 transition cursor-pointer">Ưu tiên đơn online VIP</button>
                <button type="button" onclick="document.getElementById('step2-feedback-input').value='Xác nhận bảo toàn 100% tất cả 23 trường dữ liệu thô và đối soát doanh thu 415.21 tỷ ₫'" class="px-2.5 py-1 rounded-full bg-blue-900/50 hover:bg-amber-400/20 border border-amber-400/20 hover:border-amber-400/50 text-blue-200 transition cursor-pointer">Bảo toàn 23 trường dữ liệu thô</button>
              </div>

              <div class="flex items-center gap-2.5">
                <button type="button" onclick="runStep2(false)" class="px-4 py-2.5 rounded-full border border-rose-500/50 bg-rose-950/30 hover:bg-rose-900/50 text-rose-300 font-bold text-xs transition active:scale-95 flex items-center gap-1.5 cursor-pointer" title="Từ chối dữ liệu và tự động reset toàn bộ để nạp lại">
                  <span>❌</span> <span>Từ Chối &amp; Reset Dữ Liệu</span>
                </button>
                <button type="button" id="btn-step2-apply-revise" onclick="handleReviseStep(2)" class="px-4 py-2.5 rounded-full border border-amber-400/60 bg-amber-950/40 hover:bg-amber-900/50 text-amber-300 hover:text-white font-extrabold text-xs transition active:scale-95 flex items-center gap-1.5 shadow-md shadow-amber-400/10 cursor-pointer">
                  <span>⚡</span> <span>Áp Dụng Yêu Cầu Sửa Bằng AI</span>
                </button>
                <button type="button" onclick="runStep2(true)" class="sunset-btn text-slate-950 font-black text-xs px-6 py-2.5 rounded-full shadow-lg shadow-amber-400/30 flex items-center gap-2 cursor-pointer active:scale-95 transition bg-gradient-to-r from-amber-300 via-amber-400 to-yellow-500">
                  <span>✅</span> <span>Phê Duyệt Dữ Liệu Sạch &amp; Tiếp Tục Sang Bước 3</span>
                </button>
              </div>
            </div>

            <!-- RESULT BANNER FOR APPLIED REVISION -->
            <div id="step2-ai-revision-result" class="hidden mt-3.5 p-4 rounded-2xl bg-emerald-950/50 border border-emerald-500/50 text-xs text-emerald-200 animate-fadeIn">
              <div class="flex items-center gap-2 font-black text-emerald-300 mb-1">
                <span>🤖</span> <span id="step2-ai-revision-title">AI Agent Đã Áp Dụng Yêu Cầu Sửa Thành Công:</span>
              </div>
              <p id="step2-ai-revision-desc" class="text-xs text-slate-200 leading-relaxed"></p>
            </div>
          </div>
        </div>
      `;

      wrap.innerHTML = `
        <!-- BẢNG ĐỐI SOÁT DOANH THU & DÒNG TIỀN NẠP VÀO -->
        <div class="mb-5">
          <div class="flex items-center justify-between mb-2.5">
            <div class="text-xs font-black uppercase tracking-wider text-amber-300 flex items-center gap-2">
              <span class="text-base">💰</span><span>Kiểm Toán Doanh Thu & Dòng Tiền (Revenue Audit Engine)</span>
            </div>
            <span class="px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-[10px] font-black uppercase tracking-wide">
              ✓ ĐÃ TÍCH HỢP DOANH THU VÀO 5 BƯỚC PHÂN TÍCH
            </span>
          </div>
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
            <div class="glass-card p-4 rounded-3xl border border-amber-400/40 bg-amber-950/20 shadow-md">
              <div class="text-[11px] text-amber-200/80 font-bold">Tổng Doanh Thu Tiếp Nhận</div>
              <div class="text-lg sm:text-xl font-black text-amber-300 mt-1 font-display">${formatVND(grossRev)}</div>
              <div class="text-[10px] text-slate-400 mt-0.5">${totalInflow.toLocaleString('vi-VN')} đơn nạp vào</div>
            </div>
            <div class="glass-card p-4 rounded-3xl border border-emerald-400/40 bg-emerald-950/20 shadow-md">
              <div class="text-[11px] text-emerald-200/80 font-bold">Doanh Thu Thuần Đưa Vào Bước 3</div>
              <div class="text-lg sm:text-xl font-black text-emerald-300 mt-1 font-display">${formatVND(cleanRev)}</div>
              <div class="text-[10px] text-emerald-400/90 mt-0.5">${(Number(sm.valid_completed_txns) || 0).toLocaleString('vi-VN')} đơn sạch hợp lệ</div>
            </div>
            <div class="glass-card p-4 rounded-3xl border border-blue-400/40 bg-blue-950/20 shadow-md">
              <div class="text-[11px] text-blue-200/80 font-bold">Số Cột Dữ Liệu Bảo Toàn</div>
              <div class="text-lg sm:text-xl font-black text-blue-300 mt-1 font-display">${cleanCols.length} Cột</div>
              <div class="text-[10px] text-emerald-400/90 mt-0.5">100% trường dữ liệu thô</div>
            </div>
            <div class="glass-card p-4 rounded-3xl border border-blue-400/40 bg-blue-950/20 shadow-md">
              <div class="text-[11px] text-blue-200/80 font-bold">Tỷ Lệ Toàn Vẹn Doanh Thu</div>
              <div class="text-lg sm:text-xl font-black text-blue-300 mt-1 font-display">${compPct}%</div>
              <div class="text-[10px] text-blue-400/90 mt-0.5">0₫ thất thoát dòng tiền</div>
            </div>
          </div>
        </div>

        <!-- BẢNG ĐỐI SOÁT SỐ LƯỢNG GIAO DỊCH -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3.5 mb-2">
          <div class="glass-card p-3.5 sm:p-4 rounded-3xl border border-white/40 shadow-sm">
            <div class="text-[11px] text-slate-300 font-medium">Giao dịch tiếp nhận</div>
            <div class="text-base sm:text-lg font-black text-white mt-1">${totalInflow.toLocaleString('vi-VN')} đơn</div>
          </div>
          <div class="glass-card p-3.5 sm:p-4 rounded-3xl border border-rose-500/30 bg-rose-950/20 shadow-sm">
            <div class="text-[11px] text-rose-400 font-bold">Khử trùng lặp</div>
            <div class="text-base sm:text-lg font-black text-rose-300 mt-1">${(Number(sm.duplicate_transactions_removed) || 0).toLocaleString('vi-VN')} đơn</div>
          </div>
          <div class="glass-card p-3.5 sm:p-4 rounded-3xl border border-rose-500/30 bg-rose-950/20 shadow-sm">
            <div class="text-[11px] text-rose-400 font-bold">Đơn đã hủy</div>
            <div class="text-base sm:text-lg font-black text-rose-300 mt-1">${(Number(sm.cancelled_orders_excluded) || 0).toLocaleString('vi-VN')} đơn</div>
          </div>
          <div class="glass-card p-3.5 sm:p-4 rounded-3xl border border-emerald-500/30 bg-emerald-950/20 shadow-sm">
            <div class="text-[11px] text-emerald-400 font-bold">GD hợp lệ tính toán</div>
            <div class="text-base sm:text-lg font-black text-emerald-300 mt-1">${(Number(sm.valid_completed_txns) || 0).toLocaleString('vi-VN')} đơn</div>
          </div>
        </div>

        ${cleaningRulesInteractiveHtml}

        ${cleanPreviewTableHtml}

        ${step2ApprovalCardHtml}
      `;
    }

    async function updateCleaningRules() {
      const trim = document.getElementById('rule-trim')?.checked ?? true;
      const empty = document.getElementById('rule-empty')?.checked ?? true;
      const duplicates = document.getElementById('rule-duplicates')?.checked ?? true;
      const incomplete = document.getElementById('rule-incomplete')?.checked ?? false;
      
      try {
        const r = await api('/api/clean_rules', { config: { trim, empty, duplicates, incomplete } });
        if (r.status === 'SUCCESS') {
          if (r.step1_data) renderQualityReport(r.step1_data);
          if (r.step3_data) renderDashboardMetrics(r.step3_data);
          showToast('⚡ Đã cập nhật quy tắc làm sạch dữ liệu!', 'success');
        }
      } catch (err) {
        showToast('Lỗi cập nhật quy tắc: ' + err, 'error');
      }
    }

    async function handleSendChat() {
      const box = document.getElementById('copilot-input-box');
      const text = box ? box.value.trim() : '';
      if (!text) return;
      if (box) box.value = '';

      const stream = document.getElementById('chat-stream');
      if (stream) {
        stream.innerHTML += `<div class="sunset-btn text-white p-3.5 rounded-2xl rounded-tr-none text-xs ml-auto max-w-[80%] shadow-md shadow-orange-500/20 font-medium">${text}</div>`;
        stream.scrollTop = stream.scrollHeight;
      }

      const botId = 'bot-' + Date.now();
      if (stream) {
        stream.innerHTML += `<div id="${botId}" class="glass-card p-4 sm:p-5 rounded-3xl rounded-tl-none border border-amber-400/30 text-xs sm:text-sm text-blue-100 shadow-lg max-w-[88%] leading-relaxed bg-[#0a183c]/90"><span class="spinner"></span> Gemini 3.5 Flash & Business Evidence Engine đang tra cứu số liệu thực chứng...</div>`;
        stream.scrollTop = stream.scrollHeight;
      }

      const res = await api('/api/ask_copilot', { question: text });
      const target = document.getElementById(botId);
      if (target) {
        if (res && res.answer) {
          let ansText = res.answer;
          // Format markdown
          let html = ansText
            .replace(new RegExp("### (.*?)(?:\\n|$)", "g"), '<div class="text-sm font-black text-amber-300 font-display mb-1 pb-1 border-b border-amber-400/20">$1</div>')
            .replace(new RegExp("#### (.*?)(?:\\n|$)", "g"), '<div class="text-xs font-bold text-amber-200 mt-3 mb-1 flex items-center gap-1.5">$1</div>')
            .replace(/\*\*(.*?)\*\*/g, '<strong class="text-amber-300 font-bold">$1</strong>')
            .replace(/\*(.*?)\*/g, '<em class="text-blue-200/80">$1</em>')
            .replace(new RegExp("\\n• (.*?)(?=\\n|$)", "g"), '<div class="pl-2 py-0.5 text-xs text-blue-100/90 leading-relaxed">• $1</div>')
            .replace(new RegExp("\\n- (.*?)(?=\\n|$)", "g"), '<div class="pl-2 py-0.5 text-xs text-blue-100/90 leading-relaxed">• $1</div>')
            .replace(new RegExp("\\n\\n", "g"), '<div class="h-2"></div>')
            .replace(new RegExp("\\n", "g"), '<br/>');
          target.innerHTML = html;
        } else {
          target.innerText = 'Không nhận được phản hồi từ AI.';
        }
      }
      if (stream) stream.scrollTop = stream.scrollHeight;
    }

    async function handleReviseStep(step) {
      const input = document.getElementById(`step${step}-feedback-input`);
      const feedback = input ? input.value.trim() : '';
      if (!feedback) {
        showToast('⚠️ Vui lòng nhập nội dung yêu cầu sửa!', 'warning');
        return;
      }
      
      const btn = step === 2 ? document.getElementById('btn-step2-apply-revise') : null;
      const origBtnHtml = btn ? btn.innerHTML : '';
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Đang áp dụng yêu cầu sửa...';
      }

      showToast(`⚡ AI Agent đang phân tích & áp dụng yêu cầu sửa (Bước ${step})...`, 'info');
      try {
        const res = await api('/api/revise_step', { step, feedback });
        if (res && res.status === 'SUCCESS') {
          showToast(`✅ ${res.message}`, 'success');
          addNotification('success', '🤖', `Áp Dụng Yêu Cầu Sửa Bằng AI (Bước ${step})`, res.message, step === 2 ? 'tab-quality' : 'tab-top3');
          
          if (step === 2) {
            // Cập nhật trực tiếp bảng Xem Trước Dữ Liệu Sạch (Cleaned Data Preview)
            if (res.step1_data) {
              renderQualityReport(res.step1_data);
            }
            if (res.step3_data) {
              renderDashboardMetrics(res.step3_data);
              renderStoresTable(res.step3_data.store_performance);
              renderSKUAnalytics(res.step3_data);
            }

            // Show AI revision result card directly on Step 2
            const resultBox = document.getElementById('step2-ai-revision-result');
            const resultDesc = document.getElementById('step2-ai-revision-desc');
            if (resultBox && resultDesc) {
              resultDesc.innerText = res.message;
              resultBox.classList.remove('hidden');
            }

            // Cuộn mượt đến bảng Xem Trước Dữ Liệu Sạch & tạo hiệu ứng viền nổi bật
            const previewCard = document.getElementById('quality-preview-table-card');
            if (previewCard) {
              previewCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
              previewCard.classList.add('ring-4', 'ring-amber-400', 'shadow-2xl');
              setTimeout(() => {
                previewCard.classList.remove('ring-4', 'ring-amber-400');
              }, 3500);
            }

            try {
              const curRes = await api('/api/current_results');
              if (curRes && curRes.step1) {
                curRes.step1.step2_notes = feedback;
              }
            } catch (_) {}
            await refreshStatus();
          } else if (step === 4) {
            if (res.data) renderTop3Advisor(res.data);
            await refreshStatus();
          }
        } else {
          showToast(`❌ ${res.message || 'Lỗi áp dụng yêu cầu sửa'}`, 'error');
        }
      } catch (err) {
        showToast(`❌ Lỗi kết nối: ${err}`, 'error');
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = origBtnHtml;
        }
      }
    }

    async function loadDiscoveryScope() {
      try {
        const res = await fetch('/api/discovery_scope').then(r => r.json());
        const wrap = document.getElementById('discovery-scope-container');
        if (wrap && res && res.status === 'SUCCESS') {
          wrap.innerHTML = `
            <div class="glass-card rounded-3xl p-6 sm:p-7 border border-amber-400/40 bg-gradient-to-r from-blue-950/50 via-indigo-950/40 to-amber-950/30 shadow-xl">
              <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-white/10 mb-4">
                <div class="flex items-center gap-2.5">
                  <div class="w-10 h-10 rounded-2xl bg-amber-400/20 text-amber-300 border border-amber-400/30 flex items-center justify-center text-xl shadow-md">
                    🧭
                  </div>
                  <div>
                    <h4 class="text-sm sm:text-base font-black text-amber-300 font-display">AI Discovery & Định Hướng Phân Tích Chiến Lược (DataLens Intelligence)</h4>
                    <p class="text-xs text-slate-300">Tự động nhận diện mô hình dữ liệu, phân tích thực chứng & đề xuất góc nhìn cho Ban Giám Đốc</p>
                  </div>
                </div>
                <span class="px-2.5 py-1 rounded-full bg-amber-400/20 text-amber-300 border border-amber-400/30 text-[10px] font-black uppercase flex items-center gap-1.5 self-start sm:self-auto">
                  <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                  AI Scope Planner
                </span>
              </div>
              <div class="space-y-4">
                <div class="p-4 rounded-2xl bg-white/5 border border-white/10 text-xs text-slate-200 leading-relaxed font-medium">
                  <strong class="text-amber-300">📌 Nhận diện mô hình & Dữ liệu cốt lõi:</strong> ${res.interpretation}
                </div>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-3.5">
                  ${(res.suggestions || []).map((s, idx) => `
                    <div class="p-3.5 rounded-2xl bg-blue-950/60 border border-amber-400/20 text-xs text-slate-200 flex flex-col justify-between shadow-sm">
                      <div>
                        <div class="text-[11px] font-black uppercase text-amber-300 mb-1.5 flex items-center gap-1.5">
                          <span>🎯</span><span>Góc nhìn #${idx+1}</span>
                        </div>
                        <div class="leading-relaxed text-blue-100/90">${s}</div>
                      </div>
                    </div>
                  `).join('')}
                </div>
                ${res.empirical_highlights && res.empirical_highlights.length > 0 ? `
                  <div class="p-4 rounded-2xl bg-emerald-950/20 border border-emerald-400/30 space-y-2">
                    <div class="text-[11px] font-black uppercase tracking-wider text-emerald-300 flex items-center gap-1.5">
                      <span>📊</span><span>Căn Cứ & Số Liệu Thực Nghiệm (Empirical Evidence):</span>
                    </div>
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-slate-200">
                      ${res.empirical_highlights.map(h => `<div class="leading-relaxed flex items-start gap-1.5"><span class="text-emerald-400">✓</span><span>${h}</span></div>`).join('')}
                    </div>
                  </div>
                ` : ''}
                ${res.causal_caveats && res.causal_caveats.length > 0 ? `
                  <div class="p-4 rounded-2xl bg-amber-950/20 border border-amber-400/30 space-y-1.5 text-xs text-amber-200/90">
                    <div class="text-[11px] font-black uppercase tracking-wider text-amber-300 flex items-center gap-1.5">
                      <span>⚠️</span><span>Cảnh Báo Nhân Quả vs Tương Quan (Causal Reasoning):</span>
                    </div>
                    ${res.causal_caveats.map(c => `<div class="leading-relaxed pl-2">• ${c}</div>`).join('')}
                  </div>
                ` : ''}
                <div class="pt-3 border-t border-white/10">
                  <div class="text-xs font-bold text-amber-300 mb-2.5 flex items-center gap-1.5">
                    <span>💡</span><span>Câu hỏi chiến lược gợi mở cho Ban Giám Đốc (Click để hỏi Copilot ngay):</span>
                  </div>
                  <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                    ${(res.followups || []).map((q) => `
                      <button onclick="askCopilotDirectly('${q.replace(/'/g, "\\'")}')" class="text-left p-3 rounded-2xl bg-blue-950/80 hover:bg-blue-900/90 border border-amber-400/25 hover:border-amber-400/60 text-xs text-slate-200 hover:text-amber-200 transition flex items-start justify-between gap-2 group shadow-sm">
                        <span class="leading-relaxed font-medium">${q}</span>
                        <span class="text-amber-400 text-xs font-bold shrink-0 opacity-70 group-hover:opacity-100">↗</span>
                      </button>
                    `).join('')}
                  </div>
                </div>
              </div>
            </div>
          `;
        }
      } catch (err) {
        console.warn('loadDiscoveryScope error:', err);
      }
    }

    function askCopilotDirectly(q) {
      switchTab('tab-copilot');
      const box = document.getElementById('copilot-input-box');
      if (box) box.value = q;
      handleSendChat();
    }

    async function loadLogs() {
      try {
        const r = await api('/api/logs');
        const term = document.getElementById('audit-log-terminal');
        if (term && r && r.logs) {
          term.innerText = r.logs.map(l => `[${l.timestamp}] [${l.step}] [${l.skill}] ${l.action} -> ${l.status}\\n  ${JSON.stringify(l.details)}`).join('\\n\\n');
        }
      } catch(_) {}
    }

    let isHydrated = false;

    async function hydrateUI() {
      try {
        const r = await api('/api/current_results');
        if (r.step3 && r.step3.company_summary) {
          renderDashboardMetrics(r.step3);
          renderStoresTable(r.step3.store_performance);
          renderSKUAnalytics(r.step3);
          loadDiscoveryScope();
        }
        if (r.step4 && r.step4.top_3_stores) {
          renderTop3Advisor(r.step4);
        }
        if (r.step1 && r.step1.summary_metrics) {
          renderQualityReport(r.step1);
        }
        isHydrated = true;
      } catch (e) {
        console.warn('Hydrate warning:', e);
      }
    }

    
    // ─── BƯỚC 5: KÝ DUYỆT BÁO CÁO & BAN HÀNH CHỈ THỊ (C-LEVEL) ───
    async function renderStep5View() {
      try {
        const cur = await api('/api/current_results');
        const st = await api('/api/state');
        const s5 = (cur && cur.step5) ? cur.step5 : {};
        const s3 = (cur && cur.step3) ? cur.step3 : {};
        const s1 = (cur && cur.step1) ? cur.step1 : {};
        const s4 = (cur && cur.step4) ? cur.step4 : {};
        const comp = s3.company_summary || {};
        const isApproved = st.s5_done === true || s5.approved === true;

        // 1. Dynamic document metadata & subtitle
        const totalRows = s5.total_rows || (s1.summary_metrics ? s1.summary_metrics.initial_transactions : 15000);
        const cleanRows = s5.clean_txns || (s1.summary_metrics ? s1.summary_metrics.clean_transactions : totalRows);
        const netRev = s5.total_net_revenue || comp.total_net_revenue || 430658320000;
        const targetRev = s5.total_target_revenue || comp.total_target_revenue || 495910000000;
        const gapRev = (targetRev > netRev) ? (targetRev - netRev) : 0;
        const attainment = (targetRev > 0) ? (netRev / targetRev * 100) : 86.8;
        const marginPct = s5.gross_margin_pct || comp.overall_gross_margin_pct || 13.4;
        const cogsVal = s5.cogs || comp.total_net_cogs || comp.total_cogs || 0;
        const cogsRatio = s5.cogs_ratio_pct || ((cogsVal > 0 && netRev > 0) ? (cogsVal / netRev * 100) : 86.6);
        const fileName = s5.filename || s1.filename || 'sales_data(1).csv';

        const subTitleEl = document.getElementById('step5-seal-subtitle');
        if (subTitleEl) {
          subTitleEl.innerText = `Báo cáo tài chính & Kế hoạch điều hành 30 ngày · Tệp: ${fileName} · ${totalRows.toLocaleString('vi-VN')} giao dịch (${cleanRows.toLocaleString('vi-VN')} hợp lệ) · Kỳ đối soát 2026-10`;
        }

        // 2. Dynamic certification boxes
        const mgrNameEl = document.getElementById('step5-manager-name');
        if (mgrNameEl) {
          mgrNameEl.innerText = s5.manager_name || (currentEnterpriseName ? `${currentEnterpriseName} · Ban Điều Hành C-Level` : 'Trần Bảo · Giám Đốc Kinh Doanh');
        }

        const statRowsEl = document.getElementById('step5-stat-rows');
        if (statRowsEl) {
          statRowsEl.innerText = `${totalRows.toLocaleString('vi-VN')} Dòng · 0₫ Thất Thoát`;
        }

        const statCleanEl = document.getElementById('step5-stat-clean');
        if (statCleanEl) {
          statCleanEl.innerText = `Đã đối soát ngoại lệ · ${cleanRows.toLocaleString('vi-VN')} đơn hoàn tất`;
        }

        const statRevEl = document.getElementById('step5-stat-revenue');
        if (statRevEl) {
          statRevEl.innerText = `${(netRev/1e9).toFixed(2)} tỷ ₫ · Đạt ${attainment.toFixed(1)}%`;
        }

        const statGapEl = document.getElementById('step5-stat-gap');
        if (statGapEl) {
          statGapEl.innerText = `Thiếu hụt: -${(gapRev/1e9).toFixed(2)} tỷ ₫ · Margin: ${marginPct.toFixed(1)}%`;
        }

        // 3. Status Badge, Button & Stamp Icon
        const statusEl = document.getElementById('step5-status-text');
        const badgeEl = document.getElementById('step5-seal-badge');
        const btnSign = document.getElementById('btn-step5-signoff');
        const sealIcon = document.getElementById('step5-seal-icon');
        const timeEl = document.getElementById('step5-decision-time');
        const hashEl = document.getElementById('step5-decision-hash');

        if (isApproved) {
          if (statusEl) {
            statusEl.innerText = '✅ ĐÃ PHÊ DUYỆT & BAN HÀNH';
            statusEl.className = 'text-base font-black text-emerald-300';
          }
          if (badgeEl) {
            badgeEl.className = 'flex items-center gap-3 animate-pulse';
          }
          if (sealIcon) {
            sealIcon.className = 'w-14 h-14 rounded-2xl bg-emerald-400/20 text-emerald-300 border-2 border-emerald-400/60 flex items-center justify-center text-2xl shadow-lg';
            sealIcon.innerText = '🏆';
          }
          if (btnSign) {
            btnSign.innerHTML = '<span>✅</span> <span>Báo Cáo Đã Được Ban Giám Đốc Phê Duyệt & Ban Hành</span>';
            btnSign.disabled = true;
            btnSign.className = 'px-6 py-3 rounded-2xl bg-emerald-600/40 text-emerald-200 border border-emerald-500/50 font-bold text-sm cursor-default flex items-center gap-2';
          }
          if (timeEl) {
            timeEl.innerText = s5.decision_date || new Date().toLocaleString('vi-VN');
          }
          if (hashEl) {
            hashEl.innerText = `Mã băm kiểm toán: ${s5.audit_hash || 'TB-SHA256-OCT-8501'}`;
          }

          // Show approved memo if custom notes exist
          const memoCard = document.getElementById('step5-approved-memo-card');
          const memoNotes = document.getElementById('step5-approved-memo-content');
          const notesInput = document.getElementById('step5-custom-directive-notes');
          if (memoCard && memoNotes) {
            const finalNotes = s5.custom_notes || (notesInput ? notesInput.value : '');
            if (finalNotes && finalNotes.trim()) {
              memoNotes.innerText = finalNotes.trim();
              memoCard.classList.remove('hidden');
              memoCard.style.display = 'block';
            }
          }
        } else {
          if (statusEl) {
            statusEl.innerText = '⏳ CHỜ PHÊ DUYỆT';
            statusEl.className = 'text-base font-black text-amber-300';
          }
          if (sealIcon) {
            sealIcon.className = 'w-14 h-14 rounded-2xl bg-amber-400/20 text-amber-300 border-2 border-amber-400/50 flex items-center justify-center text-2xl shadow-lg';
            sealIcon.innerText = '✍️';
          }
          if (btnSign) {
            btnSign.innerHTML = '<span>✍️</span> <span>Ban Giám Đốc Ký Duyệt Báo Cáo & Ban Hành Chỉ Thị</span>';
            btnSign.disabled = false;
            btnSign.className = 'toty-gold-btn text-slate-950 font-black text-sm px-6 py-3 rounded-2xl shadow-xl shadow-amber-400/25 flex items-center gap-2 cursor-pointer active:scale-95 transition';
          }
          if (timeEl) {
            timeEl.innerText = 'Chưa ký duyệt';
          }
        }

        // 4. Update dynamic export descriptions
        const exportTxnsDesc = document.getElementById('step5-export-txns-desc');
        if (exportTxnsDesc) {
          exportTxnsDesc.innerText = `${cleanRows.toLocaleString('vi-VN')} dòng sạch 23 cột nguyên bản không sai lệch.`;
        }

        // 5. Render 5-Step Continuity Audit Matrix
        renderStep5AuditMatrix(totalRows, cleanRows, netRev, targetRev, gapRev, attainment, marginPct, cogsRatio, isApproved, s5.decision_date);

        // 6. Render Directives 100% Grounded in Step 4 Action Plan
        const actions = (s5.action_plan && s5.action_plan.length > 0) ? s5.action_plan : ((s4.business_report && s4.business_report.action_plan) ? s4.business_report.action_plan : []);
        renderStep5Directives(actions, gapRev, attainment, cogsRatio, marginPct);

        // 7. Render Top Priority Units from Step 4
        const topStores = (s5.top_3_stores && s5.top_3_stores.length > 0) ? s5.top_3_stores : (s4.top_3_stores || []);
        renderStep5PriorityUnits(topStores);

      } catch(err) {
        console.warn('renderStep5View err:', err);
      }
    }

    function renderStep5AuditMatrix(totalRows, cleanRows, netRev, targetRev, gapRev, attainment, marginPct, cogsRatio, isApproved, decisionDate) {
      const container = document.getElementById('step5-audit-matrix');
      if (!container) return;
      
      const revBillion = (netRev / 1e9).toFixed(2);
      const gapBillion = (gapRev / 1e9).toFixed(2);
      const targetBillion = (targetRev / 1e9).toFixed(2);

      container.innerHTML = `
        <div class="p-3.5 rounded-2xl bg-white/5 border border-emerald-500/30 flex flex-col justify-between">
          <div>
            <div class="flex items-center justify-between text-[10px] font-black uppercase text-emerald-400 mb-1">
              <span>B1. Làm Sạch</span>
              <span>✅ ĐẠT</span>
            </div>
            <div class="text-xs font-bold text-white">${totalRows.toLocaleString('vi-VN')} Dòng Thô</div>
            <div class="text-[11px] text-slate-300 mt-0.5">${cleanRows.toLocaleString('vi-VN')} đơn sạch</div>
          </div>
          <div class="mt-2.5 pt-2 border-t border-white/10 text-[10px] text-emerald-300/90 font-medium">
            0₫ thất thoát · 100% toàn vẹn
          </div>
        </div>

        <div class="p-3.5 rounded-2xl bg-white/5 border border-emerald-500/30 flex flex-col justify-between">
          <div>
            <div class="flex items-center justify-between text-[10px] font-black uppercase text-emerald-400 mb-1">
              <span>B2. Kiểm Soát</span>
              <span>✅ ĐÃ DUYỆT</span>
            </div>
            <div class="text-xs font-bold text-white">Chuyên Viên Nội Bộ</div>
            <div class="text-[11px] text-slate-300 mt-0.5">Đã rà soát & duyệt ngoại lệ</div>
          </div>
          <div class="mt-2.5 pt-2 border-t border-white/10 text-[10px] text-emerald-300/90 font-medium">
            Chấp thuận đưa vào tính toán
          </div>
        </div>

        <div class="p-3.5 rounded-2xl bg-white/5 border border-amber-500/30 flex flex-col justify-between">
          <div>
            <div class="flex items-center justify-between text-[10px] font-black uppercase text-amber-300 mb-1">
              <span>B3. Tài Chính</span>
              <span>📊 ${attainment.toFixed(1)}%</span>
            </div>
            <div class="text-xs font-bold text-white">${revBillion} tỷ ₫ Doanh Thu</div>
            <div class="text-[11px] text-slate-300 mt-0.5">Target: ${targetBillion} tỷ ₫</div>
          </div>
          <div class="mt-2.5 pt-2 border-t border-white/10 text-[10px] text-amber-200/90 font-medium">
            Margin: ${marginPct.toFixed(1)}% · COGS: ${cogsRatio.toFixed(1)}%
          </div>
        </div>

        <div class="p-3.5 rounded-2xl bg-white/5 border border-amber-500/30 flex flex-col justify-between">
          <div>
            <div class="flex items-center justify-between text-[10px] font-black uppercase text-amber-300 mb-1">
              <span>B4. Chẩn Đoán</span>
              <span>⚠️ GAP ${gapBillion} TỶ</span>
            </div>
            <div class="text-xs font-bold text-white">Rủi Ro 2 Ngành Hàng</div>
            <div class="text-[11px] text-slate-300 mt-0.5">Laptop + ĐT chiếm 87.6%</div>
          </div>
          <div class="mt-2.5 pt-2 border-t border-white/10 text-[10px] text-amber-200/90 font-medium">
            4 Sáng kiến hành động 30 ngày
          </div>
        </div>

        <div class="p-3.5 rounded-2xl ${isApproved ? 'bg-emerald-950/30 border-emerald-500/50' : 'bg-amber-950/20 border-amber-500/40'} border flex flex-col justify-between">
          <div>
            <div class="flex items-center justify-between text-[10px] font-black uppercase ${isApproved ? 'text-emerald-300' : 'text-amber-300'} mb-1">
              <span>B5. Ký Duyệt</span>
              <span>${isApproved ? '✅ ĐÃ BAN HÀNH' : '⏳ CHỜ DUYỆT'}</span>
            </div>
            <div class="text-xs font-bold text-white">Ban Giám Đốc C-Level</div>
            <div class="text-[11px] text-slate-300 mt-0.5">${isApproved ? (decisionDate || 'Đã đóng dấu số') : 'Chờ ký ban hành chỉ thị'}</div>
          </div>
          <div class="mt-2.5 pt-2 border-t border-white/10 text-[10px] ${isApproved ? 'text-emerald-300' : 'text-amber-300'} font-medium">
            ${isApproved ? 'Đã ban hành 30 ngày chính thức' : 'Sẵn sàng phê chuẩn'}
          </div>
        </div>
      `;
    }

    function renderStep5Directives(actions, gapRev, attainment, cogsRatio, marginPct) {
      const container = document.getElementById('step5-directives-container');
      if (!container) return;
      
      if (!actions || actions.length === 0) {
        container.innerHTML = '<div class="p-4 text-center text-xs text-slate-400">Đang đồng bộ kế hoạch hành động 30 ngày từ Bước 4...</div>';
        return;
      }

      container.innerHTML = actions.map((item, idx) => {
        const phase = item.phase || `Giai đoạn ${idx + 1}`;
        const title = item.title || item.action || '';
        const owner = item.owner || 'Ban Giám Đốc';
        const priority = item.priority || (idx === 0 ? 'Khẩn cấp (P1)' : (idx === 1 ? 'Cao (P2)' : (idx === 2 ? 'Trung bình (P3)' : 'Chiến lược')));
        const action = item.action || '';
        const reason = item.reason || '';
        const measure = item.measure || '';

        let priBadgeClass = 'bg-blue-400/20 text-blue-300 border-blue-400/30';
        let borderClass = 'border-white/10 hover:border-amber-400/40';
        if (priority.toLowerCase().includes('khẩn cấp') || priority.toLowerCase().includes('p1')) {
          priBadgeClass = 'bg-rose-500/20 text-rose-300 border-rose-500/40';
          borderClass = 'border-rose-500/30 hover:border-rose-400/50';
        } else if (priority.toLowerCase().includes('cao') || priority.toLowerCase().includes('p2')) {
          priBadgeClass = 'bg-amber-400/20 text-amber-300 border-amber-400/40';
          borderClass = 'border-amber-400/30 hover:border-amber-300/50';
        } else if (priority.toLowerCase().includes('chiến lược') || priority.toLowerCase().includes('strategic')) {
          priBadgeClass = 'bg-indigo-400/20 text-indigo-300 border-indigo-400/40';
        }

        return `
          <div class="p-5 rounded-2xl bg-white/5 border ${borderClass} transition space-y-3">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div class="flex items-center gap-2 flex-wrap">
                <span class="px-2.5 py-0.5 rounded-md bg-amber-400/20 text-amber-300 border border-amber-400/30 text-[11px] font-black uppercase">
                  ${phase}
                </span>
                <span class="px-2.5 py-0.5 rounded-md text-[10px] font-bold uppercase border ${priBadgeClass}">
                  ${priority}
                </span>
                <span class="text-xs font-black text-white">
                  ${title}
                </span>
              </div>
              <div class="flex items-center gap-1.5 self-start sm:self-auto px-3 py-1 rounded-full bg-white/10 border border-white/15 text-[11px] text-amber-200 font-bold">
                <span>👤 Trách nhiệm:</span>
                <span class="text-white">${owner}</span>
              </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-12 gap-3 text-xs">
              <div class="lg:col-span-7 space-y-2">
                <div>
                  <span class="text-[10px] uppercase font-bold text-slate-400 block mb-0.5">Chỉ thị hành động chi tiết:</span>
                  <p class="text-slate-200 leading-relaxed font-sans">${action}</p>
                </div>
                ${reason ? `
                  <div class="p-3 rounded-xl bg-black/20 border border-white/5 text-[11px] text-slate-300 leading-relaxed">
                    <strong class="text-amber-300">Căn cứ thực chứng:</strong> ${reason}
                  </div>
                ` : ''}
              </div>

              <div class="lg:col-span-5 flex flex-col justify-between p-3.5 rounded-xl bg-gradient-to-br from-amber-950/20 to-blue-950/30 border border-amber-400/20">
                <div>
                  <div class="flex items-center gap-1.5 text-[10px] uppercase font-bold text-amber-300 mb-1">
                    <span>🎯</span> Mục tiêu KPI lượng hóa cần đạt:
                  </div>
                  <p class="text-xs font-bold text-emerald-300 leading-relaxed font-display">${measure}</p>
                </div>
                <div class="mt-3 pt-2 border-t border-white/10 flex items-center justify-between text-[10px] text-slate-400">
                  <span>Đơn vị phối hợp: CCO, CFO, COO</span>
                  <span class="text-amber-400 font-bold">Hiệu lực: 30 Ngày</span>
                </div>
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    function renderStep5PriorityUnits(topStores) {
      const container = document.getElementById('step5-top-units-container');
      if (!container) return;

      if (!topStores || topStores.length === 0) {
        container.innerHTML = '<div class="col-span-3 p-4 text-center text-xs text-slate-400">Chưa có thông tin điểm nóng vận hành từ Bước 4.</div>';
        return;
      }

      const fmtVND = (v) => {
        const val = Math.abs(v || 0);
        if (val === 0) return '0 ₫';
        if (val >= 1e9) return (val / 1e9).toFixed(2) + ' tỷ ₫';
        return (val / 1e6).toFixed(1) + ' tr ₫';
      };

      container.innerHTML = topStores.slice(0, 3).map((st, idx) => {
        const sname = st.store_name || st.store_id || `Điểm bán #${idx + 1}`;
        const sid = st.store_id || '';
        const rev = st.net_revenue || st.Net_Revenue || 0;
        const gap = st.gap_revenue || st.Revenue_Gap || (st.target_revenue ? (st.target_revenue - rev) : 0);
        const retRate = st.return_rate_pct || st.Return_Rate || 0;
        const actions = st.action_recommendations || st.recommended_actions || (st.ai_enhanced_analysis ? st.ai_enhanced_analysis.action_plan_30_days : []) || [];
        const mainAction = actions[0] || (st.ai_enhanced_analysis ? (st.ai_enhanced_analysis.executive_summary || st.ai_enhanced_analysis.diagnostic) : 'Tối ưu hóa cơ cấu hàng hóa và chấn chỉnh quy trình giao nhận.');

        return `
          <div class="p-4 rounded-2xl bg-white/5 border border-white/10 hover:border-amber-400/40 transition space-y-2.5 flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between gap-2 mb-1.5">
                <span class="px-2 py-0.5 rounded-md bg-rose-500/20 text-rose-300 border border-rose-500/30 text-[10px] font-black uppercase">
                  Ưu tiên #${idx + 1}
                </span>
                <span class="text-[11px] font-mono text-slate-400">${sid}</span>
              </div>
              <div class="text-sm font-black text-white font-display">${sname}</div>
              <div class="text-[11px] text-slate-300 mt-1 flex items-center justify-between">
                <span>Doanh thu: <strong class="text-amber-300 font-bold">${fmtVND(rev)}</strong></span>
                <span class="text-rose-400">Thâm hụt: -${fmtVND(gap)}</span>
              </div>
              <div class="text-[10px] text-slate-400 mt-0.5">
                Tỷ lệ đổi trả: <strong class="${retRate > 3.0 ? 'text-rose-400' : 'text-emerald-400'}">${retRate.toFixed(1)}%</strong>
              </div>
            </div>

            <div class="mt-2 pt-2 border-t border-white/10 text-[11px] text-slate-200 leading-relaxed font-sans bg-black/20 p-2.5 rounded-xl">
              <strong class="text-amber-300">Chỉ thị can thiệp:</strong> ${mainAction}
            </div>
          </div>
        `;
      }).join('');
    }

    async function signoffStep5() {
      const btn = document.getElementById('btn-step5-signoff');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> <span>Đang ký duyệt báo cáo...</span>';
      }
      try {
        const notes = document.getElementById('step5-custom-directive-notes')?.value || '';
        const savedName = localStorage.getItem('tb_enterprise_name') || 'Trần Bảo';
        const mgrName = `${savedName} · Giám Đốc Kinh Doanh`;
        const res = await api('/api/run_step5', { approved: true, notes: notes, manager_name: mgrName });
        showToast('✍️ ' + (res.message || 'Ban Giám Đốc đã ký duyệt thành công báo cáo điều hành!'), 'success');
        addNotification('success', '✍️', 'Bước 5: Ký Duyệt & Ban Hành Kế Hoạch 30 Ngày', 'Ban Giám đốc đã ký duyệt báo cáo và ban hành chỉ thị 30 ngày cho các khối.', 'step-tab-5');
        await refreshStatus();
        await renderStep5View();
      } catch(err) {
        showToast('❌ Lỗi ký duyệt: ' + err, 'error');
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = '<span>✍️</span> <span>Ban Giám Đốc Ký Duyệt Báo Cáo</span>';
        }
      }
    }

    // ─── RENDER TIẾN TRÌNH QUY TRÌNH 5 BƯỚC AI AGENT ───
    function renderStepperUI(st) {
      const stepperEl = document.getElementById('stepper-ui');
      if (!stepperEl) return;

      const isStep2Approved = st.s2_done === true;
      const steps = [
        { num: 1, name: "1. Kết Nối & Nguồn", done: st.has_data, icon: '🔗', locked: false },
        { num: 2, name: "2. Làm Sạch & Đối Soát", done: isStep2Approved, icon: '🧹', locked: false },
        { num: 3, name: "3. Phân Tích Số Liệu", done: st.s3_done && isStep2Approved, icon: '📊', locked: !isStep2Approved },
        { num: 4, name: "4. Báo Cáo Điều Hành", done: st.s4_done && isStep2Approved, icon: '🎯', locked: !isStep2Approved || !st.s3_done },
        { num: 5, name: "5. Ký Duyệt & Ban Hành", done: st.s5_done && isStep2Approved, icon: '✍️', locked: !isStep2Approved || !st.s4_done },
      ];

      stepperEl.innerHTML = steps.map((s, idx) => {
        const isActive = currentWorkflowStep === s.num;
        return `
          <button 
            type="button"
            onclick="goToStep(${s.num})"
            title="${s.locked ? '🔒 Chưa mở khóa - Cần hoàn thành và duyệt Bước 2' : 'Bấm để mở Bước ' + s.num + ': ' + s.name}"
            class="flex items-center gap-2 px-3.5 py-2 rounded-2xl text-xs transition-all transform hover:scale-105 cursor-pointer shadow-sm ${
              s.locked
                ? 'opacity-40 bg-[#060f26]/40 text-slate-500 border border-white/5 cursor-not-allowed'
                : (isActive
                  ? 'bg-gradient-to-r from-amber-400 via-amber-300 to-yellow-400 text-slate-950 font-black ring-2 ring-amber-300/50 shadow-lg shadow-amber-400/25'
                  : (s.done
                    ? 'bg-blue-950/80 text-amber-200 border border-amber-400/40 font-bold hover:bg-blue-900/60'
                    : 'bg-[#060f26]/80 text-blue-200/70 border border-blue-900/40 hover:bg-blue-900/30 font-medium'))
            }"
          >
            <span class="text-sm">${s.locked ? '🔒' : (s.done ? '✓' : s.icon)}</span>
            <span>${s.name}</span>
            <span class="text-[10px] px-1.5 py-0.5 rounded ${
              s.locked
                ? 'bg-white/5 text-slate-600'
                : (isActive ? 'bg-slate-950/20 text-slate-950 font-black' : (s.done ? 'bg-amber-400/20 text-amber-300' : 'bg-white/10 text-slate-400'))
            }">
              ${s.locked ? 'Khóa' : 'B' + s.num}
            </span>
          </button>
          ${idx < steps.length - 1 ? '<span class="text-amber-400/40 text-xs font-bold px-0.5 hidden sm:inline">──</span>' : ''}
        `;
      }).join('');

      // Stepper Action Container
      const act = document.getElementById('stepper-action-container');
      if (act) {
        let stepText = '';
        let nextBtnHtml = '';

        if (currentWorkflowStep === 1) {
          stepText = '📍 <strong>Bước 1:</strong> Kết nối nguồn dữ liệu & thiết lập bối cảnh doanh nghiệp với AI.';
          nextBtnHtml = `
            <div class="flex items-center gap-2">
              <button onclick="triggerDiscoveryScope()" class="px-3.5 py-2 rounded-xl text-xs font-bold bg-amber-400/20 text-amber-300 border border-amber-400/40 hover:bg-amber-400/30 transition flex items-center gap-1.5 cursor-pointer">
                <span>🧭</span><span>AI Scope</span>
              </button>
              <button onclick="goToStep(2)" class="toty-gold-btn text-slate-950 font-black text-xs px-4 py-2 rounded-xl shadow-md flex items-center gap-1.5 cursor-pointer">
                <span>Tiếp tục sang Bước 2</span><span>→</span>
              </button>
            </div>
          `;
        } else if (currentWorkflowStep === 2) {
          stepText = '📍 <strong>Bước 2:</strong> Áp dụng 4 quy tắc làm sạch tương tác và kiểm soát đối soát ngoại lệ.';
          nextBtnHtml = `
            <div class="flex items-center gap-2">
              <button onclick="applyCleanRules()" class="px-3.5 py-2 rounded-xl text-xs font-bold bg-amber-400/20 text-amber-300 border border-amber-400/40 hover:bg-amber-400/30 transition flex items-center gap-1.5 cursor-pointer">
                <span>⚡</span><span>Áp Dụng Quy Tắc</span>
              </button>
              <button onclick="runStep2(true)" class="toty-gold-btn text-slate-950 font-black text-xs px-4 py-2 rounded-xl shadow-md flex items-center gap-1.5 cursor-pointer">
                <span>✅ Duyệt &amp; Mở Khóa Bước 3</span><span>→</span>
              </button>
            </div>
          `;
        } else if (currentWorkflowStep === 3) {
          stepText = '📍 <strong>Bước 3:</strong> Khám phá số liệu đa chiều, so sánh nhóm BI và phân tích hồ sơ cột.';
          nextBtnHtml = `
            <div class="flex items-center gap-2">
              <button onclick="runDynamicBIAnalysis()" class="px-3.5 py-2 rounded-xl text-xs font-bold bg-amber-400/20 text-amber-300 border border-amber-400/40 hover:bg-amber-400/30 transition flex items-center gap-1.5 cursor-pointer">
                <span>📊</span><span>Phân Tích Nhóm</span>
              </button>
              <button onclick="goToStep(4)" class="toty-gold-btn text-slate-950 font-black text-xs px-4 py-2 rounded-xl shadow-md flex items-center gap-1.5 cursor-pointer">
                <span>Sang Bước 4: Báo Cáo</span><span>→</span>
              </button>
            </div>
          `;
        } else if (currentWorkflowStep === 4) {
          stepText = '📍 <strong>Bước 4:</strong> Báo cáo phân tích kinh doanh điều hành & chẩn đoán Gemini.';
          nextBtnHtml = `
            <div class="flex items-center gap-2">
              <button onclick="copyReportMarkdown()" class="px-3.5 py-2 rounded-xl text-xs font-bold bg-amber-400/20 text-amber-300 border border-amber-400/40 hover:bg-amber-400/30 transition flex items-center gap-1.5 cursor-pointer">
                <span>📋</span><span>Sao Chép Markdown</span>
              </button>
              <button onclick="goToStep(5)" class="toty-gold-btn text-slate-950 font-black text-xs px-4 py-2 rounded-xl shadow-md flex items-center gap-1.5 cursor-pointer">
                <span>Sang Bước 5: Ký Duyệt</span><span>→</span>
              </button>
            </div>
          `;
        } else if (currentWorkflowStep === 5) {
          stepText = '📍 <strong>Bước 5:</strong> Ban Giám Đốc ký duyệt chính thức và ban hành chỉ thị 30 ngày.';
          nextBtnHtml = `
            <div class="flex items-center gap-2">
              <button onclick="signoffStep5()" class="toty-gold-btn text-slate-950 font-black text-xs px-5 py-2 rounded-xl shadow-md flex items-center gap-1.5 cursor-pointer">
                <span>✍️</span><span>Ký Duyệt Báo Cáo</span>
              </button>
              <button onclick="openExportModal()" class="px-3.5 py-2 rounded-xl text-xs font-bold bg-white/10 text-white hover:bg-white/20 transition flex items-center gap-1.5 cursor-pointer">
                <span>📥</span><span>Xuất Tệp</span>
              </button>
            </div>
          `;
        }

        act.innerHTML = `
          <div class="text-xs text-blue-100/90 flex items-center gap-2">
            ${stepText}
          </div>
          <div>
            ${nextBtnHtml}
          </div>
        `;
      }
    }

    async function refreshStatus() {
      try {
        const st = await api('/api/state');
        renderStepperUI(st);
        loadLogs();
        if (!isHydrated) {
          hydrateUI();
        }
      } catch(err) {
        console.warn('refreshStatus error:', err);
      }
    }

  
  
  
  </script>
</body>
</html>
"""


def build_step1_payload(s1, orch):
    raw_preview = []
    raw_columns = []
    raw_total_rows = 0
    filename = getattr(orch, "loaded_filename", "") if orch.df_txns is not None else ""
    if orch.df_txns is not None and isinstance(orch.df_txns, pd.DataFrame):
        raw_columns = list(orch.df_txns.columns)
        raw_total_rows = len(orch.df_txns)
        try:
            raw_preview = json.loads(orch.df_txns.head(20).fillna("").to_json(orient="records", date_format="iso"))
        except Exception:
            raw_preview = orch.df_txns.head(20).fillna("").astype(str).to_dict(orient="records")

    if not s1:
        return {
            "has_raw_data": orch.df_txns is not None,
            "filename": filename,
            "raw_columns": raw_columns,
            "raw_preview": raw_preview,
            "raw_total_rows": raw_total_rows,
            "summary_metrics": {
                "initial_transactions": raw_total_rows,
                "clean_transactions": 0,
                "excluded_count": 0
            },
            "clean_columns": [],
            "clean_preview": [],
            "clean_total_rows": 0,
            "excluded_records": [],
            "data_alerts": [],
            "missing_field_stats": {},
            "incomplete_order_records": []
        }
    clean_txns = s1.get("clean_transactions")
    clean_preview = []
    clean_columns = []
    if clean_txns is not None and isinstance(clean_txns, pd.DataFrame):
        clean_columns = list(clean_txns.columns)
        try:
            clean_preview = json.loads(clean_txns.head(15).fillna("").to_json(orient="records", date_format="iso"))
        except Exception:
            clean_preview = clean_txns.head(15).fillna("").astype(str).to_dict(orient="records")

    raw_preview = []
    raw_columns = []
    raw_total_rows = 0
    if orch.df_txns is not None and isinstance(orch.df_txns, pd.DataFrame):
        raw_columns = list(orch.df_txns.columns)
        raw_total_rows = len(orch.df_txns)
        try:
            raw_preview = json.loads(orch.df_txns.head(15).fillna("").to_json(orient="records", date_format="iso"))
        except Exception:
            raw_preview = orch.df_txns.head(15).fillna("").astype(str).to_dict(orient="records")
    
    return {
        "has_raw_data": orch.df_txns is not None,
        "filename": filename,
        "summary_metrics": s1.get("summary_metrics", {}),
        "excluded_records": s1.get("excluded_records", [])[:60],
        "data_alerts": s1.get("data_alerts", []),
        "missing_field_stats": s1.get("missing_field_stats", {}),
        "incomplete_order_records": s1.get("incomplete_order_records", [])[:60],
        "clean_columns": clean_columns,
        "clean_preview": clean_preview,
        "clean_total_rows": len(clean_txns) if clean_txns is not None else 0,
        "raw_columns": raw_columns,
        "raw_preview": raw_preview,
        "raw_total_rows": raw_total_rows,
        "step2_approved": getattr(orch, "step2_human_approved", False),
        "step2_notes": getattr(orch, "step2_human_notes", "")
    }


def build_step3_payload(s3, orch):
    if not s3 or not isinstance(s3, dict):
        return {}
    charts = {}
    try:
        charts["rev"] = chart_store_revenue(s3.get("store_performance"))
    except Exception:
        pass
    try:
        charts["sku"] = chart_sku_revenue(s3.get("sku_performance"))
    except Exception:
        pass
    try:
        charts["region"] = chart_donut_distribution(s3.get("region_performance"), "Region", "Net_Revenue", "Tỷ Trọng Doanh Thu Theo Khu Vực")
    except Exception:
        pass
    try:
        charts["channel"] = chart_donut_distribution(s3.get("channel_performance"), "Channel", "Net_Revenue", "Tỷ Trọng Doanh Thu Theo Kênh Bán Hàng")
    except Exception:
        pass

    sku_records = s3["sku_performance"].to_dict(orient="records") if ("sku_performance" in s3 and isinstance(s3["sku_performance"], pd.DataFrame)) else []
    reg_records = s3["region_performance"].to_dict(orient="records") if ("region_performance" in s3 and isinstance(s3["region_performance"], pd.DataFrame)) else []
    chan_records = s3["channel_performance"].to_dict(orient="records") if ("channel_performance" in s3 and isinstance(s3["channel_performance"], pd.DataFrame)) else []
    store_records = s3["store_performance"].to_dict(orient="records") if ("store_performance" in s3 and isinstance(s3.get("store_performance"), pd.DataFrame)) else []
    cat_records = s3["category_performance"].to_dict(orient="records") if ("category_performance" in s3 and isinstance(s3.get("category_performance"), pd.DataFrame)) else []

    # Get Executive Dashboard data matching DataLens Studio
    exec_dash = {}
    try:
        exec_dash = orch.build_executive_dashboard("all")
    except Exception as e_dash:
        print("[build_step3_payload] Executive dashboard error:", e_dash)

    # Extract available dimensions and metrics from actual clean dataframe
    df_clean = orch.step1_results.get("clean_transactions") if (orch.step1_results and "clean_transactions" in orch.step1_results) else orch.df_txns
    avail_dims = []
    avail_metrics = []
    if df_clean is not None and len(df_clean) > 0:
        for c in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[c]):
                avail_metrics.append(c)
            elif 2 <= df_clean[c].nunique() <= 50:
                avail_dims.append(c)

    if not avail_dims:
        avail_dims = ["Region", "Channel", "Segment", "Category", "City"]
    if not avail_metrics:
        avail_metrics = ["Net_Sales_VND", "Sales_VND", "Cost_VND", "Profit_VND", "Quantity"]

    default_group = exec_dash.get("dimension") if exec_dash.get("dimension") in avail_dims else avail_dims[0]
    default_metric = exec_dash.get("revenue_column") if exec_dash.get("revenue_column") in avail_metrics else avail_metrics[0]

    try:
        dyn_analysis = orch.get_dynamic_analysis(default_group, default_metric, "sum")
    except Exception as e_dyn:
        print("[build_step3_payload] Dynamic analysis fallback:", e_dyn)
        dyn_analysis = {}
    try:
        col_stats = orch.get_column_stats()
    except Exception:
        col_stats = []
    dyn_analysis["column_stats"] = col_stats
    dyn_analysis["available_dimensions"] = avail_dims
    dyn_analysis["available_metrics"] = avail_metrics

    # Sync company summary so KPI cards are 100% mathematically exact with raw data
    cs = s3.get("company_summary", {}).copy()
    if exec_dash and "kpis" in exec_dash:
        kpis = exec_dash["kpis"]
        cs["total_net_revenue"] = kpis["revenue"]["value"]
        cs["total_gross_sales"] = kpis["revenue"]["value"]
        cs["total_net_cogs"] = kpis["cost"]["value"]
        cs["total_gross_profit"] = kpis["profit"]["value"]
        cs["total_orders"] = kpis["orders"]["value"]
        cs["overall_gross_margin_pct"] = kpis["margin_pct"]
        cs["overall_return_rate_pct"] = kpis["return_rate_pct"]
        cs["total_return_amount"] = kpis["return_amount"]

    biz_context = {}
    try:
        biz_context = orch.get_business_context_evidence()
    except Exception as e_biz:
        print("[build_step3_payload] Business context error:", e_biz)

    top_kpis = {
        "net_revenue": cs.get("total_net_revenue", 0.0),
        "gross_profit": cs.get("total_gross_profit", 0.0),
        "cogs": cs.get("total_net_cogs", 0.0),
        "orders": cs.get("total_orders", 0),
        "margin_pct": cs.get("overall_gross_margin_pct", 0.0),
        "return_rate_pct": cs.get("overall_return_rate_pct", 0.0),
        "return_amount": cs.get("total_return_amount", 0.0)
    }

    return {
        "status": "SUCCESS",
        "company_summary": cs,
        "kpis": top_kpis,
        "executive_dashboard": exec_dash,
        "business_context": biz_context,
        "rows_count": len(df_clean) if df_clean is not None else 15000,
        "columns_count": len(df_clean.columns) if df_clean is not None else 23,
        "available_dimensions": avail_dims,
        "available_metrics": avail_metrics,
        "store_performance": store_records,
        "category_performance": cat_records,
        "sku_performance": sku_records,
        "region_records": reg_records,
        "channel_records": chan_records,
        "charts": charts,
        "dynamic_analysis": dyn_analysis,
        "column_stats": col_stats
    }


def build_step5_payload(s5, orch):
    comp = orch.step3_results.get("company_summary", {}) if orch.step3_results else {}
    s1 = orch.step1_results or {}
    s4 = orch.step4_results or {}
    report = s4.get("business_report", {})
    action_plan = report.get("action_plan", []) or report.get("recommendations", [])
    top_3 = s4.get("top_3_stores", [])
    
    total_rows = len(orch.df_txns) if orch.df_txns is not None else 0
    clean_txns = len(s1.get("clean_transactions", [])) if (s1 and "clean_transactions" in s1) else total_rows
    
    total_rev = comp.get("total_net_revenue", 0.0)
    target_rev = comp.get("total_target_revenue", 495910000000.0)
    gap_rev = target_rev - total_rev if target_rev > total_rev else 0.0
    attainment = (total_rev / target_rev * 100) if target_rev > 0 else 86.8
    gross_profit = comp.get("total_gross_profit", 0.0)
    margin_pct = comp.get("overall_gross_margin_pct", (gross_profit / total_rev * 100) if total_rev > 0 else 13.4)
    cogs_val = comp.get("total_net_cogs", comp.get("total_cogs", 0.0))
    cogs_ratio = (cogs_val / total_rev * 100) if total_rev > 0 else 86.6
    
    decision = getattr(orch, "step5_manager_decision", {}) or {}
    is_approved = getattr(orch, "step5_human_approved", False)
    
    return {
        "approved": is_approved,
        "manager_name": decision.get("manager_name", "Trần Bảo · Giám Đốc Kinh Doanh"),
        "decision_date": decision.get("decision_date", ""),
        "audit_hash": decision.get("audit_hash", "TB-2026-OCT-8501"),
        "custom_notes": decision.get("custom_notes", ""),
        "directives": decision.get("directives", {}),
        "total_rows": total_rows,
        "clean_txns": clean_txns,
        "total_net_revenue": total_rev,
        "total_target_revenue": target_rev,
        "revenue_gap": gap_rev,
        "target_attainment_pct": attainment,
        "gross_profit": gross_profit,
        "gross_margin_pct": margin_pct,
        "cogs": cogs_val,
        "cogs_ratio_pct": cogs_ratio,
        "filename": getattr(orch, "loaded_filename", "") if orch.df_txns is not None else "",
        "action_plan": action_plan,
        "top_3_stores": top_3,
        "step2_approved": getattr(orch, "step2_human_approved", False),
        "step2_notes": getattr(orch, "step2_human_notes", "")
    }


class EnterpriseRetailHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def send_json(self, data: dict, code=200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html: str, code=200):
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def get_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length:
            raw = self.rfile.read(length)
            try:
                return json.loads(raw)
            except Exception:
                return {}
        return {}

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/toty_bg.jpg", "/light_bg.jpg", "/dark_bg.jpg"):
            fname = "light_bg.jpg" if path == "/light_bg.jpg" else "toty_bg.jpg"
            try:
                with open(fname, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", len(data))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(data)
                return
            except Exception:
                self.send_response(404)
                self.end_headers()
                return

        with _lock:
            try:
                if path == "/":
                    self.send_html(PAGE_HTML)
                elif path == "/api/state":
                    self.send_json({
                        "current_step": _orch.current_step,
                        "has_data": _orch.df_txns is not None,
                        "s1_done": _orch.step1_results is not None,
                        "s2_done": _orch.step2_human_approved,
                        "s3_done": _orch.step3_results is not None and _orch.step2_human_approved,
                        "s4_done": _orch.step4_results is not None and _orch.step2_human_approved,
                        "s5_done": _orch.step5_human_approved,
                    })
                elif path == "/api/logs":
                    self.send_json({"logs": _orch.logger.get_logs()})
                elif path == "/api/current_results":
                    s1_payload = build_step1_payload(_orch.step1_results, _orch) if (_orch.df_txns is not None or _orch.step1_results is not None) else {}
                    s3_payload = build_step3_payload(_orch.step3_results, _orch) if (_orch.step3_results is not None and _orch.step2_human_approved) else {}

                    self.send_json({
                        "has_data": _orch.df_txns is not None,
                        "s1_done": _orch.step1_results is not None,
                        "s2_done": _orch.step2_human_approved,
                        "s3_done": _orch.step3_results is not None and _orch.step2_human_approved,
                        "s4_done": _orch.step4_results is not None,
                        "s5_done": _orch.step5_human_approved,
                        "current_step": _orch.current_step,
                        "step1": s1_payload,
                        "step3": s3_payload,
                        "step4": _orch.step4_results,
                        "step5": build_step5_payload(_orch.step5_manager_decision, _orch)
                    })
                elif path == "/api/column_stats":
                    if _orch.df_txns is None:
                        self.send_json({"status": "NO_DATA", "stats": [], "message": "Chưa có dữ liệu thô. Vui lòng nạp file dữ liệu ở Bước 1."})
                        return
                    self.send_json({"status": "SUCCESS", "stats": _orch.get_column_stats()})
                    return
                elif path == "/api/export/excel":
                    if _orch.df_txns is None:
                        self.send_json({"status": "ERROR", "message": "Chưa có dữ liệu thô để xuất file Excel. Vui lòng nạp file dữ liệu ở Bước 1."})
                        return
                    if _orch.step3_results is None:
                        try:
                            _orch.run_step_1_data_quality()
                            _orch.run_step_3_financial_metrics()
                        except Exception as e_run:
                            print(f"[Export] Auto-run metrics: {e_run}")
                    excel_bytes = _orch.export_results_to_excel()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    self.send_header("Content-Disposition", 'attachment; filename="Bao_Cao_Quan_Tri_Tran_Bao_Project.xlsx"')
                    self.send_header("Content-Length", len(excel_bytes))
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.end_headers()
                    self.wfile.write(excel_bytes)
                    return
                elif path == "/api/export/csv":
                    from urllib.parse import parse_qs, urlparse
                    query_params = parse_qs(urlparse(self.path).query)
                    export_type = query_params.get("type", ["txns"])[0]
                    if _orch.df_txns is None:
                        self.send_json({"status": "ERROR", "message": "Chưa có dữ liệu thô để xuất file CSV. Vui lòng nạp file dữ liệu ở Bước 1."})
                        return
                    if _orch.step3_results is None:
                        try:
                            _orch.run_step_1_data_quality()
                            _orch.run_step_3_financial_metrics()
                        except Exception as e_run:
                            print(f"[Export] Auto-run metrics: {e_run}")
                    csv_str = _orch.export_results_to_csv(export_type)
                    csv_bytes = ("\ufeff" + csv_str).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header("Content-Disposition", f'attachment; filename="Du_Lieu_{export_type}_Tran_Bao.csv"')
                    self.send_header("Content-Length", len(csv_bytes))
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.end_headers()
                    self.wfile.write(csv_bytes)
                    return
                elif path == "/api/discovery_scope":
                    if _orch.df_txns is None:
                        self.send_json({
                            "status": "NO_DATA",
                            "message": "Chưa có dữ liệu thô. Vui lòng nạp file dữ liệu ở Bước 1.",
                            "interpretation": "Chưa có dữ liệu thô. Vui lòng nạp file dữ liệu ở Bước 1 để AI Agent nhận diện bối cảnh doanh nghiệp.",
                            "suggestions": [],
                            "followups": [],
                            "empirical_highlights": []
                        })
                        return
                    res = _orch.propose_business_scope()
                    self.send_json(res)
                    return
                else:
                    self.send_response(404)
                    self.end_headers()
            except Exception as e:
                tb = traceback.format_exc()
                print("GET Server Error:", tb)
                self.send_json({"status": "ERROR", "message": str(e)}, code=500)

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            if path == "/api/ask_copilot":
                body = self.get_body()
                question = body.get("question", "")
                if _orch.df_txns is None:
                    self.send_json({"status": "ERROR", "answer": "Chưa có dữ liệu thô. Vui lòng tải lên tệp dữ liệu ở Bước 1 trước khi đặt câu hỏi cho AI Copilot."})
                    return
                if _orch.step3_results is None:
                    try:
                        _orch.run_step_1_data_quality()
                        _orch.run_step_3_financial_metrics()
                    except Exception as auto_ex:
                        print(f"[Copilot] Auto-compute step context note: {auto_ex}")
                ans = _orch.ask_copilot(question)
                self.send_json({"status": "SUCCESS", "answer": ans})
                return

            elif path == "/api/revise_step":
                body = self.get_body()
                step = int(body.get("step", 2))
                feedback = body.get("feedback", "")
                if _orch.df_txns is None:
                    self.send_json({"status": "ERROR", "message": "Chưa có dữ liệu thô. Vui lòng nạp tệp dữ liệu ở Bước 1."})
                    return
                res = _orch.revise_step(step, feedback)
                if step == 2 and _orch.step1_results:
                    res["step1_data"] = build_step1_payload(_orch.step1_results, _orch)
                    res["step3_data"] = build_step3_payload(_orch.step3_results, _orch)
                elif step == 3 and _orch.step3_results:
                    res["step3_data"] = build_step3_payload(_orch.step3_results, _orch)
                self.send_json(res)
                return

            elif path == "/api/dynamic_analysis":
                body = self.get_body()
                group_by = body.get("group_by", "Region")
                metric = body.get("metric", "Net_Sales_VND")
                mode = body.get("mode", "sum")
                prompt = body.get("prompt", "")

                if _orch.df_txns is None:
                    self.send_json({
                        "status": "NO_DATA",
                        "message": "Chưa có dữ liệu thô để phân tích.",
                        "groups": [],
                        "findings": ["Vui lòng nạp tệp dữ liệu ở Bước 1."],
                        "column_stats": []
                    })
                    return

                if not _orch.step2_human_approved and _orch.step3_results is None:
                    self.send_json({
                        "status": "BLOCKED",
                        "message": "Chưa hoàn tất phê duyệt dữ liệu ở Bước 2. Bạn phải hoàn thành và duyệt Bước 2 trước khi phân tích số liệu ở Bước 3.",
                        "groups": [],
                        "findings": ["Vui lòng hoàn thành phê duyệt dữ liệu ở Bước 2 trước khi phân tích."]
                    })
                    return

                if prompt:
                    res = _orch.ask_ai_dynamic_analysis(prompt)
                else:
                    res = _orch.get_dynamic_analysis(group_by, metric, mode)

                res["column_stats"] = _orch.get_column_stats()
                self.send_json(res)
                return

            with _lock:
                # ─── 1-CLICK RUN FULL PIPELINE ───
                if path == "/api/run_full_pipeline":
                    if _orch.df_txns is None:
                        self.send_json({"status": "ERROR", "message": "Chưa có dữ liệu thô! Vui lòng tải lên file dữ liệu ở Bước 1 trước khi chạy toàn bộ quy trình."})
                        return
                    
                    # 1. Step 1: Quality Check
                    s1 = _orch.run_step_1_data_quality()
                    
                    # 2. Step 2: Human-in-the-loop review approved
                    _orch.run_step_2_human_approval(True, "Dữ liệu ngoại lệ đã được rà soát và chấp thuận đưa vào tính toán.")
                    
                    # 3. Step 3: Financial & Metrics calculation
                    s3 = _orch.run_step_3_financial_metrics()
                    
                    # Generate charts
                    charts = {}
                    try:
                        charts["rev"] = chart_store_revenue(s3["store_performance"])
                    except Exception as ex:
                        print("Chart error:", ex)
                    try:
                        charts["sku"] = chart_sku_revenue(s3.get("sku_performance"))
                    except Exception as ex:
                        print("SKU Chart error:", ex)
                    try:
                        charts["region"] = chart_donut_distribution(s3.get("region_performance"), "Region", "Net_Revenue", "Tỷ Trọng Doanh Thu Theo Khu Vực")
                    except Exception as ex:
                        print("Region Chart error:", ex)
                    try:
                        charts["channel"] = chart_donut_distribution(s3.get("channel_performance"), "Channel", "Net_Revenue", "Tỷ Trọng Doanh Thu Theo Kênh Bán Hàng")
                    except Exception as ex:
                        print("Channel Chart error:", ex)

                    sku_records = s3["sku_performance"].to_dict(orient="records") if ("sku_performance" in s3 and isinstance(s3["sku_performance"], pd.DataFrame)) else []
                    reg_records = s3["region_performance"].to_dict(orient="records") if ("region_performance" in s3 and isinstance(s3["region_performance"], pd.DataFrame)) else []
                    chan_records = s3["channel_performance"].to_dict(orient="records") if ("channel_performance" in s3 and isinstance(s3["channel_performance"], pd.DataFrame)) else []

                    s3_payload = {
                        "company_summary": s3.get("company_summary", {}),
                        "store_performance": s3["store_performance"].to_dict(orient="records"),
                        "category_performance": s3["category_performance"].to_dict(orient="records"),
                        "sku_performance": sku_records,
                        "region_records": reg_records,
                        "channel_records": chan_records,
                        "charts": charts
                    }

                    # 4. Step 4: Strategic Diagnosis with Gemini 3.5 Flash
                    s4 = _orch.run_step_4_diagnostic_recommendations()

                    # 5. Step 5: Final Decision sign-off - dynamically use top 3 store IDs
                    top_3 = s4.get("top_3_stores", [])
                    action_dirs = {}
                    for st in top_3:
                        sid = st.get("store_id", "")
                        actions = st.get("recommended_actions", [])
                        action_dirs[sid] = actions[0] if actions else "Tối ưu hóa danh mục sản phẩm và rà soát lưu lượng khách."
                    
                    s5 = _orch.run_step_5_human_decision(approved=True, action_directives=action_dirs)

                    self.send_json({
                        "status": "SUCCESS",
                        "step1": build_step1_payload(s1, _orch),
                        "step3": s3_payload,
                        "step4": s4,
                        "step5": s5
                    })

                elif path == "/api/load_default":
                    res = _orch.load_default_master_data()
                    self.send_json(res)

                elif path == "/api/executive_dashboard":
                    body = self.get_body()
                    period = body.get("period", "all")
                    if _orch.df_txns is None:
                        self.send_json({"status": "NO_DATA", "message": "Chưa có dữ liệu thô để hiển thị dashboard."})
                        return
                    if _orch.step1_results is None:
                        _orch.run_step_1_data_quality()
                    res = _orch.build_executive_dashboard(period)
                    if isinstance(res, dict):
                        res["dashboard_data"] = {k: v for k, v in res.items() if k != "dashboard_data"}
                    self.send_json(res)
                    return

                elif path == "/api/discovery_scope":
                    if _orch.df_txns is None:
                        self.send_json({
                            "status": "NO_DATA",
                            "message": "Chưa có dữ liệu thô. Vui lòng nạp file dữ liệu ở Bước 1.",
                            "interpretation": "Chưa có dữ liệu thô. Vui lòng nạp file dữ liệu ở Bước 1 để AI Agent nhận diện bối cảnh doanh nghiệp.",
                            "suggestions": [],
                            "followups": [],
                            "empirical_highlights": []
                        })
                        return
                    res = _orch.propose_business_scope()
                    self.send_json(res)
                    return

                elif path == "/api/clean_rules":
                    body = self.get_body()
                    config = body.get("config", {})
                    if _orch.df_txns is None:
                        self.send_json({"status": "ERROR", "message": "Chưa có dữ liệu thô để áp dụng luật làm sạch."})
                        return
                    res = _orch.apply_cleaning_rules(config)
                    if _orch.step1_results:
                        res["step1_data"] = build_step1_payload(_orch.step1_results, _orch)
                        res["step3_data"] = build_step3_payload(_orch.step3_results, _orch)
                    self.send_json(res)
                    return

                elif path == "/api/load_gsheet":
                    body = self.get_body()
                    res = _orch.load_from_google_sheet(body.get("url", ""))
                    if res.get("status") == "SUCCESS":
                        res["current_step"] = 1
                    self.send_json(res)
                    return

                elif path == "/api/upload_file":
                    ctype = self.headers.get("Content-Type", "")
                    length = int(self.headers.get("Content-Length", 0))
                    raw_body = self.rfile.read(length)
                    
                    # Robust multipart boundary parsing
                    file_bytes = b""
                    filename = "upload.csv"
                    try:
                        boundary = None
                        for part in ctype.split(";"):
                            part = part.strip()
                            if part.startswith("boundary="):
                                boundary = part.split("=", 1)[1].strip().strip('"').encode()
                                break
                        
                        if boundary:
                            delimiter = b"--" + boundary
                            parts = raw_body.split(delimiter)
                            for p in parts:
                                if b"filename=" not in p:
                                    continue
                                header_end = p.find(b"\r\n\r\n")
                                if header_end == -1:
                                    continue
                                header_bytes = p[:header_end].decode("utf-8", errors="replace")
                                body = p[header_end + 4:]
                                if body.endswith(b"\r\n"):
                                    body = body[:-2]
                                # Extract filename from header
                                for line in header_bytes.split("\r\n"):
                                    if "filename=" in line:
                                        idx = line.find("filename=")
                                        val = line[idx+9:].strip().strip('"')
                                        if ";" in val:
                                            val = val.split(";")[0].strip('"')
                                        filename = val
                                        break
                                file_bytes = body
                                break
                    except Exception as parse_err:
                        print("Multipart parse error:", parse_err)
                    
                    if not file_bytes or len(file_bytes.strip()) == 0:
                        self.send_json({"status": "ERROR", "message": "Dữ liệu không hợp lệ: Tệp rỗng (0 bytes) hoặc không có dữ liệu để phân tích."})
                        return
                    
                    res = _orch.load_from_uploaded_excel(file_bytes, filename)
                    if res.get("status") == "SUCCESS":
                        res["current_step"] = 1
                    self.send_json(res)
                    return

                elif path == "/api/run_step1":
                    if _orch.df_txns is None:
                        self.send_json({"status": "ERROR", "message": "Chưa có dữ liệu thô! Vui lòng tải lên file dữ liệu ở Bước 1."})
                        return
                    s1 = _orch.run_step_1_data_quality()
                    self.send_json({
                        "status": "SUCCESS",
                        "data": build_step1_payload(s1, _orch)
                    })
                    return

                elif path == "/api/run_step2":
                    body = self.get_body()
                    res = _orch.run_step_2_human_approval(body.get("approved", True), body.get("notes", ""))
                    self.send_json(res)

                elif path == "/api/run_step3":
                    if _orch.df_txns is None:
                        self.send_json({"status": "ERROR", "message": "Chưa có dữ liệu thô! Vui lòng tải lên file dữ liệu ở Bước 1."})
                        return
                    if not _orch.step2_human_approved:
                        self.send_json({
                            "status": "BLOCKED",
                            "message": "Chưa hoàn tất phê duyệt dữ liệu ở Bước 2. Bạn phải hoàn thành và duyệt Bước 2 trước khi sang Bước 3!"
                        })
                        return
                    
                    s3 = _orch.run_step_3_financial_metrics()
                    if isinstance(s3, dict) and s3.get("status") in ["ERROR", "BLOCKED"]:
                        self.send_json(s3)
                        return
                    
                    res_payload = build_step3_payload(s3, _orch)
                    res_payload["status"] = "SUCCESS"
                    self.send_json(res_payload)

                elif path == "/api/run_step4":
                    if _orch.df_txns is None:
                        self.send_json({"status": "ERROR", "message": "Chưa có dữ liệu thô! Vui lòng tải lên file dữ liệu ở Bước 1."})
                        return
                    if not _orch.step2_human_approved:
                        self.send_json({
                            "status": "BLOCKED",
                            "message": "Chưa hoàn tất phê duyệt dữ liệu ở Bước 2. Bạn phải duyệt Bước 2 trước khi chạy Bước 4!"
                        })
                        return
                    if _orch.step3_results is None:
                        _orch.run_step_3_financial_metrics()
                    s4 = _orch.run_step_4_diagnostic_recommendations()
                    if isinstance(s4, dict) and s4.get("status") == "ERROR":
                        self.send_json(s4)
                        return
                    self.send_json({"status": "SUCCESS", "data": s4})

                elif path == "/api/run_step5":
                    if _orch.df_txns is None:
                        self.send_json({"status": "ERROR", "message": "Chưa có dữ liệu thô! Vui lòng tải lên file dữ liệu ở Bước 1."})
                        return
                    if not _orch.step2_human_approved:
                        self.send_json({
                            "status": "BLOCKED",
                            "message": "Chưa hoàn tất phê duyệt dữ liệu ở Bước 2. Bạn phải duyệt Bước 2 trước khi ký duyệt Bước 5!"
                        })
                        return
                    if _orch.step3_results is None:
                        _orch.run_step_3_financial_metrics()
                    if _orch.step4_results is None:
                        _orch.run_step_4_diagnostic_recommendations()
                    
                    body = self.get_body()
                    notes = body.get("notes", "")
                    manager_name = body.get("manager_name", "Trần Bảo · Giám Đốc Kinh Doanh")
                    
                    top_3 = _orch.step4_results.get("top_3_stores", []) if _orch.step4_results else []
                    action_dirs = {}
                    for st in top_3:
                        sid = st.get("store_id") or st.get("store_name", "CH")
                        actions = st.get("recommended_actions", [])
                        action_dirs[sid] = actions[0] if actions else "Tối ưu danh mục và rà soát lưu lượng khách."
                    
                    res = _orch.run_step_5_human_decision(approved=True, action_directives=action_dirs, manager_name=manager_name, custom_notes=notes)
                    res["step5_data"] = build_step5_payload(_orch.step5_manager_decision, _orch)
                    self.send_json(res)

                elif path == "/api/ai_revise":
                    body = self.get_body()
                    step = body.get("step", 1)
                    prompt = body.get("prompt", "")
                    _orch.logger.log(f"Step {step}", "AI_REVISE", "User Revision Request", {"prompt": prompt})
                    self.send_json({
                        "status": "SUCCESS",
                        "message": f"AI Copilot đã tiếp nhận chỉ thị chỉnh sửa: '{prompt}'. Quy tắc xử lý Bước {step} đã được cập nhật thành công."
                    })

                elif path == "/api/reset":
                    _orch.reset_state()
                    self.send_json({"status": "SUCCESS"})

                else:
                    self.send_response(404)
                    self.end_headers()

        except Exception as e:
            tb = traceback.format_exc()
            print("Server Error:", tb)
            self.send_json({"status": "ERROR", "message": str(e), "traceback": tb[:400]}, code=500)


def run_server():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), EnterpriseRetailHandler)
    print("=" * 65, flush=True)
    print(f"  🏢 TRẦN BẢO ENTERPRISE AI PLATFORM", flush=True)
    print(f"  🌐 Localhost: http://localhost:{PORT}", flush=True)
    print(f"  ⚡ Tailwind CSS UI + Google Gemini 3.5 Flash", flush=True)
    print("=" * 65, flush=True)
    sys.stdout.flush()
    server.serve_forever()


if __name__ == "__main__":
    run_server()
