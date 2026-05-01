
"""
Pile Foundation Designer — Streamlit App

Author: ChatGPT
Purpose:
    Preliminary/professional engineering design aid for reinforced concrete pile caps
    from 2 to 12 piles using ACI-style strength design, sectional checks, and
    practical pile-cap / strut-and-tie engineering workflow.

Important:
    This app is a calculation aid. It is not a replacement for the responsible
    engineer's judgment, project specifications, geotechnical report, local
    amendments, or a licensed design review. ACI 318-25 should be consulted
    directly for final design, detailing, strength reduction factors, seismic
    provisions, anchorage, development, shear friction, and deep foundation
    requirements.

Units:
    - Geometry inputs: mm
    - Forces: kN
    - Moments: kN-m
    - Concrete strength fc': MPa
    - Steel yield fy: MPa
    - Output reinforcement areas: mm²

Run:
    pip install streamlit pandas numpy matplotlib openpyxl fpdf
    streamlit run pile_foundation_app.py
"""

from __future__ import annotations

import io
import math
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

try:
    from fpdf import FPDF
except ImportError:  # PDF export is optional until requirements are installed.
    FPDF = None


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

APP_TITLE = "Pile Foundation Designer"
APP_SUBTITLE = "ACI 318-25 style RC pile-cap design aid | 2 to 12 piles | SAP2000 reactions or manual input"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CONSTANTS
# =============================================================================

BAR_DATABASE_MM = {
    "DB10": {"diameter": 10.0, "area": 78.5},
    "DB12": {"diameter": 12.0, "area": 113.1},
    "DB16": {"diameter": 16.0, "area": 201.1},
    "DB20": {"diameter": 20.0, "area": 314.2},
    "DB25": {"diameter": 25.0, "area": 490.9},
    "DB28": {"diameter": 28.0, "area": 615.8},
    "DB32": {"diameter": 32.0, "area": 804.2},
    "DB36": {"diameter": 36.0, "area": 1017.9},
    "DB40": {"diameter": 40.0, "area": 1256.6},
}

DEFAULT_STRENGTH_COMBOS = {
    "ULS-Manual": {
        "Pu_kN": 5000.0,
        "Mux_kNm": 0.0,
        "Muy_kNm": 0.0,
        "Vux_kN": 0.0,
        "Vuy_kN": 0.0,
        "Tuz_kNm": 0.0,
    }
}

ACI_REFERENCES = {
    "general": "ACI CODE-318-25 should be used directly for final design and detailing.",
    "flexure": "ACI-style strength design: phi Mn >= Mu. Rectangular section, singly-reinforced simplification.",
    "one_way_shear": "ACI-style concrete shear expression for nonprestressed members; verify exact ACI 318-25 modifiers.",
    "two_way_shear": "ACI-style two-way shear/punching check around loaded area; verify beta, alpha_s and edge/corner cases.",
    "development": "ACI-style tension development estimate; engineer must verify all modifiers and confinement.",
    "stm": "MacGregor-style practical load-path/strut-and-tie advisory for deep pile caps; not a substitute for full STM.",
}

PILE_LAYOUT_TEMPLATE_VERSION = "reference-arrangements-2026-05"


# =============================================================================
# STYLING
# =============================================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --card-bg: rgba(250,250,250,0.74);
            --card-border: rgba(100,100,100,0.16);
        }
        html, body, [class*="css"] {
            font-size: 14px;
        }
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }
        .title-wrap {
            padding: 1rem 1.2rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(64,64,64,0.08), rgba(64,64,64,0.02));
            border: 1px solid var(--card-border);
            margin-bottom: 1rem;
        }
        .title-main {
            font-size: 2.0rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin: 0;
        }
        .title-sub {
            font-size: 0.95rem;
            opacity: 0.78;
            margin-top: 0.25rem;
        }
        .soft-card {
            padding: 1rem;
            border-radius: 18px;
            border: 1px solid var(--card-border);
            background: var(--card-bg);
            margin-bottom: 0.8rem;
        }
        .mini {
            font-size: 0.82rem;
            opacity: 0.78;
        }
        .ok-box, .warn-box, .bad-box, .info-box {
            border-radius: 14px;
            padding: 0.75rem 0.9rem;
            border: 1px solid rgba(90,90,90,0.2);
            margin: 0.4rem 0;
        }
        .ok-box { background: rgba(50, 180, 90, 0.10); }
        .warn-box { background: rgba(255, 190, 0, 0.13); }
        .bad-box { background: rgba(255, 70, 70, 0.12); }
        .info-box { background: rgba(70, 130, 255, 0.10); }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(100,100,100,0.15);
            border-radius: 16px;
            padding: 0.75rem;
            background: rgba(255,255,255,0.48);
        }
        .small-table table {
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Material:
    fc_MPa: float = 35.0
    fy_MPa: float = 500.0
    lambda_c: float = 1.0
    gamma_conc_kN_m3: float = 24.0
    phi_flexure: float = 0.90
    phi_shear: float = 0.75
    phi_bearing: float = 0.65
    phi_stm_tie: float = 0.75

@dataclass
class Geometry:
    n_piles: int = 4
    pile_diameter_mm: float = 600.0
    pile_capacity_comp_kN: float = 1800.0
    pile_capacity_tension_kN: float = 400.0
    cap_thickness_mm: float = 1200.0
    bottom_cover_mm: float = 100.0
    top_cover_mm: float = 75.0
    side_cover_mm: float = 75.0
    column_bx_mm: float = 800.0
    column_by_mm: float = 800.0
    pedestal_bx_mm: float = 800.0
    pedestal_by_mm: float = 800.0
    edge_from_pile_edge_mm: float = 600.0
    spacing_x_mm: float = 1800.0
    spacing_y_mm: float = 1800.0
    use_pedestal_for_shear: bool = True

@dataclass
class Reinforcement:
    main_bar_x: str = "DB25"
    main_bar_y: str = "DB25"
    top_bar: str = "DB16"
    spacing_x_mm: float = 150.0
    spacing_y_mm: float = 150.0
    top_spacing_mm: float = 200.0
    side_face_bar: str = "DB16"
    side_face_spacing_mm: float = 250.0
    hook_extension_mm: float = 300.0
    preferred_spacing_step_mm: float = 25.0

@dataclass
class LoadCase:
    name: str = "ULS-Manual"
    Pu_kN: float = 5000.0
    Mux_kNm: float = 0.0
    Muy_kNm: float = 0.0
    Vux_kN: float = 0.0
    Vuy_kN: float = 0.0
    Tuz_kNm: float = 0.0

@dataclass
class PilePoint:
    label: str
    x_mm: float
    y_mm: float
    diameter_mm: float
    reaction_kN: float = 0.0

@dataclass
class CheckResult:
    name: str
    demand: float
    capacity: float
    ratio: float
    unit: str
    status: str
    note: str = ""

@dataclass
class DesignState:
    material: Material
    geometry: Geometry
    reinforcement: Reinforcement
    loadcase: LoadCase
    piles: List[PilePoint]
    cap_length_x_mm: float
    cap_width_y_mm: float
    effective_depth_x_mm: float
    effective_depth_y_mm: float
    self_weight_kN: float = 0.0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def fmt(value: float, nd: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:,.{nd}f}"


def status_from_ratio(ratio: float, warn_limit: float = 0.95) -> str:
    if ratio <= warn_limit:
        return "PASS"
    if ratio <= 1.0:
        return "NEAR"
    return "FAIL"


def status_badge(status: str) -> str:
    s = status.upper()
    if s == "PASS":
        return "✅ PASS"
    if s == "NEAR":
        return "⚠️ NEAR"
    if s == "FAIL":
        return "❌ FAIL"
    return s


def bar_area(bar_name: str) -> float:
    return BAR_DATABASE_MM.get(bar_name, BAR_DATABASE_MM["DB25"])["area"]


def bar_diameter(bar_name: str) -> float:
    return BAR_DATABASE_MM.get(bar_name, BAR_DATABASE_MM["DB25"])["diameter"]


def round_up_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.ceil(value / step) * step


def round_down_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step


def mm_to_m(value_mm: float) -> float:
    return value_mm / 1000.0


def kN_to_N(value_kN: float) -> float:
    return value_kN * 1000.0


def kNm_to_Nmm(value_kNm: float) -> float:
    return value_kNm * 1_000_000.0


def Nmm_to_kNm(value_Nmm: float) -> float:
    return value_Nmm / 1_000_000.0


def kN_from_N(value_N: float) -> float:
    return value_N / 1000.0


def get_session_default(key: str, value: Any) -> Any:
    if key not in st.session_state:
        st.session_state[key] = value
    return st.session_state[key]


# =============================================================================
# PILE LAYOUT GENERATION
# =============================================================================

def template_layout(n: int, sx: float, sy: float) -> List[Tuple[float, float]]:
    """
    Auto-generate pile coordinates around centroid using the reference
    arrangement sketches.

    Coordinates:
        x positive to right, y positive upward in plan.
        Units: mm.

    The typical spacing inputs represent the 3D center-to-center spacing shown
    in the reference sketches. Layouts with diagonal/triangular geometry derive
    their offsets from that spacing with sqrt(2) or sqrt(3) factors.
    """
    if n < 2 or n > 12:
        raise ValueError("n_piles must be between 2 and 12")

    rt2 = math.sqrt(2.0)
    rt3 = math.sqrt(3.0)

    # Reference templates from the attached pile arrangement sketches. User can
    # still edit the generated coordinates before running the design.
    templates: Dict[int, List[Tuple[float, float]]] = {
        # 2 piles: vertical pair.
        2: [(0.0, -0.5 * sy), (0.0, 0.5 * sy)],
        # 3 piles: equilateral triangle, centered at its centroid.
        3: [(-0.5 * sx, -rt3 * sy / 6.0), (0.5 * sx, -rt3 * sy / 6.0), (0.0, rt3 * sy / 3.0)],
        4: [(-0.5 * sx, -0.5 * sy), (0.5 * sx, -0.5 * sy), (-0.5 * sx, 0.5 * sy), (0.5 * sx, 0.5 * sy)],
        # 5 piles: four corner piles plus one center pile; corner spacing is 3*sqrt(2)D.
        5: [(-sx / rt2, -sy / rt2), (sx / rt2, -sy / rt2), (0.0, 0.0), (-sx / rt2, sy / rt2), (sx / rt2, sy / rt2)],
        6: [(-sx, -0.5 * sy), (0.0, -0.5 * sy), (sx, -0.5 * sy), (-sx, 0.5 * sy), (0.0, 0.5 * sy), (sx, 0.5 * sy)],
        # 7 piles: center pile plus six piles on a regular hexagonal ring.
        7: [(0.0, -sy), (-rt3 * sx / 2.0, -0.5 * sy), (rt3 * sx / 2.0, -0.5 * sy),
            (0.0, 0.0), (-rt3 * sx / 2.0, 0.5 * sy), (rt3 * sx / 2.0, 0.5 * sy), (0.0, sy)],
        # 8 piles: compact staggered 3-2-3 layout from the reference sketch.
        8: [(-rt2 * sx, -sy / rt2), (0.0, -sy / rt2), (rt2 * sx, -sy / rt2),
            (-sx / rt2, 0.0), (sx / rt2, 0.0),
            (-rt2 * sx, sy / rt2), (0.0, sy / rt2), (rt2 * sx, sy / rt2)],
        9: [(-sx, -sy), (0.0, -sy), (sx, -sy), (-sx, 0.0), (0.0, 0.0), (sx, 0.0), (-sx, sy), (0.0, sy), (sx, sy)],
        # 10 and 11 piles: staggered triangular-grid layouts with total row
        # spacing of 3*sqrt(3)D between exterior rows.
        10: [(-sx, -rt3 * sy / 2.0), (0.0, -rt3 * sy / 2.0), (sx, -rt3 * sy / 2.0),
             (-1.5 * sx, 0.0), (-0.5 * sx, 0.0), (0.5 * sx, 0.0), (1.5 * sx, 0.0),
             (-sx, rt3 * sy / 2.0), (0.0, rt3 * sy / 2.0), (sx, rt3 * sy / 2.0)],
        11: [(-1.5 * sx, -rt3 * sy / 2.0), (-0.5 * sx, -rt3 * sy / 2.0), (0.5 * sx, -rt3 * sy / 2.0), (1.5 * sx, -rt3 * sy / 2.0),
             (-sx, 0.0), (0.0, 0.0), (sx, 0.0),
             (-1.5 * sx, rt3 * sy / 2.0), (-0.5 * sx, rt3 * sy / 2.0), (0.5 * sx, rt3 * sy / 2.0), (1.5 * sx, rt3 * sy / 2.0)],
        12: [(-1.5 * sx, -sy), (-0.5 * sx, -sy), (0.5 * sx, -sy), (1.5 * sx, -sy),
             (-1.5 * sx, 0.0), (-0.5 * sx, 0.0), (0.5 * sx, 0.0), (1.5 * sx, 0.0),
             (-1.5 * sx, sy), (-0.5 * sx, sy), (0.5 * sx, sy), (1.5 * sx, sy)],
    }
    return templates[n]


def make_piles(n: int, sx: float, sy: float, pile_dia: float) -> List[PilePoint]:
    coords = template_layout(n, sx, sy)
    piles = []
    for i, (x, y) in enumerate(coords, start=1):
        piles.append(PilePoint(label=f"P{i}", x_mm=float(x), y_mm=float(y), diameter_mm=float(pile_dia)))
    return piles


def normalize_pile_centroid(piles: List[PilePoint]) -> List[PilePoint]:
    if not piles:
        return piles
    xbar = np.mean([p.x_mm for p in piles])
    ybar = np.mean([p.y_mm for p in piles])
    out = []
    for p in piles:
        out.append(PilePoint(p.label, p.x_mm - xbar, p.y_mm - ybar, p.diameter_mm, p.reaction_kN))
    return out


def piles_to_dataframe(piles: List[PilePoint]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Pile": p.label,
                "x_mm": p.x_mm,
                "y_mm": p.y_mm,
                "diameter_mm": p.diameter_mm,
                "reaction_kN": p.reaction_kN,
            }
            for p in piles
        ]
    )


def dataframe_to_piles(df: pd.DataFrame, default_dia: float) -> List[PilePoint]:
    piles = []
    for i, row in df.iterrows():
        label = str(row.get("Pile", f"P{i + 1}"))
        x = safe_float(row.get("x_mm", 0.0))
        y = safe_float(row.get("y_mm", 0.0))
        dia = safe_float(row.get("diameter_mm", default_dia), default_dia)
        r = safe_float(row.get("reaction_kN", 0.0))
        piles.append(PilePoint(label=label, x_mm=x, y_mm=y, diameter_mm=dia, reaction_kN=r))
    return normalize_pile_centroid(piles)


def cap_dimensions_from_piles(piles: List[PilePoint], edge_mm: float) -> Tuple[float, float]:
    if not piles:
        return 0.0, 0.0
    min_x = min(p.x_mm - p.diameter_mm / 2 for p in piles) - edge_mm
    max_x = max(p.x_mm + p.diameter_mm / 2 for p in piles) + edge_mm
    min_y = min(p.y_mm - p.diameter_mm / 2 for p in piles) - edge_mm
    max_y = max(p.y_mm + p.diameter_mm / 2 for p in piles) + edge_mm
    return max_x - min_x, max_y - min_y


# =============================================================================
# LOAD DISTRIBUTION
# =============================================================================

def distribute_vertical_load_to_piles(
    piles: List[PilePoint],
    Pu_kN: float,
    Mux_kNm: float,
    Muy_kNm: float,
    include_self_weight_kN: float = 0.0,
    sign_convention: str = "Compression positive",
) -> List[PilePoint]:
    """
    Elastic rigid pile-cap distribution.

    Formula:
        R_i = P/n + Mx*y_i/sum(y²) + My*x_i/sum(x²)

    Notes:
        - x, y in m.
        - Mx is moment about X axis causing gradient along y.
        - My is moment about Y axis causing gradient along x.
        - Positive R means compression on pile.
        - Sign convention can be changed by applying load signs in input.
    """
    if not piles:
        return []
    n = len(piles)
    P_total = Pu_kN + include_self_weight_kN
    xs_m = np.array([mm_to_m(p.x_mm) for p in piles], dtype=float)
    ys_m = np.array([mm_to_m(p.y_mm) for p in piles], dtype=float)
    sum_x2 = float(np.sum(xs_m**2))
    sum_y2 = float(np.sum(ys_m**2))

    base = P_total / n
    reactions = np.full(n, base, dtype=float)

    if sum_y2 > 1e-12:
        reactions += Mux_kNm * ys_m / sum_y2
    if sum_x2 > 1e-12:
        reactions += Muy_kNm * xs_m / sum_x2

    out = []
    for p, r in zip(piles, reactions):
        out.append(PilePoint(p.label, p.x_mm, p.y_mm, p.diameter_mm, float(r)))
    return out


def pile_group_result_table(piles: List[PilePoint], geom: Geometry) -> pd.DataFrame:
    rows = []
    for p in piles:
        comp_ratio = max(p.reaction_kN, 0) / max(geom.pile_capacity_comp_kN, 1e-9)
        tension = max(-p.reaction_kN, 0)
        tension_ratio = tension / max(geom.pile_capacity_tension_kN, 1e-9)
        rows.append(
            {
                "Pile": p.label,
                "x (mm)": p.x_mm,
                "y (mm)": p.y_mm,
                "R, compression + (kN)": p.reaction_kN,
                "Compression ratio": comp_ratio,
                "Tension ratio": tension_ratio,
                "Status": status_badge("FAIL" if max(comp_ratio, tension_ratio) > 1.0 else "PASS"),
            }
        )
    return pd.DataFrame(rows)

# =============================================================================
# STATE IMPORT / EXPORT
# =============================================================================

def key_value_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{"Key": key, "Value": value} for key, value in data.items()])


def dataframe_to_key_values(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty or "Key" not in df.columns or "Value" not in df.columns:
        return {}
    return {str(row["Key"]): row["Value"] for _, row in df.iterrows() if str(row.get("Key", "")).strip()}


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def make_saved_state_xlsx(state: DesignState, results: Dict[str, Any], include_self_weight: bool) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        key_value_dataframe({"version": 1, "include_self_weight": include_self_weight}).to_excel(writer, sheet_name="settings", index=False)
        key_value_dataframe(asdict(state.material)).to_excel(writer, sheet_name="material", index=False)
        key_value_dataframe(asdict(state.geometry)).to_excel(writer, sheet_name="geometry", index=False)
        key_value_dataframe(asdict(state.reinforcement)).to_excel(writer, sheet_name="reinforcement", index=False)
        key_value_dataframe(asdict(state.loadcase)).to_excel(writer, sheet_name="loadcase", index=False)
        piles_to_dataframe(state.piles).to_excel(writer, sheet_name="piles", index=False)
        checks_to_dataframe(results["checks"]).to_excel(writer, sheet_name="checks", index=False)
        pile_group_result_table(state.piles, state.geometry).to_excel(writer, sheet_name="pile_reactions", index=False)
        pd.DataFrame(
            [
                {"Item": "Cap length X", "Value": state.cap_length_x_mm, "Unit": "mm"},
                {"Item": "Cap width Y", "Value": state.cap_width_y_mm, "Unit": "mm"},
                {"Item": "Self weight", "Value": state.self_weight_kN, "Unit": "kN"},
                {"Item": "Effective depth X", "Value": results["d_x_mm"], "Unit": "mm"},
                {"Item": "Effective depth Y", "Value": results["d_y_mm"], "Unit": "mm"},
                {"Item": "Pu total", "Value": results["Pu_total_kN"], "Unit": "kN"},
                {"Item": "X bars required As", "Value": results["flex_x"]["As_req_mm2"], "Unit": "mm2"},
                {"Item": "Y bars required As", "Value": results["flex_y"]["As_req_mm2"], "Unit": "mm2"},
            ]
        ).to_excel(writer, sheet_name="summary", index=False)
    return buf.getvalue()


def read_saved_state_xlsx(uploaded_file) -> Optional[Dict[str, Any]]:
    if uploaded_file is None:
        return None
    try:
        sheets = pd.read_excel(uploaded_file, sheet_name=None)
        payload = {
            "settings": dataframe_to_key_values(sheets.get("settings", pd.DataFrame())),
            "material": dataframe_to_key_values(sheets.get("material", pd.DataFrame())),
            "geometry": dataframe_to_key_values(sheets.get("geometry", pd.DataFrame())),
            "reinforcement": dataframe_to_key_values(sheets.get("reinforcement", pd.DataFrame())),
            "loadcase": dataframe_to_key_values(sheets.get("loadcase", pd.DataFrame())),
            "piles": sheets.get("piles", pd.DataFrame()),
        }
        if payload["geometry"] or payload["material"] or not payload["piles"].empty:
            return payload
        st.warning("Saved state workbook is missing expected sheets.")
        return None
    except Exception as exc:
        st.error(f"Could not load saved state workbook: {exc}")
        return None


def value_from_saved(section: str, key: str, default: Any) -> Any:
    saved = st.session_state.get("saved_state_payload") or {}
    value = saved.get(section, {}).get(key, default)
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def saved_float(section: str, key: str, default: float) -> float:
    return safe_float(value_from_saved(section, key, default), default)


def saved_int(section: str, key: str, default: int) -> int:
    return int(round(saved_float(section, key, float(default))))


def saved_bool(section: str, key: str, default: bool) -> bool:
    return coerce_bool(value_from_saved(section, key, default), default)


def saved_choice(section: str, key: str, default: str, choices: List[str]) -> str:
    value = str(value_from_saved(section, key, default))
    return value if value in choices else default


def apply_saved_pile_layout_if_needed(default_dia: float) -> None:
    saved = st.session_state.get("saved_state_payload") or {}
    piles_df = saved.get("piles")
    if piles_df is None or piles_df.empty or st.session_state.get("saved_piles_applied"):
        return
    required = ["Pile", "x_mm", "y_mm", "diameter_mm"]
    if all(col in piles_df.columns for col in required):
        df = piles_df.copy()
        if "reaction_kN" not in df.columns:
            df["reaction_kN"] = 0.0
        st.session_state["pile_layout_df"] = df[["Pile", "x_mm", "y_mm", "diameter_mm", "reaction_kN"]]
        st.session_state["saved_piles_applied"] = True



# =============================================================================
# RC DESIGN FORMULAS
# =============================================================================

def beta1_aci(fc_MPa: float) -> float:
    """
    ACI-style beta1 expression in MPa.
    0.85 for fc' <= 28 MPa, decreasing 0.05 per 7 MPa to minimum 0.65.
    Verify against the governing code edition.
    """
    if fc_MPa <= 28.0:
        return 0.85
    return max(0.65, 0.85 - 0.05 * ((fc_MPa - 28.0) / 7.0))


def flexural_As_required(
    Mu_kNm: float,
    b_mm: float,
    d_mm: float,
    fc_MPa: float,
    fy_MPa: float,
    phi: float = 0.90,
) -> Dict[str, float]:
    """
    Required tensile steel area for singly reinforced rectangular section.

    Solves:
        phi * As * fy * (d - a/2) >= Mu
        a = As * fy / (0.85 fc b)

    Units:
        Mu: kN-m
        b, d: mm
        fc, fy: MPa = N/mm²
        As: mm²
    """
    Mu_Nmm = abs(kNm_to_Nmm(Mu_kNm))
    if Mu_Nmm <= 1e-9:
        As_min = max(0.25 * math.sqrt(fc_MPa) / fy_MPa, 1.4 / fy_MPa) * b_mm * d_mm
        return {
            "As_req_mm2": As_min,
            "As_strength_mm2": 0.0,
            "As_min_mm2": As_min,
            "rho": As_min / (b_mm * d_mm),
            "a_mm": 0.0,
            "phiMn_kNm": 0.0,
        }

    A = phi * fy_MPa * fy_MPa / (2.0 * 0.85 * fc_MPa * b_mm)
    B = -phi * fy_MPa * d_mm
    C = Mu_Nmm

    disc = B * B - 4.0 * A * C
    if disc < 0:
        # Section is too small or heavily reinforced. Return high number as signal.
        As_strength = np.nan
    else:
        root1 = (-B - math.sqrt(disc)) / (2.0 * A)
        root2 = (-B + math.sqrt(disc)) / (2.0 * A)
        candidates = [r for r in (root1, root2) if r > 0]
        As_strength = min(candidates) if candidates else np.nan

    As_min = max(0.25 * math.sqrt(fc_MPa) / fy_MPa, 1.4 / fy_MPa) * b_mm * d_mm
    As_req = max(As_min, As_strength if np.isfinite(As_strength) else 1e99)
    a = As_req * fy_MPa / (0.85 * fc_MPa * b_mm)
    phiMn = phi * As_req * fy_MPa * (d_mm - a / 2.0)
    return {
        "As_req_mm2": As_req,
        "As_strength_mm2": As_strength,
        "As_min_mm2": As_min,
        "rho": As_req / (b_mm * d_mm),
        "a_mm": a,
        "phiMn_kNm": Nmm_to_kNm(phiMn),
    }


def flexural_capacity(
    As_mm2: float,
    b_mm: float,
    d_mm: float,
    fc_MPa: float,
    fy_MPa: float,
    phi: float = 0.90,
) -> Dict[str, float]:
    a = As_mm2 * fy_MPa / (0.85 * fc_MPa * b_mm)
    Mn_Nmm = As_mm2 * fy_MPa * max(d_mm - a / 2.0, 0.0)
    phiMn_Nmm = phi * Mn_Nmm
    c = a / max(beta1_aci(fc_MPa), 1e-9)
    eps_t = 0.003 * (d_mm - c) / max(c, 1e-9)
    return {
        "a_mm": a,
        "c_mm": c,
        "eps_t": eps_t,
        "Mn_kNm": Nmm_to_kNm(Mn_Nmm),
        "phiMn_kNm": Nmm_to_kNm(phiMn_Nmm),
    }


def one_way_shear_capacity(
    b_mm: float,
    d_mm: float,
    fc_MPa: float,
    lambda_c: float,
    phi: float = 0.75,
    vc_factor: float = 0.17,
) -> Dict[str, float]:
    """
    ACI-style one-way concrete shear:
        Vc = vc_factor * lambda * sqrt(fc') * b * d
    Units N -> kN.
    """
    Vc_N = vc_factor * lambda_c * math.sqrt(fc_MPa) * b_mm * d_mm
    return {"Vc_kN": kN_from_N(Vc_N), "phiVc_kN": kN_from_N(phi * Vc_N)}


def two_way_shear_capacity(
    bo_mm: float,
    d_mm: float,
    fc_MPa: float,
    lambda_c: float,
    phi: float = 0.75,
    vc_factor: float = 0.33,
) -> Dict[str, float]:
    """
    Simplified interior punching shear:
        Vc = vc_factor * lambda * sqrt(fc') * bo * d

    For edge/corner columns or high unbalanced moment, engineer must modify.
    """
    Vc_N = vc_factor * lambda_c * math.sqrt(fc_MPa) * bo_mm * d_mm
    return {"Vc_kN": kN_from_N(Vc_N), "phiVc_kN": kN_from_N(phi * Vc_N)}


def concrete_bearing_capacity(
    area_loaded_mm2: float,
    fc_MPa: float,
    phi: float = 0.65,
    bearing_factor: float = 0.85,
    area_ratio_factor: float = 1.0,
) -> Dict[str, float]:
    Pn_N = bearing_factor * fc_MPa * area_loaded_mm2 * area_ratio_factor
    return {"Pn_kN": kN_from_N(Pn_N), "phiPn_kN": kN_from_N(phi * Pn_N)}


def development_length_tension_estimate(
    db_mm: float,
    fc_MPa: float,
    fy_MPa: float,
    lambda_c: float = 1.0,
    psi_t: float = 1.0,
    psi_e: float = 1.0,
    psi_s: float = 1.0,
    confinement_factor: float = 1.0,
    min_ld_mm: float = 300.0,
) -> float:
    """
    Editable ACI-style development length estimate.

    This is intentionally transparent and conservative for app-level preliminary design.
    Final design must check the exact ACI 318-25 development provisions and all modifiers.
    """
    confinement_factor = min(max(confinement_factor, 1.0), 2.5)
    ld = (fy_MPa * psi_t * psi_e * psi_s / (1.1 * lambda_c * math.sqrt(fc_MPa))) * db_mm / confinement_factor
    return max(min_ld_mm, ld)


def spacing_for_As(
    As_req_mm2: float,
    bar: str,
    strip_width_mm: float,
    preferred_step_mm: float = 25.0,
    s_min_mm: float = 75.0,
    s_max_mm: float = 300.0,
) -> Dict[str, float]:
    """
    Convert required total steel area across a strip width to a spacing.
    """
    Ab = bar_area(bar)
    if As_req_mm2 <= 0:
        return {"spacing_req_mm": s_max_mm, "spacing_use_mm": s_max_mm, "n_bars": 0, "As_prov_mm2": 0.0}
    s_req = Ab * strip_width_mm / As_req_mm2
    s_use = min(s_max_mm, max(s_min_mm, round_down_to_step(s_req, preferred_step_mm)))
    n_bars = int(math.floor(strip_width_mm / s_use)) + 1
    As_prov = n_bars * Ab
    return {"spacing_req_mm": s_req, "spacing_use_mm": s_use, "n_bars": n_bars, "As_prov_mm2": As_prov}


# =============================================================================
# FOOTING / PILE CAP DESIGN ACTIONS
# =============================================================================

def effective_depths(geom: Geometry, reinf: Reinforcement) -> Tuple[float, float]:
    """
    Effective depth to bottom reinforcement in each orthogonal layer.
    Assume X-direction bars are the lower layer, Y-direction bars above them.
    """
    db_x = bar_diameter(reinf.main_bar_x)
    db_y = bar_diameter(reinf.main_bar_y)
    d_x = geom.cap_thickness_mm - geom.bottom_cover_mm - db_x / 2.0
    d_y = geom.cap_thickness_mm - geom.bottom_cover_mm - db_x - db_y / 2.0
    return max(d_x, 1.0), max(d_y, 1.0)


def self_weight_cap_kN(cap_x_mm: float, cap_y_mm: float, thickness_mm: float, gamma_kN_m3: float) -> float:
    vol_m3 = mm_to_m(cap_x_mm) * mm_to_m(cap_y_mm) * mm_to_m(thickness_mm)
    return vol_m3 * gamma_kN_m3


def design_moments_from_pile_reactions(
    piles: List[PilePoint],
    geom: Geometry,
    cap_x_mm: float,
    cap_y_mm: float,
) -> Dict[str, float]:
    """
    Simplified cantilever sectional moments at face of column/pedestal.

    For each direction, take pile reactions outside the loaded area face and
    multiply by lever arm to the face. The max of left/right or top/bottom is used.

    Mx-design here means bending requiring bars parallel to X axis, produced by
    cantilever action in Y direction. My-design means bars parallel to Y axis,
    produced by cantilever action in X direction.

    We report:
        M_for_X_bars: moments from piles above/below y face -> reinforcement along X.
        M_for_Y_bars: moments from piles left/right x face -> reinforcement along Y.
    """
    loaded_bx = geom.pedestal_bx_mm if geom.use_pedestal_for_shear else geom.column_bx_mm
    loaded_by = geom.pedestal_by_mm if geom.use_pedestal_for_shear else geom.column_by_mm

    x_face = loaded_bx / 2.0
    y_face = loaded_by / 2.0

    M_right = sum(max(p.reaction_kN, 0.0) * max(p.x_mm - x_face, 0.0) / 1000.0 for p in piles)
    M_left = sum(max(p.reaction_kN, 0.0) * max(-p.x_mm - x_face, 0.0) / 1000.0 for p in piles)

    M_top = sum(max(p.reaction_kN, 0.0) * max(p.y_mm - y_face, 0.0) / 1000.0 for p in piles)
    M_bottom = sum(max(p.reaction_kN, 0.0) * max(-p.y_mm - y_face, 0.0) / 1000.0 for p in piles)

    return {
        "M_right_for_Y_bars_kNm": M_right,
        "M_left_for_Y_bars_kNm": M_left,
        "M_top_for_X_bars_kNm": M_top,
        "M_bottom_for_X_bars_kNm": M_bottom,
        "M_for_Y_bars_kNm": max(M_right, M_left),
        "M_for_X_bars_kNm": max(M_top, M_bottom),
    }


def one_way_shear_demands(
    piles: List[PilePoint],
    geom: Geometry,
    d_x_mm: float,
    d_y_mm: float,
) -> Dict[str, float]:
    """
    One-way shear at d from face of loaded area in both directions.

    Vx section: vertical shear for strip spanning in X direction, section at x face + d_y.
    Vy section: vertical shear for strip spanning in Y direction, section at y face + d_x.

    We return maximum of two sides.
    """
    loaded_bx = geom.pedestal_bx_mm if geom.use_pedestal_for_shear else geom.column_bx_mm
    loaded_by = geom.pedestal_by_mm if geom.use_pedestal_for_shear else geom.column_by_mm

    x_sec = loaded_bx / 2.0 + d_y_mm
    y_sec = loaded_by / 2.0 + d_x_mm

    V_right = sum(max(p.reaction_kN, 0.0) for p in piles if p.x_mm > x_sec)
    V_left = sum(max(p.reaction_kN, 0.0) for p in piles if p.x_mm < -x_sec)
    V_top = sum(max(p.reaction_kN, 0.0) for p in piles if p.y_mm > y_sec)
    V_bottom = sum(max(p.reaction_kN, 0.0) for p in piles if p.y_mm < -y_sec)

    return {
        "V_right_kN": V_right,
        "V_left_kN": V_left,
        "V_top_kN": V_top,
        "V_bottom_kN": V_bottom,
        "V_for_Y_bars_direction_kN": max(V_right, V_left),
        "V_for_X_bars_direction_kN": max(V_top, V_bottom),
    }


def punching_shear_demand(
    piles: List[PilePoint],
    geom: Geometry,
    d_avg_mm: float,
    Pu_total_kN: float,
) -> Dict[str, float]:
    """
    Simplified punching shear around loaded area at d/2.

    In pile caps, pile reactions inside the critical perimeter reduce punching demand.
    This uses a rectangular perimeter around the column/pedestal.
    """
    loaded_bx = geom.pedestal_bx_mm if geom.use_pedestal_for_shear else geom.column_bx_mm
    loaded_by = geom.pedestal_by_mm if geom.use_pedestal_for_shear else geom.column_by_mm

    x_lim = loaded_bx / 2.0 + d_avg_mm / 2.0
    y_lim = loaded_by / 2.0 + d_avg_mm / 2.0

    inside = []
    outside = []
    for p in piles:
        if abs(p.x_mm) <= x_lim and abs(p.y_mm) <= y_lim:
            inside.append(p)
        else:
            outside.append(p)

    R_inside = sum(max(p.reaction_kN, 0.0) for p in inside)
    R_outside = sum(max(p.reaction_kN, 0.0) for p in outside)

    Vu_by_column_minus_inside = max(Pu_total_kN - R_inside, 0.0)
    Vu_by_outside = R_outside
    Vu = min(max(Vu_by_column_minus_inside, Vu_by_outside), max(Pu_total_kN, R_outside))

    bo = 2.0 * (loaded_bx + d_avg_mm + loaded_by + d_avg_mm)

    return {
        "Vu_punch_kN": Vu,
        "R_inside_kN": R_inside,
        "R_outside_kN": R_outside,
        "bo_mm": bo,
        "x_crit_half_mm": x_lim,
        "y_crit_half_mm": y_lim,
        "piles_inside": ", ".join([p.label for p in inside]) if inside else "-",
    }


def stm_advisory(
    piles: List[PilePoint],
    geom: Geometry,
    mat: Material,
    d_avg_mm: float,
) -> pd.DataFrame:
    """
    Practical MacGregor-style STM advisory table.

    The load path is idealized from loaded area centroid to each pile head.
    For each pile:
        a = horizontal distance from column/pedestal face to pile center, not less than small number.
        theta = atan(d / a)
        T_est = R * cot(theta), an indicative tie force along bottom reinforcement direction.

    The vector is split into x/y components based on pile plan location.
    """
    loaded_bx = geom.pedestal_bx_mm if geom.use_pedestal_for_shear else geom.column_bx_mm
    loaded_by = geom.pedestal_by_mm if geom.use_pedestal_for_shear else geom.column_by_mm

    rows = []
    for p in piles:
        dx_out = max(abs(p.x_mm) - loaded_bx / 2.0, 0.0)
        dy_out = max(abs(p.y_mm) - loaded_by / 2.0, 0.0)
        a = math.sqrt(dx_out**2 + dy_out**2)
        if a < 1.0:
            a = max(0.5 * max(loaded_bx, loaded_by), 1.0)
        theta_rad = math.atan2(d_avg_mm, a)
        theta_deg = math.degrees(theta_rad)
        cot_theta = 1.0 / max(math.tan(theta_rad), 1e-9)
        R = max(p.reaction_kN, 0.0)
        T = R * cot_theta
        ux = dx_out / max(a, 1e-9)
        uy = dy_out / max(a, 1e-9)
        Tx = T * ux
        Ty = T * uy

        if theta_deg < 25.0:
            note = "Shallow strut; increase depth or revise layout"
            stt = "WARN"
        elif theta_deg > 65.0:
            note = "Steep strut; check node geometry/detailing"
            stt = "WARN"
        else:
            note = "Practical STM angle range"
            stt = "OK"

        rows.append(
            {
                "Pile": p.label,
                "R (kN)": R,
                "a horizontal (mm)": a,
                "theta (deg)": theta_deg,
                "T tie estimate (kN)": T,
                "Tx component (kN)": Tx,
                "Ty component (kN)": Ty,
                "Status": stt,
                "Note": note,
            }
        )

    return pd.DataFrame(rows)


def design_all(state: DesignState, use_self_weight: bool = True) -> Dict[str, Any]:
    geom = state.geometry
    mat = state.material
    reinf = state.reinforcement

    cap_x = state.cap_length_x_mm
    cap_y = state.cap_width_y_mm
    d_x, d_y = effective_depths(geom, reinf)
    d_avg = 0.5 * (d_x + d_y)

    Pu_total = state.loadcase.Pu_kN + (state.self_weight_kN if use_self_weight else 0.0)

    moment_demands = design_moments_from_pile_reactions(state.piles, geom, cap_x, cap_y)
    shear_demands = one_way_shear_demands(state.piles, geom, d_x, d_y)
    punch_demand = punching_shear_demand(state.piles, geom, d_avg, Pu_total)

    # Flexure:
    # X bars resist cantilever action along y; section width is cap_x.
    # Y bars resist cantilever action along x; section width is cap_y.
    flex_x = flexural_As_required(
        moment_demands["M_for_X_bars_kNm"], cap_x, d_x, mat.fc_MPa, mat.fy_MPa, mat.phi_flexure
    )
    flex_y = flexural_As_required(
        moment_demands["M_for_Y_bars_kNm"], cap_y, d_y, mat.fc_MPa, mat.fy_MPa, mat.phi_flexure
    )

    sp_x = spacing_for_As(
        flex_x["As_req_mm2"],
        reinf.main_bar_x,
        cap_y,
        reinf.preferred_spacing_step_mm,
        s_min_mm=75.0,
        s_max_mm=min(300.0, 3.0 * geom.cap_thickness_mm),
    )
    sp_y = spacing_for_As(
        flex_y["As_req_mm2"],
        reinf.main_bar_y,
        cap_x,
        reinf.preferred_spacing_step_mm,
        s_min_mm=75.0,
        s_max_mm=min(300.0, 3.0 * geom.cap_thickness_mm),
    )

    cap_xbars = flexural_capacity(sp_x["As_prov_mm2"], cap_x, d_x, mat.fc_MPa, mat.fy_MPa, mat.phi_flexure)
    cap_ybars = flexural_capacity(sp_y["As_prov_mm2"], cap_y, d_y, mat.fc_MPa, mat.fy_MPa, mat.phi_flexure)

    # One-way shear capacities:
    # For shear in x direction, critical section length across full y dimension.
    ow_ybars = one_way_shear_capacity(cap_y, d_y, mat.fc_MPa, mat.lambda_c, mat.phi_shear)
    ow_xbars = one_way_shear_capacity(cap_x, d_x, mat.fc_MPa, mat.lambda_c, mat.phi_shear)

    # Punching capacity:
    punch_cap = two_way_shear_capacity(
        punch_demand["bo_mm"], d_avg, mat.fc_MPa, mat.lambda_c, mat.phi_shear
    )

    # Bearing at column/pedestal:
    loaded_area = (
        (geom.pedestal_bx_mm if geom.use_pedestal_for_shear else geom.column_bx_mm)
        * (geom.pedestal_by_mm if geom.use_pedestal_for_shear else geom.column_by_mm)
    )
    bearing = concrete_bearing_capacity(loaded_area, mat.fc_MPa, mat.phi_bearing)

    # Development lengths:
    ld_x = development_length_tension_estimate(
        bar_diameter(reinf.main_bar_x), mat.fc_MPa, mat.fy_MPa, mat.lambda_c, psi_t=1.0, psi_e=1.0, psi_s=1.0
    )
    ld_y = development_length_tension_estimate(
        bar_diameter(reinf.main_bar_y), mat.fc_MPa, mat.fy_MPa, mat.lambda_c, psi_t=1.0, psi_e=1.0, psi_s=1.0
    )

    # STM advisory:
    stm = stm_advisory(state.piles, geom, mat, d_avg)
    T_x_total = stm["Tx component (kN)"].sum() if not stm.empty else 0.0
    T_y_total = stm["Ty component (kN)"].sum() if not stm.empty else 0.0
    As_stm_x = T_y_total * 1000.0 / max(mat.phi_stm_tie * mat.fy_MPa, 1e-9)  # X bars cross y action
    As_stm_y = T_x_total * 1000.0 / max(mat.phi_stm_tie * mat.fy_MPa, 1e-9)  # Y bars cross x action

    # Check results:
    checks = []
    checks.append(
        CheckResult(
            "Pile compression",
            demand=max([p.reaction_kN for p in state.piles] + [0.0]),
            capacity=geom.pile_capacity_comp_kN,
            ratio=max([p.reaction_kN for p in state.piles] + [0.0]) / max(geom.pile_capacity_comp_kN, 1e-9),
            unit="kN",
            status=status_from_ratio(max([p.reaction_kN for p in state.piles] + [0.0]) / max(geom.pile_capacity_comp_kN, 1e-9)),
            note="Maximum compression pile reaction.",
        )
    )
    checks.append(
        CheckResult(
            "Pile tension/uplift",
            demand=max([-p.reaction_kN for p in state.piles] + [0.0]),
            capacity=geom.pile_capacity_tension_kN,
            ratio=max([-p.reaction_kN for p in state.piles] + [0.0]) / max(geom.pile_capacity_tension_kN, 1e-9),
            unit="kN",
            status=status_from_ratio(max([-p.reaction_kN for p in state.piles] + [0.0]) / max(geom.pile_capacity_tension_kN, 1e-9)),
            note="Maximum uplift reaction.",
        )
    )
    checks.append(
        CheckResult(
            "Flexure - X bars",
            demand=moment_demands["M_for_X_bars_kNm"],
            capacity=cap_xbars["phiMn_kNm"],
            ratio=moment_demands["M_for_X_bars_kNm"] / max(cap_xbars["phiMn_kNm"], 1e-9),
            unit="kN-m",
            status=status_from_ratio(moment_demands["M_for_X_bars_kNm"] / max(cap_xbars["phiMn_kNm"], 1e-9)),
            note="Bottom bars parallel to X; cantilever action in Y.",
        )
    )
    checks.append(
        CheckResult(
            "Flexure - Y bars",
            demand=moment_demands["M_for_Y_bars_kNm"],
            capacity=cap_ybars["phiMn_kNm"],
            ratio=moment_demands["M_for_Y_bars_kNm"] / max(cap_ybars["phiMn_kNm"], 1e-9),
            unit="kN-m",
            status=status_from_ratio(moment_demands["M_for_Y_bars_kNm"] / max(cap_ybars["phiMn_kNm"], 1e-9)),
            note="Bottom bars parallel to Y; cantilever action in X.",
        )
    )
    checks.append(
        CheckResult(
            "One-way shear - section normal to Y",
            demand=shear_demands["V_for_X_bars_direction_kN"],
            capacity=ow_xbars["phiVc_kN"],
            ratio=shear_demands["V_for_X_bars_direction_kN"] / max(ow_xbars["phiVc_kN"], 1e-9),
            unit="kN",
            status=status_from_ratio(shear_demands["V_for_X_bars_direction_kN"] / max(ow_xbars["phiVc_kN"], 1e-9)),
            note="Critical section at d from loaded area face; full cap width used.",
        )
    )
    checks.append(
        CheckResult(
            "One-way shear - section normal to X",
            demand=shear_demands["V_for_Y_bars_direction_kN"],
            capacity=ow_ybars["phiVc_kN"],
            ratio=shear_demands["V_for_Y_bars_direction_kN"] / max(ow_ybars["phiVc_kN"], 1e-9),
            unit="kN",
            status=status_from_ratio(shear_demands["V_for_Y_bars_direction_kN"] / max(ow_ybars["phiVc_kN"], 1e-9)),
            note="Critical section at d from loaded area face; full cap width used.",
        )
    )
    checks.append(
        CheckResult(
            "Two-way punching shear",
            demand=punch_demand["Vu_punch_kN"],
            capacity=punch_cap["phiVc_kN"],
            ratio=punch_demand["Vu_punch_kN"] / max(punch_cap["phiVc_kN"], 1e-9),
            unit="kN",
            status=status_from_ratio(punch_demand["Vu_punch_kN"] / max(punch_cap["phiVc_kN"], 1e-9)),
            note=f"bo={punch_demand['bo_mm']:.0f} mm; piles inside perimeter: {punch_demand['piles_inside']}",
        )
    )
    checks.append(
        CheckResult(
            "Column/pedestal bearing",
            demand=Pu_total,
            capacity=bearing["phiPn_kN"],
            ratio=Pu_total / max(bearing["phiPn_kN"], 1e-9),
            unit="kN",
            status=status_from_ratio(Pu_total / max(bearing["phiPn_kN"], 1e-9)),
            note="Simplified concrete bearing under loaded area.",
        )
    )

    lateral_demand = max(abs(state.loadcase.Vux_kN), abs(state.loadcase.Vuy_kN), abs(state.loadcase.Tuz_kNm))
    if lateral_demand > 1e-9:
        checks.append(
            CheckResult(
                "Lateral/torsion design scope",
                demand=lateral_demand,
                capacity=0.0,
                ratio=np.inf,
                unit="kN or kN-m",
                status="FAIL",
                note="Vux, Vuy, and Tuz are imported/stored but require separate lateral pile group, pile head, and torsion design checks.",
            )
        )

    return {
        "cap_x_mm": cap_x,
        "cap_y_mm": cap_y,
        "d_x_mm": d_x,
        "d_y_mm": d_y,
        "d_avg_mm": d_avg,
        "Pu_total_kN": Pu_total,
        "moment_demands": moment_demands,
        "shear_demands": shear_demands,
        "punch_demand": punch_demand,
        "flex_x": flex_x,
        "flex_y": flex_y,
        "spacing_x": sp_x,
        "spacing_y": sp_y,
        "cap_xbars": cap_xbars,
        "cap_ybars": cap_ybars,
        "one_way_x": ow_xbars,
        "one_way_y": ow_ybars,
        "punch_cap": punch_cap,
        "bearing": bearing,
        "ld_x_mm": ld_x,
        "ld_y_mm": ld_y,
        "stm": stm,
        "As_stm_x_mm2": As_stm_x,
        "As_stm_y_mm2": As_stm_y,
        "checks": checks,
    }


# =============================================================================
# SAP2000 IMPORT
# =============================================================================

def read_uploaded_table(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)
        st.warning("Unsupported file type. Please upload CSV or XLSX Excel.")
        return None
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return None


def find_matching_column(columns: List[str], aliases: List[str]) -> Optional[str]:
    lower_map = {c.lower().strip(): c for c in columns}
    for a in aliases:
        key = a.lower().strip()
        if key in lower_map:
            return lower_map[key]
    # More flexible contains search
    for c in columns:
        normalized = c.lower().replace(" ", "").replace("_", "")
        for a in aliases:
            aa = a.lower().replace(" ", "").replace("_", "")
            if normalized == aa or normalized.endswith(aa):
                return c
    return None


def sap2000_column_map(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    return {
        "joint": find_matching_column(cols, ["Joint", "JointElm", "Point", "PointElm", "Unique Name"]),
        "case": find_matching_column(cols, ["OutputCase", "Load Case", "Case", "LoadCase", "Combo", "Combination"]),
        "step": find_matching_column(cols, ["StepType", "Step", "Step Number", "StepNum"]),
        "F1": find_matching_column(cols, ["F1", "FX", "GlobalFX", "Reaction F1"]),
        "F2": find_matching_column(cols, ["F2", "FY", "GlobalFY", "Reaction F2"]),
        "F3": find_matching_column(cols, ["F3", "FZ", "GlobalFZ", "Reaction F3"]),
        "M1": find_matching_column(cols, ["M1", "MX", "GlobalMX", "Reaction M1"]),
        "M2": find_matching_column(cols, ["M2", "MY", "GlobalMY", "Reaction M2"]),
        "M3": find_matching_column(cols, ["M3", "MZ", "GlobalMZ", "Reaction M3"]),
    }


def build_loadcase_from_sap_row(
    row: pd.Series,
    cmap: Dict[str, Optional[str]],
    vertical_col: str,
    mx_col: str,
    my_col: str,
    vx_col: Optional[str],
    vy_col: Optional[str],
    torsion_col: Optional[str],
    vertical_sign: float,
    moment_sign: float,
    name: str,
) -> LoadCase:
    Pu = vertical_sign * safe_float(row.get(vertical_col, 0.0))
    Mux = moment_sign * safe_float(row.get(mx_col, 0.0))
    Muy = moment_sign * safe_float(row.get(my_col, 0.0))
    Vux = safe_float(row.get(vx_col, 0.0)) if vx_col else 0.0
    Vuy = safe_float(row.get(vy_col, 0.0)) if vy_col else 0.0
    Tuz = safe_float(row.get(torsion_col, 0.0)) if torsion_col else 0.0
    return LoadCase(name=name, Pu_kN=Pu, Mux_kNm=Mux, Muy_kNm=Muy, Vux_kN=Vux, Vuy_kN=Vuy, Tuz_kNm=Tuz)


def sap_envelope_table(df: pd.DataFrame, cmap: Dict[str, Optional[str]]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    case_col = cmap.get("case")
    if case_col and case_col in df.columns:
        group_cols = [case_col]
    else:
        group_cols = []

    numeric_cols = [c for c in [cmap.get("F1"), cmap.get("F2"), cmap.get("F3"), cmap.get("M1"), cmap.get("M2"), cmap.get("M3")] if c]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    if not numeric_cols:
        return pd.DataFrame()

    if group_cols:
        out_rows = []
        for key, g in df.groupby(group_cols):
            row = {"Case": key if isinstance(key, str) else key[0]}
            for c in numeric_cols:
                vals = pd.to_numeric(g[c], errors="coerce")
                if vals.dropna().empty:
                    continue
                idx_abs = vals.abs().idxmax()
                row[f"{c} abs max"] = vals.loc[idx_abs]
                row[f"{c} max"] = vals.max()
                row[f"{c} min"] = vals.min()
            out_rows.append(row)
        return pd.DataFrame(out_rows)
    else:
        row = {"Case": "All rows"}
        for c in numeric_cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.dropna().empty:
                continue
            idx_abs = vals.abs().idxmax()
            row[f"{c} abs max"] = vals.loc[idx_abs]
            row[f"{c} max"] = vals.max()
            row[f"{c} min"] = vals.min()
        return pd.DataFrame([row])


# =============================================================================
# DRAWING FUNCTIONS
# =============================================================================

def add_dim_line(ax, p1, p2, text, offset=(0, 0), text_offset=(0, 0), fontsize=8):
    x1, y1 = p1
    x2, y2 = p2
    ox, oy = offset
    tx, ty = text_offset
    ax.plot([x1 + ox, x2 + ox], [y1 + oy, y2 + oy], color="black", lw=0.8)
    ax.add_patch(FancyArrowPatch((x1 + ox, y1 + oy), (x1, y1), arrowstyle="-", mutation_scale=6, lw=0.6))
    ax.add_patch(FancyArrowPatch((x2 + ox, y2 + oy), (x2, y2), arrowstyle="-", mutation_scale=6, lw=0.6))
    ax.text((x1 + x2) / 2 + ox + tx, (y1 + y2) / 2 + oy + ty, text, ha="center", va="center", fontsize=fontsize)


def plot_plan(state: DesignState, results: Dict[str, Any], show_rebar: bool = True, show_reactions: bool = True):
    geom = state.geometry
    reinf = state.reinforcement
    cap_x = state.cap_length_x_mm
    cap_y = state.cap_width_y_mm

    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    ax.set_aspect("equal", adjustable="box")

    # Cap boundary
    ax.add_patch(Rectangle((-cap_x / 2, -cap_y / 2), cap_x, cap_y, fill=False, lw=2.0))
    ax.text(-cap_x / 2, cap_y / 2 + 170, f"PILE CAP {cap_x:.0f} x {cap_y:.0f} x {geom.cap_thickness_mm:.0f} mm", fontsize=10, weight="bold")

    # Column/pedestal
    ax.add_patch(Rectangle((-geom.column_bx_mm / 2, -geom.column_by_mm / 2), geom.column_bx_mm, geom.column_by_mm, fill=False, lw=1.8, linestyle="-"))
    ax.text(0, 0, "COLUMN", ha="center", va="center", fontsize=8)

    if geom.use_pedestal_for_shear and (geom.pedestal_bx_mm != geom.column_bx_mm or geom.pedestal_by_mm != geom.column_by_mm):
        ax.add_patch(Rectangle((-geom.pedestal_bx_mm / 2, -geom.pedestal_by_mm / 2), geom.pedestal_bx_mm, geom.pedestal_by_mm, fill=False, lw=1.2, linestyle="--"))
        ax.text(0, -geom.pedestal_by_mm / 2 - 100, "PEDESTAL / LOADED AREA", ha="center", va="top", fontsize=7)

    # Critical punching perimeter
    pdm = results.get("punch_demand", {})
    if pdm:
        xlim = pdm["x_crit_half_mm"]
        ylim = pdm["y_crit_half_mm"]
        ax.add_patch(Rectangle((-xlim, -ylim), 2 * xlim, 2 * ylim, fill=False, lw=1.0, linestyle=":", alpha=0.8))
        ax.text(xlim + 50, ylim + 50, "d/2 punching perimeter", fontsize=7, va="bottom")

    # Piles
    for p in state.piles:
        ax.add_patch(Circle((p.x_mm, p.y_mm), p.diameter_mm / 2, fill=False, lw=1.6))
        ax.plot(p.x_mm, p.y_mm, marker="x", ms=5)
        label = f"{p.label}"
        if show_reactions:
            label += f"\nR={p.reaction_kN:.0f} kN"
        ax.text(p.x_mm, p.y_mm, label, ha="center", va="center", fontsize=7)

    # Rebar schematic
    if show_rebar:
        cover = geom.side_cover_mm
        sx = results["spacing_x"]["spacing_use_mm"]
        sy = results["spacing_y"]["spacing_use_mm"]

        # X bars: horizontal lines across cap, spaced along y
        y_values = np.arange(-cap_y / 2 + cover, cap_y / 2 - cover + 1, sx)
        for y in y_values:
            ax.plot([-cap_x / 2 + cover, cap_x / 2 - cover], [y, y], lw=0.35, alpha=0.5)
        # Y bars: vertical lines across cap, spaced along x
        x_values = np.arange(-cap_x / 2 + cover, cap_x / 2 - cover + 1, sy)
        for x in x_values:
            ax.plot([x, x], [-cap_y / 2 + cover, cap_y / 2 - cover], lw=0.35, alpha=0.5)

        ax.text(
            -cap_x / 2 + 80,
            -cap_y / 2 + 80,
            f"BOT. X: {reinf.main_bar_x}@{sx:.0f}\nBOT. Y: {reinf.main_bar_y}@{sy:.0f}",
            fontsize=8,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.75),
        )

    # Axes
    axis_len = min(cap_x, cap_y) * 0.18
    ax.arrow(0, 0, axis_len, 0, head_width=60, head_length=80, length_includes_head=True)
    ax.arrow(0, 0, 0, axis_len, head_width=60, head_length=80, length_includes_head=True)
    ax.text(axis_len + 80, 0, "X", va="center")
    ax.text(0, axis_len + 80, "Y", ha="center")

    # Dimensions
    dim_offset_y = -cap_y / 2 - 350
    add_dim_line(ax, (-cap_x / 2, -cap_y / 2), (cap_x / 2, -cap_y / 2), f"{cap_x:.0f}", offset=(0, -250), text_offset=(0, -35))
    add_dim_line(ax, (cap_x / 2, -cap_y / 2), (cap_x / 2, cap_y / 2), f"{cap_y:.0f}", offset=(250, 0), text_offset=(55, 0))

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True, alpha=0.25, lw=0.5)
    margin = max(800, 0.12 * max(cap_x, cap_y))
    ax.set_xlim(-cap_x / 2 - margin, cap_x / 2 + margin)
    ax.set_ylim(-cap_y / 2 - margin, cap_y / 2 + margin)
    ax.set_title("Drawing-style Plan View", fontsize=12, weight="bold")
    return fig


def plot_elevation(state: DesignState, results: Dict[str, Any], direction: str = "X"):
    geom = state.geometry
    reinf = state.reinforcement
    cap_x = state.cap_length_x_mm if direction.upper() == "X" else state.cap_width_y_mm
    cap_y = state.cap_width_y_mm if direction.upper() == "X" else state.cap_length_x_mm
    thickness = geom.cap_thickness_mm

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_aspect("equal", adjustable="box")
    ax.add_patch(Rectangle((-cap_x / 2, 0), cap_x, thickness, fill=False, lw=2.0))
    ax.text(-cap_x / 2, thickness + 120, f"ELEVATION {direction.upper()}-DIRECTION", fontsize=10, weight="bold")

    # Column projection
    col_width = geom.column_bx_mm if direction.upper() == "X" else geom.column_by_mm
    ax.add_patch(Rectangle((-col_width / 2, thickness), col_width, thickness * 0.35, fill=False, lw=1.5))
    ax.text(0, thickness * 1.18, "COLUMN", ha="center", va="center", fontsize=8)

    # Piles along selected axis shown if near centerline in perpendicular axis.
    if direction.upper() == "X":
        axis_coord = [(p.x_mm, p.y_mm, p) for p in state.piles]
        perp_tol = max(geom.spacing_y_mm * 0.55, geom.pile_diameter_mm)
    else:
        axis_coord = [(p.y_mm, p.x_mm, p) for p in state.piles]
        perp_tol = max(geom.spacing_x_mm * 0.55, geom.pile_diameter_mm)

    for x, perp, p in axis_coord:
        if abs(perp) <= perp_tol:
            ax.add_patch(Rectangle((x - p.diameter_mm / 2, -thickness * 0.50), p.diameter_mm, thickness * 0.50, fill=False, lw=1.2))
            ax.text(x, -thickness * 0.25, p.label, ha="center", va="center", fontsize=7)

    # Bottom and top bars schematic
    cover_bot = geom.bottom_cover_mm
    cover_top = geom.top_cover_mm
    ax.plot([-cap_x / 2 + geom.side_cover_mm, cap_x / 2 - geom.side_cover_mm], [cover_bot, cover_bot], lw=2.0)
    ax.plot([-cap_x / 2 + geom.side_cover_mm, cap_x / 2 - geom.side_cover_mm], [thickness - cover_top, thickness - cover_top], lw=1.0, linestyle="--")

    bot_label = f"BOT. MAIN {reinf.main_bar_x if direction.upper() == 'X' else reinf.main_bar_y}"
    ax.text(-cap_x / 2 + 80, cover_bot + 60, bot_label, fontsize=8)
    ax.text(-cap_x / 2 + 80, thickness - cover_top - 100, f"TOP TEMP. {reinf.top_bar}@{reinf.top_spacing_mm:.0f}", fontsize=8)

    # Depth dimension
    add_dim_line(ax, (cap_x / 2, 0), (cap_x / 2, thickness), f"h = {thickness:.0f}", offset=(250, 0), text_offset=(70, 0))
    add_dim_line(ax, (-cap_x / 2, 0), (cap_x / 2, 0), f"{cap_x:.0f}", offset=(0, -220), text_offset=(0, -40))

    ax.set_xlim(-cap_x / 2 - 700, cap_x / 2 + 850)
    ax.set_ylim(-thickness * 0.65, thickness * 1.55)
    ax.axis("off")
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# REPORT GENERATION
# =============================================================================

def checks_to_dataframe(checks: List[CheckResult]) -> pd.DataFrame:
    rows = []
    for c in checks:
        rows.append(
            {
                "Check": c.name,
                "Demand": c.demand,
                "Capacity": c.capacity,
                "Ratio": c.ratio,
                "Unit": c.unit,
                "Status": c.status,
                "Note": c.note,
            }
        )
    return pd.DataFrame(rows)



def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows._"

    display = df.copy()
    for col in display.columns:
        display[col] = display[col].map(lambda value: fmt(value, 3) if isinstance(value, float) else str(value))

    columns = [str(col) for col in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[col]).replace("\n", " ").replace("|", "\\|") for col in display.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_markdown_report(state: DesignState, results: Dict[str, Any]) -> str:
    checks_df = checks_to_dataframe(results["checks"])
    pile_df = pile_group_result_table(state.piles, state.geometry)
    stm_df = results["stm"]

    lines = []
    lines.append(f"# {APP_TITLE} Report")
    lines.append("")
    lines.append("## Design Basis")
    lines.append(f"- Design approach: ACI 318-25 style strength design plus practical MacGregor-style pile-cap load-path checks.")
    lines.append("- This report is a calculation aid and must be verified by the responsible engineer.")
    lines.append("")
    lines.append("## Materials")
    lines.append(f"- fc' = {state.material.fc_MPa:.1f} MPa")
    lines.append(f"- fy = {state.material.fy_MPa:.1f} MPa")
    lines.append(f"- lambda = {state.material.lambda_c:.2f}")
    lines.append(f"- phi flexure = {state.material.phi_flexure:.2f}")
    lines.append(f"- phi shear = {state.material.phi_shear:.2f}")
    lines.append("")
    lines.append("## Geometry")
    lines.append(f"- Number of piles = {state.geometry.n_piles}")
    lines.append(f"- Pile diameter = {state.geometry.pile_diameter_mm:.0f} mm")
    lines.append(f"- Cap size = {state.cap_length_x_mm:.0f} x {state.cap_width_y_mm:.0f} x {state.geometry.cap_thickness_mm:.0f} mm")
    lines.append(f"- Column = {state.geometry.column_bx_mm:.0f} x {state.geometry.column_by_mm:.0f} mm")
    lines.append(f"- Effective depth X bars = {results['d_x_mm']:.0f} mm")
    lines.append(f"- Effective depth Y bars = {results['d_y_mm']:.0f} mm")
    lines.append("")
    lines.append("## Loads")
    lc = state.loadcase
    lines.append(f"- Load case = {lc.name}")
    lines.append(f"- Pu = {lc.Pu_kN:.2f} kN")
    lines.append(f"- Mux = {lc.Mux_kNm:.2f} kN-m")
    lines.append(f"- Muy = {lc.Muy_kNm:.2f} kN-m")
    lines.append(f"- Self weight included in distribution = {state.self_weight_kN:.2f} kN")
    lines.append(f"- Total Pu used in checks = {results['Pu_total_kN']:.2f} kN")
    lines.append("")
    lines.append("## Main Reinforcement")
    lines.append(f"- Bottom X bars: {state.reinforcement.main_bar_x} @ {results['spacing_x']['spacing_use_mm']:.0f} mm")
    lines.append(f"- Bottom Y bars: {state.reinforcement.main_bar_y} @ {results['spacing_y']['spacing_use_mm']:.0f} mm")
    lines.append(f"- Development estimate X bars = {results['ld_x_mm']:.0f} mm")
    lines.append(f"- Development estimate Y bars = {results['ld_y_mm']:.0f} mm")
    lines.append("")
    lines.append("## Check Summary")
    lines.append(dataframe_to_markdown_table(checks_df))
    lines.append("")
    lines.append("## Pile Reaction Table")
    lines.append(dataframe_to_markdown_table(pile_df))
    lines.append("")
    lines.append("## STM Advisory")
    lines.append(dataframe_to_markdown_table(stm_df))
    lines.append("")
    lines.append("## Notes")
    lines.append("- Verify ACI 318-25 clauses directly for final design.")
    lines.append("- For pile caps with small shear span-to-depth ratio, use a full strut-and-tie model.")
    lines.append("- For seismic regions, pile anchorage, confinement, tie forces, shear friction, and ductile detailing may govern.")
    lines.append("- Check geotechnical pile compression, uplift, lateral load, settlement, group effects, and construction tolerances.")
    return "\n".join(lines)


def pdf_safe(value: Any) -> str:
    text_value = str(value)
    replacements = {
        "φ": "phi",
        "λ": "lambda",
        "²": "2",
        "³": "3",
        "—": "-",
        "–": "-",
        "×": "x",
        "°": " deg",
        "✅": "PASS",
        "⚠️": "WARN",
        "❌": "FAIL",
    }
    for src, dst in replacements.items():
        text_value = text_value.replace(src, dst)
    return text_value.encode("latin-1", "replace").decode("latin-1")


def pdf_add_table(pdf, df: pd.DataFrame, columns: List[str], widths: List[float], max_rows: int = 40) -> None:
    pdf.set_font("Arial", "B", 8)
    for col, width in zip(columns, widths):
        pdf.cell(width, 6, pdf_safe(col), border=1)
    pdf.ln()
    pdf.set_font("Arial", "", 8)
    for _, row in df.head(max_rows).iterrows():
        for col, width in zip(columns, widths):
            value = row.get(col, "")
            if isinstance(value, float):
                value = fmt(value, 3 if abs(value) < 10 else 1)
            pdf.cell(width, 6, pdf_safe(value)[:32], border=1)
        pdf.ln()



def make_simple_pdf(lines: List[str]) -> bytes:
    escaped_lines = []
    for line in lines:
        safe = pdf_safe(line).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        escaped_lines.append(safe[:115])

    stream_parts = ["BT", "/F1 10 Tf", "50 790 Td", "14 TL"]
    first = True
    for line in escaped_lines:
        if not first:
            stream_parts.append("T*")
        stream_parts.append(f"({line}) Tj")
        first = False
    stream_parts.append("ET")
    stream = "\n".join(stream_parts).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


def make_fallback_calculation_pdf(state: DesignState, results: Dict[str, Any]) -> bytes:
    lines = [
        APP_TITLE,
        "Calculation report - preliminary design aid",
        "",
        "Input Summary",
        f"Load case: {state.loadcase.name}",
        f"Pu: {state.loadcase.Pu_kN:.2f} kN",
        f"Mux: {state.loadcase.Mux_kNm:.2f} kN-m",
        f"Muy: {state.loadcase.Muy_kNm:.2f} kN-m",
        f"Vux/Vuy/Tuz: {state.loadcase.Vux_kN:.2f} kN / {state.loadcase.Vuy_kN:.2f} kN / {state.loadcase.Tuz_kNm:.2f} kN-m",
        f"fc': {state.material.fc_MPa:.2f} MPa, fy: {state.material.fy_MPa:.2f} MPa",
        f"Cap: {state.cap_length_x_mm:.0f} x {state.cap_width_y_mm:.0f} x {state.geometry.cap_thickness_mm:.0f} mm",
        "",
        "Design Checks",
    ]
    for check in results["checks"]:
        lines.append(f"{check.name}: D={fmt(check.demand, 2)} {check.unit}, C={fmt(check.capacity, 2)}, Ratio={fmt(check.ratio, 3)}, {check.status}")
    lines.extend(["", "Pile Reactions"])
    for pile in state.piles:
        lines.append(f"{pile.label}: x={pile.x_mm:.0f} mm, y={pile.y_mm:.0f} mm, R={pile.reaction_kN:.2f} kN")
    lines.extend(
        [
            "",
            "Reinforcement Summary",
            f"X bars: {state.reinforcement.main_bar_x} @ {results['spacing_x']['spacing_use_mm']:.0f} mm, As req {results['flex_x']['As_req_mm2']:.0f} mm2",
            f"Y bars: {state.reinforcement.main_bar_y} @ {results['spacing_y']['spacing_use_mm']:.0f} mm, As req {results['flex_y']['As_req_mm2']:.0f} mm2",
            "",
            "Formula Notes",
            "R_i = P/n + Mx*y_i/sum(y^2) + My*x_i/sum(x^2)",
            "Flexure: phi As fy (d - a/2) >= Mu",
            "One-way shear and punching use simplified ACI-style concrete expressions.",
            "Lateral load and torsion are flagged separately and need project-specific checks.",
        ]
    )
    return make_simple_pdf(lines)


def make_calculation_pdf(state: DesignState, results: Dict[str, Any]) -> bytes:
    if FPDF is None:
        return make_fallback_calculation_pdf(state, results)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 9, pdf_safe(APP_TITLE), ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(0, 5, pdf_safe("Calculation report - preliminary ACI-style pile-cap design aid. Verify final design against governing code, geotechnical report, and project specifications."))
    pdf.ln(2)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, "Input Summary", ln=True)
    pdf_add_table(
        pdf,
        pd.DataFrame(
            [
                {"Item": "Load case", "Value": state.loadcase.name, "Unit": ""},
                {"Item": "Pu", "Value": state.loadcase.Pu_kN, "Unit": "kN"},
                {"Item": "Mux", "Value": state.loadcase.Mux_kNm, "Unit": "kN-m"},
                {"Item": "Muy", "Value": state.loadcase.Muy_kNm, "Unit": "kN-m"},
                {"Item": "Vux", "Value": state.loadcase.Vux_kN, "Unit": "kN"},
                {"Item": "Vuy", "Value": state.loadcase.Vuy_kN, "Unit": "kN"},
                {"Item": "Tuz", "Value": state.loadcase.Tuz_kNm, "Unit": "kN-m"},
                {"Item": "fc'", "Value": state.material.fc_MPa, "Unit": "MPa"},
                {"Item": "fy", "Value": state.material.fy_MPa, "Unit": "MPa"},
                {"Item": "Cap size", "Value": f"{state.cap_length_x_mm:.0f} x {state.cap_width_y_mm:.0f} x {state.geometry.cap_thickness_mm:.0f}", "Unit": "mm"},
                {"Item": "Pile count", "Value": state.geometry.n_piles, "Unit": ""},
                {"Item": "Pile diameter", "Value": state.geometry.pile_diameter_mm, "Unit": "mm"},
            ]
        ),
        ["Item", "Value", "Unit"],
        [55, 85, 30],
    )
    pdf.ln(3)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, "Design Checks", ln=True)
    checks_df = checks_to_dataframe(results["checks"])
    pdf_add_table(pdf, checks_df, ["Check", "Demand", "Capacity", "Ratio", "Status"], [58, 30, 30, 25, 25])
    pdf.ln(3)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, "Pile Reactions", ln=True)
    pile_df = pile_group_result_table(state.piles, state.geometry)
    pdf_add_table(pdf, pile_df, ["Pile", "x (mm)", "y (mm)", "R, compression + (kN)", "Status"], [22, 28, 28, 55, 35])
    pdf.ln(3)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, "Reinforcement Summary", ln=True)
    reinf_df = pd.DataFrame(
        [
            {"Direction": "X bars", "Bar": state.reinforcement.main_bar_x, "As req": results["flex_x"]["As_req_mm2"], "Use spacing": results["spacing_x"]["spacing_use_mm"], "As prov": results["spacing_x"]["As_prov_mm2"], "phiMn": results["cap_xbars"]["phiMn_kNm"]},
            {"Direction": "Y bars", "Bar": state.reinforcement.main_bar_y, "As req": results["flex_y"]["As_req_mm2"], "Use spacing": results["spacing_y"]["spacing_use_mm"], "As prov": results["spacing_y"]["As_prov_mm2"], "phiMn": results["cap_ybars"]["phiMn_kNm"]},
        ]
    )
    pdf_add_table(pdf, reinf_df, ["Direction", "Bar", "As req", "Use spacing", "As prov", "phiMn"], [30, 22, 28, 30, 28, 28])

    pdf.add_page()
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, "Formula Notes", ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(
        0,
        5,
        pdf_safe(
            "Pile reaction: R_i = P/n + Mx*y_i/sum(y^2) + My*x_i/sum(x^2)\n"
            "Flexure: phi * As * fy * (d - a/2) >= Mu; a = As*fy/(0.85*fc'*b)\n"
            "One-way shear: phi Vc = phi * 0.17 * lambda * sqrt(fc') * b * d\n"
            "Two-way punching: phi Vc = phi * 0.33 * lambda * sqrt(fc') * bo * d\n"
            "STM advisory: theta = atan(d/a), T_est = R*cot(theta)\n\n"
            "Lateral load and torsion are flagged separately. Add project-specific lateral pile group and torsional checks before using those actions for final design."
        ),
    )
    data = pdf.output(dest="S")
    if isinstance(data, str):
        return data.encode("latin-1")
    return bytes(data)



def dataframe_download_button(df: pd.DataFrame, label: str, filename: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")


# =============================================================================
# INPUT UI
# =============================================================================

def sidebar_inputs() -> Tuple[Material, Geometry, Reinforcement, bool]:
    with st.sidebar:
        st.header("Design Settings")

        state_upload = st.file_uploader("Load saved editable state (.xlsx)", type=["xlsx"], key="saved_state_upload")
        loaded_payload = read_saved_state_xlsx(state_upload)
        if loaded_payload is not None:
            st.session_state["saved_state_payload"] = loaded_payload
            st.session_state["saved_piles_applied"] = False
            st.success("Saved state loaded. Inputs below now use workbook values.")

        st.caption("Use project-specific code amendments and verify final calculations with ACI 318-25.")

        include_self_weight = st.toggle("Include pile-cap self weight in pile reactions", value=saved_bool("settings", "include_self_weight", True))

        with st.expander("Materials", expanded=True):
            fc = st.number_input("Concrete fc' (MPa)", min_value=15.0, max_value=100.0, value=saved_float("material", "fc_MPa", 35.0), step=1.0)
            fy = st.number_input("Rebar fy (MPa)", min_value=275.0, max_value=700.0, value=saved_float("material", "fy_MPa", 500.0), step=10.0)
            lam = st.number_input("λ lightweight factor", min_value=0.50, max_value=1.00, value=saved_float("material", "lambda_c", 1.00), step=0.05)
            gamma = st.number_input("Concrete unit weight (kN/m³)", min_value=20.0, max_value=28.0, value=saved_float("material", "gamma_conc_kN_m3", 24.0), step=0.5)
            phi_flex = st.number_input("φ flexure", min_value=0.60, max_value=0.95, value=saved_float("material", "phi_flexure", 0.90), step=0.01)
            phi_shear = st.number_input("φ shear / punching", min_value=0.50, max_value=0.90, value=saved_float("material", "phi_shear", 0.75), step=0.01)
            phi_bearing = st.number_input("φ bearing", min_value=0.50, max_value=0.90, value=saved_float("material", "phi_bearing", 0.65), step=0.01)

        with st.expander("Pile and Cap Geometry", expanded=True):
            n_piles = st.number_input("Number of piles", min_value=2, max_value=12, value=saved_int("geometry", "n_piles", 4), step=1)
            pile_dia = st.number_input("Pile diameter / equivalent width (mm)", min_value=200.0, max_value=2500.0, value=saved_float("geometry", "pile_diameter_mm", 600.0), step=50.0)
            cap_h = st.number_input("Pile cap thickness h (mm)", min_value=300.0, max_value=4000.0, value=saved_float("geometry", "cap_thickness_mm", 1200.0), step=50.0)
            edge = st.number_input("Edge from pile edge to cap edge (mm)", min_value=100.0, max_value=2000.0, value=saved_float("geometry", "edge_from_pile_edge_mm", 600.0), step=50.0)
            sx = st.number_input("Typical pile spacing X (mm)", min_value=500.0, max_value=6000.0, value=saved_float("geometry", "spacing_x_mm", 1800.0), step=50.0)
            sy = st.number_input("Typical pile spacing Y (mm)", min_value=500.0, max_value=6000.0, value=saved_float("geometry", "spacing_y_mm", 1800.0), step=50.0)
            pile_comp = st.number_input("Allowable/design pile compression capacity (kN)", min_value=1.0, max_value=50000.0, value=saved_float("geometry", "pile_capacity_comp_kN", 1800.0), step=50.0)
            pile_ten = st.number_input("Allowable/design pile tension capacity (kN)", min_value=0.0, max_value=50000.0, value=saved_float("geometry", "pile_capacity_tension_kN", 400.0), step=25.0)

        with st.expander("Column / Pedestal", expanded=True):
            col_bx = st.number_input("Column size Bx (mm)", min_value=200.0, max_value=4000.0, value=saved_float("geometry", "column_bx_mm", 800.0), step=50.0)
            col_by = st.number_input("Column size By (mm)", min_value=200.0, max_value=4000.0, value=saved_float("geometry", "column_by_mm", 800.0), step=50.0)
            ped_bx = st.number_input("Pedestal / loaded area Bx (mm)", min_value=200.0, max_value=6000.0, value=saved_float("geometry", "pedestal_bx_mm", 800.0), step=50.0)
            ped_by = st.number_input("Pedestal / loaded area By (mm)", min_value=200.0, max_value=6000.0, value=saved_float("geometry", "pedestal_by_mm", 800.0), step=50.0)
            use_ped = st.toggle("Use pedestal size for shear/bearing checks", value=saved_bool("geometry", "use_pedestal_for_shear", True))

        with st.expander("Concrete Cover and Reinforcement", expanded=True):
            bot_cover = st.number_input("Bottom cover (mm)", min_value=40.0, max_value=300.0, value=saved_float("geometry", "bottom_cover_mm", 100.0), step=5.0)
            top_cover = st.number_input("Top cover (mm)", min_value=30.0, max_value=200.0, value=saved_float("geometry", "top_cover_mm", 75.0), step=5.0)
            side_cover = st.number_input("Side cover (mm)", min_value=30.0, max_value=200.0, value=saved_float("geometry", "side_cover_mm", 75.0), step=5.0)
            bars = list(BAR_DATABASE_MM.keys())
            main_x_default = saved_choice("reinforcement", "main_bar_x", "DB25", bars)
            main_x = st.selectbox("Bottom bars parallel X", bars, index=bars.index(main_x_default))
            main_y_default = saved_choice("reinforcement", "main_bar_y", "DB25", bars)
            main_y = st.selectbox("Bottom bars parallel Y", bars, index=bars.index(main_y_default))
            top_bar_default = saved_choice("reinforcement", "top_bar", "DB16", bars)
            top_bar = st.selectbox("Top temperature bars", bars, index=bars.index(top_bar_default))
            top_spacing = st.number_input("Top bar spacing (mm)", min_value=75.0, max_value=400.0, value=saved_float("reinforcement", "top_spacing_mm", 200.0), step=25.0)
            side_bar_default = saved_choice("reinforcement", "side_face_bar", "DB16", bars)
            side_bar = st.selectbox("Side face bars", bars, index=bars.index(side_bar_default))
            side_spacing = st.number_input("Side face bar spacing (mm)", min_value=100.0, max_value=400.0, value=saved_float("reinforcement", "side_face_spacing_mm", 250.0), step=25.0)

    mat = Material(
        fc_MPa=fc,
        fy_MPa=fy,
        lambda_c=lam,
        gamma_conc_kN_m3=gamma,
        phi_flexure=phi_flex,
        phi_shear=phi_shear,
        phi_bearing=phi_bearing,
    )
    geom = Geometry(
        n_piles=int(n_piles),
        pile_diameter_mm=pile_dia,
        pile_capacity_comp_kN=pile_comp,
        pile_capacity_tension_kN=pile_ten,
        cap_thickness_mm=cap_h,
        bottom_cover_mm=bot_cover,
        top_cover_mm=top_cover,
        side_cover_mm=side_cover,
        column_bx_mm=col_bx,
        column_by_mm=col_by,
        pedestal_bx_mm=ped_bx,
        pedestal_by_mm=ped_by,
        edge_from_pile_edge_mm=edge,
        spacing_x_mm=sx,
        spacing_y_mm=sy,
        use_pedestal_for_shear=use_ped,
    )
    reinf = Reinforcement(
        main_bar_x=main_x,
        main_bar_y=main_y,
        top_bar=top_bar,
        top_spacing_mm=top_spacing,
        side_face_bar=side_bar,
        side_face_spacing_mm=side_spacing,
    )
    return mat, geom, reinf, include_self_weight


def load_input_ui() -> LoadCase:
    st.subheader("Load Input")
    mode = st.radio("Load source", ["Manual input", "SAP2000 joint reaction import"], horizontal=True)

    if mode == "Manual input":
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Load case name", value=str(value_from_saved("loadcase", "name", "ULS-Manual")))
            Pu = st.number_input("Pu compression + (kN)", value=saved_float("loadcase", "Pu_kN", 5000.0), step=100.0)
        with c2:
            Mux = st.number_input("Mux about X (kN-m)", value=saved_float("loadcase", "Mux_kNm", 0.0), step=50.0)
            Muy = st.number_input("Muy about Y (kN-m)", value=saved_float("loadcase", "Muy_kNm", 0.0), step=50.0)
        with c3:
            Vux = st.number_input("Vux (kN)", value=saved_float("loadcase", "Vux_kN", 0.0), step=25.0)
            Vuy = st.number_input("Vuy (kN)", value=saved_float("loadcase", "Vuy_kN", 0.0), step=25.0)
            Tuz = st.number_input("Tuz (kN-m)", value=saved_float("loadcase", "Tuz_kNm", 0.0), step=25.0)
        return LoadCase(name=name, Pu_kN=Pu, Mux_kNm=Mux, Muy_kNm=Muy, Vux_kN=Vux, Vuy_kN=Vuy, Tuz_kNm=Tuz)

    uploaded = st.file_uploader("Upload SAP2000 joint reaction table (CSV/XLSX)", type=["csv", "xlsx"])
    df = read_uploaded_table(uploaded)
    if df is None or df.empty:
        st.info("Upload SAP2000 reaction output. Until then, manual default load is used.")
        return LoadCase()

    st.write("Preview")
    st.dataframe(df.head(30), use_container_width=True)

    cmap = sap2000_column_map(df)
    with st.expander("Detected SAP2000 columns", expanded=False):
        st.json(cmap)

    env = sap_envelope_table(df, cmap)
    if not env.empty:
        st.write("Quick envelope by case")
        st.dataframe(env, use_container_width=True)

    columns = list(df.columns)
    c1, c2, c3 = st.columns(3)
    with c1:
        vertical_default = cmap.get("F3") or columns[0]
        vertical_col = st.selectbox("Vertical reaction column", columns, index=columns.index(vertical_default) if vertical_default in columns else 0)
        vertical_sign = st.selectbox("Vertical sign multiplier", [1.0, -1.0], index=0, help="Use -1 if SAP support reaction is negative for compression.")
    with c2:
        mx_default = cmap.get("M1") or columns[0]
        my_default = cmap.get("M2") or columns[0]
        mx_col = st.selectbox("Moment about X column", columns, index=columns.index(mx_default) if mx_default in columns else 0)
        my_col = st.selectbox("Moment about Y column", columns, index=columns.index(my_default) if my_default in columns else 0)
        moment_sign = st.selectbox("Moment sign multiplier", [1.0, -1.0], index=0)
    with c3:
        vx_col = st.selectbox("Vx column", ["- none -"] + columns, index=0)
        vy_col = st.selectbox("Vy column", ["- none -"] + columns, index=0)
        torsion_col = st.selectbox("Torsion column", ["- none -"] + columns, index=0)

    selected_index = st.number_input("Use table row index for design", min_value=0, max_value=max(len(df) - 1, 0), value=0, step=1)
    row = df.iloc[int(selected_index)]

    case_name = "SAP2000"
    case_col = cmap.get("case")
    if case_col and case_col in df.columns:
        case_name += f" - {row.get(case_col)}"

    lc = build_loadcase_from_sap_row(
        row=row,
        cmap=cmap,
        vertical_col=vertical_col,
        mx_col=mx_col,
        my_col=my_col,
        vx_col=None if vx_col == "- none -" else vx_col,
        vy_col=None if vy_col == "- none -" else vy_col,
        torsion_col=None if torsion_col == "- none -" else torsion_col,
        vertical_sign=float(vertical_sign),
        moment_sign=float(moment_sign),
        name=case_name,
    )

    st.success(
        f"Selected load: Pu={lc.Pu_kN:.2f} kN, Mux={lc.Mux_kNm:.2f} kN-m, Muy={lc.Muy_kNm:.2f} kN-m"
    )
    return lc


def pile_layout_ui(geom: Geometry) -> List[PilePoint]:
    st.subheader("Pile Layout")
    st.caption("Generated coordinates follow the reference pile arrangement sketches. You can still edit coordinates directly; the app recenters the pile group at its centroid.")

    regenerate = st.button("Regenerate layout from template", use_container_width=False)

    key = "pile_layout_df"
    apply_saved_pile_layout_if_needed(geom.pile_diameter_mm)
    expected_n = int(geom.n_piles)
    template_changed = st.session_state.get("pile_layout_template_version") != PILE_LAYOUT_TEMPLATE_VERSION
    if regenerate or template_changed or key not in st.session_state or len(st.session_state[key]) != expected_n:
        piles = make_piles(expected_n, geom.spacing_x_mm, geom.spacing_y_mm, geom.pile_diameter_mm)
        st.session_state[key] = piles_to_dataframe(piles)
        st.session_state["pile_layout_template_version"] = PILE_LAYOUT_TEMPLATE_VERSION

    editable = st.data_editor(
        st.session_state[key],
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Pile": st.column_config.TextColumn("Pile"),
            "x_mm": st.column_config.NumberColumn("x (mm)", step=50.0, format="%.1f"),
            "y_mm": st.column_config.NumberColumn("y (mm)", step=50.0, format="%.1f"),
            "diameter_mm": st.column_config.NumberColumn("Diameter (mm)", step=50.0, format="%.1f"),
            "reaction_kN": st.column_config.NumberColumn("Reaction (kN)", disabled=True),
        },
    )
    st.session_state[key] = editable
    return dataframe_to_piles(editable, geom.pile_diameter_mm)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def render_header():
    st.markdown(
        f"""
        <div class="title-wrap">
            <div class="title-main">🏗️ {APP_TITLE}</div>
            <div class="title-sub">{APP_SUBTITLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-box">
        <b>Design responsibility note:</b> This tool is for preliminary and office-check calculations.
        Final design must be verified directly against ACI 318-25, project specifications, geotechnical criteria,
        seismic requirements, and local regulations.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(state: DesignState, results: Dict[str, Any]):
    checks = results["checks"]
    max_ratio = max([c.ratio for c in checks if np.isfinite(c.ratio)] + [0.0])
    fails = sum(1 for c in checks if c.status == "FAIL")
    near = sum(1 for c in checks if c.status == "NEAR")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall max D/C", f"{max_ratio:.2f}", "FAIL" if fails else ("NEAR" if near else "PASS"))
    c2.metric("Cap size X × Y", f"{state.cap_length_x_mm:.0f} × {state.cap_width_y_mm:.0f} mm")
    c3.metric("Thickness h", f"{state.geometry.cap_thickness_mm:.0f} mm")
    c4.metric("Pu used", f"{results['Pu_total_kN']:.0f} kN")
    c5.metric("Max pile R", f"{max(p.reaction_kN for p in state.piles):.0f} kN")



def format_display_dataframe(df: pd.DataFrame, formats: Dict[str, str]) -> pd.DataFrame:
    display = df.copy()
    for col, pattern in formats.items():
        if col in display.columns:
            display[col] = display[col].map(
                lambda value, p=pattern: p.format(value) if isinstance(value, (int, float, np.floating)) and np.isfinite(value) else value
            )
    return display


def render_check_cards(checks: List[CheckResult]):
    df = checks_to_dataframe(checks)
    def color_status(val):
        if val == "PASS":
            return "background-color: rgba(60,180,90,0.16)"
        if val == "NEAR":
            return "background-color: rgba(255,190,0,0.18)"
        if val == "FAIL":
            return "background-color: rgba(255,70,70,0.18)"
        return ""

    display_df = format_display_dataframe(df, {"Demand": "{:,.2f}", "Capacity": "{:,.2f}", "Ratio": "{:.3f}"})

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


def render_design_summary(state: DesignState, results: Dict[str, Any]):
    st.subheader("Design Summary")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Main bottom reinforcement")
        summary = pd.DataFrame(
            [
                {
                    "Direction": "X bars",
                    "Bar": state.reinforcement.main_bar_x,
                    "Required As (mm²)": results["flex_x"]["As_req_mm2"],
                    "Strength As (mm²)": results["flex_x"]["As_strength_mm2"],
                    "Minimum As (mm²)": results["flex_x"]["As_min_mm2"],
                    "Use spacing (mm)": results["spacing_x"]["spacing_use_mm"],
                    "Bars count": results["spacing_x"]["n_bars"],
                    "Provided As (mm²)": results["spacing_x"]["As_prov_mm2"],
                    "φMn (kN-m)": results["cap_xbars"]["phiMn_kNm"],
                },
                {
                    "Direction": "Y bars",
                    "Bar": state.reinforcement.main_bar_y,
                    "Required As (mm²)": results["flex_y"]["As_req_mm2"],
                    "Strength As (mm²)": results["flex_y"]["As_strength_mm2"],
                    "Minimum As (mm²)": results["flex_y"]["As_min_mm2"],
                    "Use spacing (mm)": results["spacing_y"]["spacing_use_mm"],
                    "Bars count": results["spacing_y"]["n_bars"],
                    "Provided As (mm²)": results["spacing_y"]["As_prov_mm2"],
                    "φMn (kN-m)": results["cap_ybars"]["phiMn_kNm"],
                },
            ]
        )
        st.dataframe(
            format_display_dataframe(
                summary,
                {
                    "Required As (mm²)": "{:,.0f}",
                    "Strength As (mm²)": "{:,.0f}",
                    "Minimum As (mm²)": "{:,.0f}",
                    "Use spacing (mm)": "{:,.0f}",
                    "Provided As (mm²)": "{:,.0f}",
                    "φMn (kN-m)": "{:,.1f}",
                },
            ),
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.markdown("#### Shear and detailing")
        detail = pd.DataFrame(
            [
                {"Item": "Effective depth X bars", "Value": results["d_x_mm"], "Unit": "mm"},
                {"Item": "Effective depth Y bars", "Value": results["d_y_mm"], "Unit": "mm"},
                {"Item": "One-way φVc, section normal to Y", "Value": results["one_way_x"]["phiVc_kN"], "Unit": "kN"},
                {"Item": "One-way φVc, section normal to X", "Value": results["one_way_y"]["phiVc_kN"], "Unit": "kN"},
                {"Item": "Punching bo", "Value": results["punch_demand"]["bo_mm"], "Unit": "mm"},
                {"Item": "Punching φVc", "Value": results["punch_cap"]["phiVc_kN"], "Unit": "kN"},
                {"Item": "Development estimate X bars", "Value": results["ld_x_mm"], "Unit": "mm"},
                {"Item": "Development estimate Y bars", "Value": results["ld_y_mm"], "Unit": "mm"},
                {"Item": "STM tie As advisory X bars", "Value": results["As_stm_x_mm2"], "Unit": "mm²"},
                {"Item": "STM tie As advisory Y bars", "Value": results["As_stm_y_mm2"], "Unit": "mm²"},
            ]
        )
        st.dataframe(format_display_dataframe(detail, {"Value": "{:,.1f}"}), use_container_width=True, hide_index=True)


def render_engineering_notes(results: Dict[str, Any]):
    notes = []
    stm = results["stm"]
    if not stm.empty and (stm["theta (deg)"] < 25.0).any():
        notes.append("Some STM strut angles are below about 25°. A deeper cap or revised pile spacing may be required.")
    if not stm.empty and (stm["theta (deg)"] > 65.0).any():
        notes.append("Some STM strut angles are steep. Verify nodal zone geometry, anchorage, and local bearing.")
    if results["punch_demand"]["R_inside_kN"] <= 0:
        notes.append("No pile reaction is inside the punching perimeter. Punching demand may be severe; verify critical section logic.")
    if results["flex_x"]["rho"] > 0.02 or results["flex_y"]["rho"] > 0.02:
        notes.append("Flexural reinforcement ratio is high. Increase cap thickness or use a full STM.")
    if not notes:
        notes.append("No automatic advisory warnings were triggered. Engineer verification is still required.")

    st.markdown("#### Engineering advisories")
    for n in notes:
        st.markdown(f"- {n}")


def render_code_basis():
    st.subheader("Basis / Assumptions")
    st.markdown(
        """
        This app intentionally keeps its formulas visible and editable. It follows an ACI-style design workflow suitable for
        professional preliminary pile-cap sizing:

        1. Rigid pile-cap vertical load distribution to piles from axial load and biaxial moment.
        2. Pile compression/uplift capacity check from the geotechnical design values entered by the user.
        3. Simplified flexural design at the face of the column/pedestal using pile reactions outside the section.
        4. One-way shear at a distance d from the loaded area.
        5. Two-way punching shear at a perimeter d/2 from the loaded area.
        6. Practical strut-and-tie advisory based on the load path from column/pedestal to pile heads.
        7. Drawing-style output for plan and elevation review.

        Critical assumptions to verify:
        - Compression-positive sign convention.
        - Exact SAP2000 reaction direction/signs.
        - Pile-cap behavior: sectional method may be inappropriate for very deep caps; use STM.
        - ACI 318-25 modifiers for shear, punching, development, seismic detailing, and anchorage.
        - Pile head fixity, pile tolerance, eccentricity, lateral load, settlement, and group effects.
        """
    )
    with st.expander("Formula notes"):
        st.code(
            """
Pile reaction:
    R_i = P/n + Mx*y_i/sum(y^2) + My*x_i/sum(x^2)

Flexure:
    phi * As * fy * (d - a/2) >= Mu
    a = As*fy/(0.85*fc'*b)

Minimum flexural steel:
    As,min = max(0.25*sqrt(fc')/fy, 1.4/fy) * b*d

One-way shear:
    phi Vc = phi * 0.17 * lambda * sqrt(fc') * b * d

Two-way punching, simplified interior:
    phi Vc = phi * 0.33 * lambda * sqrt(fc') * bo * d

STM advisory:
    theta = atan(d/a)
    T_est = R*cot(theta)
            """,
            language="text",
        )


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    inject_css()
    render_header()

    mat, geom, reinf, include_self_weight = sidebar_inputs()

    tab_input, tab_results, tab_drawing, tab_stm, tab_report, tab_basis = st.tabs(
        ["1 Input", "2 Design Results", "3 Drawing Output", "4 STM Advisory", "5 Report / Export", "6 Basis"]
    )

    with tab_input:
        left, right = st.columns([1.05, 0.95], gap="large")
        with left:
            loadcase = load_input_ui()
        with right:
            piles_base = pile_layout_ui(geom)

        cap_x, cap_y = cap_dimensions_from_piles(piles_base, geom.edge_from_pile_edge_mm)
        sw = self_weight_cap_kN(cap_x, cap_y, geom.cap_thickness_mm, mat.gamma_conc_kN_m3)

        piles_with_reactions = distribute_vertical_load_to_piles(
            piles_base,
            loadcase.Pu_kN,
            loadcase.Mux_kNm,
            loadcase.Muy_kNm,
            include_self_weight_kN=sw if include_self_weight else 0.0,
        )

        # Update state dataframe reaction display
        reaction_df = piles_to_dataframe(piles_with_reactions)
        st.session_state["pile_layout_df"]["reaction_kN"] = reaction_df["reaction_kN"].values

        state = DesignState(
            material=mat,
            geometry=geom,
            reinforcement=reinf,
            loadcase=loadcase,
            piles=piles_with_reactions,
            cap_length_x_mm=cap_x,
            cap_width_y_mm=cap_y,
            effective_depth_x_mm=effective_depths(geom, reinf)[0],
            effective_depth_y_mm=effective_depths(geom, reinf)[1],
            self_weight_kN=sw if include_self_weight else 0.0,
        )

        results = design_all(state, use_self_weight=include_self_weight)
        st.session_state["last_state_json"] = json.dumps(asdict(state), default=str)
        st.session_state["last_report"] = make_markdown_report(state, results)

        st.markdown("---")
        render_metric_row(state, results)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Pile reactions")
            st.dataframe(
                format_display_dataframe(
                    pile_group_result_table(state.piles, geom),
                    {
                        "x (mm)": "{:,.0f}",
                        "y (mm)": "{:,.0f}",
                        "R, compression + (kN)": "{:,.1f}",
                        "Compression ratio": "{:.3f}",
                        "Tension ratio": "{:.3f}",
                    },
                ),
                use_container_width=True,
                hide_index=True,
            )
        with c2:
            st.markdown("#### Generated cap dimensions")
            dim_df = pd.DataFrame(
                [
                    {"Item": "Cap length X", "Value": cap_x, "Unit": "mm"},
                    {"Item": "Cap width Y", "Value": cap_y, "Unit": "mm"},
                    {"Item": "Cap thickness", "Value": geom.cap_thickness_mm, "Unit": "mm"},
                    {"Item": "Self weight", "Value": state.self_weight_kN, "Unit": "kN"},
                    {"Item": "Effective depth X bars", "Value": state.effective_depth_x_mm, "Unit": "mm"},
                    {"Item": "Effective depth Y bars", "Value": state.effective_depth_y_mm, "Unit": "mm"},
                ]
            )
            st.dataframe(format_display_dataframe(dim_df, {"Value": "{:,.1f}"}), use_container_width=True, hide_index=True)

    # Use latest state/results even when user opens other tabs first.
    if "last_state_runtime" not in st.session_state:
        try:
            piles_base = make_piles(geom.n_piles, geom.spacing_x_mm, geom.spacing_y_mm, geom.pile_diameter_mm)
            cap_x, cap_y = cap_dimensions_from_piles(piles_base, geom.edge_from_pile_edge_mm)
            sw = self_weight_cap_kN(cap_x, cap_y, geom.cap_thickness_mm, mat.gamma_conc_kN_m3)
            loadcase = LoadCase()
            piles_with_reactions = distribute_vertical_load_to_piles(
                piles_base, loadcase.Pu_kN, loadcase.Mux_kNm, loadcase.Muy_kNm, sw if include_self_weight else 0.0
            )
            state = DesignState(
                material=mat,
                geometry=geom,
                reinforcement=reinf,
                loadcase=loadcase,
                piles=piles_with_reactions,
                cap_length_x_mm=cap_x,
                cap_width_y_mm=cap_y,
                effective_depth_x_mm=effective_depths(geom, reinf)[0],
                effective_depth_y_mm=effective_depths(geom, reinf)[1],
                self_weight_kN=sw if include_self_weight else 0.0,
            )
            results = design_all(state, use_self_weight=include_self_weight)
        except Exception:
            state = None
            results = None

    # Because Streamlit reruns top-to-bottom, state/results from tab_input are available after the tab block.
    try:
        st.session_state["last_state_runtime"] = state
        st.session_state["last_results_runtime"] = results
    except Exception:
        pass

    state = st.session_state.get("last_state_runtime", None)
    results = st.session_state.get("last_results_runtime", None)

    if state is None or results is None:
        st.stop()

    with tab_results:
        render_metric_row(state, results)
        st.markdown("### Strength / service check table")
        render_check_cards(results["checks"])
        render_design_summary(state, results)
        render_engineering_notes(results)

        with st.expander("Detailed demand values", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("##### Moment demands")
                st.json({k: round(v, 3) for k, v in results["moment_demands"].items()})
            with c2:
                st.markdown("##### One-way shear demands")
                st.json({k: round(v, 3) for k, v in results["shear_demands"].items()})
            with c3:
                st.markdown("##### Punching demand")
                st.json({k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in results["punch_demand"].items()})

    with tab_drawing:
        st.subheader("Drawing-style Output")
        show_rebar = st.toggle("Show schematic bottom reinforcement", value=True)
        show_reactions = st.toggle("Show pile reactions on plan", value=True)

        fig_plan = plot_plan(state, results, show_rebar=show_rebar, show_reactions=show_reactions)
        st.pyplot(fig_plan, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_ex = plot_elevation(state, results, direction="X")
            st.pyplot(fig_ex, use_container_width=True)
        with c2:
            fig_ey = plot_elevation(state, results, direction="Y")
            st.pyplot(fig_ey, use_container_width=True)

        png_plan = fig_to_png_bytes(fig_plan)
        st.download_button("Download plan PNG", png_plan, "pile_cap_plan.png", "image/png")
        plt.close(fig_plan)
        plt.close(fig_ex)
        plt.close(fig_ey)

        st.markdown(
            """
            <div class="mini">
            Drawing is a calculation sketch, not a full construction drawing. Add project title block, bar marks,
            lap locations, pile cutoff level, pile dowels, construction joints, waterproofing, lean concrete,
            and site-specific notes before issue.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_stm:
        st.subheader("Strut-and-Tie / Deep Pile Cap Advisory")
        st.markdown(
            """
            This tab gives a practical load-path check inspired by the MacGregor reinforced concrete design workflow:
            confirm that load can travel from the column/pedestal node to pile nodes through compression struts,
            and that bottom reinforcement can act as tension ties. This is not a replacement for a formal 2D/3D STM.
            """
        )

        stm_df = results["stm"].copy()
        st.dataframe(
            format_display_dataframe(
                stm_df,
                {
                    "R (kN)": "{:,.1f}",
                    "a horizontal (mm)": "{:,.0f}",
                    "theta (deg)": "{:,.1f}",
                    "T tie estimate (kN)": "{:,.1f}",
                    "Tx component (kN)": "{:,.1f}",
                    "Ty component (kN)": "{:,.1f}",
                },
            ),
            use_container_width=True,
            hide_index=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("STM advisory As, X bars", f"{results['As_stm_x_mm2']:,.0f} mm²")
        c2.metric("STM advisory As, Y bars", f"{results['As_stm_y_mm2']:,.0f} mm²")
        c3.metric("Average d", f"{results['d_avg_mm']:,.0f} mm")

        st.markdown("#### STM practical detailing checklist")
        st.markdown(
            """
            - Place bottom main bars through the pile heads and fully anchor past exterior pile centerlines.
            - Provide confinement around column/pedestal node and pile head nodes where high compression struts enter.
            - Check local bearing at pile heads and column/pedestal load introduction.
            - For two-pile caps, the tie force may govern over sectional flexure; check direct truss equilibrium.
            - If strut angle is shallow, increase cap depth or reduce pile spacing.
            - For seismic design, verify pile anchorage, ductile detailing, shear friction, and collector/tie forces.
            """
        )

    with tab_report:
        st.subheader("Report / Export")
        report_md = make_markdown_report(state, results)
        st.download_button("Download Markdown report", report_md.encode("utf-8"), "pile_foundation_report.md", "text/markdown")

        checks_df = checks_to_dataframe(results["checks"])
        pile_df = pile_group_result_table(state.piles, state.geometry)
        flex_df = pd.DataFrame(
            [
                {"Item": "X bars As required", "Value": results["flex_x"]["As_req_mm2"], "Unit": "mm²"},
                {"Item": "X bars spacing use", "Value": results["spacing_x"]["spacing_use_mm"], "Unit": "mm"},
                {"Item": "Y bars As required", "Value": results["flex_y"]["As_req_mm2"], "Unit": "mm²"},
                {"Item": "Y bars spacing use", "Value": results["spacing_y"]["spacing_use_mm"], "Unit": "mm"},
                {"Item": "Punching demand", "Value": results["punch_demand"]["Vu_punch_kN"], "Unit": "kN"},
                {"Item": "Punching capacity", "Value": results["punch_cap"]["phiVc_kN"], "Unit": "kN"},
            ]
        )

        saved_xlsx = make_saved_state_xlsx(state, results, include_self_weight)
        try:
            pdf_report = make_calculation_pdf(state, results)
            pdf_error = None
        except Exception as exc:
            pdf_report = None
            pdf_error = str(exc)

        c1, c2, c3 = st.columns(3)
        with c1:
            dataframe_download_button(checks_df, "Download checks CSV", "pile_cap_checks.csv")
        with c2:
            dataframe_download_button(pile_df, "Download pile reactions CSV", "pile_reactions.csv")
        with c3:
            dataframe_download_button(flex_df, "Download design summary CSV", "pile_cap_design_summary.csv")

        c4, c5 = st.columns(2)
        with c4:
            st.download_button("Save editable state XLSX", saved_xlsx, "pile_foundation_state.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c5:
            if pdf_report is not None:
                st.download_button("Download calculation PDF", pdf_report, "pile_foundation_calculation.pdf", "application/pdf")
            else:
                st.warning(pdf_error)

        with st.expander("Preview Markdown report", expanded=False):
            st.markdown(report_md)

        with st.expander("JSON model state", expanded=False):
            st.json(asdict(state))

    with tab_basis:
        render_code_basis()


if __name__ == "__main__":
    main()
