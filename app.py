import io
import os
from typing import Dict, List, Optional, Tuple

import chardet
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# TODO: AgGrid / DuckDB UPSERT / kaleido PNG / 大分類平均・全体平均比較 / DuPont分解 / スパークライン

st.set_page_config(page_title="中小企業向け経営分析ダッシュボード", layout="wide")

FONT_FAMILY = "'Hiragino Sans','Noto Sans JP','Meiryo',sans-serif"
DEFAULT_CSV_PATH = "/mnt/data/産業構造マップ_中小企業経営分析_推移　bs pl　従業員数.csv"
KEY_COLUMNS = [
    "集計年",
    "産業大分類コード",
    "産業大分類名",
    "業種中分類コード",
    "業種中分類名",
    "集計形式",
]
DEPRECIATION_COLUMNS = [
    "減価償却費（百万円）",
    "減価償却費（百万円）.1",
]
OPTIONAL_FTE_COLUMNS = [
    "合計_正社員・正職員以外（就業時間換算人数）",
    "他社からの出向従業者（出向役員を含む）及び派遣従業者の合計",
]

st.markdown(
    f"""
    <style>
    :root {{
        --app-font-family: {FONT_FAMILY};
    }}
    html, body, [class*="st-"], [class^="st-"], section {{
        font-family: var(--app-font-family) !important;
    }}
    .metric-value, .metric-label {{
        font-family: var(--app-font-family) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _detect_encoding(file_bytes: bytes) -> Optional[str]:
    if not file_bytes:
        return None
    detection = chardet.detect(file_bytes)
    encoding = detection.get("encoding") if detection else None
    if encoding:
        return encoding.lower()
    return None


def _read_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    if not file_bytes:
        return pd.DataFrame()
    encodings_to_try: List[str] = ["cp932"]
    detected = _detect_encoding(file_bytes)
    if detected and detected not in encodings_to_try:
        encodings_to_try.append(detected)
    encodings_to_try.append("utf-8-sig")

    last_error: Optional[Exception] = None
    for enc in encodings_to_try:
        try:
            buffer = io.BytesIO(file_bytes)
            return pd.read_csv(buffer, encoding=enc)
        except UnicodeDecodeError as exc:  # pragma: no cover - defensive
            last_error = exc
            continue
    if last_error:
        raise last_error
    return pd.DataFrame()


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    missing_keys = [col for col in KEY_COLUMNS if col not in df.columns]
    if missing_keys:
        raise ValueError(f"必須列が不足しています: {', '.join(missing_keys)}")

    df["集計年"] = pd.to_numeric(df["集計年"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["集計年", "産業大分類コード", "業種中分類名"])
    df["産業大分類コード"] = df["産業大分類コード"].astype(str).str.strip()
    df["産業大分類名"] = df["産業大分類名"].astype(str).str.strip()
    df["業種中分類名"] = df["業種中分類名"].astype(str).str.strip()
    df["業種中分類コード"] = pd.to_numeric(
        df["業種中分類コード"], errors="coerce"
    ).astype("Int64")
    df["集計形式"] = df["集計形式"].astype(str).str.strip()

    for col in df.columns:
        if col in KEY_COLUMNS:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["産業大分類コード", "業種中分類コード", "集計年"])
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_default_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as file:
        file_bytes = file.read()
    df = _read_csv_from_bytes(file_bytes)
    return _prepare_dataframe(df)


def load_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    file_bytes = uploaded_file.getvalue()
    df = _read_csv_from_bytes(file_bytes)
    return _prepare_dataframe(df)


def get_query_params() -> Dict[str, List[str]]:
    if hasattr(st, "query_params"):
        return st.query_params
    return st.experimental_get_query_params()


def update_query_params(params: Dict[str, str]) -> None:
    safe_params = {k: v for k, v in params.items() if v is not None}
    if hasattr(st, "query_params"):
        st.query_params.update(safe_params)
    else:
        st.experimental_set_query_params(**safe_params)


def safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return None
    return numerator / denominator


def format_currency(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(round(value)):,} 百万円"


def format_ratio(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.1f}%"


def format_number(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.1f}"


def format_delta(value: Optional[float], as_percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    if as_percent:
        return f"{value * 100:.1f}%"
    return f"{value:+,.1f}"


def compute_ebitda(row: pd.Series) -> Optional[float]:
    if row.get("営業利益（百万円）") is None or pd.isna(row.get("営業利益（百万円）")):
        return None
    depreciation_sum = 0.0
    valid = False
    for col in DEPRECIATION_COLUMNS:
        if col in row and not pd.isna(row[col]):
            depreciation_sum += float(row[col])
            valid = True
    profit = row.get("営業利益（百万円）")
    if pd.isna(profit):
        return None
    if not valid and depreciation_sum == 0:
        return float(profit)
    return float(profit) + depreciation_sum


def compute_fte(row: pd.Series) -> Optional[float]:
    base = row.get("常用雇用者")
    if base is None or pd.isna(base):
        return None
    total = float(base)
    for col in OPTIONAL_FTE_COLUMNS:
        if col in row and not pd.isna(row[col]):
            total += float(row[col])
    return total


def compute_cagr(series: pd.Series, periods: int = 3) -> Optional[float]:
    if series.empty or len(series.dropna()) <= periods:
        return None
    latest = series.dropna().iloc[-1]
    past = series.dropna().iloc[-(periods + 1)]
    if past in (None, 0) or pd.isna(past) or pd.isna(latest):
        return None
    if past <= 0 or latest <= 0:
        return None
    return (latest / past) ** (1 / periods) - 1


def compute_yoy(series: pd.Series) -> Optional[float]:
    if series.empty or len(series.dropna()) < 2:
        return None
    current = series.dropna().iloc[-1]
    previous = series.dropna().iloc[-2]
    if previous in (None, 0) or pd.isna(previous) or pd.isna(current):
        return None
    return (current - previous) / previous


def compute_difference(series: pd.Series) -> Optional[float]:
    if series.empty or len(series.dropna()) < 2:
        return None
    current = series.dropna().iloc[-1]
    previous = series.dropna().iloc[-2]
    if pd.isna(previous) or pd.isna(current):
        return None
    return current - previous


def render_metric_card(
    column, label: str, value: Optional[float], formatter, delta: Optional[float], delta_percent: bool
) -> None:
    formatted_value = formatter(value)
    formatted_delta = format_delta(delta, as_percent=delta_percent) if delta is not None else "—"
    column.metric(label=label, value=formatted_value, delta=formatted_delta)


def build_kpi_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    metrics = pd.DataFrame({"集計年": df["集計年"].astype(int)})
    metrics = metrics.drop_duplicates().set_index("集計年")

    def add_column(name: str, series: pd.Series) -> None:
        metrics[name] = series

    if "売上高（百万円）" in df:
        add_column("売上高（百万円）", df.set_index("集計年")["売上高（百万円）"])
        metrics["売上高YoY"] = metrics["売上高（百万円）"].pct_change()
    if "営業利益（百万円）" in df:
        add_column("営業利益（百万円）", df.set_index("集計年")["営業利益（百万円）"])
        metrics["営業利益率"] = safe_series_division(
            metrics["営業利益（百万円）"], metrics.get("売上高（百万円）")
        )
    if "売上総利益（百万円）" in df:
        add_column("売上総利益（百万円）", df.set_index("集計年")["売上総利益（百万円）"])
        metrics["総利益率"] = safe_series_division(
            metrics["売上総利益（百万円）"], metrics.get("売上高（百万円）")
        )
    if "経常利益（経常損失）（百万円）" in df:
        add_column(
            "経常利益（経常損失）（百万円）",
            df.set_index("集計年")["経常利益（経常損失）（百万円）"],
        )
        metrics["経常利益率"] = safe_series_division(
            metrics["経常利益（経常損失）（百万円）"], metrics.get("売上高（百万円）")
        )
    if "付加価値額（百万円）" in df:
        add_column("付加価値額（百万円）", df.set_index("集計年")["付加価値額（百万円）"])
    if "人件費（百万円）" in df:
        add_column("人件費（百万円）", df.set_index("集計年")["人件費（百万円）"])
    if "資産（百万円）" in df:
        add_column("資産（百万円）", df.set_index("集計年")["資産（百万円）"])
    if "純資産（百万円）" in df:
        add_column("純資産（百万円）", df.set_index("集計年")["純資産（百万円）"])
    if "支払利息・割引料（百万円）" in df:
        add_column(
            "支払利息・割引料（百万円）",
            df.set_index("集計年")["支払利息・割引料（百万円）"],
        )

    ebitda_series = df.apply(compute_ebitda, axis=1)
    if not ebitda_series.isna().all():
        metrics["EBITDA（百万円）"] = pd.Series(
            ebitda_series.values, index=df["集計年"].values
        )

    fte_series = df.apply(compute_fte, axis=1)
    if not fte_series.isna().all():
        metrics["FTE"] = pd.Series(fte_series.values, index=df["集計年"].values)

    if "付加価値額（百万円）" in metrics and "FTE" in metrics:
        metrics["労働生産性"] = safe_series_division(metrics["付加価値額（百万円）"], metrics["FTE"])
    if "人件費（百万円）" in metrics and "付加価値額（百万円）" in metrics:
        metrics["労働分配率"] = safe_series_division(
            metrics["人件費（百万円）"], metrics["付加価値額（百万円）"]
        )
    if "営業利益（百万円）" in metrics and "売上高（百万円）" in metrics:
        metrics["営業利益率"] = safe_series_division(
            metrics["営業利益（百万円）"], metrics["売上高（百万円）"]
        )
    if "純資産（百万円）" in metrics and "資産（百万円）" in metrics:
        metrics["自己資本比率"] = safe_series_division(
            metrics["純資産（百万円）"], metrics["資産（百万円）"]
        )
    if "売上高（百万円）" in metrics and "資産（百万円）" in metrics:
        metrics["総資本回転率"] = safe_series_division(
            metrics["売上高（百万円）"], metrics["資産（百万円）"]
        )
    if "EBITDA（百万円）" in metrics and "支払利息・割引料（百万円）" in metrics:
        metrics["Interest Coverage"] = safe_series_division(
            metrics["EBITDA（百万円）"], metrics["支払利息・割引料（百万円）"]
        )

    metrics = metrics.sort_index()
    return metrics


def safe_series_division(numerator: pd.Series, denominator: Optional[pd.Series]) -> pd.Series:
    if denominator is None:
        return pd.Series([np.nan] * len(numerator), index=numerator.index)
    result = numerator / denominator.replace({0: np.nan})
    return result


def render_time_series_chart(df: pd.DataFrame, value_columns: Dict[str, str], title: str) -> go.Figure:
    fig = go.Figure()
    for column, label in value_columns.items():
        if column not in df:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["集計年"],
                y=df[column],
                name=label,
                mode="lines+markers",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="集計年",
        yaxis_title="百万円",
        hovermode="x unified",
        template="plotly_white",
        font=dict(family=FONT_FAMILY, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_ratio_chart(df: pd.DataFrame, ratio_columns: Dict[str, str], title: str) -> go.Figure:
    fig = go.Figure()
    for column, label in ratio_columns.items():
        if column not in df:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["集計年"],
                y=df[column] * 100,
                name=label,
                mode="lines+markers",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="集計年",
        yaxis_title="%",
        hovermode="x unified",
        template="plotly_white",
        font=dict(family=FONT_FAMILY, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_bs_composition_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    required_cols = [
        "資産（百万円）",
        "流動資産（百万円）",
        "固定資産（百万円）",
        "負債（百万円）",
        "流動負債（百万円）",
        "固定負債（百万円）",
        "純資産（百万円）",
    ]
    available_cols = [col for col in required_cols if col in df]
    if len(available_cols) < 3:
        return None

    composition = pd.DataFrame({"集計年": df["集計年"]})
    composition = composition.set_index("集計年")

    def ratio(col: str) -> pd.Series:
        return safe_series_division(df[col], df["資産（百万円）"]) if col in df else pd.Series(dtype=float)

    if "流動資産（百万円）" in df:
        composition["流動資産"] = ratio("流動資産（百万円）")
    if "固定資産（百万円）" in df:
        composition["固定資産"] = ratio("固定資産（百万円）")
    if "流動負債（百万円）" in df:
        composition["流動負債"] = ratio("流動負債（百万円）")
    if "固定負債（百万円）" in df:
        composition["固定負債"] = ratio("固定負債（百万円）")
    if "純資産（百万円）" in df:
        composition["純資産"] = ratio("純資産（百万円）")

    composition = composition.fillna(0) * 100

    fig = go.Figure()
    for column in composition.columns:
        fig.add_trace(
            go.Bar(
                y=composition.index.astype(str),
                x=composition[column],
                name=column,
                orientation="h",
                hovertemplate="%{y}: %{x:.1f}%<extra>{column}</extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        title="BS構成比推移",
        xaxis_title="構成比（%）",
        yaxis_title="集計年",
        template="plotly_white",
        font=dict(family=FONT_FAMILY, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_ebitda_interest_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    if "EBITDA（百万円）" not in df and "支払利息・割引料（百万円）" not in df:
        return None
    fig = go.Figure()
    if "EBITDA（百万円）" in df:
        fig.add_trace(
            go.Bar(
                x=df["集計年"],
                y=df["EBITDA（百万円）"],
                name="EBITDA",
                marker_color="#3E7CB1",
            )
        )
    if "支払利息・割引料（百万円）" in df:
        fig.add_trace(
            go.Scatter(
                x=df["集計年"],
                y=df["支払利息・割引料（百万円）"],
                name="支払利息・割引料",
                mode="lines+markers",
                yaxis="y2",
                marker=dict(color="#F18F01"),
            )
        )
    fig.update_layout(
        title="EBITDAと支払利息推移",
        xaxis_title="集計年",
        yaxis_title="EBITDA（百万円）",
        template="plotly_white",
        font=dict(family=FONT_FAMILY, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if "支払利息・割引料（百万円）" in df:
        fig.update_layout(
            yaxis2=dict(
                title="支払利息（百万円）",
                overlaying="y",
                side="right",
            )
        )
    return fig


def format_table(metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    display_df = metrics.copy()
    for column in display_df.columns:
        if "百万円" in column or column in ["EBITDA（百万円）"]:
            display_df[column] = display_df[column].apply(format_currency)
        elif column in {
            "自己資本比率",
            "総資本回転率",
            "総利益率",
            "営業利益率",
            "経常利益率",
            "労働分配率",
        }:
            display_df[column] = display_df[column].apply(format_ratio)
        elif column in {"労働生産性", "Interest Coverage", "FTE"}:
            display_df[column] = display_df[column].apply(format_number)
        elif column.endswith("YoY"):
            display_df[column] = display_df[column].apply(
                lambda x: format_ratio(x) if pd.notna(x) else "—"
            )
        else:
            display_df[column] = display_df[column].apply(
                lambda x: format_number(x) if pd.notna(x) else "—"
            )
    display_df = display_df.reset_index().rename(columns={"集計年": "年"})
    download_df = metrics.reset_index().rename(columns={"集計年": "年"})
    return display_df, download_df


def main() -> None:
    query_params = get_query_params()

    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.title("中小企業向け経営分析ダッシュボード")
        st.caption("産業構造マップ（中小企業向け）データから主要KPIを可視化します。")
    with header_right:
        uploaded_file = st.file_uploader("CSV差し替え", type=["csv"])

    data_source_message = "既定データを読み込みました。"
    data_frame = pd.DataFrame()

    if uploaded_file is not None:
        try:
            data_frame = load_uploaded_dataset(uploaded_file)
            data_source_message = "アップロードされたCSVを表示しています。"
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"CSVの読み込みに失敗しました: {exc}")
            return
    else:
        try:
            data_frame = load_default_dataset(DEFAULT_CSV_PATH)
        except FileNotFoundError:
            st.info("既定のCSVが見つかりません。右上のアップローダからファイルを指定してください。")
            data_frame = pd.DataFrame()
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"既定のCSVの読み込みに失敗しました: {exc}")
            return

    if data_frame.empty:
        st.warning("利用可能なデータがありません。")
        return

    st.success(data_source_message)

    major_codes = sorted(data_frame["産業大分類コード"].dropna().unique())
    selected_major = query_params.get("maj", [None])[0]
    if selected_major not in major_codes:
        selected_major = major_codes[0] if major_codes else None

    major_select = st.selectbox(
        "産業大分類コード",
        options=major_codes,
        index=major_codes.index(selected_major) if selected_major in major_codes else 0,
    )

    filtered_by_major = data_frame[data_frame["産業大分類コード"] == major_select]
    if filtered_by_major.empty:
        st.warning("選択した産業大分類コードにデータがありません。別のコードを選択してください。")
        return

    mid_options = (
        filtered_by_major[
            ["業種中分類名", "業種中分類コード", "産業大分類コード"]
        ]
        .drop_duplicates()
        .sort_values(["業種中分類名", "業種中分類コード"])
    )
    mid_values = list(mid_options.itertuples(index=False, name=None))

    selected_mid_name = query_params.get("mid", [None])[0]
    mid_index = 0
    for idx, (name, code, _) in enumerate(mid_values):
        if name == selected_mid_name:
            mid_index = idx
            break
    else:
        if mid_values:
            selected_mid_name = mid_values[mid_index][0]

    selected_mid = st.selectbox(
        "業種中分類",
        options=mid_values,
        index=mid_index,
        format_func=lambda x: f"{x[0]}（{x[1]}）",
    )
    selected_mid_name = selected_mid[0]

    filtered_df = filtered_by_major[
        filtered_by_major["業種中分類名"] == selected_mid_name
    ]
    if filtered_df.empty:
        st.warning("業種中分類に該当するデータがありません。")
        return

    years = filtered_df["集計年"].dropna().astype(int)
    min_year = int(years.min())
    max_year = int(years.max())

    default_y1 = query_params.get("y1", [None])[0]
    default_y2 = query_params.get("y2", [None])[0]
    try:
        default_y1 = int(default_y1) if default_y1 is not None else min_year
    except ValueError:
        default_y1 = min_year
    try:
        default_y2 = int(default_y2) if default_y2 is not None else max_year
    except ValueError:
        default_y2 = max_year
    default_range = (
        max(min_year, min(default_y1, default_y2)),
        min(max_year, max(default_y1, default_y2)),
    )

    year_range = st.slider(
        "表示する年範囲",
        min_value=min_year,
        max_value=max_year,
        value=default_range,
    )

    update_query_params(
        {
            "maj": major_select,
            "mid": selected_mid_name,
            "y1": str(year_range[0]),
            "y2": str(year_range[1]),
        }
    )

    mask = (filtered_df["集計年"] >= year_range[0]) & (
        filtered_df["集計年"] <= year_range[1]
    )
    filtered_df = filtered_df.loc[mask].sort_values("集計年")

    if filtered_df.empty:
        st.warning("指定した期間にデータが存在しません。年範囲を変更してください。")
        return

    st.markdown(
        f"**対象**: {major_select} {filtered_df['産業大分類名'].iloc[0]} / {selected_mid_name}"
    )

    metrics_df = build_kpi_dataframe(filtered_df)

    latest_year = metrics_df.index.max()
    latest_row = metrics_df.loc[latest_year]

    revenue_series = metrics_df.get("売上高（百万円）", pd.Series(dtype=float))
    op_profit_series = metrics_df.get("営業利益（百万円）", pd.Series(dtype=float))
    ebitda_series = metrics_df.get("EBITDA（百万円）", pd.Series(dtype=float))
    equity_ratio_series = metrics_df.get("自己資本比率", pd.Series(dtype=float))
    asset_turnover_series = metrics_df.get("総資本回転率", pd.Series(dtype=float))
    labor_productivity_series = metrics_df.get("労働生産性", pd.Series(dtype=float))

    revenue_yoy = compute_yoy(revenue_series)
    op_profit_yoy = compute_yoy(op_profit_series)
    ebitda_yoy = compute_yoy(ebitda_series)
    equity_ratio_diff = compute_difference(equity_ratio_series)
    asset_turnover_diff = compute_difference(asset_turnover_series)
    labor_productivity_yoy = compute_yoy(labor_productivity_series)
    revenue_cagr = compute_cagr(revenue_series)

    metrics_top = st.columns(3)
    metrics_bottom = st.columns(3)

    render_metric_card(
        metrics_top[0],
        "売上高",
        latest_row.get("売上高（百万円）"),
        format_currency,
        revenue_yoy,
        True,
    )
    render_metric_card(
        metrics_top[1],
        "営業利益",
        latest_row.get("営業利益（百万円）"),
        format_currency,
        op_profit_yoy,
        True,
    )
    render_metric_card(
        metrics_top[2],
        "EBITDA",
        latest_row.get("EBITDA（百万円）"),
        format_currency,
        ebitda_yoy,
        True,
    )
    render_metric_card(
        metrics_bottom[0],
        "自己資本比率",
        latest_row.get("自己資本比率"),
        format_ratio,
        equity_ratio_diff,
        False,
    )
    render_metric_card(
        metrics_bottom[1],
        "総資本回転率",
        latest_row.get("総資本回転率"),
        format_ratio,
        asset_turnover_diff,
        False,
    )
    render_metric_card(
        metrics_bottom[2],
        "労働生産性",
        latest_row.get("労働生産性"),
        format_number,
        labor_productivity_yoy,
        True,
    )

    col1, col2 = st.columns(2)

    with col1:
        fig_sales = render_time_series_chart(
            filtered_df,
            {
                "売上高（百万円）": "売上高",
                "営業利益（百万円）": "営業利益",
                "経常利益（経常損失）（百万円）": "経常利益",
            },
            "売上・利益推移",
        )
        st.plotly_chart(fig_sales, use_container_width=True)

    with col2:
        fig_margin = render_ratio_chart(
            metrics_df.reset_index(),
            {
                "総利益率": "総利益率",
                "営業利益率": "営業利益率",
                "経常利益率": "経常利益率",
            },
            "利益率推移",
        )
        st.plotly_chart(fig_margin, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig_bs = render_bs_composition_chart(filtered_df)
        if fig_bs is not None:
            st.plotly_chart(fig_bs, use_container_width=True)
        else:
            st.info("BS構成比を表示するための列が不足しています。")

    with col4:
        fig_ebitda = render_ebitda_interest_chart(metrics_df.reset_index())
        if fig_ebitda is not None:
            st.plotly_chart(fig_ebitda, use_container_width=True)
        else:
            st.info("EBITDAまたは支払利息のデータが不足しています。")

    display_df, download_df = format_table(metrics_df)

    st.subheader("主要KPI一覧")
    st.dataframe(display_df, use_container_width=True)

    csv_buffer = io.StringIO()
    download_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        "CSVダウンロード",
        data=csv_buffer.getvalue(),
        file_name="kpi_summary.csv",
        mime="text/csv",
    )

    if revenue_cagr is not None:
        st.caption(f"直近3年間の売上CAGR: {revenue_cagr * 100:.1f}%")
    else:
        st.caption("直近3年間の売上CAGR: —")


if __name__ == "__main__":
    main()
