"""에이전트 도구 함수 모음.

SnowflakeClient와 ML 모듈을 래핑하여 에이전트가 사용할 수 있는
포맷된 문자열을 반환하는 10개 도구 함수를 제공한다.

모든 함수는 try/except로 감싸져 있으며,
실패 시 사용자 친화적인 폴백 메시지를 반환한다.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from snowflake.snowpark import Session

from data.snowflake_client import SnowflakeClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

_NO_DATA_MSG = "[데이터 없음] 해당 조건에 맞는 데이터가 없습니다."
_ERROR_PREFIX = "[도구 오류]"

# ---------------------------------------------------------------------------
# ML 모델 싱글턴 캐시 (세션당 1회만 학습)
# ---------------------------------------------------------------------------
_cached_model: object | None = None
_cached_model_session_id: int | None = None


def _get_trained_model(session: "Session"):
    """세션별로 캐싱된 ConversionModel을 반환. 최초 1회만 train()."""
    global _cached_model, _cached_model_session_id
    sid = id(session)
    if _cached_model is not None and _cached_model_session_id == sid:
        return _cached_model
    try:
        from ml.conversion_model import ConversionModel
        model = ConversionModel(session)
        model.train()
        _cached_model = model
        _cached_model_session_id = sid
        return model
    except Exception as exc:
        logger.warning("ML 모델 초기화 실패: %s", exc)
        return None


def _df_to_summary(df: pd.DataFrame, max_rows: int = 20) -> str:
    """DataFrame을 에이전트가 읽기 좋은 텍스트로 변환.

    Args:
        df: 변환할 DataFrame
        max_rows: 표시할 최대 행 수

    Returns:
        포맷된 요약 문자열
    """
    if df.empty:
        return _NO_DATA_MSG

    total_rows = len(df)
    display_df = df.head(max_rows)
    summary = display_df.to_string(index=False)

    if total_rows > max_rows:
        summary += f"\n... (총 {total_rows}행 중 상위 {max_rows}행 표시)"

    return summary


def _safe_json(obj: object) -> str:
    """객체를 JSON 문자열로 안전하게 변환.

    Args:
        obj: 직렬화할 객체

    Returns:
        JSON 문자열. 실패 시 str() 폴백.
    """
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return str(obj)


def _compute_structured_summary(df: pd.DataFrame, key_cols: dict | None = None) -> str:
    """DataFrame의 핵심 통계를 미리 계산하여 구조화된 요약 반환.

    LLM에 원본 테이블 대신 사전 계산된 사실(facts)을 전달한다.

    Args:
        df: 소스 DataFrame
        key_cols: {"name": col, "value": col, "metric": col} 매핑 (선택)

    Returns:
        구조화된 요약 문자열
    """
    if df.empty:
        return _NO_DATA_MSG

    result_parts: list[str] = []

    year_month = df.get("YEAR_MONTH", pd.Series(dtype=str))
    if not year_month.empty and year_month.notna().any():
        result_parts.append(
            f"총 {len(df)}행, 기간: {year_month.min()} ~ {year_month.max()}"
        )
    else:
        result_parts.append(f"총 {len(df)}행")

    # 수치 컬럼 자동 탐지 및 통계 계산
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols[:5]:
        result_parts.append(
            f"  {col}: 평균={df[col].mean():.2f}, "
            f"최소={df[col].min():.2f}, 최대={df[col].max():.2f}"
        )

    return "\n".join(result_parts)


# ---------------------------------------------------------------------------
# 퍼널 도구
# ---------------------------------------------------------------------------


def query_funnel_data(
    session: Session,
    category: str,
    months: int = 6,
) -> str:
    """퍼널 시계열 데이터를 조회하여 포맷된 문자열로 반환.

    Args:
        session: Snowpark 세션
        category: 상품 카테고리명
        months: 조회할 최근 개월 수 (기본 6)

    Returns:
        퍼널 CVR 시계열 요약 문자열
    """
    try:
        client = SnowflakeClient(session)
        df = client.load_funnel_timeseries(category=category)

        if df.empty:
            return f"[퍼널 데이터] '{category}' 카테고리의 퍼널 데이터가 없습니다."

        # 최근 N개월 필터
        if "YEAR_MONTH" in df.columns:
            df = df.sort_values("YEAR_MONTH", ascending=False).head(months)
            df = df.sort_values("YEAR_MONTH")

        findings: list[str] = []

        # 1. 현재 전체 전환율
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        if "OVERALL_CVR" in df.columns:
            cvr = float(latest.get("OVERALL_CVR", 0))
            findings.append(f"현재 전체 전환율: {cvr:.1f}%")

            # 2. MoM 변화
            prev_cvr = float(prev.get("OVERALL_CVR", 0))
            mom = cvr - prev_cvr
            direction = "상승" if mom > 0 else "하락"
            findings.append(f"전월 대비 {abs(mom):.1f}%p {direction}")

            # 3. 3개월 추세
            if len(df) >= 3:
                recent3 = df.tail(3)["OVERALL_CVR"]
                trend = float(recent3.iloc[-1] - recent3.iloc[0])
                trend_dir = "개선" if trend > 0 else "악화"
                findings.append(f"3개월 추세: {abs(trend):.1f}%p {trend_dir}")

        # 4. 단계별 CVR (있는 경우)
        stage_labels = {
            "CVR_CONSULT_REQUEST": "상담전환",
            "CVR_OPEN": "개통전환",
            "CVR_PAYEND": "납입전환",
        }
        for stage, label in stage_labels.items():
            if stage in latest.index:
                val = float(latest[stage])
                findings.append(f"{label}: {val:.1f}%")

        # 5. 기간 정보
        if "YEAR_MONTH" in df.columns:
            findings.append(
                f"분석 기간: {df['YEAR_MONTH'].min()} ~ {df['YEAR_MONTH'].max()} "
                f"({len(df)}개월)"
            )

        header = f"[퍼널 분석 결과 — {category}]"
        return header + "\n" + "\n".join(f"- {f}" for f in findings)

    except Exception as exc:
        logger.exception("query_funnel_data 실패: category=%s", category)
        return f"{_ERROR_PREFIX} 퍼널 데이터 조회 실패: {exc}"


def query_funnel_bottlenecks(
    session: Session,
    category: Optional[str] = None,
) -> str:
    """퍼널 병목 분석 결과를 조회하여 포맷된 문자열로 반환.

    Args:
        session: Snowpark 세션
        category: 상품 카테고리 필터 (None이면 전체)

    Returns:
        병목 구간 분석 요약 문자열
    """
    try:
        client = SnowflakeClient(session)
        df = client.load_funnel_bottlenecks()

        if df.empty:
            return "[병목 데이터] 퍼널 병목 분석 결과가 없습니다."

        if category is not None and "CATEGORY" in df.columns:
            filtered = df[df["CATEGORY"] == category]
            df = filtered if not filtered.empty else df

        if category is not None and "MAIN_CATEGORY_NAME" in df.columns:
            filtered = df[df["MAIN_CATEGORY_NAME"] == category]
            df = filtered if not filtered.empty else df

        findings: list[str] = []

        # 1. 이탈률/전환율 컬럼에서 최대 병목 구간 추출
        drop_cols = [c for c in df.columns if "DROP" in c.upper() or "이탈" in c]
        stage_col = next(
            (c for c in ["STAGE", "STAGE_NAME", "WORST_BOTTLENECK_STAGE", "구간"] if c in df.columns),
            None,
        )

        for dc in drop_cols[:3]:
            if dc in df.columns and not df[dc].isna().all():
                worst_idx = df[dc].idxmax()
                worst_row = df.loc[worst_idx]
                stage_name = worst_row[stage_col] if stage_col else f"행 {worst_idx}"
                findings.append(
                    f"최대 병목 구간: {stage_name} ({dc}={worst_row[dc]:.1f}%)"
                )

                # 2. 이탈로 인한 손실 건수 추출
                count_col = next(
                    (c for c in df.columns if "COUNT" in c.upper() or "건수" in c),
                    None,
                )
                if count_col and count_col in worst_row.index:
                    lost = float(worst_row[count_col])
                    findings.append(f"해당 구간 손실 건수: {lost:,.0f}건")

        # 3. 전체 구간 이탈률 순위
        if drop_cols and stage_col:
            primary_drop = drop_cols[0]
            sorted_df = df.sort_values(primary_drop, ascending=False)
            for _, row in sorted_df.head(5).iterrows():
                s_name = row[stage_col] if stage_col else "N/A"
                d_val = float(row[primary_drop])
                findings.append(f"  {s_name}: 이탈률 {d_val:.1f}%")

        if not findings:
            findings.append(f"총 {len(df)}개 구간 분석 완료, 주요 이탈 컬럼 미감지")

        header = f"[퍼널 병목 분석 결과 — {category or '전체'}]"
        return header + "\n" + "\n".join(f"- {f}" for f in findings)

    except Exception as exc:
        logger.exception("query_funnel_bottlenecks 실패")
        return f"{_ERROR_PREFIX} 병목 데이터 조회 실패: {exc}"


# ---------------------------------------------------------------------------
# 채널 도구
# ---------------------------------------------------------------------------


def query_channel_performance(
    session: Session,
    category: str,
    months: int = 6,
) -> str:
    """채널별 성과 데이터를 조회하여 포맷된 문자열로 반환.

    Args:
        session: Snowpark 세션
        category: 상품 카테고리명
        months: 조회할 최근 개월 수 (기본 6)

    Returns:
        채널 성과 순위 요약 문자열
    """
    try:
        client = SnowflakeClient(session)
        df = client.load_channel_performance(category=category)

        if df.empty:
            return f"[채널 성과] '{category}' 카테고리의 채널 데이터가 없습니다."

        if "YEAR_MONTH" in df.columns:
            df = df.sort_values("YEAR_MONTH", ascending=False).head(months * 15)
            df = df.sort_values("YEAR_MONTH")

        findings: list[str] = []

        has_cvr = "PAYEND_CVR" in df.columns
        has_contract = "CONTRACT_COUNT" in df.columns
        has_path = "RECEIVE_PATH_NAME" in df.columns

        if has_cvr and has_contract and has_path:
            # 최신 월 기준으로 채널별 집계
            if "YEAR_MONTH" in df.columns:
                latest_month = df["YEAR_MONTH"].max()
                latest_df = df[df["YEAR_MONTH"] == latest_month]
            else:
                latest_df = df

            total = latest_df["CONTRACT_COUNT"].sum()
            n_channels = latest_df["RECEIVE_PATH_NAME"].nunique()
            findings.append(f"총 {n_channels}개 채널, 총 계약 {total:,.0f}건")

            # 전환율 Top 3
            top3_cvr = latest_df.nlargest(3, "PAYEND_CVR")
            for _, row in top3_cvr.iterrows():
                findings.append(
                    f"전환율 상위: {row['RECEIVE_PATH_NAME']} "
                    f"(CVR {row['PAYEND_CVR']:.1f}%, "
                    f"{row['CONTRACT_COUNT']:,.0f}건)"
                )

            # 볼륨 Top 3
            top3_vol = latest_df.nlargest(3, "CONTRACT_COUNT")
            for _, row in top3_vol.iterrows():
                share = row["CONTRACT_COUNT"] / max(total, 1) * 100
                findings.append(
                    f"볼륨 상위: {row['RECEIVE_PATH_NAME']} "
                    f"({row['CONTRACT_COUNT']:,.0f}건, "
                    f"점유율 {share:.1f}%)"
                )

            # 상위 채널 점유율 집중도
            top1_share = (
                latest_df.nlargest(1, "CONTRACT_COUNT")["CONTRACT_COUNT"].iloc[0]
                / max(total, 1) * 100
            )
            findings.append(f"1위 채널 점유율: {top1_share:.1f}%")
        else:
            findings.append(f"총 {len(df)}행 데이터 (세부 컬럼 부족)")

        header = f"[채널 성과 분석 결과 — {category}]"
        return header + "\n" + "\n".join(f"- {f}" for f in findings)

    except Exception as exc:
        logger.exception("query_channel_performance 실패: category=%s", category)
        return f"{_ERROR_PREFIX} 채널 성과 조회 실패: {exc}"


def query_channel_efficiency(
    session: Session,
    category: str,
) -> str:
    """채널 효율성 점수를 조회하여 포맷된 문자열로 반환.

    Args:
        session: Snowpark 세션
        category: 상품 카테고리명

    Returns:
        채널 효율성 점수 요약 문자열
    """
    try:
        client = SnowflakeClient(session)
        df = client.load_channel_efficiency(category=category)

        if df.empty:
            return f"[채널 효율] '{category}' 카테고리의 효율성 데이터가 없습니다."

        findings: list[str] = []

        eff_col = next(
            (c for c in df.columns if "EFFICIENCY" in c.upper() or "효율" in c),
            None,
        )
        path_col = next(
            (c for c in df.columns if "PATH" in c.upper() or "CHANNEL" in c.upper()),
            None,
        )

        if eff_col and path_col and len(df) >= 2:
            # 최고/최저 효율 채널
            best = df.nlargest(1, eff_col).iloc[0]
            worst = df.nsmallest(1, eff_col).iloc[0]
            findings.append(
                f"최고 효율 채널: {best[path_col]} (효율={best[eff_col]:.2f})"
            )
            findings.append(
                f"최저 효율 채널: {worst[path_col]} (효율={worst[eff_col]:.2f})"
            )

            # 효율 평균과 표준편차
            avg_eff = float(df[eff_col].mean())
            std_eff = float(df[eff_col].std()) if len(df) > 1 else 0.0
            findings.append(f"효율 평균: {avg_eff:.2f} (표준편차: {std_eff:.2f})")

            # HHI 계산 (CONTRACT_COUNT 기반)
            count_col = next(
                (c for c in df.columns if "CONTRACT" in c.upper() or "COUNT" in c.upper()),
                None,
            )
            if count_col and not df[count_col].isna().all():
                total = df[count_col].sum()
                if total > 0:
                    shares = df[count_col] / total
                    hhi = float((shares ** 2).sum())
                    hhi_level = (
                        "과도 집중" if hhi > 0.25
                        else "주의" if hhi > 0.15
                        else "양호"
                    )
                    findings.append(f"채널 집중도(HHI): {hhi:.3f} ({hhi_level})")
        else:
            findings.append(f"총 {len(df)}개 채널 데이터 (효율 컬럼 미감지)")

        header = f"[채널 효율성 분석 결과 — {category}]"
        return header + "\n" + "\n".join(f"- {f}" for f in findings)

    except Exception as exc:
        logger.exception("query_channel_efficiency 실패: category=%s", category)
        return f"{_ERROR_PREFIX} 채널 효율성 조회 실패: {exc}"


# ---------------------------------------------------------------------------
# 마케팅 도구
# ---------------------------------------------------------------------------


def query_marketing(
    session: Session,
    months: int = 6,
) -> str:
    """GA4 마케팅 어트리뷰션 데이터를 조회하여 포맷된 문자열로 반환.

    Args:
        session: Snowpark 세션
        months: 조회할 최근 개월 수 (기본 6)

    Returns:
        마케팅 성과 요약 문자열
    """
    try:
        client = SnowflakeClient(session)
        df = client.load_marketing()

        if df.empty:
            return "[마케팅] GA4 마케팅 데이터가 없습니다."

        if "YEAR_MONTH" in df.columns:
            df = df.sort_values("YEAR_MONTH", ascending=False).head(months * 10)
            df = df.sort_values("YEAR_MONTH")

        findings: list[str] = []

        # 소스/매체별 성과 집계
        source_col = next(
            (c for c in df.columns if "SOURCE" in c.upper() or "소스" in c),
            None,
        )
        session_col = next(
            (c for c in df.columns if "SESSION" in c.upper() or "세션" in c),
            None,
        )

        if source_col and session_col:
            source_agg = df.groupby(source_col)[session_col].sum().sort_values(ascending=False)
            total_sessions = source_agg.sum()
            findings.append(f"총 세션: {total_sessions:,.0f}")

            for src, sess in source_agg.head(5).items():
                share = sess / max(total_sessions, 1) * 100
                findings.append(f"소스 상위: {src} ({sess:,.0f}세션, {share:.1f}%)")
        else:
            findings.append(f"총 {len(df)}행 마케팅 데이터")

        # CVR 관련 컬럼 (있으면)
        cvr_col = next(
            (c for c in df.columns if "CVR" in c.upper() or "전환" in c),
            None,
        )
        if cvr_col and source_col:
            cvr_agg = df.groupby(source_col)[cvr_col].mean().sort_values(ascending=False)
            for src, cvr_val in cvr_agg.head(3).items():
                findings.append(f"전환율 상위: {src} (CVR {cvr_val:.1f}%)")

        # 기간 정보
        if "YEAR_MONTH" in df.columns:
            findings.append(
                f"분석 기간: {df['YEAR_MONTH'].min()} ~ {df['YEAR_MONTH'].max()}"
            )

        header = f"[GA4 마케팅 분석 결과 — 최근 {months}개월]"
        return header + "\n" + "\n".join(f"- {f}" for f in findings)

    except Exception as exc:
        logger.exception("query_marketing 실패")
        return f"{_ERROR_PREFIX} 마케팅 데이터 조회 실패: {exc}"


# ---------------------------------------------------------------------------
# 지역 도구
# ---------------------------------------------------------------------------


def query_regional_demand(
    session: Session,
    state: Optional[str] = None,
) -> str:
    """지역별 수요 점수를 조회하여 포맷된 문자열로 반환.

    Args:
        session: Snowpark 세션
        state: 시/도 필터 (None이면 전체)

    Returns:
        지역 수요 점수 요약 문자열
    """
    try:
        client = SnowflakeClient(session)
        df = client.load_regional_demand(state=state)

        if df.empty:
            msg = "[지역 수요] 지역별 수요 데이터가 없습니다."
            if state:
                msg = f"[지역 수요] '{state}'의 수요 데이터가 없습니다."
            return msg

        findings: list[str] = []

        score_col = next(
            (c for c in df.columns if "SCORE" in c.upper() or "DEMAND" in c.upper()),
            None,
        )
        region_col = next(
            (c for c in df.columns if "STATE" in c.upper() or "CITY" in c.upper()
             or "지역" in c),
            None,
        )

        if score_col and region_col and len(df) >= 2:
            # Top 3 / Bottom 3
            top3 = df.nlargest(3, score_col)
            bottom3 = df.nsmallest(3, score_col)

            for _, row in top3.iterrows():
                findings.append(
                    f"수요 상위: {row[region_col]} (점수={row[score_col]:.2f})"
                )
            for _, row in bottom3.iterrows():
                findings.append(
                    f"수요 하위: {row[region_col]} (점수={row[score_col]:.2f})"
                )

            # 전체 통계
            avg_score = float(df[score_col].mean())
            findings.append(f"전체 평균 수요 점수: {avg_score:.2f}")
            findings.append(f"분석 대상 지역 수: {len(df)}개")

            # 성장 도시 (CONTRACT_COUNT 컬럼 있으면)
            growth_col = next(
                (c for c in df.columns if "GROWTH" in c.upper() or "성장" in c),
                None,
            )
            if growth_col:
                growing = df[df[growth_col] > 0].nlargest(3, growth_col)
                for _, row in growing.iterrows():
                    findings.append(
                        f"성장 도시: {row[region_col]} "
                        f"(성장률={row[growth_col]:.1f}%)"
                    )
        else:
            findings.append(f"총 {len(df)}개 지역 데이터 (수요 점수 컬럼 미감지)")

        header = f"[지역 수요 분석 결과 — {state or '전체'}]"
        return header + "\n" + "\n".join(f"- {f}" for f in findings)

    except Exception as exc:
        logger.exception("query_regional_demand 실패: state=%s", state)
        return f"{_ERROR_PREFIX} 지역 수요 조회 실패: {exc}"


def query_regional_growth(session: Session) -> str:
    """지역별 성장 트렌드를 조회하여 포맷된 문자열로 반환.

    히트맵 데이터에서 MoM 성장률을 계산한다.

    Args:
        session: Snowpark 세션

    Returns:
        지역 성장 트렌드 요약 문자열
    """
    try:
        client = SnowflakeClient(session)
        df = client.load_regional_heatmap()

        if df.empty:
            return "[지역 성장] 지역 히트맵 데이터가 없습니다."

        findings: list[str] = []

        if "INSTALL_STATE" in df.columns and "YEAR_MONTH" in df.columns:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                grouped = (
                    df.groupby(["INSTALL_STATE", "YEAR_MONTH"])[numeric_cols]
                    .sum()
                    .reset_index()
                )
                grouped = grouped.sort_values(
                    ["INSTALL_STATE", "YEAR_MONTH"]
                )

                months_list = sorted(grouped["YEAR_MONTH"].unique())
                if len(months_list) >= 2:
                    prev_m = months_list[-2]
                    curr_m = months_list[-1]
                    prev_data = grouped[grouped["YEAR_MONTH"] == prev_m]
                    curr_data = grouped[grouped["YEAR_MONTH"] == curr_m]

                    value_col = numeric_cols[0]
                    growth_records: list[dict] = []
                    for st in grouped["INSTALL_STATE"].unique():
                        p = prev_data.loc[
                            prev_data["INSTALL_STATE"] == st, value_col
                        ]
                        c = curr_data.loc[
                            curr_data["INSTALL_STATE"] == st, value_col
                        ]
                        if not p.empty and not c.empty and p.iloc[0] > 0:
                            p_val = float(p.iloc[0])
                            c_val = float(c.iloc[0])
                            mom = (c_val - p_val) / p_val * 100
                            growth_records.append({
                                "state": st,
                                "mom": round(mom, 1),
                                "volume": c_val,
                            })

                    if growth_records:
                        growth_df = pd.DataFrame(growth_records)

                        # 최소 볼륨 필터 (>=200 계약)
                        significant = growth_df[growth_df["volume"] >= 200]
                        if significant.empty:
                            significant = growth_df

                        top = significant.nlargest(3, "mom")
                        bottom = significant.nsmallest(3, "mom")

                        findings.append(
                            f"기간: {prev_m} -> {curr_m}"
                        )
                        for _, row in top.iterrows():
                            findings.append(
                                f"성장 상위: {row['state']} "
                                f"(MoM {row['mom']:+.1f}%, "
                                f"계약 {row['volume']:,.0f}건)"
                            )
                        for _, row in bottom.iterrows():
                            findings.append(
                                f"성장 하위: {row['state']} "
                                f"(MoM {row['mom']:+.1f}%, "
                                f"계약 {row['volume']:,.0f}건)"
                            )

                        findings.append(
                            f"분석 대상: {len(significant)}개 시도 "
                            f"(볼륨 200건 이상)"
                        )

                if not findings:
                    findings.append("성장률 계산에 필요한 기간 데이터 부족")

                header = "[지역 성장 트렌드 분석 결과]"
                return header + "\n" + "\n".join(f"- {f}" for f in findings)

        findings.append(f"총 {len(df)}행 (필수 컬럼 미감지)")
        header = "[지역 성장 트렌드 분석 결과]"
        return header + "\n" + "\n".join(f"- {f}" for f in findings)

    except Exception as exc:
        logger.exception("query_regional_growth 실패")
        return f"{_ERROR_PREFIX} 지역 성장 데이터 조회 실패: {exc}"


# ---------------------------------------------------------------------------
# ML 도구
# ---------------------------------------------------------------------------


def get_ml_prediction(
    session: Session,
    category: str,
    channel: Optional[str] = None,
) -> str:
    """ML 모델 예측 결과를 반환.

    ConversionModel을 사용하여 카테고리/채널 조합의 전환율 예측을 수행한다.

    Args:
        session: Snowpark 세션
        category: 상품 카테고리명
        channel: 채널명 (None이면 주요 채널 전체 예측)

    Returns:
        ML 예측 결과 요약 문자열
    """
    try:
        model = _get_trained_model(session)
        if model is None:
            raise RuntimeError("ML 모델 초기화 실패")
        result = model.predict(category, channel)

        # 모델 결과를 구조화된 findings으로 변환
        findings: list[str] = []
        if isinstance(result, dict):
            for key, val in result.items():
                if isinstance(val, (int, float)):
                    findings.append(f"{key}: {val:.2f}")
                else:
                    findings.append(f"{key}: {val}")
        elif isinstance(result, list):
            for item in result[:10]:
                findings.append(str(item))
        else:
            findings.append(str(result))

        header = f"[ML 전환율 예측 결과 — {category}"
        if channel:
            header += f" / {channel}"
        header += "]"
        return header + "\n" + "\n".join(f"- {f}" for f in findings)

    except (ImportError, Exception) as exc:
        # ML 모델 불가 시 규칙 기반 평가 폴백
        logger.warning(
            "ML 모델 불가, 규칙 기반 평가 사용: category=%s, err=%s",
            category, exc,
        )
        try:
            client = SnowflakeClient(session)
            df = client.load_funnel_timeseries(category=category)
            if df.empty:
                return "[ML 예측] 데이터 부족으로 예측 불가"

            df = df.sort_values("YEAR_MONTH")
            findings = [f"[규칙 기반 전환율 평가 — {category}] (ML 모델 미사용)"]

            if "OVERALL_CVR" in df.columns and len(df) >= 3:
                recent = df.tail(3)["OVERALL_CVR"]
                avg_cvr = float(recent.mean())
                trend = float(recent.iloc[-1] - recent.iloc[0])
                trend_dir = "개선 추세" if trend > 0 else "악화 추세"
                findings.append(f"- 최근 3개월 평균 CVR: {avg_cvr:.1f}%")
                findings.append(f"- CVR 추세: {trend_dir} ({trend:+.1f}%p)")

                # 단순 추세 기반 예측
                projected = avg_cvr + trend / 3
                findings.append(f"- 추세 기반 다음 달 추정: {projected:.1f}%")
            else:
                findings.append("- 전환율 추세 데이터 부족")

            return "\n".join(findings)
        except Exception as fallback_exc:
            logger.exception("규칙 기반 평가도 실패")
            return f"{_ERROR_PREFIX} ML 예측 및 규칙 기반 평가 모두 실패: {fallback_exc}"


def run_what_if(
    session: Session,
    category: str,
    scenario: dict[str, float],
) -> str:
    """What-if 시나리오 시뮬레이션을 실행하여 결과를 반환.

    SimulationEngine을 사용하여 채널 변경 시나리오의 영향을 예측한다.

    Args:
        session: Snowpark 세션
        category: 상품 카테고리명
        scenario: 채널별 변경 비율 딕셔너리
            예: {"인바운드": +30, "플랫폼": -5}

    Returns:
        시뮬레이션 결과 요약 문자열
    """
    try:
        from ml.simulation_engine import SimulationEngine

        model = _get_trained_model(session)
        sim = SimulationEngine(session, model)
        result_df = sim.run_scenario(category, scenario)

        header = (
            f"=== What-if 시뮬레이션: {category} ===\n"
            f"시나리오: {_safe_json(scenario)}\n"
        )

        if isinstance(result_df, pd.DataFrame):
            return header + _df_to_summary(result_df)
        return header + _safe_json(result_df)

    except ImportError as exc:
        return (
            f"[시뮬레이션] ML 모듈 로드 실패: {exc}. "
            "ml/conversion_model.py, ml/simulation_engine.py를 확인하세요."
        )
    except Exception as exc:
        logger.exception(
            "run_what_if 실패: category=%s, scenario=%s",
            category,
            scenario,
        )
        return f"{_ERROR_PREFIX} 시뮬레이션 실패: {exc}"


# ---------------------------------------------------------------------------
# Cortex FORECAST / ANOMALY 도구
# ---------------------------------------------------------------------------


def query_forecast(
    session: Session,
    state: Optional[str] = None,
) -> str:
    """Cortex FORECAST 시도별 CONTRACT_COUNT 예측 결과를 조회.

    Args:
        session: Snowpark 세션
        state: 시도 필터 (None이면 전체)

    Returns:
        CONTRACT_COUNT 예측 요약 문자열
    """
    try:
        client = SnowflakeClient(session)
        df = client.load_forecast(metric="CONTRACT_COUNT")
        if df.empty:
            return "[FORECAST 데이터 없음] CONTRACT_COUNT 예측이 없습니다."

        if state and "SERIES_KEY" in df.columns:
            df = df[df["SERIES_KEY"] == state]

        if df.empty:
            return f"[FORECAST] '{state}' 시도의 예측 데이터 없음"

        findings: list[str] = []

        # 시도별 예측 요약
        series_list = df["SERIES_KEY"].unique() if "SERIES_KEY" in df.columns else []
        for s in sorted(series_list):
            s_df = df[df["SERIES_KEY"] == s].sort_values("TS")
            total_forecast = float(s_df["FORECAST"].sum()) if "FORECAST" in s_df.columns else 0
            findings.append(
                f"{s}: 향후 {len(s_df)}개월 예측 합계 {total_forecast:,.0f}건"
            )

            # 월별 상세 (최대 3개월)
            for _, row in s_df.head(3).iterrows():
                ts = row.get("TS", "")
                fc = float(row.get("FORECAST", 0))
                lo = float(row.get("LOWER", 0))
                hi = float(row.get("UPPER", 0))
                findings.append(
                    f"  {ts}: {fc:,.0f}건 (95% CI: {lo:,.0f}~{hi:,.0f})"
                )

        # 전체 합계
        if "FORECAST" in df.columns:
            grand_total = float(df["FORECAST"].sum())
            findings.append(f"전체 예측 합계: {grand_total:,.0f}건")

        header = "[Cortex FORECAST — 시도별 CONTRACT_COUNT 예측]"
        return header + "\n" + "\n".join(f"- {f}" for f in findings)
    except Exception as exc:
        logger.exception("query_forecast 실패")
        return f"[FORECAST 조회 실패: {exc}]"


def query_anomalies(
    session: Session,
    state: Optional[str] = None,
) -> str:
    """Cortex ANOMALY 탐지 결과를 조회하여 사전 계산된 이상 요약 반환.

    Cortex 결과가 있으면 사용하고, 없으면 MoM 변화 > 2.5 std로 직접 계산.

    Args:
        session: Snowpark 세션
        state: 시도 필터 (None이면 전체)

    Returns:
        이상 탐지 결과 요약 문자열
    """
    try:
        client = SnowflakeClient(session)
        df = client.load_anomalies()

        # Cortex ANOMALY 결과가 있는 경우
        if not df.empty:
            if state and "SERIES_KEY" in df.columns:
                df = df[df["SERIES_KEY"] == state]

            is_anomaly_col = df.get("IS_ANOMALY", pd.Series(dtype=bool))
            anomalies = df[is_anomaly_col == True]  # noqa: E712

            if anomalies.empty:
                return "[이상 탐지 결과] 모든 지역 정상 범위 내"

            findings: list[str] = []
            for _, row in anomalies.iterrows():
                series = row.get("SERIES_KEY", "")
                ts = row.get("TS", "")
                observed = float(row.get("OBSERVED", 0))
                expected = float(row.get("EXPECTED", 0))
                deviation = observed - expected
                pct = (deviation / max(expected, 1)) * 100
                direction = "초과" if deviation > 0 else "미달"
                findings.append(
                    f"{series} ({ts}): 실측 {observed:,.0f}건 vs "
                    f"기대 {expected:,.0f}건 ({direction} {abs(pct):.1f}%)"
                )

            header = f"[이상 탐지 결과 — {len(anomalies)}건 감지]"
            return header + "\n" + "\n".join(f"- {f}" for f in findings)

        # Cortex 결과가 없으면 MoM 변화 기반 이상 탐지 폴백
        heatmap = client.load_regional_heatmap()
        if heatmap.empty:
            return "[ANOMALY 데이터 없음]"

        if "INSTALL_STATE" not in heatmap.columns or "YEAR_MONTH" not in heatmap.columns:
            return "[ANOMALY] 이상 탐지에 필요한 컬럼 부족"

        numeric_cols = heatmap.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            return "[ANOMALY] 수치 컬럼 없음"

        value_col = numeric_cols[0]
        grouped = (
            heatmap.groupby(["INSTALL_STATE", "YEAR_MONTH"])[value_col]
            .sum()
            .reset_index()
        )
        grouped = grouped.sort_values(["INSTALL_STATE", "YEAR_MONTH"])
        grouped["MOM_CHANGE"] = grouped.groupby("INSTALL_STATE")[value_col].pct_change()

        mom_std = float(grouped["MOM_CHANGE"].std())
        mom_mean = float(grouped["MOM_CHANGE"].mean())
        threshold = 2.5

        outliers = grouped[
            grouped["MOM_CHANGE"].notna()
            & (
                (grouped["MOM_CHANGE"] > mom_mean + threshold * mom_std)
                | (grouped["MOM_CHANGE"] < mom_mean - threshold * mom_std)
            )
        ]

        if outliers.empty:
            return "[이상 탐지 결과] MoM 변화 기준 이상치 없음 (2.5 std 이내)"

        findings = []
        for _, row in outliers.iterrows():
            pct = float(row["MOM_CHANGE"]) * 100
            direction = "급증" if pct > 0 else "급감"
            findings.append(
                f"{row['INSTALL_STATE']} ({row['YEAR_MONTH']}): "
                f"MoM {direction} {abs(pct):.1f}%"
            )

        header = f"[MoM 기반 이상 탐지 — {len(outliers)}건 (>2.5 std)]"
        return header + "\n" + "\n".join(f"- {f}" for f in findings)

    except Exception as exc:
        logger.exception("query_anomalies 실패")
        return f"[ANOMALY 조회 실패: {exc}]"


def get_feature_importance(session: Session) -> str:
    """ML 모델의 피처 중요도를 반환.

    ModelExplainer를 사용하여 모델이 학습한 피처 중요도를 조회한다.

    Args:
        session: Snowpark 세션

    Returns:
        피처 중요도 요약 문자열
    """
    try:
        from ml.explainer import ModelExplainer

        model = _get_trained_model(session)
        if model is None:
            return "[피처 중요도] ML 모델 초기화 실패"
        explainer = ModelExplainer(model)
        importance_df = explainer.feature_importance()

        header = "=== ML 모델 피처 중요도 ===\n"

        if isinstance(importance_df, pd.DataFrame):
            return header + _df_to_summary(importance_df)
        return header + _safe_json(importance_df)

    except ImportError as exc:
        return (
            f"[피처 중요도] ML 모듈 로드 실패: {exc}. "
            "ml/explainer.py를 확인하세요."
        )
    except Exception as exc:
        logger.exception("get_feature_importance 실패")
        return f"{_ERROR_PREFIX} 피처 중요도 조회 실패: {exc}"
