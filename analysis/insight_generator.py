"""데이터 기반 자동 인사이트 생성 모듈.

각 페이지의 차트 데이터를 분석하여 핵심 발견사항과 추천 액션을
자동으로 생성한다. 그래프만 나열하지 않고 "So what?"에 답한다.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

_STAGE_LABELS = {
    "CONSULT_REQUEST": "상담요청",
    "SUBSCRIPTION": "가입신청",
    "REGISTEND": "접수완료",
    "OPEN": "개통",
    "PAYEND": "납입완료",
}


# =========================================================================
# Theme 1: 퍼널 인사이트
# =========================================================================

def generate_funnel_insights(
    stage_drop_df: pd.DataFrame,
    bottleneck_df: pd.DataFrame,
    funnel_ts_df: pd.DataFrame,
    category: Optional[str] = None,
) -> dict:
    """퍼널 데이터에서 핵심 인사이트를 자동 생성.

    Returns:
        {
            "headline": str,        # 한 줄 핵심 발견
            "severity": str,        # "critical" | "warning" | "good"
            "findings": list[str],  # 세부 발견사항 3~5개
            "actions": list[str],   # 추천 액션 2~3개
            "metrics": dict,        # 핵심 수치
        }
    """
    result = {
        "headline": "",
        "severity": "warning",
        "findings": [],
        "actions": [],
        "metrics": {},
    }

    if stage_drop_df.empty:
        result["headline"] = "퍼널 데이터가 없습니다."
        return result

    # 카테고리 필터 — "전체"일 때 소규모 카테고리(상조/부동산/이사/다이렉트자보) 제외
    _MAJOR_CATS = ["인터넷", "렌탈", "모바일", "알뜰 요금제", "유심만", "기업용인터넷"]
    df = stage_drop_df.copy()
    if category:
        cat_col = "MAIN_CATEGORY_NAME" if "MAIN_CATEGORY_NAME" in df.columns else "CATEGORY"
        df = df[df[cat_col] == category]
    else:
        cat_col = "MAIN_CATEGORY_NAME" if "MAIN_CATEGORY_NAME" in df.columns else "CATEGORY"
        if cat_col in df.columns:
            df = df[df[cat_col].isin(_MAJOR_CATS)]

    if df.empty:
        result["headline"] = f"{category or '전체'} 카테고리에 데이터가 없습니다."
        return result

    # 최근 월 데이터
    latest_month = df["YEAR_MONTH"].max()
    latest = df[df["YEAR_MONTH"] == latest_month]

    # 최대 병목 단계 찾기 — 건수 기반 직접 계산 (DROP_RATE 컬럼 신뢰 불가)
    stage_agg = (
        latest.groupby(["STAGE_NAME", "STAGE_ORDER"])
        .agg(CURR_STAGE_COUNT=("CURR_STAGE_COUNT", "sum"))
        .reset_index()
        .sort_values("STAGE_ORDER")
    )
    stage_agg["PREV_COUNT"] = stage_agg["CURR_STAGE_COUNT"].shift(1)
    stage_agg["CALC_DROP_RATE"] = (
        (stage_agg["PREV_COUNT"] - stage_agg["CURR_STAGE_COUNT"])
        / stage_agg["PREV_COUNT"].replace(0, float("nan"))
    ).clip(lower=0, upper=1)

    drop_valid = stage_agg[stage_agg["CALC_DROP_RATE"].notna() & (stage_agg["CALC_DROP_RATE"] > 0)]
    if not drop_valid.empty:
        worst = drop_valid.loc[drop_valid["CALC_DROP_RATE"].idxmax()]
        worst_stage = _STAGE_LABELS.get(worst["STAGE_NAME"], worst["STAGE_NAME"])
        worst_drop = float(worst["CALC_DROP_RATE"]) * 100
        worst_count = int(worst.get("PREV_COUNT", 0) - worst.get("CURR_STAGE_COUNT", 0))

        result["metrics"]["worst_stage"] = worst_stage
        result["metrics"]["worst_drop_pct"] = worst_drop
        result["metrics"]["lost_customers"] = worst_count

        # 심각도 판정 (텔레콤 퍼널 특성: 개통 단계 20-30% 이탈은 업계 평균)
        if worst_drop > 40:
            result["severity"] = "critical"
            result["headline"] = f"{worst_stage} 단계에서 {worst_drop:.0f}% 이탈 — 즉각 개선 필요"
        elif worst_drop > 20:
            result["severity"] = "warning"
            result["headline"] = f"{worst_stage} 단계 병목 감지 (이탈률 {worst_drop:.0f}%) — 개선 여지 존재"
        else:
            result["severity"] = "good"
            result["headline"] = f"퍼널 전반 양호 — 최대 이탈 {worst_drop:.0f}% ({worst_stage})"

        # 세부 발견
        result["findings"].append(
            f"최대 병목: **{worst_stage}** 단계에서 {worst_drop:.1f}% 이탈 (약 {worst_count:,}건 손실)"
        )

    # 전체 전환율
    if not funnel_ts_df.empty:
        ts = funnel_ts_df.copy()
        if category:
            ts = ts[ts.get("MAIN_CATEGORY_NAME", pd.Series()) == category]
        if ts.empty and not funnel_ts_df.empty:
            ts = funnel_ts_df[funnel_ts_df["MAIN_CATEGORY_NAME"] == "인터넷"]

        if not ts.empty and "OVERALL_CVR" in ts.columns:
            ts = ts.sort_values("YEAR_MONTH")
            latest_cvr = float(ts["OVERALL_CVR"].iloc[-1])
            avg_cvr = float(ts["OVERALL_CVR"].mean())
            result["metrics"]["current_cvr"] = latest_cvr
            result["metrics"]["avg_cvr"] = avg_cvr

            if latest_cvr < avg_cvr * 0.9:
                result["findings"].append(
                    f"현재 전환율 **{latest_cvr:.1f}%**는 평균({avg_cvr:.1f}%)보다 **낮음** — 하락 추세"
                )
            elif latest_cvr > avg_cvr * 1.1:
                result["findings"].append(
                    f"현재 전환율 **{latest_cvr:.1f}%**는 평균({avg_cvr:.1f}%)보다 **높음** — 개선 추세"
                )
            else:
                result["findings"].append(
                    f"현재 전환율 **{latest_cvr:.1f}%** (평균 {avg_cvr:.1f}% — 안정적)"
                )

            # 3개월 트렌드
            if len(ts) >= 3:
                recent_3 = ts.tail(3)["OVERALL_CVR"]
                trend = float(recent_3.iloc[-1] - recent_3.iloc[0])
                if trend > 1:
                    result["findings"].append(f"최근 3개월 전환율 **+{trend:.1f}%p 상승** 추세")
                elif trend < -1:
                    result["findings"].append(f"최근 3개월 전환율 **{trend:.1f}%p 하락** 추세")

    # 카테고리 비교 (전체일 때)
    if not category and not bottleneck_df.empty:
        worst_cat = bottleneck_df.loc[bottleneck_df["AVG_DROP_RATE"].idxmax()]
        result["findings"].append(
            f"카테고리 중 **{worst_cat['MAIN_CATEGORY_NAME']}**의 병목이 가장 심각 "
            f"({_STAGE_LABELS.get(worst_cat['WORST_BOTTLENECK_STAGE'], worst_cat['WORST_BOTTLENECK_STAGE'])} 단계, "
            f"평균 이탈 {float(worst_cat['AVG_DROP_RATE'])*100:.0f}%)"
        )

    # 추천 액션
    if result["severity"] == "critical":
        result["actions"] = [
            f"**{result['metrics'].get('worst_stage', '병목')} 단계** 프로세스 긴급 점검",
            "해당 단계의 고객 이탈 원인 정성 조사 (설문, 콜로그 분석)",
            "자동화/간소화로 이탈 장벽 제거 (예: 서류 축소, 온라인 처리 전환)",
        ]
    elif result["severity"] == "warning":
        result["actions"] = [
            f"{result['metrics'].get('worst_stage', '병목')} 단계 전환율 월간 모니터링 강화",
            "이탈 고객 리마인더 자동 발송 검토",
        ]
    else:
        result["actions"] = [
            "현재 양호한 퍼널 유지, 월간 대시보드 모니터링 지속",
            "추가 개선 여지가 있는 카테고리에 집중",
        ]

    return result


# =========================================================================
# Theme 2: 채널 인사이트
# =========================================================================

def generate_channel_insights(
    channel_df: pd.DataFrame,
    ch_col: str = "RECEIVE_PATH_NAME",
) -> dict:
    """채널 데이터에서 핵심 인사이트를 자동 생성."""
    result = {
        "headline": "",
        "severity": "warning",
        "findings": [],
        "actions": [],
        "metrics": {},
    }

    if channel_df.empty:
        result["headline"] = "채널 데이터가 없습니다."
        return result

    df = channel_df.copy()

    # 최근 6개월 집계
    if "YEAR_MONTH" in df.columns:
        df["YEAR_MONTH"] = pd.to_datetime(df["YEAR_MONTH"])
        recent = df["YEAR_MONTH"].nlargest(6).min()
        df = df[df["YEAR_MONTH"] >= recent]

    # 채널별 합산
    ch_agg = df.groupby(ch_col).agg(
        CONTRACT_COUNT=("CONTRACT_COUNT", "sum"),
        PAYEND_CVR=("PAYEND_CVR", "mean"),
        AVG_NET_SALES=("AVG_NET_SALES", "mean"),
    ).reset_index()

    if ch_agg.empty:
        result["headline"] = "채널 집계 데이터가 없습니다."
        return result

    total = ch_agg["CONTRACT_COUNT"].sum()
    top_channel = ch_agg.loc[ch_agg["CONTRACT_COUNT"].idxmax()]
    top_name = str(top_channel[ch_col])
    top_share = float(top_channel["CONTRACT_COUNT"]) / max(total, 1) * 100
    top_cvr = float(top_channel["PAYEND_CVR"])

    result["metrics"]["top_channel"] = top_name
    result["metrics"]["top_share"] = top_share
    result["metrics"]["total_channels"] = len(ch_agg)

    # HHI 계산
    shares = ch_agg["CONTRACT_COUNT"].astype(float) / max(total, 1)
    hhi = float((shares ** 2).sum())
    result["metrics"]["hhi"] = hhi

    # 최고 효율 채널 (CVR 기준, 최소 볼륨 이상)
    min_vol = ch_agg["CONTRACT_COUNT"].quantile(0.25)
    qualified = ch_agg[ch_agg["CONTRACT_COUNT"] >= min_vol]
    if not qualified.empty:
        best_eff = qualified.loc[qualified["PAYEND_CVR"].idxmax()]
        best_name = str(best_eff[ch_col])
        best_cvr = float(best_eff["PAYEND_CVR"])
        result["metrics"]["best_efficiency_channel"] = best_name
        result["metrics"]["best_efficiency_cvr"] = best_cvr

    # 헤드라인
    if hhi > 0.25:
        result["severity"] = "critical"
        result["headline"] = f"**{top_name}** 채널에 과도 의존 (점유율 {top_share:.0f}%, HHI={hhi:.2f}) — 다각화 시급"
    elif top_share > 40:
        result["severity"] = "warning"
        result["headline"] = f"**{top_name}** 채널 의존도 높음 ({top_share:.0f}%) — 대안 채널 육성 권장"
    else:
        result["severity"] = "good"
        result["headline"] = f"채널 다각화 양호 (HHI={hhi:.2f}) — {top_name}이 리드 ({top_share:.0f}%)"

    # 세부 발견
    result["findings"].append(
        f"총 **{len(ch_agg)}개** 채널 운영 중, 상위 채널 **{top_name}**이 전체의 **{top_share:.0f}%** 차지"
    )
    result["findings"].append(
        f"채널 집중도 HHI = **{hhi:.3f}** "
        f"({'과도 집중' if hhi > 0.25 else '적정' if hhi > 0.15 else '건전한 분산'})"
    )
    if "best_efficiency_channel" in result["metrics"]:
        result["findings"].append(
            f"전환율 최고 채널: **{result['metrics']['best_efficiency_channel']}** "
            f"(CVR {result['metrics']['best_efficiency_cvr']:.1f}%)"
        )
        if result["metrics"]["best_efficiency_channel"] != top_name:
            result["findings"].append(
                f"볼륨 1위({top_name})와 효율 1위({result['metrics']['best_efficiency_channel']})가 다름 "
                f"→ 효율 채널에 예산 이동 시 전체 CVR 개선 가능"
            )

    # 추천 액션
    if hhi > 0.25:
        result["actions"] = [
            f"**{top_name}** 의존도 축소 — 2~3순위 채널 예산 확대",
            f"전환율 높은 **{result['metrics'].get('best_efficiency_channel', '대안 채널')}** 채널 스케일업",
            "신규 채널 테스트 (분기 1회 이상)",
        ]
    elif top_share > 40:
        result["actions"] = [
            f"**{result['metrics'].get('best_efficiency_channel', '고효율 채널')}**에 예산 10~20% 재배치",
            "채널별 ROI 월간 모니터링 체계 구축",
        ]
    else:
        result["actions"] = [
            "현재 채널 믹스 유지, 분기별 효율성 리밸런싱",
            "저효율 채널 예산 축소 + 고효율 채널 확대",
        ]

    return result


# =========================================================================
# Theme 3: 지역 인사이트
# =========================================================================

def generate_regional_insights(
    heatmap_df: pd.DataFrame,
) -> dict:
    """지역 데이터에서 핵심 인사이트를 자동 생성."""
    result = {
        "headline": "",
        "severity": "warning",
        "findings": [],
        "actions": [],
        "metrics": {},
    }

    if heatmap_df.empty:
        result["headline"] = "지역 데이터가 없습니다."
        return result

    df = heatmap_df.copy()

    # 시도별 집계
    state_agg = df.groupby("INSTALL_STATE").agg(
        CONTRACT_COUNT=("CONTRACT_COUNT", "sum"),
        DEMAND_SCORE=("DEMAND_SCORE", "mean"),
    ).reset_index()

    # 도시별 성장
    growth_col = "GROWTH_3M" if "GROWTH_3M" in df.columns else None
    city_col = "INSTALL_CITY" if "INSTALL_CITY" in df.columns else None

    # 상위/하위 시도
    top_state = state_agg.loc[state_agg["DEMAND_SCORE"].idxmax()]
    bot_state = state_agg.loc[state_agg["DEMAND_SCORE"].idxmin()]
    result["metrics"]["top_state"] = str(top_state["INSTALL_STATE"])
    result["metrics"]["top_score"] = float(top_state["DEMAND_SCORE"])
    result["metrics"]["bottom_state"] = str(bot_state["INSTALL_STATE"])
    result["metrics"]["total_cities"] = len(df)

    # 성장 도시
    growth_cities = []
    if growth_col and city_col:
        growing = df[df[growth_col] > 0.1]
        if not growing.empty:
            top_growth = growing.nlargest(3, growth_col)
            for _, row in top_growth.iterrows():
                growth_cities.append({
                    "city": f"{row['INSTALL_STATE']} {row[city_col]}",
                    "growth": float(row[growth_col]),
                })
    result["metrics"]["growth_cities"] = growth_cities

    # 번들 비율
    if "BUNDLE_RATIO" in df.columns:
        avg_bundle = float(df["BUNDLE_RATIO"].mean())
        result["metrics"]["avg_bundle_ratio"] = avg_bundle

    # 헤드라인
    if growth_cities:
        cities_str = ", ".join([c["city"] for c in growth_cities[:2]])
        result["severity"] = "good"
        result["headline"] = f"성장 핫스팟 발견: **{cities_str}** — 마케팅 집중 추천"
    else:
        result["headline"] = f"수요 1위 **{result['metrics']['top_state']}**, 최하위 **{result['metrics']['bottom_state']}**"

    # 세부 발견
    result["findings"].append(
        f"총 **{len(df)}개** 시군구 분석, 수요 점수 1위 **{result['metrics']['top_state']}** "
        f"(점수 {result['metrics']['top_score']:.2f})"
    )

    gap = float(top_state["DEMAND_SCORE"]) - float(bot_state["DEMAND_SCORE"])
    result["findings"].append(
        f"지역 간 수요 격차: {gap:.1f}점 — "
        f"{'격차가 크므로 지역별 차별 전략 필요' if gap > 3 else '비교적 균등한 분포'}"
    )

    if growth_cities:
        for gc in growth_cities:
            pct = gc["growth"]
            result["findings"].append(
                f"**{gc['city']}**: 3개월 성장률 "
                f"{'**' + f'{pct:.0%}' + '**' if pct < 10 else '**' + f'{pct:.1f}' + '**'}"
            )

    if "avg_bundle_ratio" in result["metrics"]:
        br = result["metrics"]["avg_bundle_ratio"] * 100
        result["findings"].append(
            f"평균 번들 비율 **{br:.0f}%** — "
            f"{'번들 판매 강화 여지 있음' if br < 50 else '번들 침투율 양호'}"
        )

    # 추천 액션
    if growth_cities:
        result["actions"] = [
            f"**{growth_cities[0]['city']}**에 마케팅 예산 우선 배정",
            "성장 지역에 맞는 상품 번들 프로모션 기획",
            "하위 지역은 비용 효율 채널(온라인) 중심으로 전환",
        ]
    else:
        result["actions"] = [
            "지역별 수요 점수 기반 예산 재배치",
            "번들 비율 낮은 지역에 결합상품 프로모션 추진",
        ]

    return result
