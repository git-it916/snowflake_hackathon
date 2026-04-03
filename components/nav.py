"""Top navigation bar -- SiS 호환 (st.page_link 없는 환경 지원)."""
from __future__ import annotations

import streamlit as st

# st.page_link 지원 여부 확인 (SiS 구버전 대응)
_HAS_PAGE_LINK = hasattr(st, "page_link")


def safe_page_link(path: str, label: str, icon: str = "") -> None:
    """st.page_link 호환 래퍼. SiS 구버전에서는 caption으로 폴백."""
    if _HAS_PAGE_LINK:
        kwargs = {"page": path, "label": label}
        if icon:
            kwargs["icon"] = icon
        st.page_link(**kwargs)
    else:
        display = f"{icon} {label}" if icon else label
        st.caption(display)


_PAGES = {
    "랜딩": "app.py",
    "진단": "pages/1_진단.py",
    "기회 분석": "pages/2_기회_분석.py",
    "AI 전략": "pages/3_AI_전략.py",
}


def render_top_nav(active: str = "랜딩") -> None:
    cols = st.columns([2, 1, 1, 1, 1, 2])
    with cols[0]:
        st.markdown(
            '<span style="font-weight:600;font-size:1rem;color:#22d3ee;">'
            'Cortex Analytics</span>',
            unsafe_allow_html=True,
        )
    for i, (label, path) in enumerate(_PAGES.items()):
        with cols[i + 1]:
            if label == active:
                st.markdown(
                    f'<span style="padding:4px 14px;font-size:0.85rem;color:#fff;'
                    f'background:rgba(255,255,255,0.1);border-radius:100px;'
                    f'border:1px solid rgba(255,255,255,0.15);display:inline-block;">'
                    f'{label}</span>',
                    unsafe_allow_html=True,
                )
            else:
                safe_page_link(path, label=label)
    with cols[-1]:
        st.markdown(
            '<span style="font-size:0.75rem;color:rgba(255,255,255,0.4);">'
            '🟢 Data Synced</span>',
            unsafe_allow_html=True,
        )
    st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:8px 0 24px 0;">', unsafe_allow_html=True)
