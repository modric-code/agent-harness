# -*- coding: utf-8 -*-
"""
AI Agent Harness 评测平台 —— Streamlit 可视化界面
调用同目录 harness_core.py 的任务管理 / 批量执行 / Trace / 自动评测 / 统计模块。

运行：streamlit run harness_web.py
说明：页面加载后仅展示测试任务集与「运行批量测试」按钮，不自动执行；
      点击按钮后才执行批量测试并展示结果，便于分阶段截图。
"""
import sys

# Windows 控制台编码兜底
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

import pandas as pd
import streamlit as st

from harness_core import (build_test_suite, HarnessRunner, TraceLogger,
                          Evaluator, ReportSummarizer)

st.set_page_config(page_title="AI Agent Harness 评测平台", page_icon="🧪", layout="wide")

st.title("🧪 AI Agent Harness 评测平台")
st.caption("被测对象：大学生就业求职智能体（本会话已完成） | 评测底座：harness_core.py")

# 初始化 session_state
if "ran" not in st.session_state:
    st.session_state.ran = False

# 加载测试任务集
cases = build_test_suite()
case_df = pd.DataFrame([
    {"编号": c.task_id, "类型": c.task_type, "用例名称": c.name}
    for c in cases
])

st.subheader(f"测试任务集（{len(cases)} 条）")
st.dataframe(case_df, use_container_width=True, hide_index=True)

st.markdown("---")

# 运行按钮：未点击前不执行任何批量测试
if not st.session_state.ran:
    if st.button("🚀 运行批量测试", type="primary"):
        with st.spinner("正在批量执行测试用例..."):
            runner = HarnessRunner()
            logger = TraceLogger()
            evaluator = Evaluator()
            traces = runner.run_batch(cases)
            for t in traces:
                logger.log(t)
            evals = [evaluator.evaluate(c, t) for c, t in zip(cases, traces)]
            summary = ReportSummarizer.summarize(traces, evals)
        st.session_state.ran = True
        st.session_state.traces = traces
        st.session_state.evals = evals
        st.session_state.summary = summary
        st.session_state.trace_json = logger.to_json()

# 展示结果
if st.session_state.ran:
    summary = st.session_state.summary
    traces = st.session_state.traces
    evals = st.session_state.evals
    name_map = {c.task_id: c.name for c in cases}

    st.subheader("📊 测试结果汇总")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("任务总数", summary["total"])
    c2.metric("成功", summary["success"])
    c3.metric("失败", summary["failed"])
    c4.metric("空输入拦截", summary["invalid_input"])
    c5.metric("有效成功率", f"{summary['success_rate']}%")

    st.subheader("🧾 Trace 执行记录")
    rows = []
    for t, e in zip(traces, evals):
        rows.append({
            "编号": t.task_id,
            "用例": name_map.get(t.task_id, t.task_id),
            "类型": t.task_type,
            "状态": t.status,
            "耗时(ms)": t.duration_ms,
            "得分": e["score"],
            "通过": "是" if e["passed"] else ("拦截" if e["passed"] is None else "否"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if summary["failed_list"]:
        st.subheader("❌ 失败任务清单")
        st.dataframe(pd.DataFrame(summary["failed_list"]), use_container_width=True, hide_index=True)

    st.download_button("⬇️ 下载 Trace 日志 (JSONL)",
                       data=st.session_state.trace_json,
                       file_name="harness_trace.jsonl",
                       mime="text/plain")
