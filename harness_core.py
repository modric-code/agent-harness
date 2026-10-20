# -*- coding: utf-8 -*-
"""
AI Agent Harness 评测平台 —— 核心源代码（C 类大作业）

被测对象：本会话已经开发完成的大学生就业求职智能体
          （位于 C:\\Users\\Lenovo\\b03-job-agent，以下简称「求职智能体」）。
本 Harness 不训练模型、不修改求职智能体内部代码，仅作为「评测底座」，实现：
    1) 任务管理 / 测试用例管理
    2) Agent 调度执行（批量、超时、异常、输入预处理）
    3) Trace 追踪日志（任务状态 / 执行结果 / 耗时 / 错误 / 工具调用信息）
    4) 自动评测（规则打分：简历优化完整性 / 岗位匹配准确度 / 面试题与 JD 相关性）
    5) 结果统计汇总（成功率 / 失败任务清单 / 优化前后对比）

技术栈：Python 3.8+（本核心为纯标准库）；上层可用 Streamlit 做可视化（见部署与运行说明.md）。

运行方式（本文件作为作业提交的核心源代码，自包含一个演示入口）：
    python harness_core.py
"""

import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 0. 定位被测「求职智能体」，加入 import 路径（直接调用，不复制其代码）
# ---------------------------------------------------------------------------
# 可通过环境变量 JOB_AGENT_HOME 覆盖被测智能体路径，默认指向本会话开发成果目录。
AGENT_HOME = os.environ.get("JOB_AGENT_HOME", r"C:\Users\Lenovo\b03-job-agent")
if os.path.isdir(AGENT_HOME) and AGENT_HOME not in sys.path:
    sys.path.insert(0, AGENT_HOME)

# 被测智能体公共接口（与求职智能体源码一致，详见其 README）：
#   每个 Agent 均为 (llm=..., kb=...).run(context_dict) -> dict
from agents.jd_analyzer import JDAnalyzer           # 岗位分析
from agents.resume_parser import ResumeParser       # 简历解析
from agents.resume_optimizer import ResumeOptimizer  # 简历优化
from agents.job_matcher import JobMatcher           # 岗位匹配
from agents.question_generator import QuestionGenerator  # 面试题生成
from core.rag import KnowledgeBase                  # RAG 知识库
from core.llm import get_llm                        # LLM 后端
from core.document_loader import load_document      # 文件解析
from config import SAMPLE_DIR                       # 示例数据目录


# ---------------------------------------------------------------------------
# 1. 数据结构：测试用例、Trace 记录
# ---------------------------------------------------------------------------
class InvalidInputError(ValueError):
    """输入预处理未通过（缺字段 / 空输入）时抛出的自定义异常。"""


@dataclass
class TestCase:
    """一条测试用例。"""
    task_id: str                        # 用例编号，如 T01
    task_type: str                      # 任务类型：jd_analysis / resume_parse /
                                        #           resume_optimize / job_match / question_gen
    name: str                           # 用例名称
    inputs: Dict[str, Any]              # 任务输入（jd_text / resume_text / jd_list ...）
    ground_truth: Dict[str, Any] = field(default_factory=dict)  # 评测用的期望值
    timeout_s: int = 30                 # 超时阈值（秒）


@dataclass
class TraceRecord:
    """一条 Trace 记录：记录一次任务执行的完整轨迹。"""
    task_id: str
    task_type: str
    status: str                         # success / failed / timeout / error / invalid_input
    start_time: str
    end_time: str
    duration_ms: int                    # 耗时（毫秒）
    result_summary: Dict[str, Any] = field(default_factory=dict)  # 关键结果字段
    tool_calls: List[str] = field(default_factory=list)           # 工具调用（Agent 步骤）信息
    error: str = ""                     # 错误信息（失败时）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 2. 任务管理模块
# ---------------------------------------------------------------------------
class TaskManager:
    """测试用例管理：增 / 删 / 查 / 列表。"""

    def __init__(self):
        self._cases: Dict[str, TestCase] = {}

    def add(self, case: TestCase) -> None:
        self._cases[case.task_id] = case

    def add_many(self, cases: List[TestCase]) -> None:
        for c in cases:
            self.add(c)

    def remove(self, task_id: str) -> None:
        self._cases.pop(task_id, None)

    def get(self, task_id: str) -> Optional[TestCase]:
        return self._cases.get(task_id)

    def list(self) -> List[TestCase]:
        return list(self._cases.values())

    def __len__(self) -> int:
        return len(self._cases)


# ---------------------------------------------------------------------------
# 3. Agent 调度执行模块
# ---------------------------------------------------------------------------
class HarnessRunner:
    """负责调用被测求职智能体执行任务：输入预处理 -> 分发 -> 超时/异常 -> 结果摘要。"""

    def __init__(self):
        # 知识库与 LLM 后端只初始化一次，跨任务复用（提高批量执行效率）
        self._kb = None
        self._llm = None

    def _ensure_runtime(self):
        if self._kb is None:
            self._kb = KnowledgeBase()
        if self._llm is None:
            self._llm = get_llm()

    # ---- 输入预处理（作业二改进点：空输入前置拦截、字段归一化） ----
    @staticmethod
    def _preprocess(task_type: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """校验并规范化输入；不合法时抛 InvalidInputError。"""
        out = dict(inputs)
        if task_type in ("resume_parse", "resume_optimize", "job_match", "question_gen"):
            resume = (out.get("resume_text") or "").strip()
            if not resume:
                raise InvalidInputError("resume_text 为空")
            out["resume_text"] = resume
        if task_type in ("jd_analysis", "resume_optimize", "question_gen"):
            jd = (out.get("jd_text") or "").strip()
            if not jd:
                raise InvalidInputError("jd_text 为空")
            out["jd_text"] = jd
        if task_type == "job_match":
            jd_list = out.get("jd_list") or []
            # 单条 JD 也统一包装为列表，避免下游取字段异常（作业二改进点）
            if isinstance(jd_list, str):
                jd_list = [{"title": "目标岗位", "text": jd_list}]
            if not jd_list:
                raise InvalidInputError("jd_list 为空")
            out["jd_list"] = jd_list
        return out

    # ---- 任务分发：映射到具体求职智能体 ----
    def _dispatch(self, task_type: str, inputs: Dict[str, Any]):
        """执行指定任务类型，返回 (结果字典, 工具调用步骤列表)。"""
        self._ensure_runtime()
        kb, llm = self._kb, self._llm
        steps: List[str] = []

        if task_type == "jd_analysis":
            steps.append("JDAnalyzer.run")
            result = JDAnalyzer(llm=llm, kb=kb).run({"jd_text": inputs["jd_text"]})

        elif task_type == "resume_parse":
            steps.append("ResumeParser.run")
            result = ResumeParser(llm=llm, kb=kb).run({"resume_text": inputs["resume_text"]})

        elif task_type == "resume_optimize":
            steps += ["JDAnalyzer.run", "ResumeParser.run", "ResumeOptimizer.run"]
            jd = JDAnalyzer(llm=llm, kb=kb).run({"jd_text": inputs["jd_text"]})
            r = ResumeParser(llm=llm, kb=kb).run({"resume_text": inputs["resume_text"]})
            result = ResumeOptimizer(llm=llm, kb=kb).run({"jd_analysis": jd, "resume_data": r})

        elif task_type == "job_match":
            steps += ["ResumeParser.run", "JobMatcher.run"]
            r = ResumeParser(llm=llm, kb=kb).run({"resume_text": inputs["resume_text"]})
            result = JobMatcher(llm=llm, kb=kb).run({"resume_data": r, "jd_list": inputs["jd_list"]})

        elif task_type == "question_gen":
            steps += ["JDAnalyzer.run", "ResumeParser.run", "QuestionGenerator.run"]
            jd = JDAnalyzer(llm=llm, kb=kb).run({"jd_text": inputs["jd_text"]})
            r = ResumeParser(llm=llm, kb=kb).run({"resume_text": inputs["resume_text"]})
            result = QuestionGenerator(llm=llm, kb=kb).run({"jd_analysis": jd, "resume_data": r})

        else:
            raise ValueError("未知任务类型: %s" % task_type)

        return result, steps

    # ---- 结果摘要：抽取关键字段，供日志与评测使用 ----
    @staticmethod
    def _summarize(task_type: str, result: Dict[str, Any]) -> Dict[str, Any]:
        if task_type == "jd_analysis":
            return {
                "hard_skills": result.get("hard_skills", {}),
                "soft_skills": result.get("soft_skills", []),
                "bonus_items": result.get("bonus_items", []),
                "education": result.get("education", []),
            }
        if task_type == "resume_parse":
            return {
                "personal_info": result.get("personal_info", {}),
                "skills": result.get("skills", {}),
                "projects": result.get("projects", []),
                "experience": result.get("experience", []),
            }
        if task_type == "resume_optimize":
            return {
                "optimized_resume": result.get("optimized_resume", ""),
                "suggestions": result.get("suggestions", []),
                "missing_keywords": result.get("missing_keywords", []),
            }
        if task_type == "job_match":
            return {"match_reports": result.get("match_reports", [])}
        if task_type == "question_gen":
            return {
                "technical": result.get("technical", []),
                "project": result.get("project", []),
                "behavioral": result.get("behavioral", []),
            }
        return {}

    # ---- 单任务执行（含超时与异常） ----
    def execute(self, case: TestCase) -> TraceRecord:
        start = time.time()
        start_str = datetime.now().isoformat(timespec="seconds")
        trace = TraceRecord(task_id=case.task_id, task_type=case.task_type,
                            status="running", start_time=start_str,
                            end_time=start_str, duration_ms=0)

        try:
            inputs = self._preprocess(case.task_type, case.inputs)
        except InvalidInputError as e:
            trace.status = "invalid_input"
            trace.error = str(e)
            trace.end_time = datetime.now().isoformat(timespec="seconds")
            trace.duration_ms = int((time.time() - start) * 1000)
            return trace

        def _run():
            return self._dispatch(case.task_type, inputs)

        try:
            # 硬超时：用线程池执行，超时抛 TimeoutError
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_run)
                result, steps = future.result(timeout=case.timeout_s)
            trace.tool_calls = steps
            trace.result_summary = self._summarize(case.task_type, result)
            trace.status = "success"
        except FuturesTimeout:
            trace.status = "timeout"
            trace.error = "任务超时（>%ds）" % case.timeout_s
        except Exception as e:  # noqa: BLE001
            trace.status = "error"
            trace.error = "%s: %s" % (type(e).__name__, e)
            trace.error += "\n" + traceback.format_exc(limit=3)

        trace.end_time = datetime.now().isoformat(timespec="seconds")
        trace.duration_ms = int((time.time() - start) * 1000)
        return trace

    def run_batch(self, cases: List[TestCase]) -> List[TraceRecord]:
        """批量执行测试用例，返回 Trace 列表。"""
        traces = []
        for i, case in enumerate(cases, 1):
            print("[%d/%d] 执行 %s (%s)" % (i, len(cases), case.task_id, case.name))
            traces.append(self.execute(case))
        return traces


# ---------------------------------------------------------------------------
# 4. 基础日志记录模块（Trace 存储）
# ---------------------------------------------------------------------------
class TraceLogger:
    """Trace 日志：内存收集 + 落盘为 JSONL，支持查看。"""

    def __init__(self, path: Optional[str] = None):
        self.records: List[TraceRecord] = []
        self.path = path

    def log(self, trace: TraceRecord) -> None:
        self.records.append(trace)

    def to_json(self) -> str:
        return json.dumps([r.to_dict() for r in self.records],
                          ensure_ascii=False, indent=2)

    def save(self, path: Optional[str] = None) -> str:
        target = path or self.path or "harness_trace.jsonl"
        with open(target, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        return target

    def find(self, task_id: str) -> Optional[TraceRecord]:
        for r in self.records:
            if r.task_id == task_id:
                return r
        return None


# ---------------------------------------------------------------------------
# 5. 自动评测模块（规则打分，不依赖大模型）
# ---------------------------------------------------------------------------
class Evaluator:
    """针对求职智能体输出做规则打分，返回 (score, passed, details)。"""

    PASS_LINE = 60  # 及格线

    def evaluate(self, case: TestCase, trace: TraceRecord) -> Dict[str, Any]:
        if trace.status == "invalid_input":
            return {"score": None, "passed": None, "details": ["空输入已拦截：%s" % trace.error]}
        if trace.status in ("error", "timeout"):
            return {"score": 0, "passed": False, "details": ["执行失败：%s" % trace.error]}

        s = trace.result_summary
        if case.task_type == "jd_analysis":
            score, details = self._eval_jd(case, s)
        elif case.task_type == "resume_parse":
            score, details = self._eval_parse(case, s)
        elif case.task_type == "resume_optimize":
            score, details = self._eval_optimize(case, s)
        elif case.task_type == "job_match":
            score, details = self._eval_match(case, s)
        elif case.task_type == "question_gen":
            score, details = self._eval_question(case, s)
        else:
            score, details = 0, ["未知任务类型"]

        return {"score": score, "passed": score >= self.PASS_LINE, "details": details}

    # ---- 岗位分析：考察是否识别出硬技能 / 软技能 / 学历 / 核心要求 ----
    @staticmethod
    def _eval_jd(case, s):
        score, details = 0, []
        hard = s.get("hard_skills", {})
        # 无技能 JD：正确行为是识别不到硬技能（输入质量不足，不作为系统失败）
        if case.ground_truth.get("expect_no_skill"):
            if not hard:
                return 100, ["正确识别该 JD 无技能关键词（输入质量不足）"]
            return 0, ["无技能 JD 却识别出硬技能，属误判"]
        if hard:
            score += 40
            details.append("识别到硬技能 %d 类" % len(hard))
        else:
            details.append("未识别到硬技能")
        if s.get("soft_skills"):
            score += 20
            details.append("识别到软技能")
        if s.get("education"):
            score += 20
            details.append("识别到学历要求")
        if s.get("bonus_items"):
            score += 20
            details.append("识别到加分项")
        return score, details

    # ---- 简历解析：考察个人信息 / 技能 / 教育 / 项目抽取 ----
    @staticmethod
    def _eval_parse(case, s):
        score, details = 0, []
        pi = s.get("personal_info", {})
        if any(pi.get(k) for k in ("name", "phone", "email")):
            score += 30
            details.append("抽取到个人信息")
        else:
            details.append("个人信息缺失")
        skills = s.get("skills", {})
        if skills.get("hard_skills"):
            score += 30
            details.append("抽取到硬技能 %d 项" % len(skills["hard_skills"]))
        if s.get("education"):
            score += 20
            details.append("抽取到教育背景")
        if s.get("projects") or s.get("experience"):
            score += 20
            details.append("抽取到项目/实习经历")
        return score, details

    # ---- 简历优化：考察完整性（非空 / 量化 / 建议 / STAR） ----
    @staticmethod
    def _eval_optimize(case, s):
        score, details = 0, []
        opt = s.get("optimized_resume", "")
        sugg = s.get("suggestions", [])
        if opt.strip():
            score += 25
            details.append("产出优化简历")
        if re.search(r"\d+(\.\d+)?(%|％|万|个|人|次|倍|ms|秒)", opt):
            score += 25
            details.append("优化简历含量化数据")
        else:
            details.append("优化简历缺量化数据")
        if sugg:
            score += 25
            details.append("给出 %d 条修改建议" % len(sugg))
        if any(("STAR" in x) or ("量化" in x) for x in sugg):
            score += 25
            details.append("建议覆盖 STAR/量化要点")
        return score, details

    # ---- 岗位匹配：考察命中/短板识别是否与 ground_truth 一致（作业二修正后规则） ----
    @staticmethod
    def _eval_match(case, s):
        score, details = 0, []
        reports = s.get("match_reports", [])
        if not reports:
            return 0, ["未产出匹配报告"]
        score += 20
        r = reports[0]
        if 0 <= float(r.get("score", -1)) <= 100:
            score += 20
        d = r.get("detail", {})
        hit = d.get("hard", {}).get("hit", [])
        miss = d.get("hard", {}).get("miss", [])
        # ground_truth 给出「期望命中」的技能，命中集合包含期望即得分（不再用一刀切阈值）
        expect = case.ground_truth.get("expect_hit", [])
        if expect:
            matched = sum(1 for e in expect if e in hit)
            score += int(40 * matched / max(len(expect), 1))
            details.append("硬技能命中 %s（期望 %s）" % (hit, expect))
        if miss:
            details.append("正确识别短板 %s" % miss)
            score += 20
        return min(score, 100), details

    # ---- 面试题生成：考察三类题齐全、专业题与 JD 相关性、含参考答案 ----
    @staticmethod
    def _eval_question(case, s):
        score, details = 0, []
        tech, proj, beh = s.get("technical", []), s.get("project", []), s.get("behavioral", [])
        if tech and proj and beh:
            score += 40
            details.append("三类面试题齐全（专业%d/项目%d/行为%d）" % (len(tech), len(proj), len(beh)))
        else:
            details.append("面试题类别缺失")
        # JD 相关性：专业题题干命中 JD 硬技能关键词
        jd_skills = case.ground_truth.get("jd_skills", [])
        if jd_skills:
            hit = sum(1 for q in tech for kw in jd_skills if kw in q.get("question", ""))
            if hit >= 1:
                score += 30
                details.append("专业题与 JD 关键词相关（命中 %d 次）" % hit)
            else:
                details.append("专业题与 JD 关键词无关联")
        if all(q.get("answer") for q in tech + proj + beh):
            score += 30
            details.append("每题均附参考答案")
        return score, details


# ---------------------------------------------------------------------------
# 6. 结果统计汇总模块
# ---------------------------------------------------------------------------
class ReportSummarizer:
    """汇总批量测试结果：成功率、失败清单、耗时统计、优化前后对比。"""

    @staticmethod
    def summarize(traces: List[TraceRecord], evals: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(traces)
        success = sum(1 for e in evals if e["passed"] is True)
        failed = sum(1 for e in evals if e["passed"] is False)
        invalid = sum(1 for e in evals if e["passed"] is None)
        valid = success + failed  # 有效任务数（不含被拦截的空输入）
        rate = (success / valid * 100) if valid else 0.0

        failed_list = [
            {"task_id": t.task_id, "name": _name_of(t),
             "reason": "；".join(ev["details"])}
            for t, ev in zip(traces, evals) if ev["passed"] is False
        ]
        avg_ms = int(sum(t.duration_ms for t in traces) / total) if total else 0
        return {
            "total": total,
            "success": success,
            "failed": failed,
            "invalid_input": invalid,
            "success_rate": round(rate, 1),
            "avg_duration_ms": avg_ms,
            "failed_list": failed_list,
        }

    @staticmethod
    def compare(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        """输出优化前后对比。"""
        return {
            "success_rate_before": before["success_rate"],
            "success_rate_after": after["success_rate"],
            "failed_before": before["failed"],
            "failed_after": after["failed"],
            "delta_rate": round(after["success_rate"] - before["success_rate"], 1),
        }


def _name_of(t: TraceRecord) -> str:
    """从 Trace 反向查用例名（仅展示用，简化处理）。"""
    return t.task_id


# ---------------------------------------------------------------------------
# 7. 测试任务集（覆盖 5 类任务，不少于 10 条）
# ---------------------------------------------------------------------------
def build_test_suite() -> List[TestCase]:
    """构造测试用例集。示例数据来自求职智能体 data/samples。"""
    sample_jd = load_document(os.path.join(SAMPLE_DIR, "sample_jd.txt"))
    sample_resume = load_document(os.path.join(SAMPLE_DIR, "sample_resume.txt"))

    # 前端 JD（与后端简历不相关，用于考察匹配低分场景）
    frontend_jd = "前端开发工程师：熟悉 Vue、React、TypeScript，掌握 Webpack 工程化，有良好的沟通能力。"

    # 无技术关键词的 JD（输入质量不足场景）
    no_skill_jd = "我们是一家快速发展的公司，欢迎热爱挑战的你加入我们。"

    cases = [
        TestCase("T01", "resume_parse", "标准简历解析", {"resume_text": sample_resume},
                 {"expect_name": "张三"}),
        TestCase("T02", "resume_parse", "空简历（异常用例）", {"resume_text": ""}),
        TestCase("T03", "jd_analysis", "后端 JD 岗位分析", {"jd_text": sample_jd},
                 {"expect_skill": "Python"}),
        TestCase("T04", "jd_analysis", "无技能 JD（输入质量不足）", {"jd_text": no_skill_jd},
                 {"expect_no_skill": True}),
        TestCase("T05", "resume_optimize", "标准简历优化",
                 {"jd_text": sample_jd, "resume_text": sample_resume},
                 {"expect_quantified": True}),
        TestCase("T06", "resume_optimize", "空简历优化（异常用例）",
                 {"jd_text": sample_jd, "resume_text": ""}),
        TestCase("T07", "job_match", "简历匹配后端 JD",
                 {"resume_text": sample_resume,
                  "jd_list": [{"title": "后端开发", "text": sample_jd}]},
                 {"expect_hit": ["Python", "Java", "MySQL"]}),
        TestCase("T08", "job_match", "简历匹配前端 JD（低分场景）",
                 {"resume_text": sample_resume,
                  "jd_list": [{"title": "前端开发", "text": frontend_jd}]},
                 {"expect_hit": []}),
        TestCase("T09", "question_gen", "面试题生成",
                 {"jd_text": sample_jd, "resume_text": sample_resume},
                 {"jd_skills": ["Python", "Java", "MySQL"]}),
        TestCase("T10", "question_gen", "空 JD 生成（异常用例）",
                 {"jd_text": "", "resume_text": sample_resume}),
        TestCase("T11", "resume_parse", "无项目简历解析",
                 {"resume_text": "姓名：王五\n电话：13700000000\n技能清单：\n- 熟悉 SQL"}),
        TestCase("T12", "job_match", "简历匹配多 JD",
                 {"resume_text": sample_resume,
                  "jd_list": [{"title": "后端", "text": sample_jd},
                              {"title": "前端", "text": frontend_jd}]},
                 {"expect_hit": ["Python", "Java"]}),
    ]
    return cases


# ---------------------------------------------------------------------------
# 8. 演示入口
# ---------------------------------------------------------------------------
def main():
    """演示：加载套件 -> 批量执行 -> 自动评测 -> 汇总打印。"""
    runner = HarnessRunner()
    logger = TraceLogger()
    evaluator = Evaluator()

    cases = build_test_suite()
    traces = runner.run_batch(cases)
    for t in traces:
        logger.log(t)

    evals = [evaluator.evaluate(c, t) for c, t in zip(cases, traces)]
    summary = ReportSummarizer.summarize(traces, evals)

    print("\n================== 测试结果汇总 ==================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out = logger.save("harness_trace.jsonl")
    print("\nTrace 日志已保存：%s" % out)


if __name__ == "__main__":
    main()
