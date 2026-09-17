"""评估迭代接口（RAG L5）—— 量化系统表现并给出调优建议。

没有评估的 RAG 只能靠"我问了两句感觉还行"来判断好坏。
这一层把命中率、MRR、用户好评率变成可查询、可对比的数字，
让"换了个 embedding 之后到底变好还是变坏"这个问题有客观答案。
"""
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.rag import evaluator
from app.utils.logger import logger

router = APIRouter(prefix="/evaluate", tags=["评估迭代"])


@router.post("/retrieval")
def run_retrieval_evaluation(cases: Optional[list] = None, top_k: Optional[int] = None) -> dict:
    """跑检索评测。

    不传 cases 时使用内置基线用例。每次评测会真实调用 embedding 接口，
    用例数量请控制（每一轮都是一次真实远端往返，有成本也有耗时）。
    """
    try:
        if cases:
            eval_cases = [
                evaluator.EvalCase(
                    question=str(c.get("question", "")).strip(),
                    expect_keywords=list(c.get("expect_keywords", [])),
                    expect_source=c.get("expect_source"),
                )
                for c in cases
                if str(c.get("question", "")).strip()
            ]
            if not eval_cases:
                raise HTTPException(status_code=400, detail="未提供有效用例")
        else:
            eval_cases = None

        report = evaluator.run_retrieval_eval(eval_cases, top_k=top_k)
        return {"code": 0, **report}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("检索评测失败")
        raise HTTPException(status_code=500, detail=f"评测失败：{exc}")


@router.get("/report")
def system_report(include_eval: bool = False) -> dict:
    """系统综合报告：五层状态 + 反馈统计（include_eval=true 时附带检索评测）。"""
    try:
        return {"code": 0, **evaluator.build_report(include_eval=include_eval)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("报告生成失败")
        raise HTTPException(status_code=500, detail=f"报告生成失败：{exc}")


@router.get("/suggestions")
def improvement_suggestions(include_eval: bool = False) -> dict:
    """基于当前状态给出可执行的调优建议。"""
    try:
        report = evaluator.build_report(include_eval=include_eval)
        return {
            "code": 0,
            "suggestions": evaluator.suggest_improvements(report),
            "based_on": {
                "hit_rate": (report.get("retrieval_eval") or {}).get("hit_rate"),
                "good_rate": (report.get("feedback") or {}).get("good_rate"),
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("建议生成失败")
        raise HTTPException(status_code=500, detail=f"建议生成失败：{exc}")


@router.get("/feedback")
def feedback_stats(limit: int = 200) -> dict:
    """用户反馈统计：好评率与最近的差评样本。"""
    return {"code": 0, **evaluator.feedback_stats(limit=limit)}
