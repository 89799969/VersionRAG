"""Refresh delivery prose after the separately recorded comparison retakes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/reports/comparison_2026-10-07"


def change(path, replacements, append=""):
    text = path.read_text(encoding="utf-8")
    for before, after in replacements:
        assert before in text, str(path)
        text = text.replace(before, after)
    path.write_text(text + append, encoding="utf-8")


change(OUT / "实现与实测报告.md", [
    ("# 两专业多文档比较：实现、对照与未完成项", "# 两专业多文档比较：实现、对照与补测验收\n\n**当前状态：充值后已完成单独补测和全部122条有效回答的语义评审。** 原41条错误保持；合并视图仍保留048题两组生成失败。完整结果见[补测结果与根因分析](补测结果与根因分析.md)。下文的额度中断记录是首轮历史状态，补测状态以上述报告为准。留出适用11题模型完整成功3/11→7/11，同时要求系统无部分告警为3/11→6/11；补检无额外成功。59项离线测试与补测审计通过，不是生产或人类准确率验收。"),
    ("- 检索实测与离线审计完成；非开发题生成/语义评测因DeepSeek额度问题未完成。", "- 检索实测、补测与有效回答语义评审完成；048仍有业务状态校验失败，保留在分母。首轮额度中断记录保留。"),
    ("官方服务余额查询当前约0.10元", "首次中断时官方服务余额查询约0.10元"),
    ("余额恢复后，优先对已保存的非开发题有效回答做语义评审，并把缺失生成以单独的补测记录保存；保留首轮失败，注明重试轮次。不要直接重跑后抹去402和格式错误。当前不能宣布整个升级完成最终效果验收。", "已在用户告知充值后完成上述补测与评审，首轮文件未覆盖。复查`resume_comparison.py --phase audit`及`analyze_comparison_completed.py`不调用服务。补测41条中39恢复，048两组仍失败；合并180条视图122有效、56不适用、2错误，122有效均完成语义评审。原文件哈希、上下文、引用与评审输入审计通过。新增结果见补测报告，不把每项成功或人工准确率当成验收结论。"),
], "\n补测证据：`answers_resume.jsonl`、`answers_effective.jsonl`、`semantic_reviews_completed.jsonl`、`completed_summary.json`、`completed_audit.json`、`completed_case_analysis.json`及`逐题结果与失败理由.md`。本批模型服务113次HTTP200，200不等于业务状态校验成功；原402历史保留。\n")

change(ROOT / "README.md", [("其余生成/语义评测因DeepSeek HTTP 402未完成，不宣称完整验收。", "充值后已单独补测并完成122条有效回答语义评审；适用留出11题模型完整成功3/11→7/11，同时无部分告警为3/11→6/11。048生成仍失败，保留在诊断分母；补检无额外成功。[补测与根因](artifacts/reports/comparison_2026-10-07/补测结果与根因分析.md)公开净改善、退步和代价，不写整个项目准确率。")])

change(ROOT.parent / "RAG项目_简历亮点与证据_2026-10-06.md", [("非开发题生成/语义评测因模型额度不足未完成，暂不写整体回答提升。", "充值后已完成单独补测和122条有效回答自动语义评审；适用留出11题模型完整成功3/11→7/11，同时无部分告警3/11→6/11。是小样本同服务评审，不是人工准确率；048仍失败，补检无额外成功，详见[补测根因](D:/interview/rag-domain-study/artifacts/reports/comparison_2026-10-07/补测结果与根因分析.md)。建议简历写工程能力，数字留给面试展开。")], """
## 两专业比较：可选简历补充（补测后）

针对两专业比较问题，实现检索前资料范围过滤、按专业独立检索重排及平衡证据合并，在固定五条上下文内生成带双边原文引用的比较结论；建立检索覆盖与回答完整性分层评测，追踪无效引文删除、部分回答和最多一次补检，逐题分析证据挤占与引文改写造成的失败。

面试数字：11道适用原扩展留出题，模型完整成功3→7，额外要求无部分告警3→6；六题改善、两题退步，补检没有增加成功题。说明分母、同服务评审和单轮限制。不是通用多跳代理或训练式Self-RAG。59项离线测试通过，当前运行网页未新增比较入口。
""")

change(ROOT / "LESSON_05.md", [], """
## 充值后补测：三个记忆点

你已经独立指出：两个专业的范围内，一个专业的资料仍可能占据强势排序位置。因此分别检索、排序后还需平衡位置。这只记录本轮已表现的理解，不表示已会修改全部代码。

1. 适用留出11题同服务模型完整评审3/11→7/11，同时无部分告警3/11→6/11；完整18题的标注证据命中仍2/18，不混用分母。
2. 017资料已到模型，但引文把“中央挂广播电视总台”改成“中央广播电视总台”，校验删除共同点。不是召回丢失。
3. 补检没有新增完整成功，010的新宽查询替换掉原培养机制资料。多检索一次不自动等于优化。

具体证据见[补测结果与根因分析](artifacts/reports/comparison_2026-10-07/补测结果与根因分析.md)。
""")

entry = """## 2026-10-07 充值后完成两专业比较补测与失败分析

- 用户明确表示已充值，继续授权原服务补测；独立回答“两个专业会出现某一个专业资料占据强势的位置”。判断主线正确：限定两专业不保证双边位置。仅记一次正确提取，未记实现掌握或已稳定；不重复考同题。
- 原代码/提示/语料/标签/首轮结果/135条初始检索/原开发评审均按哈希保持。30种失败初始配置产生41条单独补测，39恢复；048拆分触发supported状态校验，两组保留失败。合并180条122有效、56不适用、2错误；原83/56/41状态和29次402不删除。
- 新81种不同输入评审，按完整哈希复用开发20条及相同拆分/补检答案。本批113次模型HTTP200、3编码/3重排HTTP200；200不保证业务校验成功。122有效结果均评审完成，58明确not_judged，180视图核验通过。
- 同服务单轮自动完整成功：开发0/5→5/5；适用原扩展留出11题3/11→7/11；适用诊断15题2/15→6/15（048失败计分母）。加系统无部分告警为0→5、3→6、1→5，保留两种口径。不是独立人工准确率或整个知识库泛化。原18留出完整标注命中仍2/18，组平均覆盖25.74%→34.63%，不与11题语义分母混。
- 留出新增成功001/004/014/016/018/032，退步010/017；004仍系统部分告警。诊断五增一退，019学科/专业排名模型判定有分歧。补检留出4触发3顺序变化、诊断5触发1变化，无新增成功；043仅列表顺序变仍多生成；010宽查询挤掉培养资料。
- 根因：017原文“中央挂广播电视总台”被引文改成“中央广播电视总台”，校验删除事实与共同点；043引文变化删事实；044所选13154缺实务课程，13155同source_id=663实际有，不是本轮字符截断。两边都有资料不代表维度完整，完全同文去重不处理近似重叠。未据留出题重新调提示。
- 59项离线测试通过，1已有依赖告警；completed_audit核验不可变文件、失败替换范围、上下文、引文、评审输入及次数上限。没有重启网页或宣称生产验收。保存补测根因、31题表及机器可查分析，更新README/第五课/简历边界。
- 工程实现与评测完成但保留一题失败。建议简历写范围过滤、独立检索重排、平衡证据、双边引用与分层评测，小样本数字面试展开。学习下一步沿010理解宽补检为什么挤掉已有关键证据；一次一个变量、一道主动回忆题。未经新工程要求不继续调参；原后端复习进度保持。

"""
change(ROOT.parent / "backend-agent-learning/PROGRESS.md", [("# 学习进度\n\n", "# 学习进度\n\n" + entry)])

for name, value in {
    "remaining_work.json": {"status": "evaluation_completed_with_retained_generation_failure", "user_choice_pending": False,
        "completed": "41 separate retake rows; 122 valid answers judged; 180-row effective view audited; 59 offline tests passed",
        "retained_failure": "048 split_balanced and paired split_recover: supported-aspect state validation failed; kept in denominator",
        "next": "Optional independent human review and new development-only targeted supplement trials; no automatic tuning"},
    "offline_validation_completed.json": {"tests": 59, "passed": 59, "warnings": 1, "completed_artifact_audit": "passed",
        "frozen_original_files_preserved": True, "generation_error_records_retained": 2, "scope": "Offline checks, not production or independent semantic proof"},
}.items():
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

print("Updated delivery, resume evidence, learning record and completion status.")
