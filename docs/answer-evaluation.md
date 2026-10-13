# 回答评测怎样执行

这是工程运行说明，不是已完成的模型质量报告。当前80题金标仍待人工确认，正式验收未执行。原53题的83.0%是历史标注证据命中率，不是本流程的回答正确率。

## 计划、回答与审核分开保存

`run_task_answers.py`在调用模型前写入`planned-trials.jsonl`，并把它的哈希写进批次`manifest.json`。实际回答、原始引用、实际输入证据和执行失败写入`answers.jsonl`；传输尝试及未知用量另记请求账本。默认不调用模型，只有明确使用远端生成选项才执行。

每条试验有题目、方案、重复序号和计划重复次数。同一问题的三次生成使用固定保存的检索结果，因此测的是给定证据下的生成稳定性，不是三次完整重检的稳定性。它们不会变成三道独立题。

审核时必须同时提供完整计划与原批次manifest。中断后只写出一部分回答，缺失项会补为not_run/INCOMPLETE，不能只算幸存的成功回答。计划哈希改变、题目或方案身份不符、完成批次的回答文件改变，都拒绝沿用原审核。

```powershell
# 把下列路径换为实际批次，输出目录必须尚不存在。
& ./.venv/Scripts/python.exe -X utf8 scripts/review_answer_batch.py prepare --trials <batch>/answers.jsonl --rubrics <tasks>/rubrics.jsonl --planned-trials <batch>/planned-trials.jsonl --batch-manifest <batch>/manifest.json --output-dir <new-review>

# 真实审核后，把submitted换为实际填写的记录文件。
& ./.venv/Scripts/python.exe -X utf8 scripts/review_answer_batch.py score --trials <batch>/answers.jsonl --rubrics <tasks>/rubrics.jsonl --planned-trials <batch>/planned-trials.jsonl --batch-manifest <batch>/manifest.json --review-dir <new-review> --submitted <submitted.jsonl> --output-dir <new-score>
```

上面的尖括号是说明占位，不能直接复制执行。已有合法批次的答案、计划和评分标准不能随意修改；更正须保留旧记录并创建新修订。

## 审核者判断什么

审核包随机隐藏方案名，保留实际问题、年份、原始回答、原始引用和模型实际收到的证据。审核者逐条摘录事实，并判断正确性、所需事实是否完整、引文是否确实支持结论。不能用另一段正确资料替模型补引用，不能把修正后的答案当成原回答。

`actual_action`由审核者判断回答是在回答、拒答还是澄清，分别填answer/refuse/clarify。它与action_correct分开：前者说明实际做了什么，后者判断该处置是否正确。模型自己的insufficient_evidence标志只是一份输出，不能替代语义判断。缺actual_action时拒答/澄清及过度拒答相关率保持未知；若实际处置与金标要求矛盾，严格评分失败。

主成绩须同时具有人工确认金标和真实人工审核来源；模型、助手及合成测试记录不能变成人工成绩。只有审核的一部分可报告已知子集，不能宣称整个批次质量。缺审核、事实盘点不完整或额外引用未审时保留未知。程序记录审核来源声明，但不能独立认证填写者身份。

## 指标的分母

仅作教学例子：两道题各生成三次，共六份回答。第一题三次通过，第二题两次通过一次失败。逐试验严格通过是5/6；三次全部通过是1/2。独立问题仍只有两道。

事实微平均把各回答的事实分子、分母分别相加；宏平均对具有非空分母的每份回答比率取平均。例如一份回答1/1，另一份0/3，微平均1/4，宏平均(1+0)/2。宏平均仍以回答试验为单位，不是独立题准确率；重复生成的题目数另外列出。

按方案、题型及方案×题型分别报告严格通过、事实正确/完整、引用支持/覆盖、正确拒答/澄清和过度拒答。三次稳定性按方案统计，不能把两个方案的六次输出合成“三次”。没有第三次或其审核未完成时，稳定性未知；已有一次明确失败则该题无法三次全过。

每个比率保留numerator、denominator、unit、unknown和value。分母为0或存在未完成判断时，完整主比率为null；已知人工子集另列，不能填成0费用或100%准确。
