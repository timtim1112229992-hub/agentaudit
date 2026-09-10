"""Surface markers used by the independent recoder.

The rule set is versioned so that a recoding can be tied to the exact criteria
that produced it. Changing any pattern requires incrementing RULESET_VERSION,
which is written into the run manifest.

Patterns are drawn from the language the corpus is written in. They describe the
function a message performs, not the label the source system attached to it, and
the recoder is never shown that label.
"""
from __future__ import annotations

RULESET_VERSION = "recode/1.0.0"

# The agent offers to supply content itself, whether as a menu of prefabricated
# options or as a sentence frame the learner only has to complete.
OFFER_TO_SUPPLY = (
    "我可以帮你", "我来帮你", "帮我写", "帮我选", "帮我判断", "帮我匹配",
    "选一个模板", "选一个开始", "点选一个", "选一个你觉得",
)

# The agent hands the next decision back, naming alternatives and asking the
# learner to choose between them. Praise alone does not qualify.
HANDS_BACK = ("你选哪个", "你想先", "或者直接", "你可以继续", "你选择哪")

# The agent names a conflict inside the learner's own record and asks for repair.
FLAGS_CONFLICT = ("不一致", "再检查", "改正", "好像不", "但你的", "重新看看")

# The agent asks the learner to generate an observation, comparison or reason.
SEEKS_GENERATION = (
    "你看到了什么", "是什么让你", "哪号", "什么颜色", "公平了吗", "怎么",
    "为什么", "有什么不同", "试着描述", "请至少选出", "先选择",
)

# The agent marks work as done and praises it, without asking or offering anything.
MARKS_DONE = ("太棒", "真棒", "很棒", "继续保持", "已经完成", "收集完成")

QUESTION_MARKS = ("?", "？")

# Functional categories the recoder may assign. The set is defined on functional
# grounds rather than copied from a source instrument, so that a fidelity
# comparison can reveal categories an instrument does not express.
CATEGORIES = ("scaffold", "release", "redirect", "probe", "affirm", "other")
