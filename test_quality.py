import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from my_bot import ask_bot
from judge import GeminiJudge

judge = GeminiJudge()

GOLDEN = [
    ("What is the return policy?", "Returns accepted within 30 days if unused and in original packaging."),
    ("How long does standard delivery take?", "3 to 5 business days."),
    ("How fast is express delivery?", "1 to 2 business days."),
    ("What warranty do the fans have?", "2-year warranty covering motor and remote defects."),
    ("When will I get my refund?", "Within 7 business days after the returned item is received."),
    ("What are the support hours?", "Monday to Saturday, 9 AM to 6 PM IST."),
    ("How can I contact support?", "Chat and email at support@breezefans.example."),
    ("Is installation free, and how soon after delivery is it done?", "Yes, within 3 days of delivery in major cities."),
    ("Can I control the fan from my phone?", "Yes, the app controls speed and timers over Wi-Fi."),
    ("Can I cancel after shipping?", "No, you must use the return process."),
    ("Can I cancel before shipping?", "Yes, orders can be cancelled before shipping."),
    ("Does the warranty cover the remote?", "Yes, remote defects are covered for 2 years."),
    ("Can I return a used product?", "Returns require the product to be unused."),
    ("Where does the refund go?", "To the original payment method."),
    ("Is there a Sunday support line?", "Support runs Monday to Saturday only."),
]


@pytest.mark.parametrize("question,expected", GOLDEN)
def test_golden(question, expected):
    answer, ctx = ask_bot(question)
    case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected,
        retrieval_context=ctx,
    )
    assert_test(case, [
        AnswerRelevancyMetric(threshold=0.7, model=judge, async_mode=False),
        FaithfulnessMetric(threshold=0.7, model=judge, async_mode=False),
    ])
