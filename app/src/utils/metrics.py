from prometheus_client import Histogram, Counter

# Metrics:
LLM_RESPONSE_TIME = Histogram(
    "llm_response_duration_seconds",
    "Time taken for LLM to respond"
)

FILTER_DURATION = Histogram(
    "filter_processing_duration_seconds",
    "Time taken for filter to process message"
)

FILTER_RESULT_COUNTER = Counter(
    "filter_result_total",
    "Count of filtered messages",
    ["status", "type"]  # status: passed / blocked, type: pre / post
)
