from evaluate import load
bertscore = load("bertscore")
# predictions = ["hello world", "general kenobi"]
# references = ["hello world", "general kenobi"]
results = bertscore.compute(predictions=["test"], references=["fds"], model_type="microsoft/deberta-xlarge-mnli")
