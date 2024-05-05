


from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

generated_summary = "jo bhi"
scores = scorer.score(summary[:200], generated_summary)

print(scores)
