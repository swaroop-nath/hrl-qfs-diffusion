from rouge import Rouge
from nltk import PorterStemmer

stemmer = PorterStemmer()
def get_rouge_score(pred_summaries, true_summaries):
    pred_summaries = [" ".join([stemmer.stem(word) for word in line.split()]) for line in pred_summaries]
    true_summaries = [" ".join([stemmer.stem(word) for word in line.split()]) for line in true_summaries]
    scorer = Rouge()
    scores = scorer.get_scores(pred_summaries, true_summaries, avg=True)

    return {'rouge-1': scores['rouge-1']['f'], 'rouge-2': scores['rouge-2']['f'], 'rouge-L': scores['rouge-l']['f']}