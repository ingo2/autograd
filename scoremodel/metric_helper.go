package scoremodel

// accuracy returns the fraction of correct predictions: (tp+tn) / (tp+tn+fp+fn).
func accuracy(tp, tn, fp, fn int) float64 {
	total := tp + tn + fp + fn
	if total == 0 {
		return 0
	}
	return float64(tp+tn) / float64(total)
}

// precision returns tp / (tp+fp): of all positive predictions, the fraction that were correct.
func precision(tp, fp int) float64 {
	if tp+fp == 0 {
		return 0
	}
	return float64(tp) / float64(tp+fp)
}

// recall returns tp / (tp+fn): of all actual positives, the fraction that were predicted.
func recall(tp, fn int) float64 {
	if tp+fn == 0 {
		return 0
	}
	return float64(tp) / float64(tp+fn)
}

// f1Score returns the harmonic mean of precision and recall.
func f1Score(precision, recall float64) float64 {
	if precision+recall == 0 {
		return 0
	}
	return 2 * precision * recall / (precision + recall)
}
