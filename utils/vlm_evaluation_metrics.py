from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider


def compute_bleu_cider_meteor_single_ref(predictions, references):
    """
    Compute BLEU-4, CIDEr, and METEOR for a batch of predictions
    when there is exactly ONE reference text per prediction.

    Args:
        predictions (list[str]): list of model-generated texts.
        references (list[str]): list of reference strings.
            references[i] is the single reference for predictions[i].

    Returns:
        dict: {
            "BLEU_4_mean": float,
            "CIDEr_mean": float,
            "METEOR_mean": float,
            "BLEU_per_sample": list[float],
            "CIDEr_per_sample": list[float],
            "METEOR_per_sample": list[float],
        }
    """
    assert len(predictions) == len(references), \
        "predictions and references must have the same length."

    # ---------- CIDEr (COCO format: dict[id] = [sentences]) ----------
    gts = {}   # ground truths
    res = {}   # results

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        key = str(i)
        res[key] = [pred]      # list with single prediction
        gts[key] = [ref]       # list with single reference

    cider_scorer = Cider()
    cider_mean, cider_per_sample = cider_scorer.compute_score(gts, res)

    # ---------- BLEU-4 & METEOR (using NLTK) ----------
    smooth_fn = SmoothingFunction().method4

    bleu_scores = []
    meteor_scores = []

    for pred, ref in zip(predictions, references):
        # Simple whitespace tokenization; replace with a better tokenizer if needed
        ref_tokens = ref.split()
        pred_tokens = pred.split()

        # For BLEU, we need: list of reference token lists
        bleu = sentence_bleu(
            [ref_tokens],        # list of 1 reference
            pred_tokens,
            smoothing_function=smooth_fn
        )
        bleu_scores.append(bleu)

        # For METEOR, pass tokenized refs and hypothesis
        # references: list of token lists, hypothesis: token list
        meteor = meteor_score([ref_tokens], pred_tokens)
        meteor_scores.append(meteor)

    bleu_mean = sum(bleu_scores) / len(bleu_scores)
    meteor_mean = sum(meteor_scores) / len(meteor_scores)

    return {
        "BLEU_4_mean": float(bleu_mean),
        "CIDEr_mean": float(cider_mean),
        "METEOR_mean": float(meteor_mean),
        "BLEU_per_sample": bleu_scores,
        "CIDEr_per_sample": cider_per_sample,
        "METEOR_per_sample": meteor_scores,
    }
