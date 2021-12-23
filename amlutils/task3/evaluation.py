import numpy as np


def evaluate(predictions, targets):
    ious = []
    for p, t in zip(predictions, targets):
        assert p['name'] == t['name']
        prediction = np.array(p['prediction'], dtype=bool)
        target = np.array(t['label'], dtype=bool)

        assert target.shape == prediction.shape
        overlap = prediction * target
        union = prediction + target

        ious.append(overlap.sum() / float(union.sum()))

    return np.median(ious)
