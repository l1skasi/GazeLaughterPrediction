import torch
from utils.helper import get_quality_metrics

def evaluation_on_set(model, loader, criterion, device):
    """
    Evaluate the model on a dataset (without gaze relation prediction).

    """
    total_loss = 0
    targets_gazes = []
    preds_gazes = []
    targets_laughters = []
    preds_laughters = []

    for X_batch, (targets_gaze, targets_laughter) in loader:

        logits_gaze, logits_laughter = model(X_batch, device)

        loss_gaze = criterion(logits_gaze, targets_gaze.to(device))
        loss_laughter = criterion(logits_laughter, targets_laughter.to(device))
        total_loss += (loss_gaze + loss_laughter).item()

        targets_gazes.extend(targets_gaze.cpu().numpy())
        preds_gazes.extend(torch.argmax(logits_gaze, dim=1).cpu().numpy())

        targets_laughters.extend(targets_laughter.cpu().numpy())
        preds_laughters.extend(torch.argmax(logits_laughter, dim=1).cpu().numpy())

    avg_loss = total_loss / len(loader)

    accuracy_gaze, gaze_precision, gaze_f1 = get_quality_metrics(preds_gazes, targets_gazes)
    accuracy_laughter, laughter_precision, laughter_f1 = get_quality_metrics(preds_laughters, targets_laughters)

    return avg_loss, loss_gaze, loss_laughter, accuracy_gaze, gaze_precision, gaze_f1, accuracy_laughter, laughter_precision, laughter_f1


def evaluation_on_set_gr(model, loader, criterion, device):
    """
    Evaluate the model on a dataset (with gaze relation prediction).

    """
    total_loss = 0
    targets_gazes = []
    preds_gazes = []
    targets_laughters = []
    preds_laughters = []
    targets_gazerelations = []
    preds_gazerelations = []

    for X_batch, (targets_gaze, targets_laughter, targets_gazerelation) in loader:
        logits_gaze, logits_laughter, logits_gazerelation = model(X_batch, device)

        loss_gaze = criterion(logits_gaze, targets_gaze.to(device))
        loss_laughter = criterion(logits_laughter, targets_laughter.to(device))
        loss_gazerelation = criterion(logits_gazerelation, targets_gazerelation.to(device))

        total_loss += (loss_gaze + loss_laughter + loss_gazerelation).item()

        targets_gazes.extend(targets_gaze.cpu().numpy())
        preds_gazes.extend(torch.argmax(logits_gaze, dim=1).cpu().numpy())

        targets_laughters.extend(targets_laughter.cpu().numpy())
        preds_laughters.extend(torch.argmax(logits_laughter, dim=1).cpu().numpy())

        targets_gazerelations.extend(targets_gazerelation.cpu().numpy())
        preds_gazerelations.extend(torch.argmax(logits_gazerelation, dim=1).cpu().numpy())

    avg_loss = total_loss / len(loader)

    accuracy_gaze, gaze_precision, gaze_f1 = get_quality_metrics(preds_gazes, targets_gazes)
    accuracy_laughter, laughter_precision, laughter_f1 = get_quality_metrics(preds_laughters, targets_laughters)
    # accuracy_gazerelation, gazerelation_precision, gazerelation_f1 = get_quality_metrics(preds_gazerelations, targets_gazerelations)

    return avg_loss, loss_gaze, loss_laughter, accuracy_gaze, gaze_precision, gaze_f1, accuracy_laughter, laughter_precision, laughter_f1


def evaluation_loop(model, loader, criterion, isGazeRelation, device):
    """
    Run evaluation on the dataset depending on whether gaze relation is included.

    Args:
        isGazeRelation: Boolean flag indicating whether gaze relation prediction is part of the model.
    """
    if isGazeRelation:
        return evaluation_on_set_gr(model, loader, criterion, device)
    else:
        return evaluation_on_set(model, loader, criterion, device)
