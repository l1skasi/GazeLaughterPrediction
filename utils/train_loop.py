def train_loop_standard(model, loader, criterion, optimizer, device):
    """
    Standard training loop for models predicting child gaze and laughter.

    """
    for X_batch, (targets_gaze, targets_laughter) in loader:
        logits_gaze, logits_laughter = model(X_batch, device)

        loss_gaze = criterion(logits_gaze, targets_gaze.to(device))
        loss_laughter = criterion(logits_laughter, targets_laughter.to(device))

        loss = loss_gaze + loss_laughter

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def train_loop_gr(model, loader, criterion, optimizer, device):
    """
    Training loop for models predicting child gaze, laughter, and gaze relation.

    """
    for X_batch, (targets_gaze, targets_laughter, targets_gazerelation) in loader:
        logits_gaze, logits_laughter, logits_gazerelation = model(X_batch, device)

        loss_gaze = criterion(logits_gaze, targets_gaze.to(device))
        loss_laughter = criterion(logits_laughter, targets_laughter.to(device))
        loss_gazerelation = criterion(logits_gazerelation, targets_gazerelation.to(device))

        loss = loss_gaze + loss_laughter + loss_gazerelation

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def train_loop(model, loader, criterion, optimizer, isGazeRelation, device):
    """
    Wrapper function to choose the appropriate training loop
    depending on whether gaze relation is predicted.

    """
    if isGazeRelation:
        return train_loop_gr(model, loader, criterion, optimizer, device)
    else:
        return train_loop_standard(model, loader, criterion, optimizer, device)
