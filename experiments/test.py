def ae_training_old(
    dataset,
    simulation_name,
    model_name,
    latent_dimensions,
    n_epochs=10,
    recalculateAE=True,
    batch_size=1,
    lr=1e-3,
    dropout_rate=0.0,           
    checkpoint_epochs=None
):
    """
    Train or load the 3D autoencoder.

    Naming:
    - final model: {simulation_name}_{n_epochs}epochs.pth
    - loss file:   {simulation_name}_{n_epochs}epochs_loss.txt

    where simulation_name is now something like:
    AE3dFCDeep_96patients_split0_20dims
    """
    device = get_device()

    final_model_path = (
        TEMPODATA_FOLDER
        / "autoencoder/"
        / simulation_name
        / f"_{n_epochs}epochs.pth"
    )

    loss_path = (
        TEMPODATA_FOLDER
        / "autoencoder/"
        / simulation_name
        / f"_{n_epochs}epochs_loss.txt"
    )

    if checkpoint_epochs is None:
        checkpoint_epochs = []
    checkpoint_epochs = set(checkpoint_epochs)

    if recalculateAE:

        model = build_autoencoder(model_name, latent_dimensions, dropout_rate=dropout_rate).to(device)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda")
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        epoch_losses = []

        print(f"Training {model_name} on {device} with batch_size={batch_size}")

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0

            for (x_batch,) in loader:
                x_batch = x_batch.to(device, non_blocking=(device.type == "cuda"))

                optimizer.zero_grad()
                x_recon, z = model(x_batch)
                loss = criterion(x_recon, x_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            epoch_losses.append(avg_loss)

            current_epoch = epoch + 1
            print(f"Epoch {current_epoch}/{n_epochs} - Loss: {avg_loss:.6f}")

            if current_epoch in checkpoint_epochs:
                checkpoint_path = (
                    TEMPODATA_FOLDER
                    / "autoencoder/"
                    / simulation_name
                    / f"_{current_epoch}epochs.pth"
                )
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

        # Save final model
        torch.save(model.state_dict(), final_model_path)

        # Save loss history
        with open(loss_path, "w") as f:
            for i, loss in enumerate(epoch_losses):
                f.write(f"Epoch {i+1}: {loss}\n")

        model.eval()

    else:
        model = build_autoencoder(model_name, latent_dimensions).to(device)
        model.load_state_dict(torch.load(final_model_path, map_location=device))
        model.eval()

    return model