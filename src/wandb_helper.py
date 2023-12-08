import wandb

def download_model_from_run(run_id, artifact_name="model"):
    api = wandb.Api()

    run = api.run(run_id)
    config = run.config

    artifacts = run.logged_artifacts()
    artifact_name = 'model'
    artifact = None
    for art in artifacts:
        if art.name.split(":")[0] == artifact_name:
            artifact = art

    if artifact is None:
        print(f'Artifact {artifact_name} not found in the used artifacts of this run.')

    return artifact, config