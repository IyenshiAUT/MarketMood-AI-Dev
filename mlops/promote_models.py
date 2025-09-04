import mlflow
from mlflow.tracking import MlflowClient

def promote_best_model(client, model_name):
    try:
        registered_model = client.get_registered_model(model_name)
    except mlflow.exceptions.RestException:
        print(f"Model '{model_name}' not found. No promotion possible.")
        return

    all_versions = client.search_model_versions(f"name='{model_name}'")

    best_f1_score = -1
    best_version_to_promote = None
    
    for version in all_versions:
        run_id = version.run_id
        if not run_id:
            continue
        
        run_data = client.get_run(run_id).data.metrics
        f1_score = run_data.get('eval_f1', 0)
        
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_version_to_promote = version.version

    if best_version_to_promote and client.get_model_version(model_name, best_version_to_promote).current_stage != "Production":
        print(f"Promoting version {best_version_to_promote} of model '{model_name}' to Production.")
        
        for version in all_versions:
            if version.current_stage == "Production":
                print(f"Archiving old production version {version.version}.")
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )
        
        client.transition_model_version_stage(
            name=model_name,
            version=best_version_to_promote,
            stage="Production"
        )
    else:
        print(f"No new version to promote for '{model_name}'.")

if __name__ == "__main__":
    client = MlflowClient()
    promote_best_model(client, "finbert-sentiment-model")
    promote_best_model(client, "bart-summarization-model")
