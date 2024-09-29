import click

from constants import MODEL_NAME, PIPELINE_NAME, PIPELINE_STEP_NAME
from pipelines.inference_pipeline import inference_pipeline_retial

from pipelines.training_pipeline import training_retail

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY,PREDICT, DEPLOY_AND_PREDICT]),
    default="deploy_and_predict",
    help="Optionally you can choose to only run the deployment "
    "pipline to trian and deploy a model (`deploy`), or to "
    "only run a prediction against the deployed model "
    "(`predict`). By default both will be run "
    "(`deploy_and_predict`)."
)
def main(
    config: str,
):
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        training_retail()

    if predict:
        inference_pipeline_retial(
            model_name=MODEL_NAME,
            pipeline_name=PIPELINE_NAME,
            step_name=PIPELINE_STEP_NAME,
        )

if __name__ == "__main__":
    main()
    