from typing import Dict, List

import pandas as pd
import numpy as np
from rich import print as rich_print
from zenml import step
from zenml.integrations.bentoml.services import BentoMLDeploymentService

@step
def predictor(
    inference_data:pd.DataFrame,
    service: BentoMLDeploymentService,
) -> np.ndarray:
    """Run an inference request against the BentoML prediction service.

    Args:
        service: the BentoML service
        data: the data to predict."""
    
    service.start(timeout=10) #should be a NOP if already started
    #current inference data to ndarray
    inference_data = inference_data.to_numpy()
    prediction = service.predict("predict_ndarray", inference_data)
    rich_print("Prediction:", prediction)
    return prediction