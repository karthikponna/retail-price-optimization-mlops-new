from zenml.logger import get_logger
from zenml import step

import pandas as pd
from typing_extensions import Annotated
from typing import Tuple

from steps.src.model_building import DataSplitter

logger = get_logger(__name__)

@step
def split_data(df:pd.DataFrame) ->Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    try:
        data_splitter = DataSplitter(df, features=df.drop("unit_price",axis=1).columns, target="unit_price")
        X_train, X_test, y_train, y_test = data_splitter.split()
        logger.info("Data spilt successfully")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(e)
        raise e
    