"""
Module for feature extraction.

"""

import pandas as pd

from transformers import pipeline


class Factors:
    """
    Factor class for hugging face feature extraction.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        on: str,
        task: str,
        model: str,
        top_k: int = None,
        batch_size: int = 16,
    ):
        """
        Initialization and execution.

        """
        self.data = data
        self.inputs = data[on].drop_duplicates().to_list()
        self.on = on
        self.task = task
        self.model = model
        self.top_k = top_k
        self.batch_size = batch_size
        self._run()

    def _run(self):
        """
        Main execution.

        """

        # Generate Hugging pipeline
        self.pipeline_constructor()

        # Inference with GPU
        self.output()

        # Generate features
        self.features()

        # Generate dataset
        self.dataset()

    def pipeline_constructor(self):
        """
        Pipeline constructor.

        """

        self.classifier = pipeline(
            task=self.task,
            model=self.model,
            tokenizer=self.model,
            top_k=self.top_k,
            device="cuda:0",
        )

    def output(self) -> list:
        """
        Output generation.

        """

        self.output = self.classifier(self.inputs, batch_size=self.batch_size)

    def features(self) -> pd.DataFrame:
        """
        Dataset generation of unique instances.

        """

        for i, out in enumerate(self.output):
            row = pd.DataFrame(out).set_index("label").transpose()

            if i == 0:
                self.features = row
            else:
                self.features = pd.concat([self.features, row], ignore_index=True)

    def dataset(self) -> pd.DataFrame:
        """
        Dataset generation re-indexed with original data.

        """

        # Copy and insert key
        self.df = self.features.copy()
        self.df[self.on] = self.inputs

        # Merge
        self.df = self.data.merge(self.df, on=self.on, how="left").drop(
            self.data.columns, axis=1
        )
