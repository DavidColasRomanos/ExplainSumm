"""
Evaluation module.

"""

import numpy as np
import pandas as pd


class EvalSummMetrics:
    """
    This class is designed to evaluate various summarization metrics in the 
    context of generated summaries. The evaluations consider the quality of 
    these summaries as well as the characteristics of the summarization models 
    that produced them. These characteristics are gauged based on the models' 
    parameter count and the specific summarization approach they use, such as 
    'BoN policy', 'Pretrained', 'RL policy', or 'Supervised'.

    """

    def __init__(
        self, policy_dict: dict,
    ):
        """
        Initialize the EvalSummMetrics class.

        """

        self.policy_dict = policy_dict

        self.df = None
        self.policies = None
        self.dff = None
        self.refs = None
        self.fraction_preferred_to_ref = None
        self.metrics = None
        self.metrics_df = None
        self.policy_classification = None
        self.comparisons_segmentation = None

        self.main()

    def main(self):
        """
        Main execution.

        """

        # 1. Load processed datasets
        self.load_data()

        # 2. Load policies
        self.load_policies()

        # 3. Filter comp dataset with known policies.
        self.filter_dataset()

        # 4. Filter dataset with only ref observations
        self.filter_refs()

        # 5. Add policy characteristics
        self.add_policy_characteristics()

        # 6. Calculate the fraction preferred to reference
        self.calculate_fraction_preferred_to_ref()

        # 7. Calculate metrics
        self.calculate_metrics()

        # 8. Classify model quality
        self.classify_model_quality()

        # 9. Classify policies
        self.classify_policies()

        # 10. Comparisons segmentation
        self.comparisons_quality_segmentation()

        # 11. Save DataFrame to csv
        self.save_dataframe_to_csv()

    def load_data(self):
        """
        Load processed comparisions dataset.

        """

        self.df = pd.concat(
            [
                pd.read_pickle("data/processed/fe_train_comparisons.pkl"),
                pd.read_pickle("data/processed/fe_validation_comparisons.pkl"),
            ],
            ignore_index=True,
        )

    def load_policies(self):
        """
        Load policies.

        """

        self.policies = pd.DataFrame(
            self.policy_dict.values(),
            index=self.policy_dict.keys(),
            columns=["Policy Type", "Parameters"],
        )

    def filter_dataset(self):
        """
        Filter comparisons dataset with known policies.

        """

        condition = (
            (self.df["policy_0"] == "ref")
            | self.df["policy_0"].isin(self.policies.index)
        ) & (
            (self.df["policy_1"] == "ref")
            | self.df["policy_1"].isin(self.policies.index)
        )

        self.dff = self.df[condition].reset_index(drop=True)

    def filter_refs(self):
        """
        Filter dataset with only ref observations.

        """

        self.refs = self.dff.loc[
            (self.dff["policy_0"] == "ref") | (self.dff["policy_1"] == "ref")
        ].reset_index(drop=True)

    def add_policy_characteristics(self):
        """
        Add policy characteristics.

        """
        self.refs["policy"] = np.where(
            self.refs["policy_0"] == "ref", self.refs["policy_1"], self.refs["policy_0"]
        )
        self.refs = self.refs.merge(
            self.policies, left_on="policy", right_index=True, how="left"
        )

    def calculate_fraction_preferred_to_ref(self):
        """
        Calculates the fraction of times each policy is preferred over the reference 
        policy and adds this to a DataFrame.

        """

        self.fraction_preferred_to_ref = pd.DataFrame(
            columns=["Model", "Parameters (B)", "Fraction preferred to ref"]
        )

        for (policy_type, parameters), group in self.refs.groupby(
            ["Policy Type", "Parameters"]
        ):

            # Preferred to ref
            preferred_to_ref = group.loc[
                ((group["policy_0"] == "ref") & (group["choice"] == 1))
                | ((group["policy_1"] == "ref") & (group["choice"] == 0))
            ]

            # Fraction preferred
            fraction_preferred_to_ref = len(preferred_to_ref) / len(group)

            # DataFrame
            self.fraction_preferred_to_ref = pd.concat(
                [
                    self.fraction_preferred_to_ref,
                    pd.DataFrame(
                        {
                            "Model": [policy_type],
                            "Parameters (B)": [float(parameters.replace("B", ""))],
                            "n": len(group),
                            "Fraction preferred to ref": [fraction_preferred_to_ref],
                        }
                    ),
                ],
                ignore_index=True,
            )

    def calculate_metrics(self):
        """
        Calculate the mean of various metrics for each policy type and parameter 
        count, and store these means in a DataFrame.

        """

        # Metrics
        self.metrics = [
            "rouge_1_f",
            "rouge_2_f",
            "rouge_l_f",
            "bleu",
            "jaccard_similarity_1",
            "jaccard_similarity_2",
            "flesch_reading_ease",
            "text_summary_xfmr_similarity",
            "ref_summary_xfmr_similarity",
        ]

        # Initialize DataFrame
        self.metrics_df = pd.DataFrame(
            columns=["Model", "Parameters (B)"] + self.metrics
        )

        # Calculate mean for each metrics
        for (policy_type, parameters), group in self.refs.groupby(
            ["Policy Type", "Parameters"]
        ):

            # Prepare data dict
            m_dict = {
                "Model": [policy_type],
                "Parameters (B)": [float(parameters.replace("B", ""))],
            }

            # Calculate mean for each metric
            for metric in self.metrics:
                m_dict[metric] = group.apply(
                    lambda x: x[f"m0_{metric}"]
                    if x["policy_0"] != "ref"
                    else x[f"m1_{metric}"],
                    axis=1,
                ).mean()

            # Update metrics DataFrame
            self.metrics_df = pd.concat(
                [self.metrics_df, pd.DataFrame(m_dict)], ignore_index=True,
            )

        # Add preferred_to_ref and n
        self.metrics_df = self.metrics_df.merge(
            self.fraction_preferred_to_ref, how="left", on=["Model", "Parameters (B)"]
        )

    def classify_model_quality(self):
        """
        Classify models into 'Low', 'Medium', or 'High' quality based on 
        'Fraction preferred to ref' value. 
        
        Models are classified as 'Low' if their fraction is not greater than 
        the median of fractions not greater than 0.5, 'Medium' if their fraction 
        is greater than the median but not greater than 0.5, and 'High' if their 
        fraction is greater than 0.5.

        """

        # Calculate the median for 'Fraction preferred to ref' values that are less than or equal to 0.5.
        median_value = self.fraction_preferred_to_ref[
            self.fraction_preferred_to_ref["Fraction preferred to ref"] <= 0.5
        ]["Fraction preferred to ref"].quantile(0.5)

        # Define the bin edges for the 'pd.cut()' function.
        bins = [0, median_value, 0.5, 1]
        labels = ["Low", "Medium", "High"]

        # Add a new 'Quality' column to the DataFrame, which classifies each model based on its 'Fraction preferred to ref' value.
        self.fraction_preferred_to_ref["Quality"] = pd.cut(
            self.fraction_preferred_to_ref["Fraction preferred to ref"],
            bins=bins,
            labels=labels,
        )

    def classify_policies(self):
        """
        Classify policies attending to the model classification.
        
        """

        # Load and transform policies
        policy_data = self.policies.copy()
        policy_data["Parameters"] = policy_data["Parameters"].apply(
            lambda x: float(x.replace("B", ""))
        )

        # Classify policies
        self.policy_classification = (
            policy_data.reset_index()
            .merge(
                self.fraction_preferred_to_ref,
                left_on=["Policy Type", "Parameters"],
                right_on=["Model", "Parameters (B)"],
                how="left",
            )
            .drop(columns=["Model", "Parameters (B)"])
            .rename(columns={"index": "Policy", "Parameters": "Parameters (B)"})
        )

    def comparisons_quality_segmentation(self):
        """
        Sample segmentation with attention to the quality of summarization 
        algorithms driven by the fraction preferred over the reference 
        calculation.

        """

        for i, quality in enumerate(["Low", "Medium", "High"]):

            # Mask
            mask = self.df["policy_0"].isin(
                self.policy_classification.loc[
                    self.policy_classification["Quality"] == quality, "Policy"
                ]
            ) & self.df["policy_1"].isin(
                self.policy_classification.loc[
                    self.policy_classification["Quality"] == quality, "Policy"
                ]
            )

            # Segmentation
            segmentation = self.df.loc[mask].copy()
            segmentation["Quality"] = quality

            if i == 0:
                self.comparisons_segmentation = segmentation
            else:
                self.comparisons_segmentation = pd.concat(
                    [self.comparisons_segmentation, segmentation]
                )

    def save_dataframe_to_csv(
        self, filename="data/segmentation/comparisons_segmentation.csv"
    ):
        """
        Save 'comparisons_segmentation' DataFrame to a csv file.

        Parameters:
        filename : str 
            The name of the file to save. Defaults to 'comparisons_segmentation.csv'.

        """

        self.comparisons_segmentation.to_csv(filename, index=False)
