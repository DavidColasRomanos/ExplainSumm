"""
Module for feature engineering.

"""

import numpy as np
import pandas as pd

import torch
import textstat

from rouge import Rouge
from typing import Dict

from nltk.util import ngrams
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from sentence_transformers import SentenceTransformer, util


class FeatureEngineering:
    """
    Feature engineering class.
    
    """

    def __init__(self):
        """
        Initialize the FeatureEngineering class.
        """
        pass

    def fe_pipeline(
        self, data: pd.DataFrame, method: str, encoder: SentenceTransformer
    ) -> pd.DataFrame:
        """
        Feature engineering pipeline.

        """
        df = data.copy()

        # 1. Add reference summary
        df = self.add_ref_summary(df, method)

        # 2. Word-overlap Metrics (WOMs): ROUGE score metrics
        df = self.add_rouge_scores(df, method)

        # 3. Word-overlap Metrics (WOMs): BLEU score metric (SmoothingFunction - method1)
        df = self.add_bleu_scores(df, method)

        # 3. Grammar-based score metrics (GBSMs):
        # 3.1. Add readability score metrics
        df = self.add_readability_scores(df, method)

        # 4.Intrinsic evaluation score metrics
        df = self.add_reference_free_scores(df, method)

        # 5. Sentence-Transformers score metrics
        df = self.add_textual_similarity_scores(df, method, encoder)

        return df

    def add_rouge_scores(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Add ROUGE score metrics.

        """
        df = data.copy()

        if method == "comparisons":
            df = self.compute_rouge_scores(df, "summary_0", "ref_summary", "m0")
            df = self.compute_rouge_scores(df, "summary_1", "ref_summary", "m1")

        elif method == "axis":
            df = self.compute_rouge_scores(df, "summary", "ref_summary", "m")
        else:
            raise ValueError("Wrong type.")

        return df

    def add_bleu_scores(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Add BLEU score metrics. Smoothing method 1: Add *epsilon* counts to precision with 0 counts.

        """
        df = data.copy()

        if method == "comparisons":
            df = self.compute_bleu_scores(df, "summary_0", "ref_summary", "m0")
            df = self.compute_bleu_scores(df, "summary_1", "ref_summary", "m1")

        elif method == "axis":
            df = self.compute_bleu_scores(df, "summary", "ref_summary", "m")
        else:
            raise ValueError("Wrong type.")

        return df

    def add_readability_scores(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Add READABILITY scores.

        """
        df = data.copy()

        if method == "comparisons":
            df = pd.concat(
                [
                    df,
                    df["summary_0"]
                    .apply(self.compute_readability_scores)
                    .apply(pd.Series)
                    .add_prefix("m0_"),
                    df["summary_1"]
                    .apply(self.compute_readability_scores)
                    .apply(pd.Series)
                    .add_prefix("m1_"),
                ],
                axis=1,
            )
        elif method == "axis":
            df = pd.concat(
                [
                    df,
                    df["summary"]
                    .apply(self.compute_readability_scores)
                    .apply(pd.Series)
                    .add_prefix("m_"),
                ],
                axis=1,
            )
        else:
            raise ValueError("Wrong type.")

        return df

    def add_reference_free_scores(
        self, data: pd.DataFrame, method: str
    ) -> pd.DataFrame:
        """
        Add REFERENCE FREE METRIC scores.

        """
        df = data.copy()

        if method == "comparisons":
            df = pd.concat(
                [
                    df,
                    df.apply(
                        lambda row: self.compute_reference_free_scores(
                            row["text"], row["summary_0"]
                        ),
                        axis=1,
                    )
                    .apply(pd.Series)
                    .add_prefix("m0_"),
                    df.apply(
                        lambda row: self.compute_reference_free_scores(
                            row["text"], row["summary_1"]
                        ),
                        axis=1,
                    )
                    .apply(pd.Series)
                    .add_prefix("m1_"),
                ],
                axis=1,
            )
        elif method == "axis":
            df = pd.concat(
                [
                    df,
                    df.apply(
                        lambda row: self.compute_reference_free_scores(
                            row["text"], row["summary"]
                        ),
                        axis=1,
                    )
                    .apply(pd.Series)
                    .add_prefix("m_"),
                ],
                axis=1,
            )
        else:
            raise ValueError("Wrong type.")

        return df

    def add_textual_similarity_scores(
        self, data: pd.DataFrame, method: str, encoder: SentenceTransformer,
    ) -> pd.DataFrame:
        """
        Add REFERENCE FREE METRIC scores.

        """
        df = data.copy()

        # Text embeddings
        text_encoded = self.compute_sentence_embeddigns(df["text"], encoder)

        # Summary ref embeddings. First dropna.
        f_ref = df["ref_summary"].dropna()
        ref_encoded = self.compute_sentence_embeddigns(f_ref.tolist(), encoder)

        if method == "comparisons":

            # Method embeddings
            summary_0_encoded = self.compute_sentence_embeddigns(
                df["summary_0"], encoder
            )
            summary_1_encoded = self.compute_sentence_embeddigns(
                df["summary_1"], encoder
            )

            # Concat
            df = pd.concat(
                [
                    df,
                    self.compute_textual_similarity_scores(
                        text_encoded, summary_0_encoded, "text", "summary"
                    ).add_prefix("m0_"),
                    self.compute_textual_similarity_scores(
                        text_encoded, summary_1_encoded, "text", "summary"
                    ).add_prefix("m1_"),
                    self.compute_textual_similarity_scores(
                        ref_encoded,
                        summary_0_encoded[f_ref.index],
                        "ref",
                        "summary",
                        f_ref.index,
                    ).add_prefix("m0_"),
                    self.compute_textual_similarity_scores(
                        ref_encoded,
                        summary_1_encoded[f_ref.index],
                        "ref",
                        "summary",
                        f_ref.index,
                    ).add_prefix("m1_"),
                ],
                axis=1,
            )
        elif method == "axis":

            # Method embeddings
            summary_encoded = self.compute_sentence_embeddigns(df["summary"], encoder)

            # Concat
            df = pd.concat(
                [
                    df,
                    self.compute_textual_similarity_scores(
                        text_encoded, summary_encoded, "text", "summary"
                    ).add_prefix("m_"),
                    self.compute_textual_similarity_scores(
                        ref_encoded,
                        summary_encoded[f_ref.index],
                        "ref",
                        "summary",
                        f_ref.index,
                    ).add_prefix("m_"),
                ],
                axis=1,
            )
        else:
            raise ValueError("Wrong type.")

        return df

    @staticmethod
    def add_ref_summary(data: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Add reference summary.
        
        """
        df = data.copy()

        if method == "comparisons":

            # Extract the rows where the policy value is equal to 'ref' and select 'text', 'summary_0', and 'summary_1'
            ref_rows = df.loc[
                (df["policy_0"] == "ref") | (df["policy_1"] == "ref"),
                ["text", "summary_0", "summary_1", "policy_0", "policy_1"],
            ]

            # Create a new column 'ref_summary' that contains the summary where the policy value is equal to 'ref'
            ref_rows["ref_summary"] = ref_rows.apply(
                lambda row: row["summary_0"]
                if row["policy_0"] == "ref"
                else row["summary_1"],
                axis=1,
            )

            # Filter and drop duplicates
            ref_rows = ref_rows.loc[:, ["text", "ref_summary"]].drop_duplicates()

        elif method == "axis":

            # Extract the rows where the policy value is equal to 'ref' and select 'text', 'summary_0', and 'summary_1'
            ref_rows = (
                df.loc[df["policy"] == "ref", ["text", "summary"]]
                .drop_duplicates()
                .rename(columns={"summary": "ref_summary"})
            )

        else:
            raise ValueError("Wrong type.")

        # Find repeated ref_summaries
        refs_to_drop = []
        for _, ref in ref_rows[
            ref_rows["text"].isin(ref_rows[ref_rows["text"].duplicated()]["text"])
        ].groupby("text")["ref_summary"]:
            refs_to_drop.append(ref.apply(len).idxmin())

        # Drop repeated ref_summaries
        ref_rows.drop(refs_to_drop, inplace=True)

        # Left join
        df = df.merge(ref_rows, on="text", how="left")

        return df

    @staticmethod
    def compute_rouge_scores(
        data: pd.DataFrame, hyp_col: str, ref_col: str, name: str
    ) -> pd.DataFrame:
        """
        Compute ROUGE score.
        
        """

        df = data.copy()

        # Initialize ROUGE scorer
        rouge = Rouge()

        # Calculate ROUGE scores for each row in the DataFrame
        scores = []
        for _, row in df.iterrows():
            if pd.notna(row[hyp_col]) and pd.notna(row[ref_col]):
                score = rouge.get_scores(row[hyp_col], row[ref_col], avg=True)
                scores.append(score)
            else:
                scores.append(None)

        # Extract ROUGE-1, ROUGE-2, and ROUGE-L scores
        df.loc[:, f"{name}_rouge_1_f"] = [
            score["rouge-1"]["f"] if score is not None else None for score in scores
        ]
        df.loc[:, f"{name}_rouge_2_f"] = [
            score["rouge-2"]["f"] if score is not None else None for score in scores
        ]
        df.loc[:, f"{name}_rouge_l_f"] = [
            score["rouge-l"]["f"] if score is not None else None for score in scores
        ]

        return df.fillna(value=np.nan)

    @staticmethod
    def compute_bleu_scores(
        data: pd.DataFrame, hyp_col: str, ref_col: str, name: str
    ) -> pd.DataFrame:
        """
        Compute ROUGE score.
        
        """

        df = data.copy()

        # Initialize SmoothingFunction
        smooth_fn = SmoothingFunction().method1

        # Calculate BLEU scores for each row in the DataFrame
        scores = []
        for _, row in df.iterrows():
            if pd.notna(row[hyp_col]) and pd.notna(row[ref_col]):
                score = sentence_bleu(
                    [row[ref_col].split()],
                    row[hyp_col].split(),
                    smoothing_function=smooth_fn,
                )
                scores.append(score)
            else:
                scores.append(None)

        # Add BLEU scores to the DataFrame
        df.loc[:, f"{name}_bleu"] = scores

        return df.fillna(value=np.nan)

    @staticmethod
    def compute_readability_scores(text: str) -> Dict:
        """
        Compute READABILITY score. Readability quantifies the difficulty with 
        which a reader understands a text. The following measures are calculated:

        1. Flesch Reading Ease score (RE) (Flesch, 1979), which calculates a 
        ratio between the number of characters per sentence, the number of words 
        per sentence, and the number of syllables per word. Higher RE score 
        indicates a less complex utterance that is easier to read and 
        understand.
        2. Syllable Count: number of syllables present in the given text.
        3. Lexicon Count: number of words present in the text.
        4. Sentence Count: number of sentences present in the given text.
        5. Character Count: number of characters present in the given text.
        6. Letter Count: number of characters present in the given text without 
        punctuation.
        7. Polysyllable Count: number of words with a syllable count greater 
        than or equal to 3.
        8. Monosyllable Count: number of words with a syllable count equal to 
        one.
        
        """
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "syllable_count": textstat.syllable_count(text),
            "lexicon_count": textstat.lexicon_count(text, removepunct=True),
            "sentence_count": textstat.sentence_count(text),
            "char_count": textstat.char_count(text, ignore_spaces=True),
            "letter_count": textstat.letter_count(text, ignore_spaces=True),
            "polysyllab_count": textstat.polysyllabcount(text),
            "monosyllab_count": textstat.monosyllabcount(text),
        }

    @staticmethod
    def compute_reference_free_scores(text: str, summary: str) -> Dict:
        """
        Compute reference-free metrics.
        
        """

        def compression_ratio() -> float:
            """
            Compression Ratio. How much the algorithm has condensed the original 
            text.
            
            """
            return len(summary) / len(text)

        def jaccard_similarity(n_grams) -> float:
            """
            Jaccard Similarity: Jaccard Similarity is a measure of the similarity 
            between two sets. It is calculated as the size of the intersection of 
            the sets divided by the size of their union. This metric measures the 
            overlap of words (tokens) between the input text and the generated 
            summary. A higher Jaccard Similarity indicates a larger degree of 
            shared words between the input and the summary. However, this metric 
            might not capture the quality of the summary, especially if the 
            algorithm rephrases or paraphrases the content.
            
            """

            # Tokens
            input_tokens = word_tokenize(text)
            summary_tokens = word_tokenize(summary)

            if n_grams > 1:
                input_set = set(ngrams(input_tokens, n_grams))
                summary_set = set(ngrams(summary_tokens, n_grams))
            else:
                input_set = set(input_tokens)
                summary_set = set(summary_tokens)

            # Intersection and union
            intersection = input_set.intersection(summary_set)
            union = input_set.union(summary_set)

            return len(intersection) / len(union)

        return {
            "compression_ratio": compression_ratio(),
            "jaccard_similarity_1": jaccard_similarity(n_grams=1),
            "jaccard_similarity_2": jaccard_similarity(n_grams=2),
        }

    @staticmethod
    def compute_textual_similarity_scores(
        a: torch.Tensor, b: torch.Tensor, a_name: str, b_name: str, ref_index=None
    ) -> pd.Series:
        """
        Compute the textual similarity between 2 text embeddings. This similarity is 
        measured as cosine similarity.

        """

        if a.shape != b.shape:
            raise ValueError("Check torch shapes.")

        # Normalize the tensors
        a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)

        # Compute the element-wise product of the normalized tensors and sum over the last dimension
        cosine_scores = torch.sum(a_norm * b_norm, dim=-1)

        # Move the tensor to the CPU and convert it to a numpy array
        cosine_scores = cosine_scores.cpu().numpy()

        return pd.DataFrame(
            cosine_scores,
            index=ref_index,
            columns=[f"{a_name}_{b_name}_xfmr_similarity"],
        )

    @staticmethod
    def compute_sentence_embeddigns(
        text: pd.Series, model: SentenceTransformer
    ) -> torch.Tensor:
        """
        Compute sentence embeddings.
        
        """
        return model.encode(
            text, batch_size=64, convert_to_tensor=True, show_progress_bar=True
        )

