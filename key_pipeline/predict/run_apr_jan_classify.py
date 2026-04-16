"""
Batch Claude classification for the apr/jan Claude-folder CSVs.
Paths match the former step 4 in article_extract/run_article_pipeline.py.

Run from repo root:
  python -m key_pipeline.predict.run_apr_jan_classify
"""

from key_pipeline.predict.claude_dxy_predict import process_csv_to_csv

DIR = "data/dxy_training/claude/"
CLN_CSV = DIR + "apr_jan_cln.csv"
OUT_CSV = DIR + "apr_jan_res.csv"
GT_CSV = DIR + "gt_example_ft.csv"


def main() -> None:
    process_csv_to_csv(
        input_csv_path=CLN_CSV,
        output_csv_path=OUT_CSV,
        ground_truth_csv_path=GT_CSV,
        max_few_shot_examples=39,
        test_row_index=None,
        verbose=False,
    )


if __name__ == "__main__":
    main()
