from datasets import Dataset


def main():
    # Load text file into a Hugging Face dataset
    data_dir = "/project/neiswang_1391/MGFM/MGFM-serving/datasets/evaluate/mteb-hv"
    input_file = f"{data_dir}/mteb-hv-class-01.txt"
    output_file = f"{data_dir}/data/mteb-hv-class-01.parquet"

    dataset = Dataset.from_csv(input_file, delimiter=",", column_names=["sequence", "source"])
    dataset.to_parquet(output_file)
    print(f"Converted {input_file} to {output_file} for Hugging Face")


if __name__ == "__main__":
    main()