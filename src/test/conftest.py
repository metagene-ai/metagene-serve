
# for sanity check
def pytest_addoption(parser):
    parser.addoption("--model_format", action="store", default="st", help="Format of the model")
    parser.addoption("--model_dir", action="store", default="./model_dir", help="Path to the model directory")
    parser.addoption("--data_path", action="store", default="./data.txt", help="Path to the data directory")
    parser.addoption("--output_dir", action="store", default="./output_dir", help="Path to the output directory")
