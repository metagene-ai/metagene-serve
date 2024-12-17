

def pytest_addoption(parser):
    """
    Registers command-line options for pytest to support sanity checks.

    Option naming pattern: <test_name>_<option_name>
        for sanity check test => sanity_check_<option_name>
        for env test => env_<option_name>
    """
    parser.addoption("--sanity_check_model_format", action="store", default="st",
                     help="Format of the model for sanity check: st, gguf, litgpt")
    parser.addoption("--sanity_check_model_dir", action="store", default="./model_dir",
                     help="Path to the model directory for sanity check")
    parser.addoption("--sanity_check_data_path", action="store", default="./data.txt",
                     help="Path to the data directory for sanity check")
    parser.addoption("--sanity_check_output_dir", action="store", default="./output_dir",
                     help="Path to the output directory for sanity check")

    parser.addoption("--env_sample_input", action="store", default="Hello World")
