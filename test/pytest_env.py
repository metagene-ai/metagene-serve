

@pytest.fixture
def sample_input(request):
    return request.config.getoption("env_sample_input")

def test_sample_input(sample_input):
    print(sample_input)
