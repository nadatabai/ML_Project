import argparse
import runpy
import sys

from prefect import flow, task


# =========================
# TASKS
# =========================
def run_module(command):
    """Execute a Python module safely."""
    old_argv = sys.argv.copy()

    try:
        sys.argv = command
        runpy.run_module(
            command[0],
            run_name="__main__",
        )
    finally:
        sys.argv = old_argv


@task(name="Install Dependencies")
def install_dependencies():
    run_module(
        [
            "pip",
            "install",
            "-r",
            "requirements.txt",
        ]
    )


@task(name="Format Code - Black")
def format_code():
    run_module(
        [
            "black",
            "main.py",
            "pipeline_prefect.py",
            "tests",
        ]
    )


@task(name="Code Quality - Flake8")
def code_quality():
    run_module(
        [
            "flake8",
            "main.py",
            "pipeline_prefect.py",
        ]
    )


@task(name="Security Check - Bandit")
def security_check():
    run_module(
        [
            "bandit",
            "-r",
            "main.py",
            "pipeline_prefect.py",
        ]
    )


@task(name="Security Check - Bandit")
def security_check():
    run_module(
        [
            "bandit",
            "-r",
            "main.py",
            "pipeline_prefect.py",
        ]
    )


@task(name="Prepare Data")
def prepare_data():
    from main import calculate_average

    values = [10, 12, 14]
    average = calculate_average(values)

    print(f"Average: {average}")
    return average


@task(name="Train Model")
def train_model(data):
    print(f"Training example using data: {data}")
    return {"average": data}


@task(name="Save Model")
def save_model(model):
    print(f"Saving example model: {model}")


@task(name="Load Model")
def load_model():
    return {"average": 12}


@task(name="Evaluate Model")
def evaluate_model(model):
    print(f"Evaluating example model: {model}")


@task(name="Predict")
def predict():
    from main import calculate_average

    result = calculate_average([10, 12, 14])
    print(f"Prediction example: {result}")


# =========================
# FLOWS
# =========================


@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="clean")
def clean_flow():
    format_code()
    code_quality()
    security_check()


@flow(name="test")
def test_flow():
    run_tests()


@flow(name="prepare")
def prepare_flow():
    return prepare_data()


@flow(name="train")
def train_flow():
    data = prepare_data()
    model = train_model(data)
    save_model(model)


@flow(name="evaluate")
def evaluate_flow():
    model = load_model()
    evaluate_model(model)


@flow(name="predict")
def predict_flow():
    predict()


@flow(name="all")
def all_flow():
    install_flow()
    clean_flow()
    test_flow()

    data = prepare_data()
    model = train_model(data)
    save_model(model)
    evaluate_model(model)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--flow",
        choices=[
            "all",
            "clean",
            "evaluate",
            "install",
            "predict",
            "prepare",
            "test",
            "train",
        ],
        required=True,
    )

    args = parser.parse_args()

    flows = {
        "all": all_flow,
        "clean": clean_flow,
        "evaluate": evaluate_flow,
        "install": install_flow,
        "predict": predict_flow,
        "prepare": prepare_flow,
        "test": test_flow,
        "train": train_flow,
    }

    flows[args.flow]()
