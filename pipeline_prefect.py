import argparse
import os
import subprocess
import sys

from prefect import flow, task, get_run_logger

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PYTHON_BIN = sys.executable
MAIN_PY = "main.py"
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def _run(cmd):
    """Lance une commande et lève une erreur claire si elle échoue."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Commande échouée : {' '.join(cmd)}\n{result.stdout}\n{result.stderr}"
        )


# ---------- TASKS ----------
@task(name="install-dependencies", retries=2, retry_delay_seconds=5)
def install_dependencies() -> None:
    logger = get_run_logger()
    _run([PYTHON_BIN, "-m", "pip", "install", "-r", "requirements.txt"])
    logger.info("Dépendances installées.")


@task(name="format-code")
def format_code() -> None:
    logger = get_run_logger()
    _run([PYTHON_BIN, "-m", "black", *CODE_FILES])
    logger.info("Code formaté (black).")


@task(name="check-quality")
def check_quality() -> None:
    logger = get_run_logger()
    _run([PYTHON_BIN, "-m", "flake8", "--max-line-length=120", *CODE_FILES])
    logger.info("Qualité du code OK (flake8).")


@task(name="check-security")
def check_security() -> None:
    logger = get_run_logger()
    _run([PYTHON_BIN, "-m", "bandit", "-q", "-ll", *CODE_FILES])
    logger.info("Sécurité du code OK (bandit).")


@task(name="unit-tests")
def run_tests() -> None:
    logger = get_run_logger()
    _run([PYTHON_BIN, "-m", "pytest", "-q", "test_model_pipeline.py"])
    logger.info("Tests unitaires OK.")


@task(name="prepare-data")
def prepare_data() -> None:
    logger = get_run_logger()
    _run([PYTHON_BIN, MAIN_PY, "--step", "prepare"])
    logger.info("Données préparées.")


@task(name="train-model")
def train_model() -> None:
    logger = get_run_logger()
    _run([PYTHON_BIN, MAIN_PY, "--step", "train"])
    logger.info("Modèle entraîné et sauvegardé.")


@task(name="evaluate-model")
def evaluate_model() -> None:
    logger = get_run_logger()
    _run([PYTHON_BIN, MAIN_PY, "--step", "evaluate"])
    logger.info("Modèle évalué.")


@task(name="load-model")
def load_model() -> None:
    logger = get_run_logger()
    _run([PYTHON_BIN, MAIN_PY, "--step", "load"])
    logger.info("Modèle chargé.")


# ---------- FLOWS ----------
@flow(name="code")
def code_flow():
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_tests()


@flow(name="train")
def train_flow():
    prepare_data()
    train_model()


@flow(name="evaluate")
def evaluate_flow():
    load_model()
    evaluate_model()


@flow(name="predict")
def predict_flow():
    load_model()


@flow(name="all")
def all_flow():
    code_flow()
    prepare_data()
    train_model()
    evaluate_model()


if __name__ == "__main__":
    flows = {
        "all": all_flow,
        "code": code_flow,
        "train": train_flow,
        "entrainement": train_flow,
        "evaluate": evaluate_flow,
        "predict": predict_flow,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=flows.keys(), default="all")
    args = parser.parse_args()
    flows[args.flow]()
