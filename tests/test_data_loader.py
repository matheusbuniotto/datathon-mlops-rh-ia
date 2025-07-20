import os
import sys

sys.path.append(os.path.abspath(".."))

import pytest

from app.data_loader import load_applicants, load_jobs, load_prospects


@pytest.fixture
def data_dir():
    return os.path.abspath("data/raw")


def test_load_applicants(data_dir):
    df = load_applicants(f"{data_dir}/applicants.json")
    assert not df.empty
    assert "codigo_candidato" in df.columns
    assert df["cv"].str.len().mean() > 10


def test_load_jobs(data_dir):
    df = load_jobs(f"{data_dir}/vagas.json")
    assert not df.empty
    assert "codigo_vaga" in df.columns
    assert "principais_atividades" in df.columns
    assert df["principais_atividades"].str.len().mean() > 10


def test_load_prospects(data_dir):
    df = load_prospects(f"{data_dir}/prospects.json")
    assert not df.empty
    assert "codigo_vaga" in df.columns
    assert "codigo_candidato" in df.columns
