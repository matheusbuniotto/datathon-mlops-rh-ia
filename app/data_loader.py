import json
from typing import Any, Dict, List
import pandas as pd
from logaru import logger

logger = logger.bind(name="data_loader")

def clean_empty_values(value: Any) -> str:
    """Sanitize null-like values (None, empty string, etc.)."""
    return value.strip() if isinstance(value, str) and value.strip() else ""


def process_applicant_record(app_id: str, app_data: Dict[str, Any]) -> Dict[str, Any]:
    basic_info = app_data.get("infos_basicas", {})
    personal_info = app_data.get("informacoes_pessoais", {})
    professional_info = app_data.get("informacoes_profissionais", {})
    education_info = app_data.get("formacao_e_idiomas", {})

    return {
        "codigo_candidato": app_id,
        "nome": clean_empty_values(basic_info.get("nome")),
        "email": clean_empty_values(basic_info.get("email")),
        "telefone": clean_empty_values(basic_info.get("telefone")),
        "nivel_ingles": clean_empty_values(education_info.get("nivel_ingles")),
        "nivel_academico": clean_empty_values(education_info.get("nivel_academico")),
        "nivel_profissional": clean_empty_values(professional_info.get("nivel_profissional")),
        "area_atuacao": clean_empty_values(professional_info.get("area_atuacao")),
        "conhecimentos_tecnicos": clean_empty_values(professional_info.get("conhecimentos_tecnicos")),
        "cv": clean_empty_values(app_data.get("cv_pt")),
    }


def process_vaga_record(vaga_id: str, vaga_data: Dict[str, Any]) -> Dict[str, Any]:
    basic_info = vaga_data.get("informacoes_basicas", {})
    profile_info = vaga_data.get("perfil_vaga", {})
    benefits_info = vaga_data.get("beneficios", {})

    return {
        "codigo_vaga": vaga_id,
        "titulo_vaga": clean_empty_values(basic_info.get("titulo_vaga")),
        "cliente": clean_empty_values(basic_info.get("cliente")),
        "solicitante_cliente": clean_empty_values(basic_info.get("solicitante_cliente")),
        "empresa_divisao": clean_empty_values(basic_info.get("empresa_divisao")),
        "requisitante": clean_empty_values(basic_info.get("requisitante")),
        "analista_responsavel": clean_empty_values(basic_info.get("analista_responsavel")),
        "tipo_contratacao": clean_empty_values(basic_info.get("tipo_contratacao")),
        "data_requicisao": clean_empty_values(basic_info.get("data_requicisao")),
        "limite_contratacao": clean_empty_values(basic_info.get("limite_esperado_para_contratacao")),
        "vaga_sap": clean_empty_values(basic_info.get("vaga_sap")),
        "objetivo_vaga": clean_empty_values(basic_info.get("objetivo_vaga")),
        "prioridade_vaga": clean_empty_values(basic_info.get("prioridade_vaga")),
        "origem_vaga": clean_empty_values(basic_info.get("origem_vaga")),
        "nivel_profissional": clean_empty_values(profile_info.get("nivel profissional")),
        "nivel_ingles": clean_empty_values(profile_info.get("nivel_ingles")),
        "areas_atuacao": clean_empty_values(profile_info.get("areas_atuacao")),
        "principais_atividades": clean_empty_values(profile_info.get("principais_atividades")),
        "competencias": clean_empty_values(profile_info.get("competencia_tecnicas_e_comportamentais")),
        "observacoes": clean_empty_values(profile_info.get("demais_observacoes")),
        "beneficios_raw": f"{clean_empty_values(benefits_info.get('valor_venda'))} "
                          f"{clean_empty_values(benefits_info.get('valor_compra_1'))} "
                          f"{clean_empty_values(benefits_info.get('valor_compra_2'))}"
    }


def process_prospect_record(prospect_id: str, prospect_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    titulo = clean_empty_values(prospect_data.get("titulo"))
    modalidade = clean_empty_values(prospect_data.get("modalidade"))
    prospects = prospect_data.get("prospects", [])

    return [{
        "codigo_vaga": prospect_id,
        "codigo_candidato": clean_empty_values(p.get("codigo")),
        "nome_candidato": clean_empty_values(p.get("nome")),
        "situacao": clean_empty_values(p.get("situacao_candidado")),
        "data_candidatura": clean_empty_values(p.get("data_candidatura")),
        "ultima_atualizacao": clean_empty_values(p.get("ultima_atualizacao")),
        "comentario": clean_empty_values(p.get("comentario")),
        "recrutador": clean_empty_values(p.get("recrutador")),
        "modalidade": modalidade,
        "titulo_vaga": titulo
    } for p in prospects]


def load_applicants(filepath: str) -> pd.DataFrame:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    records = [process_applicant_record(k, v) for k, v in data.items()]
    return pd.DataFrame(records)


def load_jobs(filepath: str) -> pd.DataFrame:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    records = [process_vaga_record(k, v) for k, v in data.items()]
    return pd.DataFrame(records)


def load_prospects(filepath: str) -> pd.DataFrame:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    all_records = []
    for prospect_id, prospect_data in data.items():
        all_records.extend(process_prospect_record(prospect_id, prospect_data))
    return pd.DataFrame(all_records)
