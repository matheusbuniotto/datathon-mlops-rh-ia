SELECT
    -- Vaga
    v.codigo_vaga::INT AS codigo_vaga,
    v.titulo_vaga,
    v.cliente,
    CASE WHEN v.vaga_sap ILIKE 'sim' THEN 1 ELSE 0 END AS vaga_sap,
    v.nivel_profissional AS nivel_profissional_vaga,
    v.nivel_ingles AS nivel_ingles_vaga,
    v.areas_atuacao AS vaga_areas_atuacao,
    v.principais_atividades,
    v.competencias,

    -- Prospecção
    p.codigo_vaga::INT AS prospect_codigo_vaga,
    p.codigo_candidato::INT AS prospect_codigo_candidato,
    p.nome_candidato AS nome_candidato,
    p.situacao AS situacao_candidato,
    p.data_candidatura,
    p.ultima_atualizacao,
    p.comentario,
    p.recrutador,

    -- Candidato
    a.codigo_candidato::INT AS applicants_codigo_candidato,
    a.nome AS nome_candidato,
    a.email,
    a.area_atuacao AS candidato_area_atuacao,
    a.nivel_profissional,
    a.nivel_academico,
    a.nivel_ingles,
    a.conhecimentos_tecnicos,
    a.cv

FROM read_parquet('data/processed/vagas.parquet') v
LEFT JOIN read_parquet('data/processed/prospects.parquet') p
    ON v.codigo_vaga = p.codigo_vaga
LEFT JOIN read_parquet('data/processed/applicants.parquet') a
    ON p.codigo_candidato = a.codigo_candidato