def clean_area_atuacao(df, column_name='vaga_areas_atuacao'):
    """
    Limpa as strings de uma coluna específica de um DataFrame,
    substituindo espaços e hífens por underlines, removendo espaços extras
    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to clean. Defaults to 'vaga_areas_atuacao'.

    Returns:
        pd.Series: The cleaned column.
    """
    cleaned_column = (
        df[column_name]
        .str.replace(r'[-\s]+', '_', regex=True)
        .str.strip('_')
        .str.strip()
    )
    return cleaned_column