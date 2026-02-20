import pandas as pd


def resolve_alias(alias: str) -> str:
    """
    Normalize a raw alias from .dat files into the canonical
    Original_Name_in_Source_Literature used in the master CSV.

    Handles:
      - Trailing underscores  (e.g. "hepta_963_" -> "hepta_963")
      - Townsend prefix format (e.g. "2020_Townsend_4540-hepta_963" -> "hepta_963")
    """
    name = alias.rstrip('_')
    if 'Townsend' in name and '-' in name:
        name = name.split('-', 1)[1]
    return name


def find_match_in_df(alias: str, all_df: pd.DataFrame):
    """
    Given a raw alias, return (match_mask, skip) for looking up the row in all_df.

    For Kelly/Naylor aliases the lookup is by CycPeptMPDB_ID.
    For everything else it is by Original_Name_in_Source_Literature.

    Returns:
        (match_mask, False)  on success
        (None, True)         if no match found (should be skipped with a warning)
    Raises AssertionError if a Kelly/Naylor ID has no match.
    """
    if alias.startswith('Kelly') or alias.startswith('Naylor'):
        cyc_id = int(alias.split('_')[1])
        mask = all_df['CycPeptMPDB_ID'] == cyc_id
        assert mask.any(), (
            f"No match found in reference for CycPeptMPDB_ID: {cyc_id}"
        )
        return mask, False

    origin_name = resolve_alias(alias)
    mask = all_df['Original_Name_in_Source_Literature'] == origin_name
    if not mask.any():
        print(
            f"Warning: No match found in all_df for "
            f"Original_Name_in_Source_Literature: {origin_name}"
        )
        return None, True
    return mask, False
