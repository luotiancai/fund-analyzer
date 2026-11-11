"""Fund classifications shared by batch jobs and the lightweight web reader."""


def is_fof(fund_type, name=None) -> bool:
    """Use type first; some FOF funds omit the label from their short name."""
    return any("FOF" in str(value).upper() or "基金中基金" in str(value)
               for value in (fund_type, name))
