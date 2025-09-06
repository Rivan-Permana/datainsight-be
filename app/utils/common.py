import io

def df_info_string(df) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()
