"""Postgres snippets and routines"""


import uuid

import geopandas as gpd


def set_primary_key(con, table, schema="public", key_col="id"):
    sql = f'ALTER TABLE "{schema}"."{table}" ADD PRIMARY KEY ("{key_col}");'
    return con.execute(sql)


def set_primary_key_auto(con, table, schema="public", key_col="id"):
    sql = f'ALTER TABLE "{schema}"."{table}" ADD COLUMN "{key_col}" SERIAL PRIMARY KEY;'
    return con.execute(sql)


def set_foreign_key(con, table, fkey, ref_table, ref_key, schema="public"):
    sql = (
        f'ALTER TABLE "{schema}"."{table}" ADD CONSTRAINT "fk_{table}_{ref_table}" '
        f'FOREIGN KEY ("{fkey}") REFERENCES "{schema}"."{ref_table}" ("{ref_key}");'
    )

    return con.execute(sql)


def create_index(con, table, on, unique=False, schema="public"):
    if type(on) == str:
        on = [
            on,
        ]

    sql = (
        f'CREATE {"UNIQUE" if unique else ""} INDEX {table}_{"_".join(on)}_idx '
        f'ON "{schema}"."{table}" ({", ".join(on)});'
    )
    return con.execute(sql)


def check_exists(con, table, schema="public"):
    if not con.execute(
        f"""SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE  table_schema = '{schema}'
            AND    table_name   = '{table}');
            """
    ).first()[0]:
        return False
    else:
        return True


def upsert_df(con, df, table, unique_constraint_cols: list[str], schema="public"):
    assert check_exists(con, table, schema=schema), "table must exist to upsert"

    temp_table_name = f"temp_{uuid.uuid4().hex[:6]}"
    if isinstance(df, gpd.GeoDataFrame):
        df.to_postgis(temp_table_name, con, index=False)
    else:
        df.to_sql(temp_table_name, con, index=False)

    columns = list(df.columns)
    headers_sql_txt = ", ".join([f'"{i}"' for i in columns])
    index_cols = ", ".join([f'"{i}"' for i in unique_constraint_cols])
    # col1 = exluded.col1, col2=excluded.col2
    update_column_stmt = ", ".join([f'"{col}" = EXCLUDED."{col}"' for col in columns])
    try:
        # For the ON CONFLICT clause, postgres requires that
        # the columns have unique constraint
        constraint = f"unq_{table}_constraint_for_upsert"
        drop_constraint = (
            f'ALTER TABLE "{table}" DROP CONSTRAINT IF EXISTS {constraint};'
        )
        add_constraint = (
            f'ALTER TABLE "{table}" ADD CONSTRAINT {constraint} UNIQUE ({index_cols});'
        )
        con.execute(drop_constraint)
        con.execute(add_constraint)

        # Compose and execute upsert query
        query_upsert = f"""
        INSERT INTO "{table}" ({headers_sql_txt})
        SELECT {headers_sql_txt} FROM "{temp_table_name}"
        ON CONFLICT ({index_cols}) DO UPDATE
        SET {update_column_stmt};
        """
        con.execute(query_upsert)
    except Exception as e:
        print(e)
    finally:
        # ensure temporary table always removed
        con.execute(f"DROP TABLE {temp_table_name}")
