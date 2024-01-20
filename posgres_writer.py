import psycopg2
from psycopg2 import sql

# Parámetros de conexión a la base de datos
dbname = 'vectorPoC'
user = 'postgres'
password = 'pochovive'
host = 'localhost'
port = '5433'

conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)

# Crear un cursor para ejecutar consultas SQL
cur = conn.cursor()

# Habilitar la extensión PostGIS en la base de datos (si no está habilitada)
cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")

# Crear una tabla espacial (si no existe)
table_name = 'mi_tabla_espacial'
create_table_query = sql.SQL("""
    CREATE TABLE IF NOT EXISTS {} (
        id SERIAL PRIMARY KEY,
        nombre VARCHAR(255),
        geom GEOMETRY(Point, 4326) -- Ejemplo con un punto geométrico en coordenadas geográficas
    )
""").format(sql.Identifier(table_name))

cur.execute(create_table_query)

# Insertar datos en la tabla
insert_data_query = sql.SQL("""
    INSERT INTO {} (nombre, geom) VALUES (%s, ST_GeomFromText(%s, 4326))
""").format(sql.Identifier(table_name))

# Datos a insertar (nombre y coordenadas del punto)
datos = [('Punto A', 'POINT(-70.12345 40.67890)'), ('Punto B', 'POINT(-71.98765 42.34567)')]

# Ejecutar la consulta de inserción para cada conjunto de datos
cur.executemany(insert_data_query, datos)

# Confirmar los cambios en la base de datos
conn.commit()

referencia_x = -71.0
referencia_y = 40.6

# Ejemplo de consulta para obtener los 5 puntos más cercanos por similitud coseno
# Ejemplo de consulta para obtener los 5 puntos más cercanos por similitud coseno
consulta_similitud_coseno = sql.SQL("""
    SELECT id, nombre, ST_AsText(geom), 
           COS(POINT(%s, %s) <-> geom) AS similitud_coseno
    FROM mi_tabla_espacial
    ORDER BY ST_Distance(geom, ST_MakePoint(%s, %s)) ASC
    LIMIT 5
""")


# Ejecutar la consulta
cur.execute(consulta_similitud_coseno, (referencia_x, referencia_y, referencia_x, referencia_y))

# Obtener los resultados
resultados = cur.fetchall()

# Imprimir los resultados
for resultado in resultados:
    print(resultado)


# Cerrar el cursor y la conexión
cur.close()
conn.close()