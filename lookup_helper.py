import sqlite3
import json

DB_PATH = "aircraft_knowledge.db"

def get_aircraft_specs(name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        aircraft.name,
        aircraft.role,
        aircraft.country,
        capabilities.threat_level,
        capabilities.summary,
        performance.max_speed_kph,
        performance.combat_range_km,
        performance.service_ceiling_m,
        armament.air_to_air,
        armament.air_to_ground
    FROM aircraft
    JOIN capabilities ON aircraft.id = capabilities.aircraft_id
    JOIN performance ON aircraft.id = performance.aircraft_id
    JOIN armament ON aircraft.id = armament.aircraft_id
    WHERE aircraft.name = ?
    """, (name,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "name": row[0],
        "role": row[1],
        "country": row[2],
        "threat_level": row[3],
        "summary": row[4],
        "performance": {
            "max_speed_kph": row[5],
            "combat_range_km": row[6],
            "service_ceiling_m": row[7]
        },
        "armament": {
            "air_to_air": json.loads(row[8]),
            "air_to_ground": json.loads(row[9])
        }
    }
