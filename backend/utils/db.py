"""MySQL connection helper and schema bootstrap."""

import mysql.connector
from mysql.connector import Error
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MYSQL_CONFIG


def get_connection(use_database=True):
    cfg = dict(MYSQL_CONFIG)
    if not use_database:
        cfg.pop("database", None)
    return mysql.connector.connect(**cfg)


def init_database():
    """Create the database and all required tables if they don't exist."""
    conn = get_connection(use_database=False)
    cur = conn.cursor()

    db_name = MYSQL_CONFIG["database"]
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
    cur.execute(f"USE `{db_name}`")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS tbl_Spatial_Clusters (
            Cluster_ID   INT          PRIMARY KEY AUTO_INCREMENT,
            Centroid_Lat  DECIMAL(10,8) NOT NULL,
            Centroid_Lon  DECIMAL(11,8) NOT NULL,
            Radius_Eps    DECIMAL(5,2)  NOT NULL,
            Incident_Count INT          NOT NULL,
            INDEX idx_centroid (Centroid_Lat, Centroid_Lon)
        ) ENGINE=InnoDB
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS tbl_Accident_Records (
            Record_ID     INT          PRIMARY KEY AUTO_INCREMENT,
            Latitude      DECIMAL(10,8) NOT NULL,
            Longitude     DECIMAL(11,8) NOT NULL,
            Timestamp     DATETIME      NOT NULL,
            Weather_Cond  VARCHAR(50)   NULL,
            Severity_Hist INT           NOT NULL,
            Cluster_ID    INT           NULL,
            INDEX idx_coords (Latitude, Longitude),
            FOREIGN KEY (Cluster_ID) REFERENCES tbl_Spatial_Clusters(Cluster_ID)
                ON DELETE SET NULL
        ) ENGINE=InnoDB
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS tbl_Risk_Assessments (
            Assessment_ID INT          PRIMARY KEY AUTO_INCREMENT,
            Cluster_ID    INT          NOT NULL,
            Pred_Severity FLOAT        NOT NULL,
            ARI_Score     FLOAT        NOT NULL,
            Risk_Tier     VARCHAR(20)  NOT NULL,
            Env_Modifier  VARCHAR(50)  NULL,
            FOREIGN KEY (Cluster_ID) REFERENCES tbl_Spatial_Clusters(Cluster_ID)
                ON DELETE CASCADE
        ) ENGINE=InnoDB
    """)

    conn.commit()
    cur.close()
    conn.close()
    print(f"[DB] Database '{db_name}' and tables initialised.")


def truncate_tables():
    """Clear all data from tables (respecting FK order)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SET FOREIGN_KEY_CHECKS = 0")
    for tbl in ("tbl_Risk_Assessments", "tbl_Accident_Records", "tbl_Spatial_Clusters"):
        cur.execute(f"TRUNCATE TABLE {tbl}")
    cur.execute("SET FOREIGN_KEY_CHECKS = 1")
    conn.commit()
    cur.close()
    conn.close()
    print("[DB] All tables truncated.")


if __name__ == "__main__":
    init_database()
