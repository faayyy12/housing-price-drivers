-- Tables: stg_transactions, fct_transactions, dim_district, rej_transactions.
--
-- Natural key [M4]: 編號 (case_id) -- confirmed unique within each city's raw
-- file in M0 recon (466/466 Taipei, 1474/1474 New Taipei; see M0_NOTES.md).
-- Cross-city uniqueness was not independently verified, so the actual key
-- used throughout is the composite (city_code, case_id).

DROP TABLE IF EXISTS rej_transactions;
DROP TABLE IF EXISTS fct_transactions;
DROP TABLE IF EXISTS dim_district;
DROP TABLE IF EXISTS stg_transactions;

-- Lightly typed parse of every raw row. Nothing dropped here -- this is the
-- audit trail; filtering/quarantine happens downstream into rej_transactions.
CREATE TABLE stg_transactions (
    city_code                      TEXT NOT NULL,
    case_id                        TEXT NOT NULL,          -- 編號
    transfer_seq                   TEXT,                   -- 移轉編號
    district                       TEXT,                   -- 鄉鎮市區
    transaction_sign                TEXT,                   -- 交易標的
    land_building_address            TEXT,                   -- 土地位置建物門牌 (de-identified)
    land_area_sqm                    NUMERIC,                -- 土地移轉總面積平方公尺
    urban_land_use_zone               TEXT,
    non_urban_land_use_zone           TEXT,
    non_urban_land_use_designation     TEXT,
    transaction_date_roc              TEXT,                   -- 交易年月日, raw ROC string e.g. 1150518
    transaction_unit_count            TEXT,                   -- 交易筆棟數, e.g. 土地1建物0車位0
    floor_raw                        TEXT,                   -- 移轉層次
    total_floors_raw                  TEXT,                   -- 總樓層數
    building_type                    TEXT,                   -- 建物型態
    main_use                         TEXT,                   -- 主要用途
    main_material                    TEXT,                   -- 主要建材
    completion_date_roc               TEXT,                   -- 建築完成年月
    building_area_sqm                 NUMERIC,                -- 建物移轉總面積平方公尺 (includes parking area, M0)
    rooms_raw                        TEXT,                   -- 建物現況格局-房
    halls_raw                        TEXT,                   -- 建物現況格局-廳
    baths_raw                        TEXT,                   -- 建物現況格局-衛
    has_partition_raw                  TEXT,                   -- 建物現況格局-隔間
    has_management_raw                 TEXT,                   -- 有無管理組織 (有/無)
    total_price                      NUMERIC,                -- 總價元
    unit_price_sqm                    NUMERIC,                -- 單價元平方公尺 (already parking-adjusted, M0)
    parking_type                     TEXT,                   -- 車位類別
    parking_area_sqm                  NUMERIC,                -- 車位移轉總面積平方公尺
    parking_price                    NUMERIC,                -- 車位總價元
    note                             TEXT,                   -- 備註
    main_building_area_sqm            NUMERIC,                -- 主建物面積
    auxiliary_building_area_sqm        NUMERIC,                -- 附屬建物面積
    balcony_area_sqm                  NUMERIC,                -- 陽台面積
    elevator_raw                     TEXT,                   -- 電梯
    source_file                      TEXT NOT NULL,
    loaded_at                        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (city_code, case_id)
);

-- Clean analytical table: one row per valid residential sale that passed all
-- filters in clean.py / quality.py, with engineered features.
CREATE TABLE fct_transactions (
    city_code               TEXT NOT NULL,
    case_id                  TEXT NOT NULL,
    district                 TEXT NOT NULL,
    txn_date                 DATE NOT NULL,
    price_per_ping            NUMERIC NOT NULL,   -- unit_price_sqm / 0.3025 (source already parking-adjusted; price per larger area unit)
    log_price_per_ping         NUMERIC NOT NULL,
    building_age              NUMERIC,
    area_ping                 NUMERIC NOT NULL,
    floor_level               INTEGER,            -- null for 全 / blank / unparseable
    total_floors              INTEGER,
    floor_ratio               NUMERIC,
    whole_building             BOOLEAN NOT NULL DEFAULT FALSE,   -- 移轉層次 = 全
    has_terrace_or_arcade       BOOLEAN NOT NULL DEFAULT FALSE,   -- comma-compound floor descriptor (M0 M7)
    building_type              TEXT,
    rooms                     INTEGER,
    halls                     INTEGER,
    baths                     INTEGER,
    has_management             BOOLEAN,
    bundled_parking            BOOLEAN NOT NULL DEFAULT FALSE,   -- 房地+車位
    parking_price_is_zero       BOOLEAN NOT NULL DEFAULT FALSE,   -- M0 caveat: area still includes parking
    loaded_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (city_code, case_id)
);

-- District-level rollups for dashboard context ONLY.
-- NEVER joined back into the model [M6]: district median price-per-ping as a
-- feature would be circular leakage on top of the district dummies already
-- used as the model's location signal.
CREATE TABLE dim_district (
    city_code              TEXT NOT NULL,
    district                TEXT NOT NULL,
    txn_count                INTEGER NOT NULL,
    median_price_per_ping     NUMERIC NOT NULL,
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (city_code, district)
);

-- Quarantined rows, kept with a reason code so data quality is reported, not
-- silently hidden. Append-only for v1 -- re-running on the same batch may
-- log a row's rejection more than once; acceptable at this scope.
CREATE TABLE rej_transactions (
    id            BIGSERIAL PRIMARY KEY,
    city_code      TEXT NOT NULL,
    case_id        TEXT NOT NULL,
    reason_code    TEXT NOT NULL,
    detail        TEXT,
    rejected_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_rej_transactions_reason ON rej_transactions (reason_code);
