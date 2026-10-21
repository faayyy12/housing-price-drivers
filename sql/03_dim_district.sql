-- Dashboard-context rollups, fully rebuilt from fct_transactions on every
-- refresh cycle (cheap full TRUNCATE + INSERT -- this table is entirely
-- derived, not user-entered, so there's nothing to upsert against).
--
-- NEVER joined back into the model [M6]: district median price-per-ping as
-- a model feature would be circular leakage on top of the district dummies
-- already used as the model's location signal. This feeds the dashboard's
-- district-comparison view only.

TRUNCATE TABLE dim_district;

INSERT INTO dim_district (city_code, district, txn_count, median_price_per_ping, updated_at)
SELECT
    city_code,
    district,
    count(*) AS txn_count,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY price_per_ping) AS median_price_per_ping,
    now() AS updated_at
FROM fct_transactions
GROUP BY city_code, district;
