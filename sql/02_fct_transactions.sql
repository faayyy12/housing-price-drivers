-- Architecture note: the design doc's "Transform" stage originally called
-- for plain SQL to build fct_transactions from stg_transactions. In
-- practice, the cleaning logic needed (Chinese-numeral floor parsing,
-- ROC-date conversion, comma-compound floor splitting -- see M0_NOTES.md)
-- is impractical to express in plain SQL without stored procedures, and
-- doing it in Python gives unit-testable, pure functions (src/clean.py)
-- instead of untestable CASE-statement spaghetti. quality.py (via
-- clean.clean_row) builds fct_transactions directly and is the real
-- transform step.
--
-- This file is a one-time setup step (run once after 01_schema.sql, not
-- part of the recurring refresh.sh cycle): indexes + constraints that
-- support the dashboard/analysis query patterns and add a DB-level safety
-- net matching the sanity bounds already enforced in quality.py.

CREATE INDEX IF NOT EXISTS idx_fct_transactions_district
    ON fct_transactions (city_code, district);

CREATE INDEX IF NOT EXISTS idx_fct_transactions_txn_date
    ON fct_transactions (txn_date);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_price_per_ping_positive'
    ) THEN
        ALTER TABLE fct_transactions
            ADD CONSTRAINT chk_price_per_ping_positive CHECK (price_per_ping > 0);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_area_ping_positive'
    ) THEN
        ALTER TABLE fct_transactions
            ADD CONSTRAINT chk_area_ping_positive CHECK (area_ping > 0);
    END IF;
END $$;
