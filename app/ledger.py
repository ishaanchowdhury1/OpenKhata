from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "openkhata.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                phone TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                transaction_type TEXT NOT NULL
                    CHECK (transaction_type IN ('credit', 'payment')),
                amount_inr INTEGER NOT NULL CHECK (amount_inr > 0),
                description TEXT,
                due_date_text TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (customer_id)
                    REFERENCES customers(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_transactions_customer
                ON transactions(customer_id);

            CREATE INDEX IF NOT EXISTS idx_transactions_created
                ON transactions(created_at);
            """
        )

        # Migration for an existing OpenKhata database.
        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(customers)"
            ).fetchall()
        }

        if "phone" not in columns:
            conn.execute(
                "ALTER TABLE customers ADD COLUMN phone TEXT"
            )


def get_or_create_customer(name: str) -> int:
    name = name.strip()

    if not name:
        raise ValueError("Customer name cannot be empty")

    init_db()

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id
            FROM customers
            WHERE name = ? COLLATE NOCASE
            """,
            (name,),
        ).fetchone()

        if row:
            return int(row["id"])

        cursor = conn.execute(
            """
            INSERT INTO customers (name)
            VALUES (?)
            """,
            (name,),
        )

        return int(cursor.lastrowid)


def update_customer_phone(
    customer_id: int,
    phone: str | None,
) -> None:
    phone = (phone or "").strip() or None

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE customers
            SET phone = ?
            WHERE id = ?
            """,
            (phone, customer_id),
        )


def add_transaction(
    customer_name: str,
    transaction_type: str,
    amount_inr: int,
    description: str | None = None,
    due_date_text: str | None = None,
) -> int:
    """
    Add an already-confirmed transaction.

    AI extraction is never performed here.
    """

    customer_name = customer_name.strip()

    if not customer_name:
        raise ValueError("Customer name cannot be empty")

    if transaction_type not in {"credit", "payment"}:
        raise ValueError(
            "transaction_type must be 'credit' or 'payment'"
        )

    if not isinstance(amount_inr, int) or amount_inr <= 0:
        raise ValueError(
            "amount_inr must be a positive integer"
        )

    customer_id = get_or_create_customer(customer_name)

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO transactions (
                customer_id,
                transaction_type,
                amount_inr,
                description,
                due_date_text
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                customer_id,
                transaction_type,
                amount_inr,
                description,
                due_date_text,
            ),
        )

        return int(cursor.lastrowid)


def get_customer_balance(customer_id: int) -> int:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(
                SUM(
                    CASE
                        WHEN transaction_type = 'credit'
                            THEN amount_inr
                        WHEN transaction_type = 'payment'
                            THEN -amount_inr
                    END
                ),
                0
            ) AS balance
            FROM transactions
            WHERE customer_id = ?
            """,
            (customer_id,),
        ).fetchone()

        return int(row["balance"])


def get_all_customers():
    """Return customers with their current balance and phone."""

    init_db()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.name,
                c.phone,
                COALESCE(
                    SUM(
                        CASE
                            WHEN t.transaction_type = 'credit'
                                THEN t.amount_inr
                            WHEN t.transaction_type = 'payment'
                                THEN -t.amount_inr
                        END
                    ),
                    0
                ) AS balance
            FROM customers c
            LEFT JOIN transactions t
                ON c.id = t.customer_id
            GROUP BY c.id, c.name, c.phone
            ORDER BY c.name COLLATE NOCASE
            """
        ).fetchall()

        return [dict(row) for row in rows]


def get_customer_transactions(customer_id: int):
    """Return transaction history for one customer."""

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                transaction_type,
                amount_inr,
                description,
                due_date_text,
                created_at
            FROM transactions
            WHERE customer_id = ?
            ORDER BY id DESC
            """,
            (customer_id,),
        ).fetchall()

        return [dict(row) for row in rows]
