"""Composable canonical writes: nested owner operations must not commit their caller."""
from contextlib import contextmanager
from uuid import uuid4


@contextmanager
def write_transaction(db):
    nested = db.in_transaction
    savepoint = "write_" + uuid4().hex
    db.execute(f"SAVEPOINT {savepoint}" if nested else "BEGIN IMMEDIATE")
    try:
        yield db
    except BaseException:
        if nested:
            db.execute(f"ROLLBACK TO {savepoint}")
            db.execute(f"RELEASE {savepoint}")
        else:
            db.execute("ROLLBACK")
        raise
    else:
        db.execute(f"RELEASE {savepoint}" if nested else "COMMIT")
