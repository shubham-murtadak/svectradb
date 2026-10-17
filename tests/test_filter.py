"""
Unit tests for the MongoDB-style metadata filtering engine.
"""

from svectra.engine.filter import match_filter, compile_filter


def test_basic_equality():
    meta = {"category": "machine_learning", "year": 2024}
    assert match_filter(meta, {"category": "machine_learning"})
    assert not match_filter(meta, {"category": "web_dev"})
    assert match_filter(meta, {"year": 2024})


def test_comparison_operators():
    meta = {"price": 49.99, "stock": 10}
    assert match_filter(meta, {"price": {"$gt": 20}})
    assert match_filter(meta, {"price": {"$gte": 49.99}})
    assert match_filter(meta, {"price": {"$lt": 100}})
    assert match_filter(meta, {"price": {"$lte": 49.99}})
    assert not match_filter(meta, {"price": {"$gt": 50}})


def test_in_and_nin_operators():
    meta = {"tags": ["ai", "python", "vectordb"]}
    assert match_filter(meta, {"tags": {"$in": ["python", "rust"]}})
    assert not match_filter(meta, {"tags": {"$in": ["golang", "ruby"]}})
    assert match_filter(meta, {"tags": {"$nin": ["java", "c++"]}})


def test_logical_operators():
    meta = {"category": "ai", "status": "active", "views": 1500}

    # $and
    assert match_filter(meta, {
        "$and": [
            {"category": "ai"},
            {"views": {"$gt": 1000}},
        ]
    })

    # $or
    assert match_filter(meta, {
        "$or": [
            {"category": "crypto"},
            {"status": "active"},
        ]
    })

    # $not
    assert match_filter(meta, {
        "$not": {"category": "crypto"}
    })


def test_string_operators():
    meta = {"title": "Introduction to Vector Search with SvectraDB"}
    assert match_filter(meta, {"title": {"$contains": "Vector Search"}})
    assert match_filter(meta, {"title": {"$startswith": "Intro"}})
    assert match_filter(meta, {"title": {"$endswith": "SvectraDB"}})
    assert not match_filter(meta, {"title": {"$contains": "PostgreSQL"}})


def test_nested_path():
    meta = {"author": {"name": "Shubham", "role": "founder"}}
    assert match_filter(meta, {"author.name": "Shubham"})
    assert match_filter(meta, {"author.role": "founder"})
    assert not match_filter(meta, {"author.name": "Alice"})
