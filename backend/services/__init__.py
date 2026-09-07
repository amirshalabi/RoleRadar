"""
Service layer: clean, orchestration-level functions the Streamlit UI
(or any other caller) uses instead of talking to backend/db/* modules
directly. A db/ module owns exactly one table; a service composes
several of them into one user-facing action (e.g. "save this role"
touches both favorites and applications).
"""
