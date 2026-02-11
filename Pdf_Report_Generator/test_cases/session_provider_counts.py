"""Print per-session unique provider counts and provider lists from filtered_df.

Usage:
    python -m test_cases.session_provider_counts
"""
from src.load_data_db import load_project_data


def main(project_id: int = 149):
    raw, filtered, project = load_project_data(project_id)

    if filtered is None or filtered.empty:
        print('No filtered data found')
        return

    # Ensure columns lower-case for safety
    cols = [c.lower() for c in filtered.columns]
    filtered.columns = cols

    if 'm_alpha_long' not in filtered.columns:
        print('m_alpha_long column not present')
        return

    session_stats = []
    for sid, g in filtered.groupby('session_id'):
        provs = sorted(set(g['m_alpha_long'].dropna().astype(str)))
        session_stats.append((int(sid), len(provs), provs))

    # Print header
    print(f"Total sessions: {len(session_stats)}")
    multiple = [s for s in session_stats if s[1] > 1]
    print(f"Sessions with >1 provider values: {len(multiple)}")
    print()

    # Print all sessions (session_id, unique_count, providers)
    for sid, cnt, provs in sorted(session_stats):
        provs_display = ', '.join(provs) if provs else '<none>'
        print(f"Session {sid}: {cnt} unique provider(s) -> {provs_display}")


if __name__ == '__main__':
    main()
