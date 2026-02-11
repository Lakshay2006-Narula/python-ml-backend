"""Compare project ref_session_id with loaded raw/filtered session IDs.

Run as:
    python -m test_cases.compare_sessions
"""
from src.load_data_db import load_project_data


def main(project_id=149):
    raw, filtered, project = load_project_data(project_id)
    ref = str(project.get('ref_session_id', ''))
    session_ids = [int(s.strip()) for s in ref.split(',') if s.strip().isdigit()]

    print('project ref_session_id:', ref)
    print('parsed session_ids:', session_ids)
    print('expected session count:', len(session_ids))

    raw_ids = sorted(set(int(x) for x in raw['session_id'].dropna().unique()))
    filt_ids = sorted(set(int(x) for x in filtered['session_id'].dropna().unique()))

    print('distinct session_ids in raw (count):', len(raw_ids))
    print('distinct session_ids in filtered (count):', len(filt_ids))

    missing_in_filtered = [s for s in session_ids if s not in filt_ids]
    extra_in_filtered = [s for s in filt_ids if s not in session_ids]

    print('expected but missing in filtered:', missing_in_filtered)
    print('sessions in filtered but not in ref list (sample 20):', extra_in_filtered[:20])

    # print summary per expected session whether present and provider counts
    print('\nPer expected session presence and provider counts:')
    for s in session_ids:
        present = s in filt_ids
        cnt = 0
        providers = []
        if present:
            g = filtered[filtered['session_id'] == s]
            providers = sorted(set(g['m_alpha_long'].dropna().astype(str)))
            cnt = len(providers)
        print(f'  Session {s}: present={present}, unique_providers={cnt}, providers={providers}')


if __name__ == '__main__':
    main()
