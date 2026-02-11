"""Diagnostic: inspect `filtered_df` provider values and provider-change counts."""
from src.load_data_db import load_project_data


def main(project_id=149):
    raw, filtered, project = load_project_data(project_id)
    print('Filtered rows:', len(filtered))
    print('Columns:', list(filtered.columns))

    if 'm_alpha_long' in filtered.columns:
        vals = filtered['m_alpha_long'].fillna('<na>').astype(str)
        unique = vals.unique()
        print('Unique providers (count):', len(unique))
        print('Top provider counts:')
        print(vals.value_counts().head(10).to_string())

        diffs = 0
        for sid, g in filtered.groupby('session_id'):
            pv = g['m_alpha_long'].fillna('<na>').astype(str).unique()
            if len(pv) > 1:
                diffs += 1
        print('Sessions with provider changes:', diffs)
    else:
        print('m_alpha_long not in filtered_df columns')

    print('\nSample rows:')
    print(filtered[['lat', 'lon', 'm_alpha_long']].head(30).to_string(index=False))


if __name__ == '__main__':
    main()
