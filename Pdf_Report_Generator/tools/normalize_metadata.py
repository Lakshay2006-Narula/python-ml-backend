import json

f = 'metadata.json'
D = json.load(open(f))
ks = D.get('kpi_details') or D.get('kpi_summary') or D.get('kpi_names') or []
kd = {}
if isinstance(ks, dict):
    kd = ks
elif isinstance(ks, list):
    for item in ks:
        if isinstance(item, str):
            kd[item] = D.get('kpi_details', {}).get(item, {})
        elif isinstance(item, dict) and len(item) == 1:
            k = list(item.keys())[0]
            kd[k] = item[k]
        elif isinstance(item, dict):
            for k, v in item.items():
                if isinstance(v, dict):
                    kd[k] = v

D['kpi_details'] = kd
D['kpi_summary'] = list(kd.keys())
D.pop('kpi_names', None)
json.dump(D, open(f, 'w'), indent=2)
print('normalized metadata.json: kpi_summary length=', len(D['kpi_summary']))
