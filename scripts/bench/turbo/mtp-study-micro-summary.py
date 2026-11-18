#!/usr/bin/env python3
"""Compare complete old/new projection event matrices; retain every control."""
import argparse
import json
from pathlib import Path
import re

def read(path):
    rows = {}
    for line in path.read_text().splitlines():
        if not line.startswith('onfly-bench '):
            continue
        fields = dict(re.findall(r'(\w+)=([^ ]+)', line))
        fields['site'] = re.search(r'site=(.*?) rows=', line)[1]
        key = (fields['site'], int(fields['rows']), fields['mode'], fields['cache'])
        if key in rows:
            raise ValueError('duplicate benchmark arm')
        if fields.get('row_policy_bitwise') != '1':
            raise ValueError('missing row-policy byte gate')
        for name in ['native_us', 'onfly_us']:
            fields[name] = float(fields[name])
            if fields[name] <= 0:
                raise ValueError('nonpositive event timing')
        fields['iu4_launches'] = int(fields['iu4_launches'])
        rows[key] = fields
    return rows

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('old', type=Path)
    p.add_argument('new', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--expected', type=int, default=104)
    a = p.parse_args()
    old, new = read(a.old), read(a.new)
    if old.keys() != new.keys() or len(old) != a.expected:
        raise ValueError(f'incomplete benchmark: {len(old)}/{len(new)} expected {a.expected}')
    records = []
    for key in sorted(old):
        x, y = old[key], new[key]
        record = dict(site=key[0], rows=key[1], mode=key[2], cache=key[3],
                      old_us=x['onfly_us'], new_us=y['onfly_us'],
                      old_native_us=x['native_us'], new_native_us=y['native_us'],
                      old_launches=x['iu4_launches'], new_launches=y['iu4_launches'],
                      raw_speedup=x['onfly_us']/y['onfly_us']-1,
                      native_control_adjusted_speedup=(x['onfly_us']/x['native_us'])/(y['onfly_us']/y['native_us'])-1)
        records.append(record)
        if key[1] == 8 and key[2] == 'dense-a8':
            print(json.dumps(record))
    a.out.write_text(json.dumps(dict(arms=len(records), row_policy_bitwise=True, records=records), indent=2)+'\n')
    print(json.dumps(dict(arms=len(records), output=str(a.out))))

if __name__ == '__main__':
    main()
