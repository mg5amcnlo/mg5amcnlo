from pathlib import Path
import gzip,json,re,math
work=Path(__file__).parent
runs={}
for name,expected in [('pool_restart_unity',40),('pool_restart_sum',20)]:
    path=work/'drell_yan/Events'/name/'events.lhe.gz'
    if not path.exists(): continue
    text=gzip.open(path,'rt').read()
    events=re.findall(r'<event(?:\s[^>]*)?>\s*(.*?)</event>',text,re.S)
    weights=[float(event.split()[2].replace('D','E')) for event in events]
    init=re.search(r'<init>\s*(.*?)</init>',text,re.S).group(1).strip().splitlines()
    alternatives=[float(weight.replace('D','E')) for weight in re.findall(r'<wgt\s[^>]*>\s*([^<]+)</wgt>',text)]
    rates=[list(map(float,row.split())) for row in init[1:]]
    manifest=json.loads((path.parent/'ampli_production.json').read_text())
    summary=(path.parent/'summary.txt').read_text()
    runs[name]={'events':len(events),'positive':sum(w>0 for w in weights),'negative':sum(w<0 for w in weights),
      'abs_min':min(map(abs,weights)),'abs_max':max(map(abs,weights)),'sum_abs_weights':sum(map(abs,weights)),
      'IDWTUP':int(init[0].split()[8]),'init_rates':rates,'rwgt_events':sum('<rwgt>' in e for e in events),
      'mgrwgt_events':sum('<mgrwgt>' in e for e in events),'alternative_weights':len(alternatives),
      'finite_alternative_weights':all(map(math.isfinite,alternatives)),
      'manifest':manifest,'summary':summary}
    assert len(events)==expected,(name,len(events))
    assert int(init[0].split()[8])==-4
    assert all(map(math.isfinite,weights+alternatives))
    assert all('<rwgt>' in event and '<mgrwgt>' in event for event in events)
    assert abs(sum(row[0] for row in rates)-manifest['cross_section'])<1.e-4
print(json.dumps({key:{k:v for k,v in value.items() if k not in ('manifest','summary')} for key,value in runs.items()},indent=2))
(work/'validation.json').write_text(json.dumps({'runs':runs,'source_hashes':json.loads((work/'source_hashes.json').read_text())},indent=2))
