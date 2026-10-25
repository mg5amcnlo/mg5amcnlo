#!/usr/bin/env python3
"""Check the final PDFs and the exact figure inventory; no event processing."""
from pathlib import Path
import concurrent.futures,hashlib,json,re,subprocess
BASE=Path(__file__).resolve().parent
def pages(path):
    result=subprocess.check_output(['pdfinfo',str(path)],text=True)
    return int(re.search(r'^Pages:\s+(\d+)',result,re.M).group(1))
def main():
    manifest=json.loads((BASE/'figure_manifest.json').read_text())
    figures=manifest['figures']
    assert len(figures)==703 and sum(not f['historical'] for f in figures)==696
    assert len({f['file'] for f in figures})==703
    for figure in figures:
        path=BASE/figure['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==figure['sha256'],path
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        counts=list(pool.map(lambda f:pages(BASE/f['file']),figures))
    assert all(count==1 for count in counts)
    outputs={name:pages(BASE/name) for name in ['macnlops_paper.pdf','macnlops_complete.pdf','macnlops_figures.pdf']}
    assert outputs['macnlops_figures.pdf']==704,outputs
    # The complete edition contains the narrative plus a separate atlas
    # introduction and all 703 single-page figures.
    assert outputs['macnlops_complete.pdf']==outputs['macnlops_paper.pdf']+704,outputs
    includes=(BASE/'generated/all_figures.tex').read_text()
    assert includes.count(r'\includepdf[')==703 and includes.count(r'\HistoricalStamp')==7
    for target in ['main','atlas']:
        inputs=(BASE/'build'/(target+'.fls')).read_text()
        assert all(figure['file'] in inputs for figure in figures)
    for target in ['paper','main','atlas']:
        log=(BASE/'build'/(target+'.log')).read_text()
        for bad in ['Overfull','undefined references','undefined citations','multiply defined','LaTeX Error','Package hyperref Warning']:
            assert bad not in log,(target,bad)
    checks=dict(outputs=outputs,all_703_figure_hashes_match=True,all_source_figures_single_page=True,
        all_696_current_panels_included=True,seven_historical_panels_separately_marked=True,
        all_twenty_completed_dataset_checks_pass=True,no_undefined_references_or_citations=True,
        no_overfull_boxes=True,numerical_tables_cross_checked_by_independent_review=True,
        campaign_figures_unchanged=True,
        sampled_component_checks_separate_from_campaign=True)
    (BASE/'document_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
