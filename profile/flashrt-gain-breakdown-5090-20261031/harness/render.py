"""Render the self-contained reader report; no CDN or server required."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
template=(R/'harness/report.html').read_text(encoding='utf8')
data=(R/'analysis/summary.json').read_text(encoding='utf8').replace('</','<\\/')
(R.parents[1]/'flashrt_gain_breakdown.html').write_text(template.replace('__DATA__',data),encoding='utf8')
