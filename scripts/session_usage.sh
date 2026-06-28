#!/bin/bash
# Report token usage + rough cost for the CURRENT Claude Code session, parsed from the
# session transcript. Lets the user check "usage" from outside (e.g. via Telegram: they
# ask, the agent runs this and replies). The official plan-quota meter (/usage) is TUI-only
# and can't be pulled here; this gives the token/cost side, which is what drives spend.
PROJDIR="$HOME/.claude/projects/-home-myoungjin-01-Research-01-Projects-RIBBO"
TR="${1:-$(ls -t "$PROJDIR"/*.jsonl 2>/dev/null | head -1)}"   # default: most recently active session
PY=/home/myoungjin/miniconda3/envs/RIBBO/bin/python
"$PY" - "$TR" <<'PY'
import json, sys, os
f = sys.argv[1]
ti=to=tcc=tcr=turns=0
with open(f, errors='replace') as fh:
    for line in fh:
        try: d=json.loads(line)
        except: continue
        msg = d.get('message') if isinstance(d.get('message'), dict) else {}
        u = msg.get('usage') or d.get('usage')
        if isinstance(u, dict):
            turns+=1
            ti+=u.get('input_tokens',0) or 0
            to+=u.get('output_tokens',0) or 0
            tcc+=u.get('cache_creation_input_tokens',0) or 0
            tcr+=u.get('cache_read_input_tokens',0) or 0
# rough cost: standard Opus rates ($15/M in, $75/M out; cache write 1.25x, cache read 0.1x)
IN,OUT = 15/1e6, 75/1e6
cost = ti*IN + to*OUT + tcc*IN*1.25 + tcr*IN*0.1
print(f"session: {os.path.basename(f)}")
print(f"turns: {turns}")
print(f"output:      {to:>12,} tok")
print(f"input(raw):  {ti:>12,} tok")
print(f"cache write: {tcc:>12,} tok")
print(f"cache read:  {tcr:>12,} tok")
print(f"~cost (est, std Opus rates): ${cost:,.2f}")
PY
