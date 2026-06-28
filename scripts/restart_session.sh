#!/usr/bin/env bash
# 채팅 세션 재시작 스크립트.
# 트리거: 텔레그램에서 "세션 새로 시작해" 류 명령 → 현재 세션이 (1) HANDOFF.md 갱신 후
#        (2) 이 스크립트 실행. 그러면 기존 telegram claude 세션을 종료하고, 부트스트랩
#        프롬프트로 새 세션을 띄운다(새 세션이 텔레그램으로 시작/파악완료 메시지 전송).
#
# 직접 터미널에서 실행해도 됨: `bash scripts/restart_session.sh`
set -uo pipefail

PROJ="/home/myoungjin/01_Research/01_Projects/RIBBO"
# claude 바이너리는 conda env(RIBBO)가 아니라 ~/.local/bin 에 있다. conda bin만 PATH에
# 붙이면 못 찾을 수 있으므로 실제 경로를 명시적으로 해석한다.
CLAUDE_BIN="$(command -v claude 2>/dev/null || true)"
[ -z "$CLAUDE_BIN" ] && CLAUDE_BIN="$HOME/.local/bin/claude"
PYBIN_DIR="/home/myoungjin/miniconda3/envs/RIBBO/bin"
# 긴 한국어 프롬프트는 파일로 두고, 짧은 1줄 프롬프트로 그 파일을 읽게 한다(쉘 따옴표 안전).
BOOT_PROMPT='scripts/restart_bootstrap.txt 파일을 읽고, 그 안의 지시를 사용자 추가 입력 없이 즉시 그대로 수행해라.'

# 기존 telegram 세션을 "완전히" 종료한다. 핵심: claude 부모뿐 아니라
# 텔레그램 polling을 도는 bun 자식(claude-plugins-official/telegram)까지 죽여야
# 새 세션의 Telegram getUpdates 가 409(conflict) 없이 시작된다.
# (이 스크립트 자신의 cmdline 은 'bash .../restart_session.sh' 라 아래 패턴에 안 걸려 self-kill 안 됨.)
# 주의: 프롬프트를 --channels 앞에 두므로(do_launch) cmdline 이 'claude <prompt> --channels ...'
# 형태다. 따라서 'claude --channels' 로는 안 잡힌다. 'claude --resume'(이 대화 같은 resume 세션)는
# 일부러 제외하려고 --channels 기준으로만 매칭한다.
PAT_CLAUDE='--channels plugin:telegram'
PAT_BUN='claude-plugins-official/telegram'

kill_old_sessions() {
  pkill -f "$PAT_CLAUDE" 2>/dev/null || true
  pkill -f "$PAT_BUN"    2>/dev/null || true
  # 프로세스가 사라지고 Telegram long-poll 슬롯이 풀릴 때까지 대기(최대 ~15s).
  local i
  for i in $(seq 1 30); do
    if ! pgrep -f "$PAT_CLAUDE" >/dev/null 2>&1 && ! pgrep -f "$PAT_BUN" >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
  # 끈질긴 잔존 프로세스 강제 종료.
  pkill -9 -f "$PAT_CLAUDE" 2>/dev/null || true
  pkill -9 -f "$PAT_BUN"    2>/dev/null || true
  # bun 연결이 끊긴 뒤 Telegram 서버가 이전 getUpdates 를 정리할 여유.
  sleep 3
}

# 기존 세션 정리 후 새 telegram 세션으로 교체(exec). 터미널/분리 양쪽에서 공유.
do_launch() {
  kill_old_sessions
  cd "$PROJ"
  export PATH="$PYBIN_DIR:$PATH"
  # 주의: --channels 는 variadic(여러 값) 옵션이라, 프롬프트를 --channels 뒤에 두면
  # 프롬프트까지 채널 항목으로 삼켜져 "--channels entries must be tagged" 오류가 난다.
  # usage 는 `claude [options] [prompt]` 이므로 프롬프트(positional)를 --channels 앞에 둔다.
  exec "$CLAUDE_BIN" "$BOOT_PROMPT" --channels plugin:telegram@claude-plugins-official
}

# 분리(setsid) 모드에서 재진입하는 내부 진입점: 부모가 완전히 반환할 시간을 잠깐 준 뒤 launch.
if [ "${1:-}" = "__launch" ]; then
  sleep 2
  do_launch
fi

cd "$PROJ"
date '+%Y-%m-%d %H:%M:%S' > "$PROJ/.last_session_restart"

if [ -t 1 ]; then
  # (A) 터미널에서 직접 실행: 기존 텔레그램 세션 종료 후 이 터미널을 새 세션으로 교체(exec).
  echo "[restart_session] 터미널 모드: 기존 telegram 세션(+bun poller) 종료 후 새 세션으로 교체합니다."
  echo "[restart_session] claude=$CLAUDE_BIN"
  do_launch
else
  # (B) 현재 claude 세션 내부에서 트리거: 프로세스 트리에서 분리(setsid)해
  #     기존 세션을 종료하고 새 세션을 시작. (분리되어 있어야 pkill로 현재 세션이 죽어도 살아남음)
  echo "[restart_session] 분리 모드: 새 세션 예약 + 기존 세션 종료 예정(약 4초 후). 로그: $PROJ/.session_restart.log"
  setsid bash "$0" __launch < /dev/null >> "$PROJ/.session_restart.log" 2>&1 &
  disown
fi
