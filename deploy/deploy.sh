#!/usr/bin/env bash
# ft-lab 教程发布脚本：把两份单文件 HTML 打包后推到 Cloudflare。
# 全部行为可用 --check（离线）与 --dry-run（不联网）验证。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="$ROOT/deploy"
OUT_DIR="$DEPLOY_DIR/public"
MAIN="$ROOT/dist/index.html"
DEEP="$ROOT/fromscratch/dist/from-scratch.html"

WRANGLER_VERSION="${WRANGLER_VERSION:-4.160.1}"
PROJECT="${PROJECT:-ft-lab-tutorials}"
BRANCH="${BRANCH:-main}"
TARGET="${TARGET:-pages}"
PYTHON="${PYTHON:-python3}"
MAX_FILE_BYTES=26214400
MAX_FILES=20000
NODE_MIN_MAJOR=22
RETRY_WAIT=2

MODE="deploy"
SITE_URL=""
DO_BUILD=0
SKIP_VERIFY=0
CHECKS=0
FAILS=0

usage() {
  cat <<'TXT'
用法： bash deploy/deploy.sh [选项]

首次配置（一次，约 32 秒）：
  0. npm install --prefix deploy              # 固定 wrangler 4.160.1 到 deploy/node_modules
  1. bash deploy/deploy.sh --login            # 浏览器 OAuth，凭据缓存在本机
  2. bash deploy/deploy.sh --check            # 离线校验，期望 checks=19 fails=0 ALL PASS
  3. bash deploy/deploy.sh --dry-run          # 打印将执行的命令，不联网
  4. bash deploy/deploy.sh                    # 发布 + 线上校验

每次更新后（Direct Upload 模式）：
  bash deploy/deploy.sh --build               # 重新构建两份 HTML 再发布
  （或改完自己跑 build 后直接 bash deploy/deploy.sh）

每次更新后（Cloudflare Git 集成模式，push 自动部署）：
  bash deploy/deploy.sh --build --prepare    # 只重建 deploy/public/，不部署
  git add deploy/public && git commit -m "update" && git push
  注意：Git 集成项目不要跑无参数 deploy.sh，那会用 wrangler 覆盖生产部署。

选项：
  --check              只做本地校验并组装 public/，不联网、不部署
  --prepare            --check 的语义别名：只准备待 git 提交的目录
  --dry-run            校验 + 打印命令，不联网、不部署
  --login              执行 wrangler login（首次或凭据过期时）
  --build              先跑 build.py 与 fromscratch/build_deep.py
  --target pages|workers   默认 pages
  --project NAME       Pages 项目名，默认 ft-lab-tutorials
  --branch NAME        生产分支，默认 main
  --url URL            覆盖校验用的站点根地址
  --verify-only URL    只做线上校验，不构建不部署
  --no-verify          部署后跳过线上校验
  --python PATH        构建用解释器，默认 python3
  --wrangler-version V 固定 wrangler 版本，默认 4.160.1
  --help               显示本说明

WRANGLER_BIN=/path/to/wrangler 可覆盖可执行文件。查找顺序：
  WRANGLER_BIN → deploy/node_modules/.bin/wrangler → npx wrangler@版本。
  本机 npx 可能报 [_npx concurrency.lock] 删除被拒；装本地依赖即可绕开。

CI 免交互：设置 CLOUDFLARE_API_TOKEN 与 CLOUDFLARE_ACCOUNT_ID，脚本跳过 whoami。
  Token 权限（Pages）：Account / Cloudflare Pages / Edit。
  Token 权限（Workers）：Account / Workers Scripts / Edit。

默认线上地址：
  https://ft-lab-tutorials.pages.dev/
  https://ft-lab-tutorials.pages.dev/from-scratch.html
  https://ft-lab-tutorials.pages.dev/tutorials.html
TXT
}

expect() {
  CHECKS=$((CHECKS + 1))
  if [ "$2" = "$3" ]; then
    printf 'PASS  %s\n' "$1"
  else
    FAILS=$((FAILS + 1))
    printf 'FAIL  %s  (got=%s want=%s)\n' "$1" "$2" "$3"
  fi
}

bytes_of() { wc -c < "$1" | tr -d ' '; }
digest_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}
count_of() { { grep -oE "$1" "$2" 2>/dev/null || true; } | wc -l | tr -d ' '; }
has_file() { if [ -f "$1" ]; then echo 1; else echo 0; fi; }
has_cmd() { if command -v "$1" >/dev/null 2>&1; then echo 1; else echo 0; fi; }
is_under() { if [ "$1" -lt "$2" ]; then echo 1; else echo 0; fi; }
is_over() { if [ "$1" -gt "$2" ]; then echo 1; else echo 0; fi; }

while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE="check"; shift ;;
    --prepare) MODE="check"; shift ;;
    --dry-run) MODE="dry"; shift ;;
    --login) MODE="login"; shift ;;
    --verify-only) MODE="verify"; SITE_URL="$2"; shift 2 ;;
    --build) DO_BUILD=1; shift ;;
    --no-verify) SKIP_VERIFY=1; shift ;;
    --target) TARGET="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --url) SITE_URL="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --wrangler-version) WRANGLER_VERSION="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) printf '未知参数：%s\n' "$1" >&2; usage; exit 2 ;;
  esac
done

if [ -n "${WRANGLER_BIN:-}" ]; then
  WRANGLER=("$WRANGLER_BIN")
elif [ -x "$DEPLOY_DIR/node_modules/.bin/wrangler" ]; then
  WRANGLER=("$DEPLOY_DIR/node_modules/.bin/wrangler")
else
  WRANGLER=(npx --yes "wrangler@$WRANGLER_VERSION")
fi

if [ "$TARGET" = "pages" ]; then
  [ -n "$SITE_URL" ] || SITE_URL="https://${PROJECT}.pages.dev"
elif [ "$TARGET" = "workers" ]; then
  :
else
  printf 'TARGET 只能是 pages 或 workers，收到：%s\n' "$TARGET" >&2; exit 2
fi

fetch_to() {
  # fetch_to URL TMPFILE：最多 5 次，返回最终 HTTP 状态码
  local url="$1" out="$2" i=1 code=000
  while [ "$i" -le 5 ]; do
    code="$(curl -sS --retry 2 --retry-delay 1 -o "$out" -w '%{http_code}' "$url" 2>/dev/null || echo 000)"
    [ "$code" = "200" ] && break
    [ "$i" -lt 5 ] && sleep "$RETRY_WAIT"
    i=$((i + 1))
  done
  printf '%s' "$code"
}

verify_site() {
  local base="$1" tmp code label
  tmp="$(mktemp)"
  for path in "/" "/from-scratch.html" "/tutorials.html"; do
    code="$(fetch_to "${base}${path}" "$tmp")"
    expect "HTTP 200 ${path}" "$code" "200"
  done
  fetch_to "${base}/" "$tmp" >/dev/null
  label="$(count_of '小模型训练实战' "$tmp")"
  expect "首页是主教程" "$(is_over "$label" 0)" "1"
  fetch_to "${base}/from-scratch.html" "$tmp" >/dev/null
  label="$(count_of '从零训练一个小语言模型' "$tmp")"
  expect "深版页面可访问" "$(is_over "$label" 0)" "1"
  rm -f "$tmp"
}

if [ "$MODE" = "login" ]; then
  printf '[login] %s login\n' "${WRANGLER[*]}"
  "${WRANGLER[@]}" login
  exit 0
fi

if [ "$MODE" = "verify" ]; then
  [ -n "$SITE_URL" ] || { printf 'verify 需要 URL\n' >&2; exit 2; }
  printf '[verify] %s\n' "$SITE_URL"
  verify_site "$SITE_URL"
  printf 'checks=%s fails=%s %s\n' "$CHECKS" "$FAILS" "$([ "$FAILS" -eq 0 ] && echo 'ALL PASS' || echo 'FAILED')"
  [ "$FAILS" -eq 0 ] || exit 1
  exit 0
fi

if [ "$DO_BUILD" -eq 1 ]; then
  printf '[build] %s build.py\n' "$PYTHON"
  ( cd "$ROOT" && "$PYTHON" build.py )
  printf '[build] %s fromscratch/build_deep.py\n' "$PYTHON"
  ( cd "$ROOT/fromscratch" && "$PYTHON" build_deep.py )
fi

printf '[check] 源文件\n'
expect "主教程存在" "$(has_file "$MAIN")" "1"
expect "独立深版存在" "$(has_file "$DEEP")" "1"
if [ "$(has_file "$MAIN")" = 1 ] && [ "$(has_file "$DEEP")" = 1 ]; then
  B_MAIN="$(bytes_of "$MAIN")"
  B_DEEP="$(bytes_of "$DEEP")"
  expect "主教程 > 100 KB" "$(is_over "$B_MAIN" 102400)" "1"
  expect "独立深版 > 100 KB" "$(is_over "$B_DEEP" 102400)" "1"
  expect "主教程 < 25 MiB" "$(is_under "$B_MAIN" "$MAX_FILE_BYTES")" "1"
  expect "独立深版 < 25 MiB" "$(is_under "$B_DEEP" "$MAX_FILE_BYTES")" "1"
  expect "主教程无未替换标记" "$(count_of '\{\{[a-z]+:' "$MAIN")" "0"
  expect "独立深版无未替换标记" "$(count_of '\{\{[a-z]+:' "$DEEP")" "0"
  expect "主教程无外部资源引用" "$(count_of '(src|href)="https?://|@import' "$MAIN")" "0"
  expect "独立深版无外部资源引用" "$(count_of '(src|href)="https?://|@import' "$DEEP")" "0"
  expect "主教程有 html 结束标签" "$(is_over "$(count_of '</html>' "$MAIN")" 0)" "1"
  expect "独立深版有 html 结束标签" "$(is_over "$(count_of '</html>' "$DEEP")" 0)" "1"
else
  B_MAIN=0; B_DEEP=0
fi

printf '[check] 环境\n'
if [ "$(has_cmd node)" = 1 ]; then
  NODE_MAJOR="$(node -p 'parseInt(process.versions.node)' 2>/dev/null || echo 0)"
else
  NODE_MAJOR=0
fi
expect "node 主版本 >= ${NODE_MIN_MAJOR}" "$(is_over "$NODE_MAJOR" "$((NODE_MIN_MAJOR - 1))")" "1"
expect "curl 可用" "$(has_cmd curl)" "1"
printf '[env] wrangler: %s\n' "${WRANGLER[*]}"

[ -d "$OUT_DIR" ] || mkdir -p "$OUT_DIR"
cp "$MAIN" "$OUT_DIR/index.html"
cp "$DEEP" "$OUT_DIR/from-scratch.html"
KB_MAIN=$((B_MAIN / 1024))
KB_DEEP=$((B_DEEP / 1024))
sed -e "s/__DATE__/$(date +%Y-%m-%d)/" \
    -e "s/__SIZE_MAIN__/${KB_MAIN} KB/" \
    -e "s/__SIZE_DEEP__/${KB_DEEP} KB/" \
    <<'HTML' > "$OUT_DIR/tutorials.html"
<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>模型训练教程 · 目录</title>
<style>
:root{--bg:#f7f6f3;--fg:#1d1d1f;--sub:#6b6b70;--card:#fff;--line:#e3e1dc;--acc:#9a3b2e}
@media (prefers-color-scheme:dark){:root{--bg:#17181a;--fg:#eeeae4;--sub:#a5a29c;--card:#212326;--line:#34363a;--acc:#e08a72}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:64px 24px 80px}
h1{font-size:30px;letter-spacing:-.02em;margin:0 0 8px}
.sub{color:var(--sub);margin:0 0 40px;font-size:15px}
a.card{display:block;background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:24px 26px;margin:0 0 16px;text-decoration:none;color:inherit;transition:border-color .15s}
a.card:hover{border-color:var(--acc)}
.tag{font-size:12px;letter-spacing:.12em;color:var(--acc);text-transform:uppercase}
h2{font-size:20px;margin:8px 0 6px}
p{color:var(--sub);font-size:14px;margin:0}
.meta{margin-top:14px;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--sub)}
footer{margin-top:44px;color:var(--sub);font-size:13px}
</style></head><body><div class="wrap">
<h1>模型训练教程</h1>
<p class="sub">两份单文件 HTML，无外部依赖，断网可打开。</p>
<a class="card" href="./index.html">
  <div class="tag">Part 01 · 主线</div>
  <h2>小模型训练实战 · 从零到专家</h2>
  <p>12 部分，微调 / LoRA / 从零训练 / 评测，Apple Silicon 16GB 路线。</p>
  <div class="meta">index.html · __SIZE_MAIN__</div>
</a>
<a class="card" href="./from-scratch.html">
  <div class="tag">Part 02 · 独立深版</div>
  <h2>从零训练一个小语言模型</h2>
  <p>9 部分 32 节，15 张原理图，71 项可复跑断言，配套 numpy 实验脚本。</p>
  <div class="meta">from-scratch.html · __SIZE_DEEP__</div>
</a>
<footer>更新 __DATE__ · 本地生成，Cloudflare 静态托管</footer>
</div></body></html>
HTML

printf '[check] 发布目录\n'
N_FILES="$(find "$OUT_DIR" -maxdepth 1 -type f | wc -l | tr -d ' ')"
TOTAL="$(find "$OUT_DIR" -maxdepth 1 -type f -print0 | xargs -0 wc -c | tail -1 | awk '{print $1}')"
expect "public 恰好 3 个文件" "$N_FILES" "3"
expect "文件数 < 20000" "$(is_under "$N_FILES" "$MAX_FILES")" "1"
expect "index.html 与源一致" "$(digest_of "$OUT_DIR/index.html")" "$(digest_of "$MAIN")"
expect "from-scratch.html 与源一致" "$(digest_of "$OUT_DIR/from-scratch.html")" "$(digest_of "$DEEP")"
expect "目录页含两个入口链接" "$(count_of 'class="card"' "$OUT_DIR/tutorials.html")" "2"

printf '[bundle] files=%s bytes=%s  main=%s B  deep=%s B  hub=%s B\n' \
  "$N_FILES" "$TOTAL" "$B_MAIN" "$B_DEEP" "$(bytes_of "$OUT_DIR/tutorials.html")"
printf 'checks=%s fails=%s %s\n' "$CHECKS" "$FAILS" "$([ "$FAILS" -eq 0 ] && echo 'ALL PASS' || echo 'FAILED')"
[ "$FAILS" -eq 0 ] || exit 1

if [ "$TARGET" = "pages" ]; then
  CMD="${WRANGLER[*]} pages deploy public --project-name=${PROJECT} --branch=${BRANCH}"
else
  CMD="${WRANGLER[*]} deploy --config ../wrangler.jsonc"
fi
printf '[plan] target=%s project=%s branch=%s\n' "$TARGET" "$PROJECT" "$BRANCH"
printf '[plan] %s\n' "$CMD"

if [ "$MODE" = "check" ] || [ "$MODE" = "dry" ]; then
  printf '[plan] 未执行（%s）\n' "$MODE"
  exit 0
fi

if [ -z "${CLOUDFLARE_API_TOKEN:-}" ]; then
  if ! "${WRANGLER[@]}" whoami >/dev/null 2>&1; then
    printf '未登录。先执行： bash deploy/deploy.sh --login\n' >&2
    exit 2
  fi
  printf '[auth] wrangler whoami OK\n'
else
  printf '[auth] CLOUDFLARE_API_TOKEN 已设置（CI 模式）\n'
fi

printf '[deploy] 开始\n'
if [ "$TARGET" = "pages" ]; then
  ( cd "$DEPLOY_DIR" && "${WRANGLER[@]}" pages deploy public --project-name="$PROJECT" --branch="$BRANCH" )
else
  ( cd "$DEPLOY_DIR" && "${WRANGLER[@]}" deploy --config ../wrangler.jsonc )
fi
printf '[deploy] 完成\n'

if [ "$SKIP_VERIFY" -eq 0 ]; then
  if [ -n "$SITE_URL" ]; then
    printf '[verify] %s\n' "$SITE_URL"
    verify_site "$SITE_URL"
    printf 'checks=%s fails=%s %s\n' "$CHECKS" "$FAILS" "$([ "$FAILS" -eq 0 ] && echo 'ALL PASS' || echo 'FAILED')"
    [ "$FAILS" -eq 0 ] || exit 1
  else
    printf '[verify] 未指定 --url，跳过（Workers 地址形如 https://NAME.SUBDOMAIN.workers.dev）\n'
  fi
fi

if [ -n "$SITE_URL" ]; then
  printf '[done] %s\n' "$SITE_URL"
  printf '[done] %s/from-scratch.html\n' "$SITE_URL"
  printf '[done] %s/tutorials.html\n' "$SITE_URL"
else
  printf '[done] 部署完成；用 --url <workers.dev 地址> 补做线上校验\n'
fi
