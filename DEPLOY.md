# 雲端上線筆記

這套系統是 FastAPI + Python 排程 + Telegram，不適合只用 GitHub Pages 靜態網站承載。

## 建議方式

1. 先把程式碼推到 GitHub。
2. 到 Render 建立 Web Service，連接 GitHub repository。
3. 使用專案根目錄的 `render.yaml` 建立服務。
4. 在 Render 環境變數填入：
   - `TELEGRAM_BOT_TOKEN`（可先留空，之後要推播再補）
   - `TELEGRAM_CHAT_ID`（可先留空，之後要推播再補）
   - `OPENAI_API_KEY`（目前可先留空）
5. 上線後會取得 `https://你的服務.onrender.com`，手機可直接開啟。

第一次啟動時，`scripts/start_web.sh` 會先啟動網站，避免免費機器因為先跑完整 AI 分析而記憶體不足。
如果 Telegram 尚未填 Token 或 Chat ID，網站仍可正常部署，只是不會送出推播。

`cloud_seed/artifacts_real/` 會提供一份本機已分析完成的壓縮預載資料；Render 第一次啟動時會自動還原到 `artifacts_real/`，讓網站先有資料可看。
Render Free 只有 512MB 記憶體，建議先讓網站直接啟動，不要在服務啟動前跑完整 AI 分析。
若未來升級付費機器或改用背景工作，可設定環境變數 `RUN_STARTUP_ANALYSIS=true`，讓服務啟動前先嘗試建立分析結果。

## 重要提醒

- `.env` 不可上傳 GitHub。
- `artifacts_real/`、`reports_real/` 是本機資料與模型輸出，不上傳 GitHub。
- `cloud_seed/` 是雲端初始資料包，可用本機最新分析結果重新產生後推送。
- 若要雲端自動更新資料，後續要把排程改成雲端背景工作或 Cron Job。
- 目前 `render.yaml` 先建立網站服務與持久化資料夾，方便後續把自動更新接到雲端。
