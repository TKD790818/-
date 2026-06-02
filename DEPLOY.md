# 雲端上線筆記

這套系統是 FastAPI + Python 排程 + Telegram，不適合只用 GitHub Pages 靜態網站承載。

## 建議方式

1. 先把程式碼推到 GitHub。
2. 到 Render 建立 Web Service，連接 GitHub repository。
3. 使用專案根目錄的 `render.yaml` 建立服務。
4. 在 Render 環境變數填入：
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `OPENAI_API_KEY`（目前可先留空）
5. 上線後會取得 `https://你的服務.onrender.com`，手機可直接開啟。

第一次啟動時，`scripts/start_web.sh` 會先嘗試建立一份雲端分析結果，再啟動網站。

## 重要提醒

- `.env` 不可上傳 GitHub。
- `artifacts_real/`、`reports_real/` 是本機資料與模型輸出，不上傳 GitHub。
- 若要雲端自動更新資料，後續要把排程改成雲端背景工作或 Cron Job。
- 目前 `render.yaml` 先建立網站服務與持久化資料夾，方便後續把自動更新接到雲端。
