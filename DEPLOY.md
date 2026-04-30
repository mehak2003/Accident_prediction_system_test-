# Deployment Guide — Free & Always-On

Backend → **Hugging Face Spaces (Docker)** · Frontend → **Vercel**

Both are free, always-on, auto-deploy from GitHub.

---

## 1. Push code to GitHub

```bash
git add -A
git commit -m "chore: deployment config"
git push
```

---

## 2. Deploy backend on Hugging Face Spaces

1. Sign up / sign in at https://huggingface.co.
2. **New Space** → name it (e.g. `accident-api`) → **SDK: Docker** → **Public** → Create.
3. Link / import this GitHub repo (Space settings → "Link to GitHub repo"),
   or `git clone` the Space and copy repo contents in.
4. Copy `README_HF.md` from this repo to the Space's root as `README.md`
   (it contains the required `sdk: docker` + `app_port: 7860` front matter).
5. Wait for build (~3–5 min). Space becomes **Running**.
6. Your API base URL is:

   ```
   https://<your-username>-<space-name>.hf.space
   ```

   Health check: `/api/health`

7. (Recommended) In **Space Settings**:
   - **Variables**: `CORS_ORIGINS` = `https://<your-vercel-app>.vercel.app`
   - **Persistent Storage**: enable the free tier so uploaded CSVs & trained
     `.joblib` models survive restarts.

---

## 3. Deploy frontend on Vercel

1. Sign in at https://vercel.com with GitHub.
2. **Add New → Project** → select this repo.
3. Settings:
   - **Root Directory**: `frontend`
   - **Framework**: Vite (auto-detected)
   - **Environment Variables**:

     | Key             | Value                                              |
     |-----------------|----------------------------------------------------|
     | `VITE_API_BASE` | `https://<your-username>-<space>.hf.space/api`     |

4. **Deploy**. You get `https://<project>.vercel.app`.
5. Back in the HF Space, set `CORS_ORIGINS` to this URL and **Factory Rebuild**.

---

## 4. Verify

- Open the Vercel URL.
- **Data Manager** → upload `road.csv` → **Run Pipeline**.
- Map / Analytics / Predict pages should populate.
- Direct API check: `https://<space>.hf.space/api/health`.

---

## Local Docker test (optional)

```bash
docker build -t accident-api .
docker run --rm -p 7860:7860 accident-api
# → http://localhost:7860/api/health
```

---

## Environment variables reference

### Backend (HF Space)
| Var            | Default | Purpose                                  |
|----------------|---------|------------------------------------------|
| `FLASK_PORT`   | `7860`  | Port (HF Spaces requires 7860)           |
| `FLASK_DEBUG`  | `0`     | Debug flag                               |
| `CORS_ORIGINS` | `*`     | Comma-separated allowed frontend origins |

### Frontend (Vercel)
| Var             | Purpose                                |
|-----------------|----------------------------------------|
| `VITE_API_BASE` | Full backend base URL incl. `/api`     |
