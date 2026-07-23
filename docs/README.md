# scCoBench website

Single-page static site for the scCoBench paper, with:

- **AI Method Recommender** — rule-based decision engine that suggests a co-expression pipeline based on your dataset characteristics (no API key needed). Answers include the manuscript's evidence with figure citations, a step-by-step protocol, the expected quantitative gain, the limitations the authors raise, and the paper's Conclusion verbatim.
- **Manuscript knowledge base** — the paper's findings, Figure 8 guideline, limitations and Conclusion are embedded in `index.html` as a single `PAPER` object. Both the rule-based recommender and the LLM advisor read from it, so they cannot drift apart. Update the paper in one place.
- **LLM dataset interpreter** — calls *your own* OpenAI / Anthropic API key. The manuscript is passed as system context (~2.8k tokens) and the model is instructed to answer only from it, cite the figure behind each claim, and say so when a question falls outside what the paper tested. Keys stay client-side; never sent to this site.
- **Interactive calculator** — paste two gene expression vectors and the page computes Pearson, Spearman, Kendall, subset (cells expressing both), log-normalised, and pseudo-bulk correlations directly in the browser.
- **Script generators** — download ready-to-run Python (AE / VAE imputation, pseudo-bulk pipeline) and R (CS-CORE) scripts pre-filled with your gene names.
- **Browsable benchmark tables** — Pearson and Spearman recovery rates across 7 plant species × 5 expression bins, colour-coded.
- **Decision guide** — step-by-step Figure 9 in HTML form.

## View locally

```bash
cd website
python -m http.server 8000
# open http://localhost:8000
```

Or just double-click `index.html` — it works as a single-file static page.

## Deploy to GitHub Pages

1. Push this `website/` directory to your repo (or a separate repo).
2. Settings → Pages → Branch: `main`, Folder: `/website` (or `/`).
3. Wait ~30 seconds; site goes live at `https://<user>.github.io/<repo>/`.

For a custom domain, add a `CNAME` file with your domain and configure DNS.

## Deploy to Netlify / Vercel

Drag-and-drop the `website/` folder onto Netlify Drop (https://app.netlify.com/drop). Live in seconds. No build step required.

## Files

- `index.html` — the entire website (HTML + CSS + JavaScript in one file). No build dependencies.
- `README.md` — this file.

## Privacy

- The page does **not** make network requests on its own.
- API keys for the LLM interpreter are stored only in the browser's JavaScript memory for the current tab. Refresh to clear.
- The rule-based recommender, calculator, and script generators run entirely client-side.

## Limitations

- Browser Kendall tau is naive O(n²) — skipped for vectors longer than 5,000 cells.
- Pseudo-bulk in the calculator uses a simple sort-and-bin proxy (not full k-means). For the real pipeline, download the generated `scCoBench_pseudobulk.py`.
- Deep-learning imputation (AE/VAE) and CS-CORE cannot run in the browser; the page generates Python/R scripts to run locally instead.
- The free-text LLM box is the only feature that needs a key, and it uses the visitor's own. Everything else — including the full manuscript-grounded recommender — runs with no key and no network access.
- Anthropic calls are sent with the `anthropic-dangerous-direct-browser-access` header, which permits direct browser requests. Keys that are not enabled for browser access will still fail CORS; use OpenAI or a small proxy in that case.
