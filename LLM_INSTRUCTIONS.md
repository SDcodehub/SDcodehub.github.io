### LLM Authoring Checklist for This Repo (Jekyll)

Use this checklist every time you create or edit content. Keep it technical, concise, and consistent with the site.

---

## General Rules

LLM Authoring Checklist (short)
- Be technical; avoid analogies. Focus on facts, reasons, data.
- Preserve existing style; minimal changes; do not reflow unrelated lines.
- Headings: prefer ####. Use bold for key phrases; keep bullets skimmable.
- Code fences: add language tags (e.g., python, bash). Keep snippets minimal.
- Files/dirs in backticks; use markdown links for URLs.
- Tables only when they aid quick comparison.
- Images: place under assets_files/blogs/<YYYY-MM-DD-slug>/ with alt text.
- No unrelated edits/renames/moves. Do not use ':' in the title.

Math rendering best practices
- Display math: use $$ ... $$, flush-left at column 1, with blank lines around.
- Inline math: use $...$; keep units/words (bytes, GB, elements) outside math.
- Avoid \\text{} in inline math; prefer plain text or \\mathrm for short symbols.
- Do not indent math blocks inside lists; avoid placing math in code fences.
- Escape underscores outside math (e.g., batch\\_size) or wrap identifiers in backticks.

