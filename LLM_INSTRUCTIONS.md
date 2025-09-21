### LLM Authoring Checklist for This Repo (Jekyll)

Use this checklist every time you create or edit content. Keep it technical, concise, and consistent with the site.

---

## General Rules

- **Be technical, no analogies:** Avoid real-life analogies; state facts, reasons, and data.
- **Preserve existing style:** Do not reflow/reformat unrelated lines. Keep current indentation and spacing.
- **Headings:** Prefer `###` and `##`. Avoid `#` top-level.
- **Emphasis & bullets:** Bold key phrases; use short, skimmable bullets.
- **Code fences:** Use proper language tags (e.g., `python`, `bash`). Keep snippets minimal and runnable if possible.
- **Files/URLs:** Wrap file and directory names in backticks; use markdown links for URLs.
- **Tables:** Use only when they aid quick comparison (e.g., "At a glance").
- **Images:** Place under `assets_files/blogs/<YYYY-MM-DD-slug>/` and reference with alt text.
- **No unrelated edits:** Don’t rename or move existing files unless explicitly asked.

---

## New Blog Post (`_posts`)

1. **Filename format:** `YYYY-MM-DD-your-slug.md`
2. **Front matter (required):**
```yaml
---
layout: post
title: Your Clear, Descriptive Title
date: YYYY-MM-DD
author: Sagar Desai
categories: [PrimaryCategory, OptionalSecondary]
tags: [tag1, tag2, tag3]
---
```
3. **Recommended structure:**
   - `## TL;DR` — 2–4 bullets with the core findings.
   - `## At a glance` — small comparison table if relevant.
   - `## Why` — brief, technical reasoning (no analogies).
   - `## Where differences grow` — contexts and boundaries.
   - `## Metric/Method` — precise definitions and formulas.
   - `## Minimal repro` — short, focused code block.
   - `## References` — repo/files/links.
   - `## Takeaways` — actionable bullets.
4. **Images:**
   - Save to: `assets_files/blogs/YYYY-MM-DD-your-slug/`.
   - Reference: `![Alt text](/assets_files/blogs/YYYY-MM-DD-your-slug/image.png)`.
5. **Internal links:** Prefer relative links to files in this repo; keep anchors short.

---

## New Page (root)

1. **Filename:** `about.md`, `projects.md`, etc.
2. **Front matter:**
```yaml
---
layout: page
title: Page Title
permalink: /custom-slug/  # optional
---
```
3. **Content:** Use `###` headings, concise bullets, and links to related posts/projects.

---

## New Project (`_projects`)

1. **Filename:** `project-name.md`
2. **Front matter:**
```yaml
---
layout: page
title: Project Name
---
```
3. **Content checklist:** overview, goals, links, media under `assets_files/projects/...` if applicable.

---

## Code & Data Blocks

- **Language tags:** `python`, `bash`, `json`, `yaml`, etc.
- **Scope:** Only include the minimal code needed; avoid inline commentary inside code.
- **Formulas:** Use inline code for simple formulas, e.g., `bytes_per_token = total_utf8_bytes / total_tokens`.

---

## Style Guardrails

- **No analogies or fluff;** keep claims testable and scoped.
- **Use precise terms;** prefer "bytes/token" over vague phrasing.
- **Use separators `---`** between major sections for readability.
- **Tables** only when they improve scannability.

---

## Asset Handling

- **Blogs:** `assets_files/blogs/YYYY-MM-DD-slug/` per post.
- **Projects:** `assets_files/projects/<project>/` if needed.
- **Use descriptive filenames** and include alt text in markdown.

---

## Quick Templates

### Post skeleton
```markdown
---
layout: post
title: Descriptive Title
date: YYYY-MM-DD
author: Sagar Desai
categories: [Category]
tags: [tag1, tag2]
---

## TL;DR
- Key point 1
- Key point 2

## At a glance
| Option | Metric A | Metric B |
| --- | --- | --- |
| X | ... | ... |

## Why
- Technical reason(s) without analogies

## Metric/Method
- Definition(s) and formula(s)

## Minimal repro
```python
# focused, runnable snippet
```

## References
- [Link text](https://example.com)

## Takeaways
- Actionable point 1
- Actionable point 2
```

### Page skeleton
```markdown
---
layout: page
title: Page Title
permalink: /custom-slug/
---

### Section
- Bullet
```
