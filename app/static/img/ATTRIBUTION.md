# Landing page imagery

Every image in this directory is **AI-generated** (Google Gemini, August 2026)
from prompts written for this project. None depicts a real person, place,
product or copyrighted work, and none is derived from show artwork.

Declared here because this is assessed work: generated assets should be
identifiable as such rather than passed off as photography or commissioned
illustration.

| File | Section | Concept |
|---|---|---|
| `hero-field.webp` | Hero | A field of luminous points — the catalogue as a feature space |
| `taxonomy-drawers.webp` | The problem | Card-catalogue drawers, most labels blank — TMDB's missing genres |
| `prism-axes.webp` | Pinned panel 1 | Light through a prism separating into measured bands — one show decomposed into named axes |
| `explanation-threads.webp` | Explanation | Two forms joined by threads, bright where they agree |
| `steering-sliders.webp` | Steering | Precision sliders, one mid-motion |
| `catalogue-grid.webp` | Pinned panel 2 | A vast grid of tiles — 500 shows, a few lit |
| `icon-vote.webp` | Feature card | Up / neutral / down controls |
| `icon-review.webp` | Feature card | Pen nib on a blank card |
| `icon-watchlist.webp` | Feature card | Bookmark in a stack |
| `icon-history.webp` | Feature card | Clock sweeping backwards |

## Processing

Originals are kept in `homepage images/` at the repository root as JPEG, at the
resolution Gemini produced. The files served here are WebP at quality 82, with
the four icons downscaled from 1024px to 256px — they render at roughly 54px,
so 1024 was 16x more pixels than any display needs.

That took the set from **4.9 MB to 0.39 MB, a 92% reduction**, with no visible
difference at display size. Regenerate with Pillow if the originals change.

## Note on the show posters

The poster thumbnails elsewhere on the landing page are **not** generated. They
come from the TMDB API under its terms of use, and the required attribution
appears in the page footer.
