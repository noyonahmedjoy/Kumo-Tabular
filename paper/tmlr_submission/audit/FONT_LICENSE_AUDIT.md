# Font and Template Licence Audit

Date: 9 November 2026. No file was modified or deleted.

## Files concerned

45 files in `paper/final_submission/latex_source/` that are **present on the public GitHub `main` branch** (commit `863db6c`), although `.gitignore` excludes them:

- `ieeeaccess.cls`, `spotcolor.sty`, `logo.png`, `notaglinelogo.png`, `bullet.png`
- 40 `t1*` files: Type 1 fonts (`.pfb`), font metrics (`.tfm`), font maps (`.map`) and font definitions (`.fd`)

## Source

All 45 files are byte-identical (SHA-256) to files in the official IEEE Access LaTeX template, `ACCESS_latex_template_20260513-1-1.zip` (SHA-256 `60c7efc9db8ac9e8bdb31c550ad4e03cb6f258a878ececc0bc690b6203e45a67`), downloaded from https://ieeeaccess.ieee.org/authors/preparing-your-article/ on 1 November 2026. The zip contains no licence, readme or notice file.

## Licence statements found in the files

| Font family | Files | Notice embedded in the font |
|---|---|---|
| Times LT Std | `t1-times*.pfb` (4) | "...copyrighted (c) 1985–2002 Adobe Systems. All Rights Reserved. This software is the property of Adobe Systems Incorporated and its licensors, and may not be reproduced, used, displayed, modified, disclosed or transferred without the express written approval of Adobe." Outline data: "(c) 1981, 1999, 2002 Heidelberger Druckmaschinen AG. All rights reserved. ... may not be reproduced, used, displayed, modified, disclosed or transferred without the express written approval of Heidelberger Druckmaschinen AG." |
| Formata (OTF, Cond, Lt, Md) | `t1-formata-*.pfb` (11 of 12) | "Copyright (c) 1991–1999 Adobe Systems Incorporated. All Rights Reserved. Formata is either a registered trademark or trademark of Berthold Types Limited..." |
| Formata symbol subset | `t1-formata-regsymb.pfb` | "Created with FontForge 2.0"; no licence notice |
| ITC Giovanni Std | `t1-giovannistd-bookitalic.pfb` | "Copyright 1991, 2001, 2002 Adobe Systems Incorporated. All rights reserved. ITC is a trademark of International Typeface Corporation..." |
| Class, style, images | `ieeeaccess.cls`, `spotcolor.sty`, 3 PNG files | No licence statement |

`.tfm`, `.map` and `.fd` files are derived metric and configuration files for these fonts.

## Distribution restrictions

The Times LT Std notice explicitly prohibits reproduction or transfer without Adobe's written approval. The Formata and Giovanni notices reserve all rights. IEEE distributes these files to authors as part of its template, but no licence found grants authors the right to redistribute them. Republishing them in a public repository is therefore not supported by any licence we could find. This is not legal advice.

## Necessity

| Use | Needs these files? |
|---|---|
| TMLR manuscript (`paper/tmlr_submission/manuscript/`) | **No.** `tmlr.sty` uses the free Latin Modern fonts (`lmodern`, T1 encoding) |
| Local compilation of the IEEE Access manuscript | Yes, locally; anyone can obtain them from IEEE's template zip |
| The public repository | No. `paper/final_submission/latex_source/README.md` already tells readers to obtain them from IEEE |

## Options for the author (no action taken)

1. Remove the 45 files from the public repository in a new commit on `main` (`git rm --cached` keeps the local copies). This needs a remote push and the author's approval. The files would remain in the history of commit `863db6c` unless the history is rewritten, which is a further decision.
2. Keep them public, accepting the licensing risk, if the author holds or obtains permission.
