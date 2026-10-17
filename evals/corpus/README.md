# Eval corpus

Three public-domain texts, rendered to PDF so the eval runs through the same
PDF extraction path as `POST /upload`.

| File | Source | Kept |
|---|---|---|
| `us_constitution.pdf` | Project Gutenberg eBook #5, *The United States Constitution* (1787) | Preamble through Article VII and George Washington's attestation; signatory list dropped |
| `declaration_of_independence.pdf` | Project Gutenberg eBook #1, *The Declaration of Independence* (1776) | Full text from "IN CONGRESS, July 4, 1776" |
| `elements_of_style.pdf` | Project Gutenberg eBook #37134, William Strunk Jr., *The Elements of Style* (1918) | Chapters I to V; spelling list and exercises dropped |

The works are in the public domain. Project Gutenberg headers, footers and
transcriber notes were removed, curly quotes were straightened and the
`_italic_` / `=bold=` markup was stripped. The cleaned text is in `source/`.

Rebuild the PDFs with:

```bash
pip install -r requirements-dev.txt
python -m evals.build_corpus
```

The structure differs on purpose: the Constitution has articles and sections,
*The Elements of Style* has chapters and numbered rules, and the Declaration is
unbroken prose. That spread is what separates the chunking strategies.
