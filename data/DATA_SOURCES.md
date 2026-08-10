# CyberAdapt-LLM — Data Sources

All content in `data/raw/` must come from one of the approved categories below.
**Never add** content that requires purchase, a paywalled login, or whose license
forbids derivative ML training use.

---

## Approved Source Categories

| # | Category | License | Notes |
|---|----------|---------|-------|
| 1 | US Government publications (NIST, CISA, NSA) | Public Domain (17 U.S.C. §105) | No copyright on USG works |
| 2 | MITRE ATT&CK Framework | CC BY 4.0 | Attribution required |
| 3 | MITRE CWE (Common Weakness Enumeration) | Public domain / free use | See mitre.org/legal |
| 4 | NVD / CVE descriptions | Public domain | Produced by NIST |
| 5 | OWASP documents | CC BY-SA 3.0 / 4.0 | Attribution + ShareAlike |
| 6 | IETF RFCs (security-related) | IETF Trust (permissive) | Redistribution allowed with attribution |
| 7 | Wikipedia cybersecurity articles | CC BY-SA 4.0 | Attribution + ShareAlike |
| 8 | Project Gutenberg texts | Public Domain | Pre-1928 or confirmed PD |

---

## Specific Sources Used in `data/raw/`

### nist_glossary_sample.txt
- **Source**: NIST SP 800-12 Rev. 1, NIST IR 7298 Rev. 3 (Glossary of Key IS Terms)
- **URL**: https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.7298r3.pdf
- **License**: Public Domain (US Government work)
- **Content**: Cybersecurity terminology definitions

### nist_sp800_12_sample.txt
- **Source**: NIST SP 800-12 Rev. 1 "An Introduction to Information Security"
- **URL**: https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-12r1.pdf
- **License**: Public Domain (US Government work)
- **Content**: Core information security principles

### owasp_top10_sample.txt
- **Source**: OWASP Top 10 — 2021 Edition
- **URL**: https://owasp.org/Top10/
- **License**: CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/)
- **Attribution**: © OWASP Foundation
- **Content**: Web application security risks descriptions

### mitre_attack_sample.txt
- **Source**: MITRE ATT&CK Enterprise Matrix v14
- **URL**: https://attack.mitre.org/
- **License**: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
- **Attribution**: © 2024 The MITRE Corporation
- **Content**: Adversary tactics and technique descriptions

### cve_sample.jsonl
- **Source**: National Vulnerability Database (NVD) — sample CVE descriptions
- **URL**: https://nvd.nist.gov/
- **License**: Public Domain (NVD/NIST, US Government work)
- **Content**: CVE description text (synthetic examples representing NVD format)

### mitre_cwe_sample.jsonl
- **Source**: MITRE Common Weakness Enumeration (CWE)
- **URL**: https://cwe.mitre.org/
- **License**: Public Domain / Free for any use (see https://cwe.mitre.org/about/termsofuse.html)
- **Content**: Weakness descriptions and extended descriptions

---

## Sources NOT Permitted

- DEFCON/Black Hat presentation slides (copyright retained by authors)
- Academic papers behind paywalls (ACM, IEEE, Springer — unless open access)
- Commercial security reports (Mandiant, CrowdStrike, etc.)
- Any dataset whose license prohibits ML training use
- Books available for purchase (e.g., "The Web Application Hacker's Handbook")

---

## Adding New Sources

Before adding any new source to `data/raw/`, verify:

1. [ ] The license explicitly allows redistribution and derivative works
2. [ ] Attribution requirements are documented here
3. [ ] No personal data or PII is present
4. [ ] The content is cybersecurity-relevant
5. [ ] Update this file with the source details

---

## Data Lineage

All processing is logged. The `source` and `license` fields in
`data/processed/cybersecurity_corpus.jsonl` trace every record back to its
origin. The pipeline script (`training/prepare_dataset.py`) never strips
metadata — it only adds it.
