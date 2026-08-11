# docs/demo-script.md
# CyberAdapt-LLM — Hackathon Demo Script
## 3–5 Minute Live Demonstration

---

> **Presenter note:** Open the dashboard at `http://localhost:3000` before you begin.  
> Backend must be running at `http://localhost:8000`.  
> All demo prompts are pre-loaded in the Live Comparison page (`/demo`).

---

## 🎯 The Flow

```
Problem
  ↓  Generic LLM limitation
  ↓  Cybersecurity-specific training data
  ↓  Domain adaptation (DAP + RAG)
  ↓  CyberAdapt-LLM
  ↓  Live side-by-side comparison
  ↓  Threat analysis demonstration
  ↓  Measured results
  ↓  Impact & limitations
```

---

## ⏱ Minute 0:30 — Hook: The Problem

**Say:**
> "Every SOC analyst today works alongside generic AI tools — but those tools were trained to write emails and summarise news, not to understand CVE databases, MITRE ATT&CK techniques, or DNS tunnelling patterns.  
> The result? Shallow answers, missed indicators, hallucinated mitigation steps."

**Show:** Navigate to `/chat` (Cyber Chat). Type:

```
What is Log4Shell and what JNDI exploit vector does it use?
```

> "This is our base model — distilgpt2 — a general-purpose LLM. Notice the response."

*(The base model gives a generic or incomplete response.)*

---

## ⏱ Minute 1:00 — Our Approach

**Say:**
> "CyberAdapt-LLM applies two complementary techniques:  
> **First** — Domain-Adaptive Pretraining (DAP): we fine-tune the base model on 
> curated cybersecurity corpora: CVE records, MITRE ATT&CK, NIST documents, OWASP.  
> **Second** — Retrieval-Augmented Generation (RAG): at inference time, we retrieve 
> the most relevant evidence chunks from our vector store and inject them into the prompt."

**Show:** Navigate to `/demo` (Live Comparison). Point to the two-panel layout.

> "Both models see the exact same RAG evidence. This isolates the contribution of domain adaptation from retrieval."

---

## ⏱ Minute 1:30 — Live Comparison Demo

**Click question 1 (CVE-2021-44228 / Log4Shell):**

```
What is CVE-2021-44228 (Log4Shell), and what attack vector does it exploit?
```

> "Watch both models answer simultaneously — same prompt, same evidence."

**Point out:**
- Left column: Base model response + latency
- Right column: CyberAdapt response + latency
- Scroll down to **Retrieved Evidence** section — show source attribution from NIST/CVE

> "The evidence panel shows exactly what cybersecurity knowledge was retrieved.  
> CyberAdapt was trained on this kind of data — so its domain vocabulary and 
> contextual reasoning are sharper."

**Scroll to Difference Summary:**
> "The model itself articulates what differs between the two answers."

---

## ⏱ Minute 2:30 — Threat Analysis Demo

**Navigate to `/threat`.**

**Paste this incident description:**
```
Unusual outbound HTTPS traffic detected from workstation WS-042 to IP 185.220.101.47 
at 3-minute intervals. Endpoint EDR flagged PowerShell execution with base64-encoded 
arguments: [System.Convert]::FromBase64String("aHR0cHM6Ly9ldmlsLmV4YW1wbGUuY29t"). 
Process parent: outlook.exe. User had received a zipped attachment 2 hours prior.
```

> "This simulates a real SOC alert. Watch CyberAdapt extract:
> - Threat classification  
> - Indicators of Compromise (IoCs)  
> - MITRE ATT&CK technique mapping  
> - Recommended containment actions"

**Point out** the IoC list, attack technique field, and evidence panel.

---

## ⏱ Minute 3:30 — Vulnerability Analysis Demo

**Navigate to `/vulnerability`.**

**Type:**
```
CVE-2021-26084 Confluence OGNL injection RCE
```

> "We query with a CVE ID. CyberAdapt returns severity, attack vector, affected components, and concrete mitigation steps — grounded in our RAG knowledge base."

---

## ⏱ Minute 4:00 — Measured Results

**Navigate to `/evaluation`.**

> "We ran our own benchmark experiments. These numbers come from our code — not the paper."

**Point to the bar chart and radar chart:**
- MCQ accuracy improvement (cybersecurity multiple-choice)
- Keyword recall on generative tasks
- Perplexity reduction (lower = better domain fit)

> "Perplexity is our clearest signal — it measures how 'surprised' the model is by cybersecurity text. CyberAdapt has lower perplexity, meaning the domain vocabulary is genuinely internalised."

**Important disclaimer:**
> "We are honest about limitations. On some questions the base model performs comparably — domain adaptation is not a universal win. This is a research demonstration, not a production system."

---

## ⏱ Minute 4:45 — Impact & What's Next

**Say:**
> "CyberAdapt-LLM demonstrates that:
> 1. Domain adaptation is measurably effective on cybersecurity tasks
> 2. RAG provides grounded, source-cited answers — reducing hallucination
> 3. The full pipeline runs on consumer hardware — no enterprise GPU cluster required
>
> Next steps: larger base model (Llama 3 8B), richer training corpus, LoRA for  
> parameter-efficient fine-tuning, and red-team evaluation against adversarial prompts."

**Close on the Dashboard `/`:**
> "Everything you've seen tonight runs locally on a single machine — backend, RAG, 
> model inference, and this interface. Thank you."

---

## 📋 Backup Prompts (if primary demos fail)

### Vulnerability fallback:
```
Describe the BlueKeep vulnerability (CVE-2019-0708) and its impact on unpatched Windows systems.
```

### Threat analysis fallback:
```
An IDS alert fired on port 4444 outbound from a Linux server. Metasploit default port 
detected. Access logs show POST /cgi-bin/login.cgi with oversized parameters 2 minutes prior.
```

### Chat fallback:
```
What is the difference between symmetric and asymmetric encryption, and when would you use each in a zero-trust network?
```

---

## ⚠️ Limitations (Be Proactive)

Mention these before judges ask:

1. **Base model is distilgpt2 (82M params)** — a small model chosen for accessibility. Production deployment would use Llama 3 or Mistral.
2. **Training corpus is a curated sample** — not exhaustive. Real deployment needs larger, continuously updated threat feeds.
3. **No real-time CVE ingestion** — the RAG knowledge base is static. Operational tools need live ingestion.
4. **Hallucination risk** — even with RAG, model may confidently state incorrect information. All output must be human-verified.
5. **Not a replacement for human analysts** — a decision-support tool, not autonomous action.

---

## 📚 Research Attribution

- **Domain-Adaptive Pre-Training:** Gururangan et al., "Don't Stop Pretraining" (ACL 2020)
- **RAG:** Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS 2020)
- **MITRE ATT&CK:** MITRE Corporation (https://attack.mitre.org/) — CC BY 4.0
- **NIST NVD:** National Vulnerability Database — Public Domain
- **OWASP:** Open Web Application Security Project — CC BY-SA 4.0
- **Sentence Transformers:** Reimers & Gurevych (2019) — Apache 2.0
- **FAISS:** Meta AI — MIT License

---

## 📦 Dataset Licenses

| Dataset | Source | License |
|---------|--------|---------|
| CVE records | NVD / MITRE | Public Domain (US Gov) |
| MITRE ATT&CK | MITRE Corporation | CC BY 4.0 |
| NIST Glossary | NIST | Public Domain |
| OWASP Top 10 | OWASP Foundation | CC BY-SA 4.0 |
| CWE | MITRE Corporation | Public Domain |

All datasets are used strictly for **defensive cybersecurity research**.

---

## 🏗 Model License

- **Base model (distilgpt2):** Hugging Face / OpenAI — MIT License  
- **Adapted weights:** Derived from distilgpt2. Same MIT License applies.  
- **Embedding model (all-MiniLM-L6-v2):** Sentence Transformers / Microsoft — Apache 2.0

---

*Generated for CyberAdapt-LLM Hackathon Demo — Phase 11*
