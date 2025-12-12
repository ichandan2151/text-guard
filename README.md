# Automated Risk & Compliance Analysis for IDB Project Documents

This project is an end-to-end NLP system that automatically analyzes **Inter-American Development Bank (IDB)** project documents to identify **risk signals, compliance issues, and relevant policy references**. It is designed to support governance and oversight teams by reducing manual document review and enabling scalable, consistent risk assessment.



## Problem Statement
IDB and similar development institutions produce thousands of unstructured documents such as supervision reports, disclosures, and audits. Manual review of these documents is time-consuming, inconsistent, and difficult to scale. This project applies modern NLP techniques to automatically flag high-risk content and compliance gaps, allowing analysts to focus on critical sections.


## Dataset Creation
Since no public dataset exists for governance and compliance risk detection, we created a **custom dataset from 50+ publicly available IDB project documents**.

- Documents were parsed and split by section structure (I, II, III, etc.)
- Each section was further chunked into ~140-character text segments
- Each chunk was manually labeled with:
  - `risk_label`: risk / no_risk
  - `risk_category` and `risk_level` (multi-label, JSON)
  - `compliance_status`: Compliant / Non-Compliant / Not Applicable
  - `policy_labels`: relevant IDB policy references

This resulted in a domain-specific dataset of ~5,000 labeled text chunks.


## Modeling Approach
Three fine-tuned **BERT-based models** were trained using Python, PyTorch, and Hugging Face Transformers:

1. **Risk Label Model**  
   Binary classifier to detect whether a text chunk indicates risk  
   *Accuracy: 0.93 | F1 (risk): 0.96*

2. **Risk Category Model**  
   Multi-label classifier to identify one or more risk types  
   *Micro F1-score: 0.61*

3. **Compliance Status Model**  
   Multi-class classifier to determine compliance status  
   *Accuracy: 0.97 | Macro F1-score: 0.96*

Predictions are generated at the chunk level and aggregated to produce document-level insights.



## AI Risk Summarization
To improve interpretability, the system integrates the **Gemini API** to generate concise, human-readable summaries explaining:
- Overall document risk profile
- Key risk drivers
- Why human review is recommended



## System Architecture
- **Frontend**: Next.js, React.js (hosted on Vercel)
- **Backend API**: FastAPI (Python) on Google Cloud Run
- **Models**: Fine-tuned BERT models
- **Model Storage**: Google Cloud Storage
- **Database**: Supabase (document metadata and predictions)

**Workflow:**  
Upload document → text extraction & chunking → BERT inference → Gemini summary → dashboard visualization



## Impact
This project demonstrates how NLP and transformer models can support **governance and compliance oversight at scale**. By automating risk detection for IDB project documents, the system improves efficiency, consistency, and transparency in document review workflows.



## Future Work
- Spanish-language document support  
- Expanded policy ontology  
- Advanced risk dashboards and visualizations  
- Cross-document risk trend analysis
