import { NextResponse } from "next/server";
import { GoogleGenAI } from "@google/genai";

export async function POST(req: Request) {
  try {
    const {
      fileName,
      createdAt,
      totalChunks,
      riskChunks,
      noRiskChunks,
      riskScore,
    } = await req.json();

    if (!process.env.GEMINI_API_KEY) {
      return NextResponse.json(
        { error: "GEMINI_API_KEY is not configured on the server." },
        { status: 500 }
      );
    }

    const ai = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
    });

    const humanReadableScore =
      typeof riskScore === "number" ? (riskScore * 100).toFixed(1) + "%" : "N/A";

    const prompt = `
You are a senior Governance, Risk, and Compliance (GRC) analyst specializing in 
multilateral development bank projects (e.g., IDB, World Bank, ADB). 
Your task is to generate a clear, reliable *Risk & Compliance Insight* for a project document
based ONLY on the metadata provided below.

The goal is to help a non-technical risk officer quickly understand 
the document’s risk profile and what actions may be needed.

--------------------------------
📌 *RISK THEMES TO ANALYZE (Always Cover All Four):*

1. *Governance & Institutional Capacity*
   - Execution capacity, project management, staffing, coordination challenges,
     administrative bottlenecks, prior experience with MDB-funded projects.

2. *Fiduciary (Procurement & Financial Management)*
   - Procurement weaknesses, fraud/corruption exposure, oversight needs,
     financial control risks, contract management risks.

3. *Environmental & Social Risks*
   - Environmental impacts, community risks, climate vulnerability,
     stakeholder engagement issues, safeguard compliance.

4. *Integrity & Policy Compliance*
   - Transparency, accountability, anti-corruption measures,
     regulatory compliance, disclosure requirements, policy adherence.

--------------------------------
📌 *HOW TO WRITE THE INSIGHT*

- Use a *short introductory paragraph* summarizing the document’s overall risk.
- Then provide *FOUR bullet paragraphs*, one for each theme above.
- Use MDB-appropriate terminology (e.g., "moderate risk", "capacity constraints", "oversight requirements").
- Do NOT invent policy numbers, exact regulations, or fictional safeguards.
- Base the tone on MDB risk summaries: concise, factual, forward-looking.
- If the risk score is extremely high (>85%), emphasize the intensity of review needed.
- If the risk score is low (<30%), emphasize that risks exist but appear manageable.
- Use the chunk counts to infer breadth:
  • Many risk chunks = pervasive concerns  
  • Few/no risk chunks = isolated or narrow issues  
- Use only information that logically follows from the metadata.

--------------------------------
📌 *DOCUMENT METADATA (the ONLY input you have):*

- File name: ${fileName || "Unknown"}
- Uploaded at: ${createdAt || "Unknown"}
- Total text chunks: ${totalChunks ?? "Unknown"}
- Risk chunks: ${riskChunks ?? "Unknown"}
- No-risk chunks: ${noRiskChunks ?? "Unknown"}
- Overall risk score: ${humanReadableScore}
- Source link: ${Document || "N/A"}

--------------------------------
📌 *OUTPUT FORMAT (MANDATORY)*

Write your answer as:

*AI Risk Insight*

[Intro paragraph – 2–3 sentences explaining overall risk profile]

* Governance & Institutional Capacity:[3–5 sentences based on metadata]  
* Fiduciary (Procurement & Financial Management): [3–5 sentences]  
* Environmental & Social Risks: [3–5 sentences]  
* Integrity & Policy Compliance: [3–5 sentences]

--------------------------------
Now generate the insight.
`.trim();

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
    });

    // `text` is a PROPERTY, not a function
    const text = (response.text ?? "") as string;

    return NextResponse.json({ text });
  } catch (err) {
    console.error("Gemini summary error:", err);
    return NextResponse.json(
      { error: "Failed to generate Gemini summary." },
      { status: 500 }
    );
  }
}
