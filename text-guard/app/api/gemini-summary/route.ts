// app/api/gemini-summary/route.ts
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
You are a senior risk & compliance analyst for multilateral development bank projects.

Given the following document metadata and model outputs, write a concise narrative
assessment for a non-technical user. Explain:

1) What the risk profile looks like overall,
2) How to interpret the risk score and risk/no-risk counts,
3) What follow-up actions or checks a human reviewer should consider.

Keep it to 2–3 short paragraphs, friendly but professional.

Document:
- File name: ${fileName || "Unknown"}
- Uploaded at: ${createdAt || "Unknown"}
- Total text chunks: ${totalChunks ?? "Unknown"}

- Risk chunks: ${riskChunks ?? "Unknown"}
- No-risk chunks: ${noRiskChunks ?? "Unknown"}
- Overall risk score: ${humanReadableScore}
`;

const response = await ai.models.generateContent({
  model: "gemini-2.5-flash",
  contents: prompt,
});

// TS thinks `text` might be undefined, so guard it
const text =
  typeof (response as any).text === "function"
    ? (response as any).text()
    : "";


    return NextResponse.json({ text });
  } catch (err) {
    console.error("Gemini summary error:", err);
    return NextResponse.json(
      { error: "Failed to generate Gemini summary." },
      { status: 500 }
    );
  }
}
